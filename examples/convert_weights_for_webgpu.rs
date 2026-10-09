use std::{collections::HashSet, path::Path};

use burn::{
    module::{Module, Quantizer},
    tensor::{
        DType, Device,
        quantization::{Calibration, QuantScheme, ScaleDtype},
    },
};
use burn_store::{
    BurnpackStore, FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter,
    SafetensorsStore,
};
use qwen_image::{
    qwen_image::QwenImageTransformerModelConfig,
    qwen_image_modules::QwenImageTransformerBlockConfig, vae::model::AutoencoderKLQwenImageConfig,
};

const TRANSFORMER_DIRECTORY: &str = "./weights/transformer";
const TRANSFORMER_SHARDS: [&str; 2] = [
    "diffusion_pytorch_model-00001-of-00002.safetensors",
    "diffusion_pytorch_model-00002-of-00002.safetensors",
];
const VAE_SAFETENSORS: &str = "./weights/vae/diffusion_pytorch_model.safetensors";
const OUTPUT_DIRECTORY: &str = "./out";
const NUM_TRANSFORMER_BLOCKS: usize = 32;
const VALUES_SHARING_ONE_QUANTIZATION_SCALE: u8 = 64;

fn main() {
    let device = Device::wgpu(Default::default());
    let output_directory = Path::new(OUTPUT_DIRECTORY);
    std::fs::create_dir_all(output_directory).unwrap();

    convert_transformer_shell_to_f32(&device, output_directory);
    convert_transformer_blocks_to_int8(&device, output_directory);
    convert_vae_to_f32(&device, output_directory);
}

fn convert_transformer_shell_to_f32(device: &Device, output_directory: &Path) {
    let mut shell = QwenImageTransformerModelConfig::new([16, 56, 56])
        .with_num_layers(0)
        .init(device);
    load_every_tensor_from_shards(&mut shell, "shell", |shard| {
        open_shard_casting_to_f32(shard).allow_partial(true)
    });
    save_as_burnpack(&shell, &output_directory.join("shell.bpk"));
}

fn convert_transformer_blocks_to_int8(device: &Device, output_directory: &Path) {
    let block_config = QwenImageTransformerBlockConfig::new(32 * 128, 32, 128, 3, 1e-6);
    let quant_scheme =
        QuantScheme::default().per_block([VALUES_SHARING_ONE_QUANTIZATION_SCALE], ScaleDtype::F32);
    let mut quantizer = Quantizer::new(Calibration::MinMax, quant_scheme);

    for block_index in 0..NUM_TRANSFORMER_BLOCKS {
        let mut block = block_config.init(device);
        load_every_tensor_from_shards(&mut block, &format!("block {block_index}"), |shard| {
            open_shard_casting_to_f32(shard)
                .with_key_remapping(&format!(r"^transformer_blocks\.{block_index}\."), "")
                .allow_partial(true)
        });
        let block = block.quantize_weights(&mut quantizer);
        save_as_burnpack(
            &block,
            &output_directory.join(format!("block_{block_index}.bpk")),
        );
    }
}

fn convert_vae_to_f32(device: &Device, output_directory: &Path) {
    let mut vae = AutoencoderKLQwenImageConfig::new().init(device);
    let mut store = SafetensorsStore::from_file(VAE_SAFETENSORS)
        .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)));
    let result = vae.load_from(&mut store).unwrap();
    assert!(
        result.missing.is_empty() && result.errors.is_empty(),
        "vae: {result}"
    );
    save_as_burnpack(&vae, &output_directory.join("vae.bpk"));
}

fn open_shard_casting_to_f32(shard: &str) -> SafetensorsStore {
    SafetensorsStore::from_file(Path::new(TRANSFORMER_DIRECTORY).join(shard))
        .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(DType::F32)))
}

fn load_every_tensor_from_shards<M: ModuleSnapshot>(
    module: &mut M,
    module_description: &str,
    open_shard: impl Fn(&str) -> SafetensorsStore,
) {
    let mut loaded_tensor_paths = HashSet::new();
    let mut tensor_paths_missing_from_some_shard = Vec::new();
    for shard in TRANSFORMER_SHARDS {
        let result = module.load_from(&mut open_shard(shard)).unwrap();
        assert!(result.errors.is_empty(), "{module_description}: {result}");
        loaded_tensor_paths.extend(result.applied);
        tensor_paths_missing_from_some_shard
            .extend(result.missing.into_iter().map(|(path, _)| path));
    }
    let tensor_paths_missing_from_every_shard: Vec<_> = tensor_paths_missing_from_some_shard
        .into_iter()
        .filter(|path| !loaded_tensor_paths.contains(path))
        .collect();
    assert!(
        tensor_paths_missing_from_every_shard.is_empty(),
        "{module_description}: no shard holds {tensor_paths_missing_from_every_shard:?}"
    );
}

fn save_as_burnpack<M: ModuleSnapshot>(module: &M, path: &Path) {
    module
        .save_into(&mut BurnpackStore::from_file(path).overwrite(true))
        .unwrap();
    println!("saved {}", path.display());
}
