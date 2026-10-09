use std::path::{Path, PathBuf};

use burn::{module::Module, store::ModuleRecord, tensor::Device};
use qwen_image::{
    qwen_image::BlockProvider,
    qwen_image_modules::{QwenImageTransformerBlock, QwenImageTransformerBlockConfig},
};

const NUM_TRANSFORMER_BLOCKS: usize = 32;

// On unified memory, holding every block in RAM as well as on the GPU pushes a 16 GB machine
// into swap; reading each block from disk when it is needed leaves the caching to the OS,
// which drops file pages instead of swapping them.
pub struct TransformerBlocksReadFromDiskWhenNeeded {
    block_files: Vec<PathBuf>,
    reusable_block_on_device: Option<QwenImageTransformerBlock>,
}

impl TransformerBlocksReadFromDiskWhenNeeded {
    pub fn new(converted_weights_directory: &Path, device: &Device) -> Self {
        let block_files: Vec<PathBuf> = (0..NUM_TRANSFORMER_BLOCKS)
            .map(|block_index| converted_weights_directory.join(format!("block_{block_index}.bpk")))
            .collect();
        if let Some(missing_block_file) = block_files.iter().find(|file| !file.exists()) {
            panic!(
                "{} is missing: run convert_weights_for_webgpu first",
                missing_block_file.display()
            );
        }
        let reusable_block_on_device =
            QwenImageTransformerBlockConfig::new(32 * 128, 32, 128, 3, 1e-6).init(device);
        Self {
            block_files,
            reusable_block_on_device: Some(reusable_block_on_device),
        }
    }
}

impl BlockProvider for TransformerBlocksReadFromDiskWhenNeeded {
    fn num_layers(&self) -> usize {
        self.block_files.len()
    }

    fn get(&mut self, layer: usize) -> &QwenImageTransformerBlock {
        let block_file = &self.block_files[layer];
        let record = ModuleRecord::load(block_file)
            .unwrap_or_else(|error| panic!("could not read {}: {error:?}", block_file.display()));
        let block = self
            .reusable_block_on_device
            .take()
            .expect("the reusable block is put back after every load");
        self.reusable_block_on_device
            .insert(block.load_record(record))
    }
}
