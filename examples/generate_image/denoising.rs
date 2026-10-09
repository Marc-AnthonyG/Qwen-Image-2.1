use std::{path::Path, time::Instant};

use burn::{
    module::Module,
    store::ModuleRecord,
    tensor::{Device, Distribution, FloatDType, Tensor, s},
};
use qwen_image::{
    qwen_image::{BlockProvider, QwenImageTransformerModel, QwenImageTransformerModelConfig},
    qwen_image_modules::{KvCacheMode, QwenImageKVCache},
};

use crate::{
    block_streaming_from_disk::TransformerBlocksReadFromDiskWhenNeeded,
    sigma_schedule::flow_match_sigmas_shifted_for_image_size,
};

const EMBEDDING_WIDTH: usize = 4096;
const LATENT_CHANNELS: usize = 64;

pub struct DiffusionTransformerWithBlocksStreamedFromDisk {
    model_without_blocks: QwenImageTransformerModel,
    blocks_read_from_disk: TransformerBlocksReadFromDiskWhenNeeded,
    device: Device,
}

impl DiffusionTransformerWithBlocksStreamedFromDisk {
    pub fn load(converted_weights_directory: &Path, device: &Device) -> Self {
        let config = QwenImageTransformerModelConfig::new([16, 56, 56]);
        let shell_record =
            ModuleRecord::load(converted_weights_directory.join("shell.bpk")).unwrap();
        let model_without_blocks = config
            .clone()
            .with_num_layers(0)
            .init(device)
            .load_record(shell_record);
        let blocks_read_from_disk =
            TransformerBlocksReadFromDiskWhenNeeded::new(converted_weights_directory, device);
        Self {
            model_without_blocks,
            blocks_read_from_disk,
            device: device.clone(),
        }
    }

    pub fn denoise_latents_from_seed(
        &mut self,
        prompt_embeddings: &[f32],
        latent_height: usize,
        latent_width: usize,
        num_steps: usize,
        seed: u64,
    ) -> Vec<f32> {
        let device = &self.device;
        let num_text_tokens = prompt_embeddings.len() / EMBEDDING_WIDTH;
        let num_image_tokens = latent_height * latent_width;
        let slots = num_image_tokens / 4;
        let img_shapes = vec![(1, latent_height, latent_width)];
        let img_mask: Vec<bool> = (0..num_text_tokens + slots)
            .map(|index| index >= num_text_tokens)
            .collect();

        let encoder_hidden_states = Tensor::<1>::from_floats(prompt_embeddings, device).reshape([
            1,
            num_text_tokens,
            EMBEDDING_WIDTH,
        ]);

        device.seed(seed);
        let mut latents = Tensor::<3>::random(
            [1, num_image_tokens, LATENT_CHANNELS],
            Distribution::Normal(0.0, 1.0),
            device,
        )
        .cast(FloatDType::F32);

        let sigmas = flow_match_sigmas_shifted_for_image_size(num_steps, num_image_tokens);
        let mut kv_cache =
            QwenImageKVCache::new(self.blocks_read_from_disk.num_layers(), 1, device);

        let all_steps_started = Instant::now();
        for step in 0..num_steps {
            let step_started = Instant::now();
            let kv_cache_mode = if step == 0 {
                KvCacheMode::EXTRACT
            } else {
                KvCacheMode::CACHED
            };
            let timestep = Tensor::<1>::from_floats([sigmas[step]], device);

            let out = self.model_without_blocks.forward(
                &mut self.blocks_read_from_disk,
                latents.clone(),
                encoder_hidden_states.clone(),
                timestep,
                img_shapes.clone(),
                &img_mask,
                None,
                Some(&mut kv_cache),
                Some(kv_cache_mode),
            );

            let num_output_tokens = out.dims()[1];
            let velocity = out
                .slice(s![
                    ..,
                    num_output_tokens - num_image_tokens..num_output_tokens,
                    ..
                ])
                .cast(FloatDType::F32);
            latents = latents + velocity * (sigmas[step + 1] - sigmas[step]);

            device.sync().unwrap();
            println!(
                "  step {:>2}/{num_steps}: {:.2?}",
                step + 1,
                step_started.elapsed()
            );
        }
        println!("  {num_steps} steps: {:.2?}", all_steps_started.elapsed());

        let latents = latents.into_data().try_into_vec::<f32>().unwrap();
        let num_not_a_number = latents.iter().filter(|value| value.is_nan()).count();
        if num_not_a_number > 0 {
            eprintln!(
                "  warning: {num_not_a_number} of {} latent values are NaN, so this image will come out blank",
                latents.len()
            );
        }
        latents
    }
}
