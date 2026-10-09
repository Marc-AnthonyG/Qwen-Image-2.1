mod block_streaming_from_disk;
mod command_line_arguments;
mod denoising;
mod image_file_naming;
mod prompt_embeddings;
mod sigma_schedule;
mod vae_decoding;

use burn::tensor::Device;

use command_line_arguments::{CommandLineArguments, PromptSource};
use denoising::DiffusionTransformerWithBlocksStreamedFromDisk;
use image_file_naming::{
    image_file_name, name_prefix_from_embeddings_file, name_prefix_from_prompt_text,
};
use prompt_embeddings::{encode_prompt_with_reference_python_script, read_prompt_embeddings};
use vae_decoding::VaeDecoder;

const PIXELS_PER_LATENT_TOKEN: usize = 16;

fn main() {
    let arguments = CommandLineArguments::parse_from_process_or_exit();
    std::fs::create_dir_all(&arguments.output_directory).unwrap();

    let (prompt_embeddings, image_name_prefix) =
        obtain_prompt_embeddings_and_image_name_prefix(&arguments);

    let device = Device::wgpu(Default::default());
    let latent_height = arguments.height_in_pixels / PIXELS_PER_LATENT_TOKEN;
    let latent_width = arguments.width_in_pixels / PIXELS_PER_LATENT_TOKEN;
    let seeds: Vec<u64> = (0..arguments.num_images as u64)
        .map(|offset| arguments.first_seed + offset)
        .collect();

    let latents_for_each_seed = denoise_every_seed_then_release_the_transformer(
        &arguments,
        &device,
        &prompt_embeddings,
        latent_height,
        latent_width,
        &seeds,
    );
    device.memory_cleanup();

    let vae_decoder = VaeDecoder::load(&arguments.converted_weights_directory, &device);
    for (seed, latents) in seeds.iter().zip(latents_for_each_seed) {
        let image = vae_decoder.decode_to_rgba_image(&latents, latent_height, latent_width);
        let path = arguments.output_directory.join(image_file_name(
            &image_name_prefix,
            arguments.width_in_pixels,
            arguments.height_in_pixels,
            arguments.num_steps,
            *seed,
        ));
        image.save(&path).unwrap();
        println!("saved {}", path.display());
    }
}

fn obtain_prompt_embeddings_and_image_name_prefix(
    arguments: &CommandLineArguments,
) -> (Vec<f32>, String) {
    match &arguments.prompt_source {
        PromptSource::Text(prompt) => {
            let image_name_prefix = name_prefix_from_prompt_text(prompt);
            let embeddings_file = arguments
                .output_directory
                .join(format!("{image_name_prefix}.prompt_embeds.bin"));
            encode_prompt_with_reference_python_script(prompt, &embeddings_file);
            (read_prompt_embeddings(&embeddings_file), image_name_prefix)
        }
        PromptSource::EmbeddingsFile(embeddings_file) => (
            read_prompt_embeddings(embeddings_file),
            name_prefix_from_embeddings_file(embeddings_file),
        ),
    }
}

fn denoise_every_seed_then_release_the_transformer(
    arguments: &CommandLineArguments,
    device: &Device,
    prompt_embeddings: &[f32],
    latent_height: usize,
    latent_width: usize,
    seeds: &[u64],
) -> Vec<Vec<f32>> {
    let mut diffusion_transformer = DiffusionTransformerWithBlocksStreamedFromDisk::load(
        &arguments.converted_weights_directory,
        device,
    );
    seeds
        .iter()
        .map(|seed| {
            println!("denoising seed {seed}");
            diffusion_transformer.denoise_latents_from_seed(
                prompt_embeddings,
                latent_height,
                latent_width,
                arguments.num_steps,
                *seed,
            )
        })
        .collect()
}
