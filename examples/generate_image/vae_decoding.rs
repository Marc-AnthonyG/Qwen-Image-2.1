use std::path::Path;

use burn::{
    module::Module,
    store::ModuleRecord,
    tensor::{Device, Tensor},
};
use image::RgbaImage;
use qwen_image::{
    LatentNorm,
    vae::model::{AutoencoderKLQwenImage, AutoencoderKLQwenImageConfig},
};

const LATENT_CHANNELS: usize = 64;
const DECODED_CHANNELS_RED_GREEN_BLUE_ALPHA: usize = 4;

pub struct VaeDecoder {
    vae: AutoencoderKLQwenImage,
    latent_norm: LatentNorm,
    device: Device,
}

impl VaeDecoder {
    pub fn load(converted_weights_directory: &Path, device: &Device) -> Self {
        let vae_record = ModuleRecord::load(converted_weights_directory.join("vae.bpk")).unwrap();
        let vae = AutoencoderKLQwenImageConfig::new()
            .init(device)
            .load_record(vae_record);
        let latent_norm = LatentNorm::new(&vae.mean, &vae.std, device);
        Self {
            vae,
            latent_norm,
            device: device.clone(),
        }
    }

    pub fn decode_to_rgba_image(
        &self,
        packed_latents: &[f32],
        latent_height: usize,
        latent_width: usize,
    ) -> RgbaImage {
        let latents = Tensor::<1>::from_floats(packed_latents, &self.device)
            .reshape([1, latent_height, latent_width, LATENT_CHANNELS])
            .permute([0, 3, 1, 2]);
        let decoded = self.vae.decode(self.latent_norm.denormalize(latents));

        let [_, channels, height, width] = decoded.dims();
        assert_eq!(channels, DECODED_CHANNELS_RED_GREEN_BLUE_ALPHA);
        let pixels_interleaved_in_minus_one_to_one = decoded.squeeze_dim::<3>(0).permute([1, 2, 0]);
        let pixels_interleaved_in_zero_to_255 = pixels_interleaved_in_minus_one_to_one
            .add_scalar(1.0)
            .mul_scalar(127.5)
            .round()
            .clamp(0.0, 255.0)
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();

        RgbaImage::from_raw(
            width as u32,
            height as u32,
            pixels_interleaved_in_zero_to_255
                .into_iter()
                .map(|value| value as u8)
                .collect(),
        )
        .unwrap()
    }
}
