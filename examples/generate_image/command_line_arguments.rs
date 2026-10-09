use std::{path::PathBuf, str::FromStr, time::SystemTime};

const USAGE: &str = "\
usage: cargo run --release --example generate_image -- (--prompt <text> | --prompt-embeds <file>) [options]

  --prompt <text>              encode this prompt with reference/encode_prompt_to_embeddings_file.py first
  --prompt-embeds <file>       use embeddings already written by that script
  --size <width>x<height>      in pixels, each a multiple of 16 (default 512x512)
  --steps <n>                  denoising steps (default 25)
  --seed <n>                   seed of the first image (default: random)
  --count <n>                  images to generate, seeds counting up from --seed (default 1)
  --output-directory <dir>     where the PNGs go (default out/images)
  --weights-directory <dir>    converted weights from convert_weights_for_webgpu (default out)";

const PIXEL_MULTIPLE_THE_LATENT_GRID_REQUIRES: usize = 16;
const FEWEST_STEPS_THE_SIGMA_SCHEDULE_ACCEPTS: usize = 2;

pub enum PromptSource {
    Text(String),
    EmbeddingsFile(PathBuf),
}

pub struct CommandLineArguments {
    pub prompt_source: PromptSource,
    pub width_in_pixels: usize,
    pub height_in_pixels: usize,
    pub num_steps: usize,
    pub first_seed: u64,
    pub num_images: usize,
    pub output_directory: PathBuf,
    pub converted_weights_directory: PathBuf,
}

impl CommandLineArguments {
    pub fn parse_from_process_or_exit() -> Self {
        let raw_arguments: Vec<String> = std::env::args().skip(1).collect();
        if raw_arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!("{USAGE}");
            std::process::exit(0);
        }
        Self::parse(raw_arguments).unwrap_or_else(|message| {
            eprintln!("{message}\n\n{USAGE}");
            std::process::exit(2);
        })
    }

    fn parse(raw_arguments: Vec<String>) -> Result<Self, String> {
        let mut prompt_text = None;
        let mut prompt_embeddings_file = None;
        let mut size_in_pixels = (512, 512);
        let mut num_steps = 25;
        let mut first_seed = None;
        let mut num_images = 1;
        let mut output_directory = PathBuf::from("out/images");
        let mut converted_weights_directory = PathBuf::from("out");

        let mut raw_arguments = raw_arguments.into_iter();
        while let Some(flag) = raw_arguments.next() {
            let value = raw_arguments
                .next()
                .ok_or_else(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--prompt" => prompt_text = Some(value),
                "--prompt-embeds" => prompt_embeddings_file = Some(PathBuf::from(value)),
                "--size" => size_in_pixels = parse_size_in_pixels(&value)?,
                "--steps" => num_steps = parse_number(&flag, &value)?,
                "--seed" => first_seed = Some(parse_number(&flag, &value)?),
                "--count" => num_images = parse_number(&flag, &value)?,
                "--output-directory" => output_directory = PathBuf::from(value),
                "--weights-directory" => converted_weights_directory = PathBuf::from(value),
                _ => return Err(format!("unknown flag {flag}")),
            }
        }

        let prompt_source = match (prompt_text, prompt_embeddings_file) {
            (Some(text), None) => PromptSource::Text(text),
            (None, Some(file)) => PromptSource::EmbeddingsFile(file),
            _ => return Err("pass exactly one of --prompt and --prompt-embeds".to_string()),
        };
        if num_steps < FEWEST_STEPS_THE_SIGMA_SCHEDULE_ACCEPTS {
            return Err(format!(
                "--steps must be at least {FEWEST_STEPS_THE_SIGMA_SCHEDULE_ACCEPTS}"
            ));
        }
        if num_images == 0 {
            return Err("--count must be at least 1".to_string());
        }

        let (width_in_pixels, height_in_pixels) = size_in_pixels;
        Ok(Self {
            prompt_source,
            width_in_pixels,
            height_in_pixels,
            num_steps,
            first_seed: first_seed.unwrap_or_else(seed_from_current_time),
            num_images,
            output_directory,
            converted_weights_directory,
        })
    }
}

fn parse_size_in_pixels(value: &str) -> Result<(usize, usize), String> {
    let (width, height) = value
        .split_once('x')
        .ok_or_else(|| format!("--size takes <width>x<height>, got {value}"))?;
    let width: usize = parse_number("--size", width)?;
    let height: usize = parse_number("--size", height)?;
    let is_whole_latent_grid =
        |pixels: usize| pixels > 0 && pixels % PIXEL_MULTIPLE_THE_LATENT_GRID_REQUIRES == 0;
    if !is_whole_latent_grid(width) || !is_whole_latent_grid(height) {
        return Err(format!(
            "--size {value}: width and height must be multiples of {PIXEL_MULTIPLE_THE_LATENT_GRID_REQUIRES}"
        ));
    }
    Ok((width, height))
}

fn parse_number<T: FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} takes a whole number, got {value}"))
}

fn seed_from_current_time() -> u64 {
    let since_epoch = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    since_epoch.as_nanos() as u64 % 1_000_000
}
