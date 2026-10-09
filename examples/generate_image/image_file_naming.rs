use std::path::Path;

const LONGEST_NAME_PREFIX_IN_CHARACTERS: usize = 60;
const EMBEDDINGS_FILE_SUFFIX: &str = ".prompt_embeds.bin";

pub fn name_prefix_from_prompt_text(prompt: &str) -> String {
    let words_of_letters_and_digits: Vec<String> = prompt
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();
    let mut name_prefix = String::new();
    for word in words_of_letters_and_digits {
        let separator_length = usize::from(!name_prefix.is_empty());
        if name_prefix.chars().count() + separator_length + word.chars().count()
            > LONGEST_NAME_PREFIX_IN_CHARACTERS
        {
            break;
        }
        if !name_prefix.is_empty() {
            name_prefix.push('-');
        }
        name_prefix.push_str(&word);
    }
    if name_prefix.is_empty() {
        "prompt".to_string()
    } else {
        name_prefix
    }
}

pub fn name_prefix_from_embeddings_file(embeddings_file: &Path) -> String {
    let file_name = embeddings_file
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    file_name
        .strip_suffix(EMBEDDINGS_FILE_SUFFIX)
        .or_else(|| file_name.strip_suffix(".bin"))
        .unwrap_or(&file_name)
        .to_string()
}

pub fn image_file_name(
    name_prefix: &str,
    width_in_pixels: usize,
    height_in_pixels: usize,
    num_steps: usize,
    seed: u64,
) -> String {
    format!("{name_prefix}_{width_in_pixels}x{height_in_pixels}_{num_steps}steps_seed{seed}.png")
}
