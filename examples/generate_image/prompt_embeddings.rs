use std::{path::Path, process::Command};

const PYTHON_WITH_TEXT_ENCODER_DEPENDENCIES: &str = "reference/.venv/bin/python";
const PROMPT_ENCODING_SCRIPT: &str = "reference/encode_prompt_to_embeddings_file.py";
const EMBEDDING_WIDTH: usize = 4096;

pub fn encode_prompt_with_reference_python_script(prompt: &str, embeddings_file: &Path) {
    println!("encoding the prompt with the Qwen3-VL text encoder (this is the slow part)");
    let status = Command::new(PYTHON_WITH_TEXT_ENCODER_DEPENDENCIES)
        .arg(PROMPT_ENCODING_SCRIPT)
        .arg(prompt)
        .arg("--output")
        .arg(embeddings_file)
        .status()
        .unwrap_or_else(|error| {
            panic!("could not start {PYTHON_WITH_TEXT_ENCODER_DEPENDENCIES}: {error}")
        });
    assert!(
        status.success(),
        "{PROMPT_ENCODING_SCRIPT} failed: {status}"
    );
}

pub fn read_prompt_embeddings(embeddings_file: &Path) -> Vec<f32> {
    let bytes = std::fs::read(embeddings_file)
        .unwrap_or_else(|error| panic!("could not read {}: {error}", embeddings_file.display()));
    let bytes_per_token = EMBEDDING_WIDTH * size_of::<f32>();
    assert!(
        !bytes.is_empty() && bytes.len() % bytes_per_token == 0,
        "{} holds {} bytes, which is not a whole number of {EMBEDDING_WIDTH}-wide f32 token embeddings",
        embeddings_file.display(),
        bytes.len()
    );
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect()
}
