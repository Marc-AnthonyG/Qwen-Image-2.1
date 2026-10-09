// Copied from the private `schedule` in src/lib.rs: keep the two in step.
pub fn flow_match_sigmas_shifted_for_image_size(
    num_steps: usize,
    num_image_tokens: usize,
) -> Vec<f32> {
    let (base_shift, max_shift) = (0.5f64, 0.9f64);
    let (base_len, max_len) = (256.0f64, 8192.0f64);
    let terminal = 0.02f64;

    let m = (max_shift - base_shift) / (max_len - base_len);
    let mu = num_image_tokens as f64 * m + (base_shift - m * base_len);
    let emu = mu.exp();

    let mut sigmas: Vec<f64> = (0..num_steps)
        .map(|i| {
            let x = 1.0 + (1.0 / num_steps as f64 - 1.0) * i as f64 / (num_steps - 1) as f64;
            emu / (emu + (1.0 / x - 1.0))
        })
        .collect();

    let scale = (1.0 - sigmas[num_steps - 1]) / (1.0 - terminal);
    for sigma in sigmas.iter_mut() {
        *sigma = 1.0 - (1.0 - *sigma) / scale;
    }
    sigmas.push(0.0);
    sigmas.into_iter().map(|sigma| sigma as f32).collect()
}
