use rand::Rng;

pub fn augment_image(image: &[f32]) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut new_image = image.to_vec();

    if rng.gen_bool(0.5) {
        if rng.gen_bool(0.5) {
            apply_noise(&mut new_image, &mut rng);
        } else {
            new_image = apply_shift(&new_image, &mut rng);
        }
    }

    new_image
}

fn apply_noise(image: &mut [f32], rng: &mut impl Rng) {
    let noise_level = 0.1;
    for pixel in image.iter_mut() {
        let noise: f32 = rng.gen_range(-noise_level..=noise_level);
        *pixel = (*pixel + noise).clamp(0.0, 1.0);
    }
}

fn apply_shift(image: &[f32], rng: &mut impl Rng) -> Vec<f32> {
    let mut shifted = vec![0.0; 784];
    let dx: i32 = rng.gen_range(-2..=2);
    let dy: i32 = rng.gen_range(-2..=2);

    for y in 0..28i32 {
        for x in 0..28i32 {
            let new_x = x + dx;
            let new_y = y + dy;

            if (0..28).contains(&new_x) && (0..28).contains(&new_y) {
                let old_idx = (y * 28 + x) as usize;
                let new_idx = (new_y * 28 + new_x) as usize;
                shifted[new_idx] = image[old_idx];
            }
        }
    }

    shifted
}
