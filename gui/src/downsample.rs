pub fn average_pooling(
    high_res: &[f32],
    high_w: usize,
    high_h: usize,
    low_w: usize,
    low_h: usize,
) -> Vec<f32> {
    let mut low_res = vec![0.0; low_w * low_h];
    let block_w = high_w / low_w;
    let block_h = high_h / low_h;

    for ly in 0..low_h {
        for lx in 0..low_w {
            let mut sum = 0.0;
            for by in 0..block_h {
                for bx in 0..block_w {
                    let hx = lx * block_w + bx;
                    let hy = ly * block_h + by;
                    sum += high_res[hy * high_w + hx];
                }
            }
            let avg = sum / (block_w * block_h) as f32;
            low_res[ly * low_w + lx] = avg;
        }
    }
    low_res
}
