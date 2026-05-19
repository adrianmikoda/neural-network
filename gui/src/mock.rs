use rand::Rng;

pub struct MockNetwork {
    num_classes: usize,
}

impl MockNetwork {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }

    pub fn predict(&self, _input: &[f32]) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let mut predictions: Vec<f32> = (0..self.num_classes)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();

        let sum: f32 = predictions.iter().sum();
        for p in &mut predictions {
            *p /= sum;
        }
        predictions
    }
}
