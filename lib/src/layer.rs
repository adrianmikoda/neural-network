use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::matrix::Matrix;
use rand::Rng;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: Activation,

    pub inputs: Option<Matrix>,
    pub z_values: Option<Matrix>,
}

impl Layer {
    pub(crate) fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let scale = match activation {
            Activation::ReLU    => (2.0 / input_size as f32).sqrt(),
            Activation::Sigmoid => (1.0 / input_size as f32).sqrt(),
            Activation::Softmax => (1.0 / input_size as f32).sqrt(),
        };

        let mut weights = Matrix::new(output_size, input_size);
        weights.map(|_| {
            let r: f32 = rand::random();
            (r * 2.0 - 1.0) * scale
        });

        let biases = Matrix::new(output_size, 1);

        Self { weights, biases, activation, inputs: None, z_values: None }
    }

    pub(crate) fn forward(&mut self, input: &Matrix) -> Matrix {
        self.inputs = Some(input.clone());
        let mut z = self.weights.dot(input);
        z.add_row_vector(&self.biases);
        self.z_values = Some(z.clone());
        self.activation.apply(&mut z);
        z
    }
}
