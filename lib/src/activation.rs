use serde::{Deserialize, Serialize};

use crate::matrix::Matrix;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Softmax,
}

impl Activation {
    pub(crate) fn apply(&self, matrix: &mut Matrix) {
        match self {
            Activation::ReLU => {
                matrix.map(|x| if x > 0.0 { x } else { 0.0 });
            }
            Activation::Sigmoid => {
                matrix.map(|x| 1.0 / (1.0 + (-x).exp()));
            }
            Activation::Softmax => {
                let max = matrix
                    .data
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                matrix.map(|x| (x - max).exp());
                let sum: f32 = matrix.data.iter().sum();
                matrix.map(|x| x / sum);
            }
        }
    }

    pub(crate) fn apply_derivative(&self, matrix: &mut Matrix) {
        match self {
            Activation::ReLU => {
                matrix.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
            }
            Activation::Sigmoid => {
                matrix.map(|x| {
                    let s = 1.0 / (1.0 + (-x).exp());
                    s * (1.0 - s)
                });
            }
            Activation::Softmax => {
                let _ = matrix;
            }
        }
    }
}
