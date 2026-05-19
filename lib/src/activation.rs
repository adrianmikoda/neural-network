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
        todo!()
    }

    pub(crate) fn apply_derivative(&self, matrix: &mut Matrix) {
        todo!()
    }
}
