use crate::matrix::Matrix;

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
