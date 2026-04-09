use crate::activation::Activation;
use crate::matrix::Matrix;

pub(crate) struct Layer {
    pub weights: Matrix,
    pub biases: Matrix,
    pub activation: Activation,

    pub inputs: Option<Matrix>,
    pub z_values: Option<Matrix>,
}

impl Layer {
    pub(crate) fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        todo!()
    }

    pub(crate) fn forward(&mut self, input: &Matrix) -> Matrix {
        todo!()
    }
}
