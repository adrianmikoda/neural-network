use crate::activation::Activation;
use crate::layer::Layer;
use crate::matrix::Matrix;

pub struct NeuralNetwork {
    layers: Vec<Layer>,
    current_output_size: usize,
}

impl NeuralNetwork {
    pub fn with_input(input_size: usize) -> Self {
        Self {
            layers: Vec::new(),
            current_output_size: input_size,
        }
    }

    pub fn add_layer(mut self, activation: Activation, neurons: usize) -> Self {
        let layer = Layer::new(self.current_output_size, neurons, activation);
        self.layers.push(layer);
        self.current_output_size = neurons;
        self
    }

    pub fn predict(&mut self, input: &[f32]) -> Vec<f32> {
        todo!()
    }

    pub fn train_on_batch(&mut self, input: &[f32], target: &[f32], learning_rate: f32) -> f32 {
        todo!()
    }
}
