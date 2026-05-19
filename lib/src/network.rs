use serde::{Deserialize, Serialize};

use crate::activation::Activation;
use crate::layer::Layer;
use crate::matrix::Matrix;

#[derive(Serialize, Deserialize, Debug, Clone)]
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

    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let serialized = bincode::serialize(self).map_err(std::io::Error::other)?;

        std::fs::write(path, serialized)?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        let buffer = std::fs::read(path)?;

        let network: NeuralNetwork = bincode::deserialize(&buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(network)
    }
}
