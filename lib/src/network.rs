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
        let mut current = Matrix::new(input.len(), 1);
        current.data.copy_from_slice(input);
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current.data.to_vec()
    }

    pub fn train_on_batch(&mut self, input: &[f32], target: &[f32], learning_rate: f32) -> f32 {
        let mut current = Matrix::new(input.len(), 1);
        current.data.copy_from_slice(input);
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        let a_l = current;

        let loss = if matches!(
            self.layers.last().map(|l| &l.activation),
            Some(Activation::Softmax)
        ) {
            let mut sum = 0.0;
            for (&val, &t) in a_l.data.iter().zip(target) {
                if t > 0.0 {
                    sum -= t * val.max(1e-15).ln();
                }
            }
            sum
        } else {
            let mut sum = 0.0;
            for (&val, &t) in a_l.data.iter().zip(target) {
                let diff = val - t;
                sum += diff * diff;
            }
            sum / a_l.data.len() as f32
        };

        let mut deltas = Vec::with_capacity(self.layers.len());
        let last_idx = self.layers.len() - 1;

        let mut delta = a_l.clone();
        for (val, &t) in delta.data.iter_mut().zip(target) {
            *val -= t;
        }
        if !matches!(self.layers[last_idx].activation, Activation::Softmax) {
            let mut z_deriv = self.layers[last_idx].z_values.clone().unwrap();
            self.layers[last_idx]
                .activation
                .apply_derivative(&mut z_deriv);
            for (val, &z) in delta.data.iter_mut().zip(&z_deriv.data) {
                *val *= z;
            }
        }
        deltas.push(delta);

        for l in (0..last_idx).rev() {
            let next_layer = &self.layers[l + 1];
            let next_delta = &deltas[deltas.len() - 1];

            let w_trans = next_layer.weights.transpose();
            let mut delta_hidden = w_trans.dot(next_delta);

            let current_layer = &self.layers[l];
            let mut z_deriv = current_layer.z_values.clone().unwrap();
            current_layer.activation.apply_derivative(&mut z_deriv);
            for (val, &z) in delta_hidden.data.iter_mut().zip(&z_deriv.data) {
                *val *= z;
            }
            deltas.push(delta_hidden);
        }

        deltas.reverse();

        for (current_layer, delta_l) in self.layers.iter_mut().zip(&deltas) {
            let inputs_trans = current_layer.inputs.clone().unwrap().transpose();
            let grad_w = delta_l.dot(&inputs_trans);
            let grad_b = delta_l;

            for (w, &gw) in current_layer.weights.data.iter_mut().zip(&grad_w.data) {
                *w -= learning_rate * gw;
            }
            for (b, &gb) in current_layer.biases.data.iter_mut().zip(&grad_b.data) {
                *b -= learning_rate * gb;
            }
        }

        loss
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
