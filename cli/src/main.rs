mod args;
mod augmentation;
mod dataset;

use args::{Cli, Commands};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use lib::Activation;
use lib::network::NeuralNetwork;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct LayerConfig {
    neurons: usize,
    activation: Activation,
}

#[derive(Deserialize, Debug)]
struct NetworkConfig {
    layers: Vec<LayerConfig>,
}

fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Train {
            config,
            data_dir,
            epochs,
            learning_rate,
            output,
        } => {
            println!("Parsing network config from {:?}...", config);
            let config_content =
                std::fs::read_to_string(config).expect("Failed to read configuration file");
            let net_config: NetworkConfig = serde_json::from_str(&config_content)
                .expect("Failed to parse network configuration JSON");

            println!("Loading training data...");
            let dataset = dataset::Dataset::load(
                data_dir.join("train-images-idx3-ubyte"),
                data_dir.join("train-labels-idx1-ubyte"),
            )
            .expect("Failed to load training dataset");

            let mut network = NeuralNetwork::with_input(784);
            for layer in &net_config.layers {
                network = network.add_layer(layer.activation.clone(), layer.neurons);
            }

            println!("Starting training for {} epochs...", epochs);

            for epoch in 1..=*epochs {
                let pb = ProgressBar::new(dataset.images.len() as u64);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("[{elapsed_precise}] Epoch {msg} [{bar:40}] {pos}/{len}")
                        .unwrap(),
                );
                pb.set_message(epoch.to_string());

                let mut total_loss = 0.0;

                for (i, image) in dataset.images.iter().enumerate() {
                    let augmented_image = augmentation::augment_image(image);

                    let mut target = [0.0f32; 10];
                    target[dataset.labels[i] as usize] = 1.0;

                    let loss = network.train_on_batch(&augmented_image, &target, *learning_rate);
                    total_loss += loss;

                    pb.inc(1);
                }
                pb.finish();
                println!(
                    "Epoch {} completed. Average Loss: {:.4}",
                    epoch,
                    total_loss / dataset.images.len() as f32
                );
            }

            network.save(output).expect("Failed to save model");
            println!("Model saved to {:?}", output);
        }

        Commands::Eval { model, data_dir } => {
            println!("Loading test data...");
            let dataset = dataset::Dataset::load(
                data_dir.join("t10k-images-idx3-ubyte"),
                data_dir.join("t10k-labels-idx1-ubyte"),
            )
            .expect("Failed to load test dataset");

            let mut network = NeuralNetwork::load(model).expect("Failed to load model weights");

            let pb = ProgressBar::new(dataset.images.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] Evaluating [{bar:40}] {pos}/{len}")
                    .unwrap(),
            );

            let mut correct = 0;

            for (i, image) in dataset.images.iter().enumerate() {
                let prediction = network.predict(image);
                let predicted_label = argmax(&prediction);

                if predicted_label == dataset.labels[i] as usize {
                    correct += 1;
                }
                pb.inc(1);
            }
            pb.finish();

            let accuracy = (correct as f32 / dataset.images.len() as f32) * 100.0;
            println!(
                "Accuracy: {:.2}% ({}/{})",
                accuracy,
                correct,
                dataset.images.len()
            );
        }
    }
}
