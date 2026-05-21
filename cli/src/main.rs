mod args;
mod augmentation;
mod dataset;

use args::{Cli, Commands};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use lib::network::NeuralNetwork;

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Train {
            config: _,
            data_dir,
            epochs,
            learning_rate: _,
            output,
        } => {
            println!("Loading training data...");
            let dataset = dataset::Dataset::load(
                data_dir.join("train-images-idx3-ubyte"),
                data_dir.join("train-labels-idx1-ubyte"),
            )
            .expect("Failed to load training dataset");

            let network = NeuralNetwork::with_input(784);

            println!("Starting training for {} epochs...", epochs);

            for epoch in 1..=*epochs {
                let pb = ProgressBar::new(dataset.images.len() as u64);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("[{elapsed_precise}] Epoch {msg} [{bar:40}] {pos}/{len}")
                        .unwrap(),
                );
                pb.set_message(epoch.to_string());

                for (i, image) in dataset.images.iter().enumerate() {
                    let _augmented_image = augmentation::augment_image(image);

                    let mut _target = [0.0f32; 10];
                    _target[dataset.labels[i] as usize] = 1.0;

                    // TODO: network.train(&_augmented_image, &_target, *learning_rate);

                    pb.inc(1);
                }
                pb.finish();
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

            let _network = NeuralNetwork::load(model).expect("Failed to load model weights");

            let pb = ProgressBar::new(dataset.images.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] Evaluating [{bar:40}] {pos}/{len}")
                    .unwrap(),
            );

            let mut correct = 0;

            for (i, _image) in dataset.images.iter().enumerate() {
                // TODO: let prediction = network.predict(_image);
                // TODO: let predicted_label = argmax(&prediction);

                let predicted_label = 0;

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
