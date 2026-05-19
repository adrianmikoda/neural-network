use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "neural-cli", version, about = "CLI for training and evaluating neural networks", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train the neural network
    Train {
        /// Path to the network configuration file
        #[arg(short, long)]
        config: PathBuf,

        /// Path to the directory containing MNIST dataset
        #[arg(short, long)]
        data_dir: PathBuf,

        /// Number of training epochs
        #[arg(short, long, default_value_t = 10)]
        epochs: usize,

        /// Learning rate for the optimizer
        #[arg(short, long, default_value_t = 0.01)]
        learning_rate: f32,

        /// Output path to save the trained model weights
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Evaluate a trained neural network
    Eval {
        /// Path to the saved model weights
        #[arg(short, long)]
        model: PathBuf,

        /// Path to the directory containing MNIST test dataset
        #[arg(short, long)]
        data_dir: PathBuf,
    },
}
