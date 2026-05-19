mod args;
mod dataset;

use args::{Cli, Commands};
use clap::Parser;

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
            println!("TODO: Execute training");
            println!("Config: {:?}", config);
            println!("Data Dir: {:?}", data_dir);
            println!("Epochs: {}", epochs);
            println!("Learning Rate: {}", learning_rate);
            println!("Output: {:?}", output);
        }
        Commands::Eval { model, data_dir } => {
            println!("TODO: Execute evaluation");
            println!("Model: {:?}", model);
            println!("Data Dir: {:?}", data_dir);
        }
    }
}
