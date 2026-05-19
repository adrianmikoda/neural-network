mod app;
mod canvas;
mod config;
mod mock;

use app::NeuralApp;
use config::ModelConfig;

fn main() -> Result<(), eframe::Error> {
    let config_data = r#"{
        "model_name": "MNIST Digits Classifier",
        "input_width": 28,
        "input_height": 28,
        "labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    }"#;

    let config: ModelConfig = serde_json::from_str(config_data).expect("Invalid config JSON");

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    let window_title = config.model_name.clone();

    eframe::run_native(
        &window_title,
        options,
        Box::new(|cc| Box::new(NeuralApp::new(cc, config))),
    )
}
