use crate::config::ModelConfig;
use crate::mock::MockNetwork;
use eframe::egui;

pub struct NeuralApp {
    config: ModelConfig,
    network: MockNetwork,
}

impl NeuralApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, config: ModelConfig) -> Self {
        let network = MockNetwork::new(config.labels.len());
        Self { config, network }
    }
}

impl eframe::App for NeuralApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::right("right_panel")
            .exact_width(250.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading(&self.config.model_name);
                ui.separator();

                if ui.button("Clear Canvas").clicked() {
                    todo!();
                }

                ui.separator();
                ui.heading("Predictions");

                let dummy_input = vec![0.0; self.config.input_width * self.config.input_height];
                let preds = self.network.predict(&dummy_input);

                let mut results: Vec<_> = self.config.labels.iter().zip(preds.iter()).collect();
                results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

                for (label, &prob) in results {
                    let progress = egui::ProgressBar::new(prob).text(format!(
                        "{}: {:.1}%",
                        label,
                        prob * 100.0
                    ));
                    ui.add(progress);
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Canvas Area");
        });
    }
}
