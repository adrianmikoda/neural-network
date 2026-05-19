use crate::canvas::CanvasState;
use crate::config::ModelConfig;
use crate::mock::MockNetwork;
use eframe::egui;
use eframe::egui::ColorImage;

pub struct NeuralApp {
    config: ModelConfig,
    network: MockNetwork,
    canvas: CanvasState,
    texture: Option<egui::TextureHandle>,
}

impl NeuralApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, config: ModelConfig) -> Self {
        let network = MockNetwork::new(config.labels.len());
        let canvas = CanvasState::new(config.input_width * 10, config.input_height * 10);

        Self {
            config,
            network,
            canvas,
            texture: None,
        }
    }

    fn build_image_from_canvas(&self) -> ColorImage {
        let mut pixels_rgba = Vec::with_capacity(self.canvas.width * self.canvas.height);

        for &val in &self.canvas.pixels {
            let color = (val * 255.0) as u8;
            pixels_rgba.push(egui::Color32::from_rgb(color, color, color));
        }

        ColorImage {
            size: [self.canvas.width, self.canvas.height],
            pixels: pixels_rgba,
        }
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
                    self.canvas.clear();
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
            ui.heading("Draw your digit:");
            ui.separator();

            let image = self.build_image_from_canvas();

            let texture = self.texture.get_or_insert_with(|| {
                ctx.load_texture(
                    "canvas_texture",
                    image.clone(),
                    egui::TextureOptions::NEAREST,
                )
            });
            texture.set(image, egui::TextureOptions::NEAREST);

            let available_size = ui.available_size();
            let side_length = available_size.x.min(available_size.y) - 20.0;

            let image_widget = egui::Image::new(&*texture)
                .fit_to_exact_size(egui::vec2(side_length, side_length))
                .sense(egui::Sense::click_and_drag());

            let response = ui.add(image_widget);

            if (response.dragged() || response.clicked())
                && let Some(pointer_pos) = response.interact_pointer_pos() {
                    let rect = response.rect;
                    if rect.contains(pointer_pos) {
                        let scale_x = self.canvas.width as f32 / rect.width();
                        let scale_y = self.canvas.height as f32 / rect.height();

                        let x_canvas = (pointer_pos.x - rect.min.x) * scale_x;
                        let y_canvas = (pointer_pos.y - rect.min.y) * scale_y;

                        self.canvas.draw_brush(x_canvas, y_canvas, 10.0);
                        ctx.request_repaint();
                    }
                }
        });
    }
}
