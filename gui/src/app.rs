use crate::canvas::CanvasState;
use crate::config::ModelConfig;
use crate::downsample::average_pooling;
use crate::mock::MockNetwork;
use eframe::egui;
use eframe::egui::ColorImage;

pub struct NeuralApp {
    config: ModelConfig,
    network: MockNetwork,
    canvas: CanvasState,
    texture: Option<egui::TextureHandle>,
    predictions: Vec<f32>,
}

impl NeuralApp {
    pub fn new(cc: &eframe::CreationContext<'_>, config: ModelConfig) -> Self {
        let network = MockNetwork::new(config.labels.len());
        let canvas = CanvasState::new(config.input_width * 10, config.input_height * 10);
        let predictions = vec![0.0; config.labels.len()];

        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(24, 24, 24);
        visuals.window_fill = egui::Color32::from_rgb(24, 24, 24);
        visuals.override_text_color = Some(egui::Color32::from_rgb(224, 224, 224));
        visuals.window_rounding = egui::Rounding::ZERO;

        visuals.widgets.noninteractive.rounding = egui::Rounding::ZERO;
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(24, 24, 24);
        visuals.widgets.noninteractive.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 80, 80));
        visuals.widgets.noninteractive.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(224, 224, 224));

        visuals.widgets.inactive.rounding = egui::Rounding::ZERO;
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(40, 40, 40);
        visuals.widgets.inactive.weak_bg_fill = egui::Color32::from_rgb(40, 40, 40);
        visuals.widgets.inactive.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 80, 80));
        visuals.widgets.inactive.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(224, 224, 224));

        visuals.widgets.hovered.rounding = egui::Rounding::ZERO;
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(60, 60, 60);
        visuals.widgets.hovered.weak_bg_fill = egui::Color32::from_rgb(60, 60, 60);
        visuals.widgets.hovered.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(180, 180, 180));
        visuals.widgets.hovered.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 255, 255));

        visuals.widgets.active.rounding = egui::Rounding::ZERO;
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(80, 80, 80);
        visuals.widgets.active.weak_bg_fill = egui::Color32::from_rgb(80, 80, 80);
        visuals.widgets.active.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(220, 220, 220));
        visuals.widgets.active.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 255, 255));

        visuals.widgets.open.rounding = egui::Rounding::ZERO;
        visuals.widgets.open.bg_fill = egui::Color32::from_rgb(24, 24, 24);
        visuals.widgets.open.weak_bg_fill = egui::Color32::from_rgb(24, 24, 24);
        visuals.widgets.open.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 80, 80));
        visuals.widgets.open.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(224, 224, 224));

        visuals.extreme_bg_color = egui::Color32::from_rgb(18, 18, 18);

        cc.egui_ctx.set_visuals(visuals);

        let mut style = (*cc.egui_ctx.style()).clone();
        for font_id in style.text_styles.values_mut() {
            font_id.family = egui::FontFamily::Monospace;
        }

        cc.egui_ctx.set_style(style);

        Self {
            config,
            network,
            canvas,
            texture: None,
            predictions,
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

    fn update_predictions(&mut self) {
        let downsampled = average_pooling(
            &self.canvas.pixels,
            self.canvas.width,
            self.canvas.height,
            self.config.input_width,
            self.config.input_height,
        );
        self.predictions = self.network.predict(&downsampled);
    }

    fn show_right_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("right_panel")
            .exact_width(280.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new("=================================")
                        .color(egui::Color32::from_rgb(100, 100, 100))
                        .monospace(),
                );
                ui.label(
                    egui::RichText::new(format!(" MODEL: {}", self.config.model_name))
                        .color(egui::Color32::from_rgb(240, 240, 240))
                        .monospace(),
                );
                ui.label(
                    egui::RichText::new("=================================")
                        .color(egui::Color32::from_rgb(100, 100, 100))
                        .monospace(),
                );
                ui.add_space(15.0);

                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label(
                        egui::RichText::new("> ")
                            .color(egui::Color32::from_rgb(160, 160, 160))
                            .monospace(),
                    );
                    if ui.button(" [ CLEAR CANVAS ] ").clicked() {
                        self.canvas.clear();
                        self.predictions.fill(0.0);
                    }
                });

                ui.add_space(15.0);
                ui.label(
                    egui::RichText::new("---------------------------------")
                        .color(egui::Color32::from_rgb(80, 80, 80))
                        .monospace(),
                );
                ui.label(
                    egui::RichText::new(" PREDICTIONS:")
                        .color(egui::Color32::from_rgb(200, 200, 200))
                        .monospace(),
                );
                ui.label(
                    egui::RichText::new("---------------------------------")
                        .color(egui::Color32::from_rgb(80, 80, 80))
                        .monospace(),
                );
                ui.add_space(10.0);

                let mut results: Vec<_> = self
                    .config
                    .labels
                    .iter()
                    .zip(self.predictions.iter())
                    .collect();
                results.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

                for (label, &prob) in results {
                    let num_chars = 12;
                    let filled = (prob * num_chars as f32).round() as usize;
                    let filled_str = "█".repeat(filled);
                    let empty_str = "░".repeat(num_chars - filled);

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        ui.label(
                            egui::RichText::new(format!("  {} [", label))
                                .color(egui::Color32::from_rgb(180, 180, 180))
                                .monospace(),
                        );
                        ui.label(
                            egui::RichText::new(filled_str)
                                .color(egui::Color32::from_rgb(224, 224, 224))
                                .monospace(),
                        );
                        ui.label(
                            egui::RichText::new(empty_str)
                                .color(egui::Color32::from_rgb(56, 56, 56))
                                .monospace(),
                        );
                        ui.label(
                            egui::RichText::new(format!("] {:>5.1}%", prob * 100.0))
                                .color(egui::Color32::from_rgb(180, 180, 180))
                                .monospace(),
                        );
                    });
                    ui.add_space(2.0);
                }

                ui.add_space(15.0);
                ui.label(
                    egui::RichText::new("=================================")
                        .color(egui::Color32::from_rgb(100, 100, 100))
                        .monospace(),
                );
            });
    }

    fn show_left_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(10.0);
            ui.label(
                egui::RichText::new(">>> DRAWING INPUT AREA")
                    .color(egui::Color32::from_rgb(240, 240, 240))
                    .monospace(),
            );
            ui.label(
                egui::RichText::new(
                    "--------------------------------------------------------------------",
                )
                .color(egui::Color32::from_rgb(80, 80, 80))
                .monospace(),
            );
            ui.add_space(10.0);

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
            let side_length = available_size.x.min(available_size.y) - 30.0;

            let image_widget = egui::Image::new(&*texture)
                .fit_to_exact_size(egui::vec2(side_length, side_length))
                .sense(egui::Sense::click_and_drag());

            let response = egui::Frame::none()
                .stroke(egui::Stroke::new(
                    2.0,
                    egui::Color32::from_rgb(120, 120, 120),
                ))
                .show(ui, |ui| ui.add(image_widget))
                .inner;

            ui.add_space(10.0);
            ui.label(
                egui::RichText::new("(Use left click + drag to draw,")
                    .color(egui::Color32::from_rgb(140, 140, 140))
                    .monospace(),
            );
            ui.label(
                egui::RichText::new("use CLEAR button on the right to reset)")
                    .color(egui::Color32::from_rgb(140, 140, 140))
                    .monospace(),
            );

            if (response.dragged() || response.clicked())
                && let Some(pointer_pos) = response.interact_pointer_pos()
            {
                let rect = response.rect;
                if rect.contains(pointer_pos) {
                    let scale_x = self.canvas.width as f32 / rect.width();
                    let scale_y = self.canvas.height as f32 / rect.height();

                    let x_canvas = (pointer_pos.x - rect.min.x) * scale_x;
                    let y_canvas = (pointer_pos.y - rect.min.y) * scale_y;

                    self.canvas.draw_brush(x_canvas, y_canvas, 10.0);
                    self.update_predictions();
                    ctx.request_repaint();
                }
            }
        });
    }
}

impl eframe::App for NeuralApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.show_right_panel(ctx);
        self.show_left_panel(ctx);
    }
}
