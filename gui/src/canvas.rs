pub struct CanvasState {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
}

impl CanvasState {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![0.0; width * height],
        }
    }

    pub fn clear(&mut self) {
        self.pixels.fill(0.0);
    }

    pub fn draw_brush(&mut self, cx: f32, cy: f32, radius: f32) {
        let r_int = radius.ceil() as i32;
        let cx_int = cx as i32;
        let cy_int = cy as i32;

        for y in (cy_int - r_int)..=(cy_int + r_int) {
            for x in (cx_int - r_int)..=(cx_int + r_int) {
                if x < 0 || x >= self.width as i32 || y < 0 || y >= self.height as i32 {
                    continue;
                }

                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let distance = (dx * dx + dy * dy).sqrt();

                if distance <= radius {
                    let index = (y as usize) * self.width + (x as usize);
                    self.pixels[index] = 1.0;
                }
            }
        }
    }
}
