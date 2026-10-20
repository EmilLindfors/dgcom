//! A small RGBA canvas for the plots drawn on the CPU (the gauge trace, the pinned
//! point's profiles): anti-aliased dots and segments blended over a transparent
//! background, uploaded as an image.

/// An RGBA8 image being drawn, row-major from the top left.
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Canvas {
    /// A transparent canvas.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            rgba: vec![0; 4 * width * height],
        }
    }

    /// Blend `colour` at `alpha` into pixel (x, y).
    pub fn stamp(&mut self, x: usize, y: usize, colour: [u8; 3], alpha: f32) {
        if x >= self.width || y >= self.height {
            return;
        }
        let p = 4 * (y * self.width + x);
        let rgba = &mut self.rgba;
        let a = rgba[p + 3] as f32 / 255.0;
        let out = alpha + a * (1.0 - alpha);
        for c in 0..3 {
            let blended =
                (colour[c] as f32 * alpha + rgba[p + c] as f32 * a * (1.0 - alpha)) / out.max(1e-6);
            rgba[p + c] = blended.round() as u8;
        }
        rgba[p + 3] = (out * 255.0).round() as u8;
    }

    /// A round dot `width` pixels across at (x, y).
    pub fn dot(&mut self, x: f32, y: f32, colour: [u8; 3], width: f32) {
        let r = 0.5 * width;
        let (x0, x1) = ((x - r).floor() as i64, (x + r).ceil() as i64);
        let (y0, y1) = ((y - r).floor() as i64, (y + r).ceil() as i64);
        for py in y0.max(0)..=y1 {
            for px in x0.max(0)..=x1 {
                let d = ((px as f32 - x).powi(2) + (py as f32 - y).powi(2)).sqrt();
                let alpha = (r + 0.5 - d).clamp(0.0, 1.0);
                if alpha > 0.0 {
                    self.stamp(px as usize, py as usize, colour, alpha);
                }
            }
        }
    }

    /// A straight segment `width` pixels wide from `a` to `b` (without its end dot).
    pub fn segment(
        &mut self,
        (xa, ya): (f32, f32),
        (xb, yb): (f32, f32),
        colour: [u8; 3],
        width: f32,
    ) {
        let r = 0.5 * width;
        let steps = ((xb - xa).abs().max((yb - ya).abs()) / (0.5 * r))
            .ceil()
            .max(1.0) as usize;
        for s in 0..steps {
            let f = s as f32 / steps as f32;
            self.dot(xa + f * (xb - xa), ya + f * (yb - ya), colour, width);
        }
    }

    /// A dashed horizontal line at row `y`, from column `x0` to `x1`.
    pub fn dashed_row(&mut self, y: usize, [x0, x1]: [usize; 2], colour: [u8; 3], alpha: f32) {
        for x in (x0..x1).step_by(6) {
            for dx in 0..3 {
                self.stamp(x + dx, y, colour, alpha);
            }
        }
    }
}
