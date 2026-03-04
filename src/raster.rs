// Software rasterizer: framebuffer, z-buffer, triangle fill via edge functions

pub struct Framebuffer {
    pub pixels: Vec<u32>,
    pub zbuf: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

impl Framebuffer {
    pub fn new(w: usize, h: usize) -> Self {
        Framebuffer {
            pixels: vec![0; w * h],
            zbuf: vec![1.0; w * h],
            w,
            h,
        }
    }

    pub fn resize(&mut self, w: usize, h: usize) {
        self.w = w;
        self.h = h;
        self.pixels.resize(w * h, 0);
        self.zbuf.resize(w * h, 1.0);
    }

    pub fn clear(&mut self, color: u32) {
        self.pixels.fill(color);
        self.zbuf.fill(1.0);
    }

    pub fn put_pixel(&mut self, x: usize, y: usize, z: f32, color: u32) {
        let idx = y * self.w + x;
        if z < self.zbuf[idx] {
            self.zbuf[idx] = z;
            self.pixels[idx] = color;
        }
    }
}

// Screen-space triangle: 3 vertices (x, y, z_ndc) + flat color
pub struct ScreenTri {
    pub v: [[f32; 3]; 3],
    pub color: u32,
}

pub fn draw_triangle(fb: &mut Framebuffer, tri: &ScreenTri) {
    let [v0, v1, v2] = tri.v;
    let w = fb.w as f32;
    let h = fb.h as f32;

    let min_x = v0[0].min(v1[0]).min(v2[0]).max(0.0) as usize;
    let max_x = (v0[0].max(v1[0]).max(v2[0]).min(w - 1.0)) as usize;
    let min_y = v0[1].min(v1[1]).min(v2[1]).max(0.0) as usize;
    let max_y = (v0[1].max(v1[1]).max(v2[1]).min(h - 1.0)) as usize;

    let area = edge(v0, v1, v2);
    if area.abs() < 0.001 { return; }
    let inv_area = 1.0 / area;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let w0 = edge_2d(v1, v2, p);
            let w1 = edge_2d(v2, v0, p);
            let w2 = edge_2d(v0, v1, p);

            if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0) {
                let b0 = w0 * inv_area;
                let b1 = w1 * inv_area;
                let b2 = w2 * inv_area;
                let z = b0 * v0[2] + b1 * v1[2] + b2 * v2[2];
                fb.put_pixel(x, y, z, tri.color);
            }
        }
    }
}

fn edge(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn edge_2d(a: [f32; 3], b: [f32; 3], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

