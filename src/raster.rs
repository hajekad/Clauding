// Software rasterizer: framebuffer, z-buffer, incremental edge-function triangle fill

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

    #[inline(always)]
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

/// Rasterize a triangle using incremental edge functions.
/// Inner loop does 3 additions per pixel step instead of 6 multiplies.
/// Uses a small fill bias to close sub-pixel gaps between adjacent triangles.
pub fn draw_triangle(fb: &mut Framebuffer, tri: &ScreenTri) {
    let [v0, mut v1, mut v2] = tri.v;

    // Sub-pixel bias: allows tiny overdraw at shared edges so no gap pixels appear.
    // Z-buffer resolves which triangle wins, so overdraw is visually invisible.
    const FILL_BIAS: f32 = 0.125;

    // Bounding box clamped to screen (expand by 1px to account for bias)
    let min_x = (v0[0].min(v1[0]).min(v2[0]) - 1.0).max(0.0) as usize;
    let max_x = ((v0[0].max(v1[0]).max(v2[0]) + 1.0).min((fb.w - 1) as f32)) as usize;
    let min_y = (v0[1].min(v1[1]).min(v2[1]) - 1.0).max(0.0) as usize;
    let max_y = ((v0[1].max(v1[1]).max(v2[1]) + 1.0).min((fb.h - 1) as f32)) as usize;
    if min_x > max_x || min_y > max_y { return; }

    // Signed 2x area
    let mut area = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]);
    if area.abs() < 0.5 { return; }

    // Normalize to CCW (positive area) by swapping v1/v2
    if area < 0.0 {
        std::mem::swap(&mut v1, &mut v2);
        area = -area;
    }
    let inv_area = 1.0 / area;

    // Edge function increments per x/y step
    // E0 = edge(v1→v2), E1 = edge(v2→v0)
    // E2 = area - E0 - E1 (not tracked, checked via e0+e1 <= area)
    let dx0 = v1[1] - v2[1];
    let dy0 = v2[0] - v1[0];
    let dx1 = v2[1] - v0[1];
    let dy1 = v0[0] - v2[0];

    // Initial edge values at pixel center (min_x+0.5, min_y+0.5)
    let px = min_x as f32 + 0.5;
    let py = min_y as f32 + 0.5;
    let mut row_e0 = (v2[0] - v1[0]) * (py - v1[1]) - (v2[1] - v1[1]) * (px - v1[0]);
    let mut row_e1 = (v0[0] - v2[0]) * (py - v2[1]) - (v0[1] - v2[1]) * (px - v2[0]);

    // Incremental z: z = (e0*(v0z-v2z) + e1*(v1z-v2z))/area + v2z
    let dz0 = v0[2] - v2[2];
    let dz1 = v1[2] - v2[2];
    let z_step_x = (dx0 * dz0 + dx1 * dz1) * inv_area;
    let z_step_y = (dy0 * dz0 + dy1 * dz1) * inv_area;
    let mut row_z = (row_e0 * dz0 + row_e1 * dz1) * inv_area + v2[2];

    let biased_area = area + FILL_BIAS;
    let w = fb.w;
    let color = tri.color;
    let pixels = fb.pixels.as_mut_ptr();
    let zbuf = fb.zbuf.as_mut_ptr();

    for y in min_y..=max_y {
        let mut e0 = row_e0;
        let mut e1 = row_e1;
        let mut z = row_z;
        let row_off = y * w;

        for x in min_x..=max_x {
            if e0 >= -FILL_BIAS && e1 >= -FILL_BIAS && e0 + e1 <= biased_area {
                // Safety: x in [0, fb.w-1], y in [0, fb.h-1], so idx in [0, fb.w*fb.h-1]
                let idx = row_off + x;
                unsafe {
                    let zp = &mut *zbuf.add(idx);
                    if z < *zp {
                        *zp = z;
                        *pixels.add(idx) = color;
                    }
                }
            }
            e0 += dx0;
            e1 += dx1;
            z += z_step_x;
        }

        row_e0 += dy0;
        row_e1 += dy1;
        row_z += z_step_y;
    }
}
