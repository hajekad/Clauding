// Shared color utility functions (ARGB u32 format)

/// Linearly interpolate between two ARGB colors
pub fn lerp_color(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let inv = 1.0 - t;
    let r = ((a >> 16) & 0xFF) as f32 * inv + ((b >> 16) & 0xFF) as f32 * t;
    let g = ((a >> 8) & 0xFF) as f32 * inv + ((b >> 8) & 0xFF) as f32 * t;
    let bl = (a & 0xFF) as f32 * inv + (b & 0xFF) as f32 * t;
    0xFF000000 | ((r as u32) << 16) | ((g as u32) << 8) | (bl as u32)
}

/// Darken an ARGB color by a factor (0.0 = black, 1.0 = unchanged)
pub fn darken(color: u32, factor: f32) -> u32 {
    let r = (((color >> 16) & 0xFF) as f32 * factor) as u32;
    let g = (((color >> 8) & 0xFF) as f32 * factor) as u32;
    let b = ((color & 0xFF) as f32 * factor) as u32;
    0xFF000000 | (r.min(255) << 16) | (g.min(255) << 8) | b.min(255)
}

/// Alpha-blend src over dst. alpha is 0..255.
pub fn alpha_blend(dst: u32, src: u32, alpha: u32) -> u32 {
    let a = alpha as f32 / 255.0;
    let inv = 1.0 - a;
    let r = (((src >> 16) & 0xFF) as f32 * a + ((dst >> 16) & 0xFF) as f32 * inv) as u32;
    let g = (((src >> 8) & 0xFF) as f32 * a + ((dst >> 8) & 0xFF) as f32 * inv) as u32;
    let b = ((src & 0xFF) as f32 * a + (dst & 0xFF) as f32 * inv) as u32;
    0xFF000000 | (r << 16) | (g << 8) | b
}
