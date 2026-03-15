// Model viewer: renders each entity type from 4 orthographic views
// Usage: cargo run --bin model_viewer [--gpu]
//   --gpu   Use Vulkan GPU pipeline instead of CPU rasterizer
// Output: debug/model_*.png

use clauding::{state, render, raster, math, mesh, gpu};
use clauding::image::save_png;
use clauding::render::clip_to_screen;
use clauding::rng::Rng;

const VIEW_W: usize = 512;
const VIEW_H: usize = 512;
const IMG_W: usize = VIEW_W * 2;  // 2x2 grid
const IMG_H: usize = VIEW_H * 2;

/// Render a set of WorldTris into a framebuffer from a given camera position
fn render_model(
    fb: &mut raster::Framebuffer,
    tris: &[state::WorldTri],
    eye: [f32; 3],
    target: [f32; 3],
) {
    let aspect = fb.w as f32 / fb.h as f32;
    let view = math::m4_look_at(eye, target, [0.0, 1.0, 0.0]);
    let proj = math::m4_perspective(60.0_f32.to_radians(), aspect, 0.01, 100.0);
    let vp = math::m4_mul(&proj, &view);
    let fw = fb.w as f32;
    let fh = fb.h as f32;

    for tri in tris {
        // Simple flat shading — use normal to get basic lighting
        let light_dir = [0.4, 0.8, -0.3]; // overhead slightly angled
        let dot = tri.normal[0] * light_dir[0] + tri.normal[1] * light_dir[1] + tri.normal[2] * light_dir[2];
        let intensity = dot.max(0.0) * 0.6 + 0.4; // ambient 0.4, diffuse 0.6

        let r = ((tri.color >> 16) & 0xFF) as f32;
        let g = ((tri.color >> 8) & 0xFF) as f32;
        let b = (tri.color & 0xFF) as f32;
        let ro = (r * intensity).min(255.0) as u32;
        let go = (g * intensity).min(255.0) as u32;
        let bo = (b * intensity).min(255.0) as u32;
        let color = 0xFF000000 | (ro << 16) | (go << 8) | bo;

        // Transform to clip space
        let c0 = math::m4_transform_no_div(&vp, tri.v[0]);
        let c1 = math::m4_transform_no_div(&vp, tri.v[1]);
        let c2 = math::m4_transform_no_div(&vp, tri.v[2]);

        let near_w = 0.01;
        if c0[3] < near_w || c1[3] < near_w || c2[3] < near_w { continue; }

        let s0 = clip_to_screen(c0, fw, fh);
        let s1 = clip_to_screen(c1, fw, fh);
        let s2 = clip_to_screen(c2, fw, fh);

        // Quick bounds check
        if s0[0].max(s1[0]).max(s2[0]) < 0.0 { continue; }
        if s0[0].min(s1[0]).min(s2[0]) >= fw { continue; }
        if s0[1].max(s1[1]).max(s2[1]) < 0.0 { continue; }
        if s0[1].min(s1[1]).min(s2[1]) >= fh { continue; }

        raster::draw_triangle(fb, &raster::ScreenTri { v: [s0, s1, s2], color });
    }
}

/// Render triangles colored by their face normal direction (normal-map visualization).
/// normal.x → Red, normal.y → Green, normal.z → Blue, each mapped from [-1,1] to [0,255].
fn render_model_normals(
    fb: &mut raster::Framebuffer,
    tris: &[state::WorldTri],
    eye: [f32; 3],
    target: [f32; 3],
) {
    let aspect = fb.w as f32 / fb.h as f32;
    let view = math::m4_look_at(eye, target, [0.0, 1.0, 0.0]);
    let proj = math::m4_perspective(60.0_f32.to_radians(), aspect, 0.01, 100.0);
    let vp = math::m4_mul(&proj, &view);
    let fw = fb.w as f32;
    let fh = fb.h as f32;

    for tri in tris {
        // Map normal components from [-1,1] to [0,255]
        let r = ((tri.normal[0] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u32;
        let g = ((tri.normal[1] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u32;
        let b = ((tri.normal[2] * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u32;
        let color = 0xFF000000 | (r << 16) | (g << 8) | b;

        let c0 = math::m4_transform_no_div(&vp, tri.v[0]);
        let c1 = math::m4_transform_no_div(&vp, tri.v[1]);
        let c2 = math::m4_transform_no_div(&vp, tri.v[2]);

        let near_w = 0.01;
        if c0[3] < near_w || c1[3] < near_w || c2[3] < near_w { continue; }

        let s0 = clip_to_screen(c0, fw, fh);
        let s1 = clip_to_screen(c1, fw, fh);
        let s2 = clip_to_screen(c2, fw, fh);

        if s0[0].max(s1[0]).max(s2[0]) < 0.0 { continue; }
        if s0[0].min(s1[0]).min(s2[0]) >= fw { continue; }
        if s0[1].max(s1[1]).max(s2[1]) < 0.0 { continue; }
        if s0[1].min(s1[1]).min(s2[1]) >= fh { continue; }

        raster::draw_triangle(fb, &raster::ScreenTri { v: [s0, s1, s2], color });
    }
}

// ── Smooth shading pipeline ────────────────────────────────────────────────

/// Compute per-vertex smooth normals by averaging face normals at coincident vertices.
/// Returns one [[f32;3]; 3] per tri — the smooth normal for each of its 3 vertices.
/// Uses spatial hashing with a crease angle threshold (70°) to preserve hard edges
/// where surfaces meet at sharp angles (e.g. overlapping body parts).
fn compute_smooth_normals(tris: &[state::WorldTri]) -> Vec<[[f32; 3]; 3]> {
    use std::collections::HashMap;

    // Quantize position to spatial hash key
    let quantize = |p: [f32; 3]| -> [i32; 3] {
        [(p[0] * 10000.0).round() as i32,
         (p[1] * 10000.0).round() as i32,
         (p[2] * 10000.0).round() as i32]
    };

    // Build map: quantized position → list of (face_index, vertex_in_face, face_normal)
    let mut pos_map: HashMap<[i32; 3], Vec<(usize, [f32; 3])>> = HashMap::new();
    for (fi, tri) in tris.iter().enumerate() {
        for vi in 0..3 {
            let key = quantize(tri.v[vi]);
            pos_map.entry(key).or_default().push((fi, tri.normal));
        }
    }

    let crease_cos = 70.0_f32.to_radians().cos(); // ~0.342

    let mut result = vec![[[0.0f32; 3]; 3]; tris.len()];
    for (fi, tri) in tris.iter().enumerate() {
        for vi in 0..3 {
            let key = quantize(tri.v[vi]);
            let face_n = tri.normal;
            let mut sum = [0.0f32; 3];

            if let Some(neighbors) = pos_map.get(&key) {
                for &(_, n) in neighbors {
                    // Dot product with this face's normal — only average if within crease angle
                    let dot = face_n[0] * n[0] + face_n[1] * n[1] + face_n[2] * n[2];
                    if dot >= crease_cos {
                        sum[0] += n[0];
                        sum[1] += n[1];
                        sum[2] += n[2];
                    }
                }
            }

            // Normalize
            let len = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
            if len > 1e-10 {
                result[fi][vi] = [sum[0] / len, sum[1] / len, sum[2] / len];
            } else {
                result[fi][vi] = face_n;
            }
        }
    }

    result
}

/// 3-light sculpture rig: key, fill, rim + ambient/hemisphere.
/// Returns intensity in [0.15, 1.0].
fn compute_lighting(normal: [f32; 3]) -> f32 {
    // Key light: warm, upper-right-front
    let key_dir: [f32; 3] = [0.4, 0.8, -0.5];
    let key_len = (key_dir[0]*key_dir[0] + key_dir[1]*key_dir[1] + key_dir[2]*key_dir[2]).sqrt();
    let key_n = [key_dir[0]/key_len, key_dir[1]/key_len, key_dir[2]/key_len];
    let key_dot = (normal[0]*key_n[0] + normal[1]*key_n[1] + normal[2]*key_n[2]).max(0.0);
    let key = key_dot * 0.55;

    // Fill light: softer, left
    let fill_dir: [f32; 3] = [-0.5, 0.3, -0.2];
    let fill_len = (fill_dir[0]*fill_dir[0] + fill_dir[1]*fill_dir[1] + fill_dir[2]*fill_dir[2]).sqrt();
    let fill_n = [fill_dir[0]/fill_len, fill_dir[1]/fill_len, fill_dir[2]/fill_len];
    let fill_dot = (normal[0]*fill_n[0] + normal[1]*fill_n[1] + normal[2]*fill_n[2]).max(0.0);
    let fill = fill_dot * 0.25;

    // Rim light: backlight, fresnel-weighted
    let rim_dir: [f32; 3] = [0.0, 0.2, 0.8];
    let rim_len = (rim_dir[0]*rim_dir[0] + rim_dir[1]*rim_dir[1] + rim_dir[2]*rim_dir[2]).sqrt();
    let rim_n = [rim_dir[0]/rim_len, rim_dir[1]/rim_len, rim_dir[2]/rim_len];
    let rim_dot = (normal[0]*rim_n[0] + normal[1]*rim_n[1] + normal[2]*rim_n[2]).max(0.0);
    let rim = rim_dot * 0.15;

    // Ambient + hemisphere (slight upward boost)
    let ambient = 0.15 + normal[1].max(0.0) * 0.05;

    (key + fill + rim + ambient).clamp(0.15, 1.0)
}

/// Apply lighting intensity to a base color, returning ARGB u32
fn apply_intensity(color: u32, intensity: f32) -> u32 {
    let r = ((color >> 16) & 0xFF) as f32;
    let g = ((color >> 8) & 0xFF) as f32;
    let b = (color & 0xFF) as f32;
    let ro = (r * intensity).min(255.0) as u32;
    let go = (g * intensity).min(255.0) as u32;
    let bo = (b * intensity).min(255.0) as u32;
    0xFF000000 | (ro << 16) | (go << 8) | bo
}

/// Render smooth-shaded model with per-vertex normals and 3-light rig
fn render_model_smooth(
    fb: &mut raster::Framebuffer,
    tris: &[state::WorldTri],
    vertex_normals: &[[[f32; 3]; 3]],
    eye: [f32; 3],
    target: [f32; 3],
) {
    let aspect = fb.w as f32 / fb.h as f32;
    let view = math::m4_look_at(eye, target, [0.0, 1.0, 0.0]);
    let proj = math::m4_perspective(60.0_f32.to_radians(), aspect, 0.01, 100.0);
    let vp = math::m4_mul(&proj, &view);
    let fw = fb.w as f32;
    let fh = fb.h as f32;

    for (fi, tri) in tris.iter().enumerate() {
        // Compute per-vertex lit colors
        let vn = &vertex_normals[fi];
        let colors = [
            apply_intensity(tri.color, compute_lighting(vn[0])),
            apply_intensity(tri.color, compute_lighting(vn[1])),
            apply_intensity(tri.color, compute_lighting(vn[2])),
        ];

        let c0 = math::m4_transform_no_div(&vp, tri.v[0]);
        let c1 = math::m4_transform_no_div(&vp, tri.v[1]);
        let c2 = math::m4_transform_no_div(&vp, tri.v[2]);

        let near_w = 0.01;
        if c0[3] < near_w || c1[3] < near_w || c2[3] < near_w { continue; }

        let s0 = clip_to_screen(c0, fw, fh);
        let s1 = clip_to_screen(c1, fw, fh);
        let s2 = clip_to_screen(c2, fw, fh);

        if s0[0].max(s1[0]).max(s2[0]) < 0.0 { continue; }
        if s0[0].min(s1[0]).min(s2[0]) >= fw { continue; }
        if s0[1].max(s1[1]).max(s2[1]) < 0.0 { continue; }
        if s0[1].min(s1[1]).min(s2[1]) >= fh { continue; }

        raster::draw_triangle_smooth(fb, &raster::ScreenTriSmooth {
            v: [s0, s1, s2],
            colors,
        });
    }
}

/// 8K smooth-shaded sheet: same layout as render_8k_sheet but uses smooth rendering
fn render_8k_sheet_smooth(
    tris: &[state::WorldTri],
    vertex_normals: &[[[f32; 3]; 3]],
    views: &[([f32; 3], [f32; 3], &str)],
    sheet_label: &str,
) -> (Vec<u32>, usize, usize) {
    let panel_w: usize = 960;
    let panel_h: usize = 2160;
    let n = views.len();
    let sheet_w = panel_w * n;
    let sheet_h = panel_h;

    let mut view_fb = raster::Framebuffer::new(panel_w, panel_h);
    let mut composite = vec![0xFF3A4455u32; sheet_w * sheet_h];

    let tri_label = format!("tris: {}", tris.len());
    for (view_idx, (eye, target, view_name)) in views.iter().enumerate() {
        view_fb.clear(0xFF445566);
        render_model_smooth(&mut view_fb, tris, vertex_normals, *eye, *target);

        // Labels at 2x scale for 4K
        for dy in 0..2_usize {
            for dx in 0..2_usize {
                draw_label_scaled(&mut view_fb, 8 + dx, 8 + dy, view_name, 2);
                draw_label_scaled(&mut view_fb, 8 + dx, 30 + dy, &tri_label, 2);
            }
        }

        let qx = view_idx * panel_w;
        for y in 0..panel_h {
            for x in 0..panel_w {
                composite[y * sheet_w + (qx + x)] = view_fb.pixels[y * panel_w + x];
            }
        }
    }

    // Panel separators (3px white)
    for pi in 1..n {
        let sx = pi * panel_w;
        for y in 0..sheet_h {
            for dx in 0..3_usize {
                if sx + dx < sheet_w { composite[y * sheet_w + sx + dx] = 0xFFFFFFFF; }
                if sx >= dx + 1 { composite[y * sheet_w + sx - 1 - dx] = 0xFFFFFFFF; }
            }
        }
    }

    eprintln!("Rendered 4K smooth: {} ({} tris, {}x{})", sheet_label, tris.len(), sheet_w, sheet_h);
    (composite, sheet_w, sheet_h)
}

/// Render model from 4 views and composite into a 2x2 grid
fn render_model_sheet(
    tris: &[state::WorldTri],
    center_y: f32,
    cam_dist: f32,
    label: &str,
) -> Vec<u32> {
    let mut view_fb = raster::Framebuffer::new(VIEW_W, VIEW_H);
    let mut composite = vec![0xFF334455u32; IMG_W * IMG_H]; // dark bg

    // 4 views: front (-Z), right (+X), back (+Z), left (-X)
    let views: [([f32; 3], &str); 4] = [
        ([0.0, center_y, -cam_dist], "Front (-Z)"),
        ([cam_dist, center_y, 0.0], "Right (+X)"),
        ([0.0, center_y, cam_dist], "Back (+Z)"),
        ([-cam_dist, center_y, 0.0], "Left (-X)"),
    ];

    let target = [0.0, center_y, 0.0];

    for (view_idx, (eye, view_name)) in views.iter().enumerate() {
        view_fb.clear(0xFF445566);
        render_model(&mut view_fb, tris, *eye, target);

        // Draw axis indicator in corner (small colored lines)
        // Red = +X, Green = +Y, Blue = +Z
        draw_label(&mut view_fb, 4, 4, view_name);
        draw_label(&mut view_fb, 4, 16, &format!("tris: {}", tris.len()));

        // Copy into composite at the right quadrant
        let qx = (view_idx % 2) * VIEW_W;
        let qy = (view_idx / 2) * VIEW_H;
        for y in 0..VIEW_H {
            for x in 0..VIEW_W {
                composite[(qy + y) * IMG_W + (qx + x)] = view_fb.pixels[y * VIEW_W + x];
            }
        }
    }

    // Draw separator lines
    for y in 0..IMG_H {
        composite[y * IMG_W + VIEW_W] = 0xFFFFFFFF;
        if VIEW_W > 1 { composite[y * IMG_W + VIEW_W - 1] = 0xFFFFFFFF; }
    }
    for x in 0..IMG_W {
        composite[VIEW_H * IMG_W + x] = 0xFFFFFFFF;
        if VIEW_H > 1 { composite[(VIEW_H - 1) * IMG_W + x] = 0xFFFFFFFF; }
    }

    eprintln!("Rendered: {} ({} tris)", label, tris.len());
    composite
}

/// Render model from 4 views using normal-direction colorization
fn render_model_sheet_normals(
    tris: &[state::WorldTri],
    center_y: f32,
    cam_dist: f32,
    label: &str,
) -> Vec<u32> {
    let mut view_fb = raster::Framebuffer::new(VIEW_W, VIEW_H);
    let mut composite = vec![0xFF334455u32; IMG_W * IMG_H];

    let views: [([f32; 3], &str); 4] = [
        ([0.0, center_y, -cam_dist], "Front (-Z)"),
        ([cam_dist, center_y, 0.0], "Right (+X)"),
        ([0.0, center_y, cam_dist], "Back (+Z)"),
        ([-cam_dist, center_y, 0.0], "Left (-X)"),
    ];

    let target = [0.0, center_y, 0.0];

    for (view_idx, (eye, view_name)) in views.iter().enumerate() {
        view_fb.clear(0xFF445566);
        render_model_normals(&mut view_fb, tris, *eye, target);

        draw_label(&mut view_fb, 4, 4, view_name);
        draw_label(&mut view_fb, 4, 16, &format!("tris: {}", tris.len()));

        let qx = (view_idx % 2) * VIEW_W;
        let qy = (view_idx / 2) * VIEW_H;
        for y in 0..VIEW_H {
            for x in 0..VIEW_W {
                composite[(qy + y) * IMG_W + (qx + x)] = view_fb.pixels[y * VIEW_W + x];
            }
        }
    }

    for y in 0..IMG_H {
        composite[y * IMG_W + VIEW_W] = 0xFFFFFFFF;
        if VIEW_W > 1 { composite[y * IMG_W + VIEW_W - 1] = 0xFFFFFFFF; }
    }
    for x in 0..IMG_W {
        composite[VIEW_H * IMG_W + x] = 0xFFFFFFFF;
        if VIEW_H > 1 { composite[(VIEW_H - 1) * IMG_W + x] = 0xFFFFFFFF; }
    }

    eprintln!("Rendered normals: {} ({} tris)", label, tris.len());
    composite
}

/// Render a set of WorldTris via the GPU pipeline into a framebuffer
fn render_model_gpu(
    ctx: &mut gpu::GpuContext,
    fb: &mut raster::Framebuffer,
    tris: &[state::WorldTri],
    eye: [f32; 3],
    target: [f32; 3],
) {
    // Convert tris to GpuVertex
    let mut verts: Vec<gpu::GpuVertex> = Vec::with_capacity(tris.len() * 3);
    for tri in tris {
        for vi in 0..3 {
            verts.push(gpu::GpuVertex {
                pos: tri.v[vi],
                color_packed: tri.color,
                normal: tri.normal,
            });
        }
    }
    ctx.upload_static_vertices(&verts);

    let aspect = fb.w as f32 / fb.h as f32;
    let view = math::m4_look_at(eye, target, [0.0, 1.0, 0.0]);
    let proj = math::m4_perspective_vk(60.0_f32.to_radians(), aspect, 0.01, 100.0);
    let vp = math::m4_mul(&proj, &view);
    let push = render::gpu_push_constants(10.0, eye, target, &vp); // 10am lighting
    let clear = [0.267, 0.333, 0.400, 1.0]; // match 0xFF445566 bg

    ctx.resize_render_target(fb.w as u32, fb.h as u32);
    // Double-buffer: render twice so second frame reads back the completed first
    ctx.render_frame(&[], &push, clear, fb.w as u32, fb.h as u32, &mut fb.pixels);
    ctx.render_frame(&[], &push, clear, fb.w as u32, fb.h as u32, &mut fb.pixels);
}

/// Render model from 4 views via GPU and composite into a 2x2 grid
fn render_model_sheet_gpu(
    ctx: &mut gpu::GpuContext,
    tris: &[state::WorldTri],
    center_y: f32,
    cam_dist: f32,
    label: &str,
) -> Vec<u32> {
    let mut view_fb = raster::Framebuffer::new(VIEW_W, VIEW_H);
    let mut composite = vec![0xFF334455u32; IMG_W * IMG_H]; // dark bg

    // 4 views: front (-Z), right (+X), back (+Z), left (-X)
    let views: [([f32; 3], &str); 4] = [
        ([0.0, center_y, -cam_dist], "Front (-Z)"),
        ([cam_dist, center_y, 0.0], "Right (+X)"),
        ([0.0, center_y, cam_dist], "Back (+Z)"),
        ([-cam_dist, center_y, 0.0], "Left (-X)"),
    ];

    let target = [0.0, center_y, 0.0];

    for (view_idx, (eye, view_name)) in views.iter().enumerate() {
        view_fb.clear(0xFF445566);
        render_model_gpu(ctx, &mut view_fb, tris, *eye, target);

        draw_label(&mut view_fb, 4, 4, view_name);
        draw_label(&mut view_fb, 4, 16, &format!("tris: {}", tris.len()));

        // Copy into composite at the right quadrant
        let qx = (view_idx % 2) * VIEW_W;
        let qy = (view_idx / 2) * VIEW_H;
        for y in 0..VIEW_H {
            for x in 0..VIEW_W {
                composite[(qy + y) * IMG_W + (qx + x)] = view_fb.pixels[y * VIEW_W + x];
            }
        }
    }

    // Draw separator lines
    for y in 0..IMG_H {
        composite[y * IMG_W + VIEW_W] = 0xFFFFFFFF;
        if VIEW_W > 1 { composite[y * IMG_W + VIEW_W - 1] = 0xFFFFFFFF; }
    }
    for x in 0..IMG_W {
        composite[VIEW_H * IMG_W + x] = 0xFFFFFFFF;
        if VIEW_H > 1 { composite[(VIEW_H - 1) * IMG_W + x] = 0xFFFFFFFF; }
    }

    eprintln!("Rendered (GPU): {} ({} tris)", label, tris.len());
    composite
}

/// Dispatch to GPU or CPU model sheet renderer
fn render_sheet(
    gpu_ctx: &mut Option<gpu::GpuContext>,
    tris: &[state::WorldTri],
    center_y: f32,
    cam_dist: f32,
    label: &str,
) -> Vec<u32> {
    if let Some(ctx) = gpu_ctx {
        render_model_sheet_gpu(ctx, tris, center_y, cam_dist, label)
    } else {
        render_model_sheet(tris, center_y, cam_dist, label)
    }
}

/// Draw text at integer scale factor (for high-res displays)
fn draw_label_scaled(fb: &mut raster::Framebuffer, x: usize, y: usize, text: &str, scale: usize) {
    let mut cx = x;
    for ch in text.bytes() {
        draw_char_scaled(fb, cx, y, ch, 0xFFFFFFFF, scale);
        cx += 6 * scale;
    }
}

fn draw_char_scaled(fb: &mut raster::Framebuffer, x: usize, y: usize, ch: u8, color: u32, scale: usize) {
    let glyph = match ch {
        b'A'..=b'Z' => FONT_UPPER[(ch - b'A') as usize],
        b'a'..=b'z' => FONT_UPPER[(ch - b'a') as usize],
        b'0'..=b'9' => FONT_DIGIT[(ch - b'0') as usize],
        b' ' => [0; 7],
        b':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        b'(' => [0b00100, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b00100],
        b')' => [0b01000, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01000],
        b'+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        b'-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        b'/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        _ => [0b11111; 7],
    };
    for row in 0..7 {
        for col in 0..5 {
            if glyph[row] & (1 << (4 - col)) != 0 {
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = x + col * scale + sx;
                        let py = y + row * scale + sy;
                        if px < fb.w && py < fb.h {
                            fb.pixels[py * fb.w + px] = color;
                        }
                    }
                }
            }
        }
    }
}

/// Minimal pixel font for labels (5x7 glyphs, ASCII subset)
fn draw_label(fb: &mut raster::Framebuffer, x: usize, y: usize, text: &str) {
    let mut cx = x;
    for ch in text.bytes() {
        draw_char(fb, cx, y, ch, 0xFFFFFFFF);
        cx += 6;
    }
}

fn draw_char(fb: &mut raster::Framebuffer, x: usize, y: usize, ch: u8, color: u32) {
    // Ultra-minimal bitmap font — just enough for labels
    let glyph = match ch {
        b'A'..=b'Z' => FONT_UPPER[(ch - b'A') as usize],
        b'a'..=b'z' => FONT_UPPER[(ch - b'a') as usize], // same as upper
        b'0'..=b'9' => FONT_DIGIT[(ch - b'0') as usize],
        b' ' => [0; 7],
        b':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        b'(' => [0b00100, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b00100],
        b')' => [0b01000, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01000],
        b'+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        b'-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        _ => [0b11111; 7], // block for unknown
    };
    for row in 0..7 {
        for col in 0..5 {
            if glyph[row] & (1 << (4 - col)) != 0 {
                let px = x + col;
                let py = y + row;
                if px < fb.w && py < fb.h {
                    fb.pixels[py * fb.w + px] = color;
                }
            }
        }
    }
}

#[rustfmt::skip]
const FONT_UPPER: [[u8; 7]; 26] = [
    [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001], // A
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110], // B
    [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110], // C
    [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100], // D
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111], // E
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000], // F
    [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111], // G
    [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001], // H
    [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], // I
    [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100], // J
    [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001], // K
    [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111], // L
    [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001], // M
    [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001], // N
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110], // O
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000], // P
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101], // Q
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001], // R
    [0b01110, 0b10001, 0b10000, 0b01110, 0b00001, 0b10001, 0b01110], // S
    [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100], // T
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110], // U
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100], // V
    [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001], // W
    [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001], // X
    [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100], // Y
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111], // Z
];

#[rustfmt::skip]
const FONT_DIGIT: [[u8; 7]; 10] = [
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110], // 0
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], // 1
    [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111], // 2
    [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110], // 3
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010], // 4
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110], // 5
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110], // 6
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000], // 7
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110], // 8
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100], // 9
];

fn make_player() -> state::Player {
    state::Player {
        x: 0.0, y: 0.0, z: 0.0, rot_y: 0.0,
        health: 100.0, stamina: 100.0, money: 0.0,
        vel_y: 0.0, on_ground: true, walk_phase: 0.3,
        sprinting: false, in_vehicle: None,
        carrying_item: false, carrying_bin: None,
        active_job: state::PlayerJob::none(),
        sitting: false, bank_balance: 0.0,
        job_menu_open: false, job_menu_cursor: 0,
        attack_cooldown: 0.0, attack_phase: 0.0, hit_flash: 0.0,
        damage_shake: 0.0,
        hunger: 0.0, thirst: 0.0,
        wanted_vehicle_hit: false, bounty: 0.0,
        is_female: false,
        terrain_normal: [0.0, 1.0, 0.0],
        body: {
            let shape = clauding::physics::CollisionShape::Capsule { radius: 0.3, half_height: 0.625 };
            let inertia = shape.inertia_diag(80.0);
            clauding::physics::RigidBody::new_dynamic([0.0, 0.0, 0.0], 80.0, inertia)
        },
        skeleton: clauding::skeleton::Skeleton::new_humanoid(),
        standing_on_vehicle: None,
        standing_on_vehicle_timer: 0.0,
        model_index: 0,
    }
}

fn make_vehicle(color: u32) -> state::Vehicle {
    state::Vehicle {
        x: 0.0, y: 0.0, z: 0.0, rot_y: 0.0,
        speed: 0.0, color,
        occupied: false, ai_active: false,
        ai_target_x: 0.0, ai_target_z: 0.0,
        rng: Rng::new(1),
        owner_npc: None,
        path: vec![],
        path_idx: 0,
        current_segment: None,
        lane_dir: state::LaneDirection::Forward,
        intersection_state: state::IntersectionState::Cruising,
        intersection_wait_timer: 0.0,
        cruise_speed: 10.0,
        target_speed: 0.0,
        parking_target: None,
        parked: true,
        idle_timer: 0.0,
        terrain_normal: [0.0, 1.0, 0.0],
        scale: 1.0,
        body: clauding::physics::RigidBody::new_static([0.0, 0.0, 0.0]),
        wheels: {
            let w = clauding::tire::WheelState::new([0.0; 3], 0.355);
            [w, w, w, w]
        },
        suspension: {
            let s = clauding::suspension::SuspensionState::new(
                clauding::suspension::SuspensionParams::default_car(),
            );
            [s, s, s, s]
        },
        drivetrain: clauding::tire::Drivetrain::new(350.0, 35.0),
        deformation: clauding::deform::VehicleDeformation::new(),
        surface_override: None,
    }
}

fn make_npc(job: state::NpcJob) -> state::Npc {
    state::Npc {
        x: 0.0, y: 0.0, z: 0.0, rot_y: 0.0,
        walk_phase: 0.3,
        target_x: 0.0, target_z: 0.0,
        shirt_color: 0xFFAA3333,
        pants_color: 0xFF333355,
        rng: Rng::new(42),
        vel_y: 0.0, on_ground: true,
        state: state::NpcState::Working,
        home_idx: 0, car_idx: 0,
        wake_hour: 7.0, state_timer: 0.0,
        money: 0.0,
        carrying_item: false, carrying_bin: None,
        target_item: None, target_bin: None,
        items_deposited_today: 0,
        in_vehicle: false,
        parked_x: 0.0, parked_z: 0.0,
        stuck_timer: 0.0,
        nav_path: Vec::new(), nav_path_idx: 0,
        nav_target_x: 0.0, nav_target_z: 0.0,
        job,
        job_timer: 0.0,
        job_target_x: 0.0, job_target_z: 0.0,
        interaction_target: None,
        interacting_with: None, interaction_timer: 0.0,
        brain_idx: 0,
        fitness_money_earned: 0.0, fitness_items_picked: 0,
        fitness_interactions: 0, fitness_distance: 0.0,
        fitness_stuck_time: 0.0,
        prev_x: 0.0, prev_z: 0.0,
        health: 100.0, attack_cooldown: 0.0, attack_phase: 0.0,
        hit_flash: 0.0, knockout_timer: 0.0,
        knockback_vx: 0.0, knockback_vz: 0.0,
        attack_intent: 0, fitness_knockouts: 0, fitness_hits_landed: 0,
        hunger: 0.0, thirst: 0.0, starving_dead: false,
        fitness_starve_time: 0.0,
        sound: [0.0; 3], fitness_sounds_made: 0, fitness_npcs_heard: 0,
        fitness_proximity: 0.0,
        ragdoll_active: false,
        ragdoll_points: [[0.0; 3]; 7],

        ragdoll_timer: 0.0,
        skeleton: clauding::skeleton::Skeleton::new_humanoid(),
        body: {
            let shape = clauding::physics::CollisionShape::Capsule { radius: 0.3, half_height: 0.625 };
            let inertia = shape.inertia_diag(75.0);
            clauding::physics::RigidBody::new_dynamic([0.0, 0.0, 0.0], 75.0, inertia)
        },
        wanted: false, bounty: 0.0, violation_timer: 0.0,
        police_target: None,
        terrain_normal: [0.0, 1.0, 0.0],
        find_item_failures: 0,
        find_bin_failures: 0,
        stuck_recoveries: 0,
        height_scale: 1.0,
        walk_speed_mult: 1.0,
    }
}

fn make_trash_bin() -> state::TrashBin {
    state::TrashBin {
        x: 0.0, y: 0.0, z: 0.0,
        items_held: 3,
        carried_by: None,
        terrain_normal: [0.0, 1.0, 0.0],
    }
}

// ── World object generators (standalone, at origin) ───────────────────────

fn gen_building(tris: &mut Vec<state::WorldTri>) {
    let w = 6.0;
    let d = 5.0;
    let h = 10.0;
    let color = 0xFF887766u32;
    let bevel = 0.15_f32.min(w * 0.1).min(d * 0.1);
    mesh::beveled_box_tris(tris, 0.0, h * 0.5, 0.0, w, h, d, bevel, color);

    // Recessed windows
    let win_color = 0xFF222244u32;
    let win_h = 1.2;
    let win_w = 0.8;
    let recess_depth = 0.15;
    let floor_height = 3.0;
    let floors = ((h - 1.0) / floor_height) as i32;
    let cols = ((w - 1.0) / 2.0) as i32;

    let mut win_holes: Vec<mesh::WallHole> = Vec::new();
    for floor in 0..floors {
        let wy = 2.0 + floor as f32 * floor_height;
        for col in 0..cols {
            let wx = 1.2 + col as f32 * 2.0;
            win_holes.push(mesh::WallHole { x: wx, y: wy, w: win_w, h: win_h });
        }
    }

    // Front face (z+)
    mesh::wall_with_holes_tris(tris,
        -w * 0.5, 0.0, d * 0.5,
        w, h, &win_holes, recess_depth, color, &[win_color], 1.0, 1.0, false);
    // Back face (z-)
    mesh::wall_with_holes_tris(tris,
        w * 0.5, 0.0, -d * 0.5,
        w, h, &win_holes, recess_depth, color, &[win_color], -1.0, -1.0, false);

    // Side windows
    let side_cols = ((d - 1.0) / 2.5) as i32;
    let mut side_holes: Vec<mesh::WallHole> = Vec::new();
    for floor in 0..floors {
        let wy = 2.0 + floor as f32 * floor_height;
        for col in 0..side_cols {
            let wz = 1.5 + col as f32 * 2.5;
            side_holes.push(mesh::WallHole { x: wz, y: wy, w: win_w, h: win_h });
        }
    }
    // Right face (x+) — swap_xz for YZ-plane wall
    mesh::wall_with_holes_tris(tris,
        -d * 0.5, 0.0, w * 0.5,
        d, h, &side_holes, recess_depth, color, &[win_color], -1.0, 1.0, true);
    // Left face (x-) — swap_xz for YZ-plane wall
    mesh::wall_with_holes_tris(tris,
        d * 0.5, 0.0, -w * 0.5,
        d, h, &side_holes, recess_depth, color, &[win_color], 1.0, -1.0, true);

    // Pitched roof
    let roof_color = 0xFF665544u32;
    mesh::pitched_roof_tris(tris, 0.0, h, 0.0, w + 0.3, d + 0.3, 2.5, roof_color);

    // Cornice
    mesh::box_tris(tris, 0.0, h - 0.15, 0.0, w + 0.3, 0.3, d + 0.3, 0xFF776655);

    // Belt course
    mesh::box_tris(tris, 0.0, h * 0.5, 0.0, w + 0.15, 0.2, d + 0.15, 0xFF776655);

    // Chimney
    mesh::cylinder_tris(tris, w * 0.3, h + 1.0, -d * 0.3, 0.25, 2.0, 6, 0xFF555555);

    // Door
    mesh::box_tris(tris, 0.0, 1.1, d * 0.5 - 0.075, 1.0, 2.2, 0.15, 0xFF443322);
}

fn gen_building_flat_roof(tris: &mut Vec<state::WorldTri>) {
    let w = 5.0;
    let d = 5.0;
    let h = 8.0;
    let color = 0xFF668877u32;
    mesh::beveled_box_tris(tris, 0.0, h * 0.5, 0.0, w, h, d, 0.12, color);

    // Windows on front only (simpler test)
    let mut win_holes: Vec<mesh::WallHole> = Vec::new();
    for floor in 0..2 {
        for col in 0..2 {
            win_holes.push(mesh::WallHole { x: 1.0 + col as f32 * 2.0, y: 2.0 + floor as f32 * 3.0, w: 0.8, h: 1.2 });
        }
    }
    mesh::wall_with_holes_tris(tris, -w * 0.5, 0.0, d * 0.5,
        w, h, &win_holes, 0.15, color, &[0xFF222244], 1.0, 1.0, false);
    mesh::wall_with_holes_tris(tris, w * 0.5, 0.0, -d * 0.5,
        w, h, &win_holes, 0.15, color, &[0xFF222244], -1.0, -1.0, false);

    // Flat roof with parapet
    mesh::box_tris(tris, 0.0, h + 0.15, 0.0, w + 0.2, 0.3, d + 0.2, 0xFF556655);
    mesh::box_tris(tris, 0.0, h - 0.15, 0.0, w + 0.3, 0.3, d + 0.3, 0xFF556655);
}

fn gen_building_hip_roof(tris: &mut Vec<state::WorldTri>) {
    let w = 7.0;
    let d = 6.0;
    let h = 12.0;
    let color = 0xFF778899u32;
    mesh::beveled_box_tris(tris, 0.0, h * 0.5, 0.0, w, h, d, 0.15, color);

    // Hip roof
    mesh::hip_roof_tris(tris, 0.0, h, 0.0, w + 0.3, d + 0.3, 2.0, 0xFF665544);
    mesh::box_tris(tris, 0.0, h - 0.15, 0.0, w + 0.3, 0.3, d + 0.3, 0xFF667788);
}

fn gen_bridge(tris: &mut Vec<state::WorldTri>) {
    let bridge_len = 20.0;
    let bridge_hw = 4.0;
    let deck_y = 2.0;

    // Beveled deck
    mesh::beveled_box_tris(tris, 0.0, deck_y - 0.2, 0.0, bridge_hw * 2.0, 0.4, bridge_len, 0.05, 0xFF888877);

    // Girder under deck
    mesh::box_tris(tris, 0.0, deck_y - 0.5, 0.0, bridge_hw * 1.5, 0.2, bridge_len, 0xFF666655);

    // Pillar supports
    for pi in 0..3 {
        let t = (pi as f32 + 0.5) / 3.0;
        let pz = (t - 0.5) * bridge_len;
        let pillar_h = deck_y + 1.0;
        mesh::cylinder_tris(tris, 0.0, -0.5 + pillar_h * 0.5, pz, 0.25, pillar_h, 6, 0xFF777766);
    }

    // Railing posts + rail bars (left side)
    let rail_x_l = bridge_hw;
    let rail_x_r = -bridge_hw;
    // Rail bars (6 segments for visibility from all angles)
    mesh::cylinder_between(tris,
        [rail_x_l, deck_y + 0.8, -bridge_len * 0.5],
        [rail_x_l, deck_y + 0.8, bridge_len * 0.5],
        0.06, 6, 0xFF777766);
    mesh::cylinder_between(tris,
        [rail_x_r, deck_y + 0.8, -bridge_len * 0.5],
        [rail_x_r, deck_y + 0.8, bridge_len * 0.5],
        0.06, 6, 0xFF777766);
    // Lower rail bars
    mesh::cylinder_between(tris,
        [rail_x_l, deck_y + 0.4, -bridge_len * 0.5],
        [rail_x_l, deck_y + 0.4, bridge_len * 0.5],
        0.04, 6, 0xFF777766);
    mesh::cylinder_between(tris,
        [rail_x_r, deck_y + 0.4, -bridge_len * 0.5],
        [rail_x_r, deck_y + 0.4, bridge_len * 0.5],
        0.04, 6, 0xFF777766);
    // Railing posts
    for pi in 0..7 {
        let t = (pi as f32 + 0.5) / 7.0 - 0.5;
        let pz = t * bridge_len;
        mesh::cylinder_tris(tris, rail_x_l, deck_y + 0.4, pz, 0.04, 0.8, 6, 0xFF777766);
        mesh::cylinder_tris(tris, rail_x_r, deck_y + 0.4, pz, 0.04, 0.8, 6, 0xFF777766);
    }
}

fn gen_suburb_house(tris: &mut Vec<state::WorldTri>) {
    let hw = 5.0;
    let hd = 5.0;
    let hh = 3.0;
    let color = 0xFF99887Au32;

    // House body
    mesh::beveled_box_tris(tris, 0.0, hh * 0.5, 0.0, hw, hh, hd, 0.08, color);

    // Pitched roof
    mesh::pitched_roof_tris(tris, 0.0, hh, 0.0, hw + 0.4, hd + 0.4, 1.55, 0xFF554433);

    // Door (front face z-)
    mesh::box_tris(tris, 0.0, 0.9, -hd * 0.5 + 0.07, 0.8, 1.8, 0.14, 0xFF553322);

    // Windows
    let win_color = 0xFF222244u32;
    mesh::box_tris(tris, -hw * 0.3, hh * 0.6, -hd * 0.5, 0.7, 0.7, 0.12, win_color);
    mesh::box_tris(tris, hw * 0.3, hh * 0.6, -hd * 0.5, 0.7, 0.7, 0.12, win_color);

    // Picket fence posts
    for fp in 0..6 {
        let t = (fp as f32 + 0.5) / 6.0 * 2.0 - 1.0;
        let fx = t * 5.0;
        mesh::cylinder_tris(tris, fx, 0.4, 0.0, 0.03, 0.8, 4, 0xFF998866);
    }
}

fn gen_market_stall(tris: &mut Vec<state::WorldTri>) {
    let sw = 3.0;
    let sd = 2.0;
    let sh = 2.5;
    let canvas_color = 0xFFCC4444u32;

    // 4 cylinder posts
    for dx in [-1.0f32, 1.0] {
        for dz in [-1.0f32, 1.0] {
            mesh::cylinder_tris(tris, dx * sw * 0.45, sh * 0.5, dz * sd * 0.45,
                0.04, sh, 4, 0xFF886644);
        }
    }

    // Canvas roof (angled quad — manual tris)
    let roof_y = sh;
    let v0 = [-sw * 0.5, roof_y + 0.3, -sd * 0.5];
    let v1 = [sw * 0.5, roof_y + 0.3, -sd * 0.5];
    let v2 = [sw * 0.5, roof_y - 0.1, sd * 0.5];
    let v3 = [-sw * 0.5, roof_y - 0.1, sd * 0.5];
    let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
    let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
    let n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
    let l = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).sqrt();
    let roof_n = if l < 1e-10 { [0.0,1.0,0.0] } else { [n[0]/l,n[1]/l,n[2]/l] };
    tris.push(state::WorldTri { v: [v0, v1, v2], normal: roof_n, color: canvas_color });
    tris.push(state::WorldTri { v: [v0, v2, v3], normal: roof_n, color: canvas_color });

    // Counter front
    mesh::beveled_box_tris(tris, 0.0, 0.5, -sd * 0.5 + 0.1, sw * 0.9, 1.0, 0.2, 0.03, 0xFF886644);
}

fn gen_bus_stop(tris: &mut Vec<state::WorldTri>) {
    let shelter_w = 2.5;
    let shelter_d = 1.5;
    let shelter_h = 2.5;
    let glass_color = 0xFF88AABB_u32;

    // Back wall
    mesh::beveled_box_tris(tris, 0.0, shelter_h * 0.5, -shelter_d * 0.5,
        shelter_w, shelter_h, 0.1, 0.02, glass_color);
    // Left wall
    mesh::beveled_box_tris(tris, -shelter_w * 0.5, shelter_h * 0.5, 0.0,
        0.1, shelter_h, shelter_d, 0.02, glass_color);
    // Right wall
    mesh::beveled_box_tris(tris, shelter_w * 0.5, shelter_h * 0.5, 0.0,
        0.1, shelter_h, shelter_d, 0.02, glass_color);
    // Roof
    mesh::beveled_box_tris(tris, 0.0, shelter_h + 0.05, 0.0, shelter_w + 0.3, 0.1, shelter_d + 0.3, 0.02, 0xFF445566);

    // Bench slats
    for si in 0..3 {
        let bsz = -0.15 + si as f32 * 0.15;
        mesh::box_tris(tris, 0.0, 0.25, bsz, 1.5, 0.04, 0.1, 0xFF886644);
    }

    // Sign post + sign
    let sign_x = shelter_w * 0.5 + 0.5;
    mesh::cylinder_tris(tris, sign_x, 1.5, 0.0, 0.04, 3.0, 4, 0xFF666666);
    mesh::beveled_box_tris(tris, sign_x, 3.1, 0.0, 0.4, 0.4, 0.08, 0.02, 0xFF2255CC);
}

fn gen_vending_machine(tris: &mut Vec<state::WorldTri>) {
    mesh::beveled_box_tris(tris, 0.0, 0.75, 0.0, 0.7, 1.5, 0.6, 0.04, 0xFFCC2222);
    // Recessed panel
    mesh::box_tris(tris, 0.0, 0.9, -0.25, 0.55, 0.7, 0.06, 0xFF888888);
}

fn gen_phone_booth(tris: &mut Vec<state::WorldTri>) {
    mesh::beveled_box_tris(tris, 0.0, 1.1, 0.0, 0.8, 2.2, 0.8, 0.06, 0xFF667788);
    // Domed roof
    mesh::sphere_tris(tris, 0.0, 2.25, 0.0, 0.45, 1, 0xFF667788);
}

fn gen_fire_hydrant(tris: &mut Vec<state::WorldTri>) {
    let profile: [[f32;2]; 6] = [
        [0.0, 0.0], [0.12, 0.0], [0.1, 0.25],
        [0.15, 0.35], [0.08, 0.5], [0.0, 0.55],
    ];
    mesh::lathe_tris(tris, 0.0, 0.0, 0.0, &profile, 6, 0xFFCC3333);
}

fn gen_picnic_table(tris: &mut Vec<state::WorldTri>) {
    // Table top
    mesh::box_tris(tris, 0.0, 0.75, 0.0, 1.8, 0.08, 0.9, 0xFF886644);
    // Two bench slabs
    mesh::box_tris(tris, 0.0, 0.3, -0.7, 1.8, 0.06, 0.25, 0xFF886644);
    mesh::box_tris(tris, 0.0, 0.3, 0.7, 1.8, 0.06, 0.25, 0xFF886644);
    // Legs
    for lx in [-0.7f32, 0.7] {
        mesh::cylinder_tris(tris, lx, 0.375, 0.0, 0.03, 0.75, 4, 0xFF886644);
    }
}

fn gen_water_tower(tris: &mut Vec<state::WorldTri>) {
    // Legs
    for (lx, lz) in [(-0.5f32, -0.5f32), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
        mesh::cylinder_tris(tris, lx, 1.5, lz, 0.08, 3.0, 5, 0xFF888888);
    }
    // Tank
    mesh::sphere_tris(tris, 0.0, 4.0, 0.0, 1.5, 1, 0xFF888888);
}

fn gen_billboard(tris: &mut Vec<state::WorldTri>) {
    mesh::cylinder_tris(tris, 0.0, 2.5, 0.0, 0.12, 5.0, 6, 0xFF666666);
    mesh::beveled_box_tris(tris, 0.0, 5.5, 0.0, 3.0, 2.0, 0.5, 0.03, 0xFFDDDDCC);
}

fn gen_tree(tris: &mut Vec<state::WorldTri>) {
    let trunk_h = 2.5;
    let trunk_r = 0.18;
    let canopy_r = 1.8;

    // Trunk
    mesh::cylinder_tris(tris, 0.0, trunk_h * 0.5, 0.0, trunk_r, trunk_h, 6, 0xFF554422);

    // Branches
    let branch_base_y = trunk_h * 0.7;
    for bi in 0..3 {
        let angle = (bi as f32 / 3.0) * std::f32::consts::TAU;
        let blen = canopy_r * 0.6;
        let bx = angle.cos() * blen * 0.5;
        let bz = angle.sin() * blen * 0.5;
        let by = branch_base_y + blen * 0.4;
        mesh::cylinder_between(tris, [0.0, branch_base_y, 0.0], [bx, by, bz], 0.06, 4, 0xFF554422);
    }

    // Canopy clusters
    mesh::sphere_tris(tris, 0.0, trunk_h + canopy_r * 0.3, 0.0, canopy_r * 0.6, 1, 0xFF338833);
    mesh::sphere_tris(tris, canopy_r * 0.3, trunk_h + canopy_r * 0.1, canopy_r * 0.2, canopy_r * 0.5, 1, 0xFF228822);
    mesh::sphere_tris(tris, -canopy_r * 0.3, trunk_h, -canopy_r * 0.2, canopy_r * 0.45, 1, 0xFF448844);
}

fn gen_wave_surface(tris: &mut Vec<state::WorldTri>) {
    mesh::wave_surface_tris(tris, -5.0, 5.0, -3.0, 3.0, 0.0, 0.5, 0.5, 10, 6, 0xFF224466);
}

fn gen_dumpster(tris: &mut Vec<state::WorldTri>) {
    mesh::beveled_box_tris(tris, 0.0, 0.5, 0.0, 1.2, 1.0, 0.8, 0.05, 0xFF334488);
    mesh::box_tris(tris, 0.0, 1.05, 0.0, 1.25, 0.08, 0.82, 0xFF445599);
}

fn gen_street_light(tris: &mut Vec<state::WorldTri>) {
    let base_color = 0xFF555555u32;
    let pole_color = 0xFF666666u32;
    // Base mounting plate (flat disc)
    for pi in 0..8u32 {
        let a0 = (pi as f32 / 8.0) * std::f32::consts::TAU;
        let a1 = ((pi + 1) as f32 / 8.0) * std::f32::consts::TAU;
        tris.push(state::WorldTri {
            v: [[0.0, 0.04, 0.0],
                [a1.cos() * 0.25, 0.04, a1.sin() * 0.25],
                [a0.cos() * 0.25, 0.04, a0.sin() * 0.25]],
            normal: [0.0, 1.0, 0.0], color: base_color,
        });
    }
    // Wider base section
    mesh::cylinder_tris(tris, 0.0, 0.2, 0.0, 0.12, 0.4, 6, base_color);
    // Main pole (8 segments for rounder appearance)
    mesh::cylinder_tris(tris, 0.0, 2.5, 0.0, 0.06, 4.2, 8, pole_color);
    // Curved arm (cylinder between)
    mesh::cylinder_between(tris, [0.0, 4.6, 0.0], [0.8, 4.7, 0.0], 0.03, 4, pole_color);
    // Lamp globe
    mesh::sphere_tris(tris, 0.8, 4.7, 0.0, 0.2, 1, 0xFFFFEE88);
}

fn gen_crane(tris: &mut Vec<state::WorldTri>) {
    let crane_h = 15.0;
    // Tower
    mesh::cylinder_tris(tris, 0.0, crane_h * 0.5, 0.0, 0.35, crane_h, 8, 0xFFCC8833);
    // Boom arm
    mesh::cylinder_between(tris, [0.0, crane_h, 0.0], [8.0, crane_h - 0.5, 0.0], 0.15, 6, 0xFFCC8833);
    // Counterweight
    mesh::beveled_box_tris(tris, -3.0, crane_h - 1.0, 0.0, 2.0, 2.0, 1.5, 0.1, 0xFF555555);
    // Cabin
    mesh::beveled_box_tris(tris, 0.0, crane_h - 2.0, 0.0, 1.5, 2.0, 1.5, 0.08, 0xFF888833);
}

fn gen_warehouse(tris: &mut Vec<state::WorldTri>) {
    let ww = 10.0;
    let wd = 8.0;
    let wh = 5.0;
    let color = 0xFF666655u32;
    mesh::beveled_box_tris(tris, 0.0, wh * 0.5, 0.0, ww, wh, wd, 0.1, color);
    // Garage door
    mesh::box_tris(tris, 0.0, 2.0, -wd * 0.5 + 0.08, ww * 0.4, 4.0, 0.16, 0xFF333322);
    // Pitched roof
    mesh::pitched_roof_tris(tris, 0.0, wh, 0.0, ww + 0.2, wd + 0.2, 1.5, 0xFF555544);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_gpu = args.iter().any(|a| a == "--gpu");

    let mut gpu_ctx = if use_gpu {
        match gpu::GpuContext::try_new() {
            Some(mut ctx) => {
                eprintln!("GPU mode: {}", ctx.device_name);
                ctx.init_graphics(VIEW_W as u32, VIEW_H as u32);
                Some(ctx)
            }
            None => {
                eprintln!("No Vulkan GPU — falling back to CPU rasterizer");
                None
            }
        }
    } else {
        None
    };

    let _ = std::fs::create_dir_all("debug");
    let mut tris: Vec<state::WorldTri> = Vec::with_capacity(8192);

    // ── Player: 8K smooth-shaded close-up inspection (3 sheets) ──
    let player = make_player();
    tris.clear();
    mesh::set_mesh_quality(2, 3); // high tessellation for smooth surfaces
    render::gen_player_mesh(&player, &mut tris);
    mesh::set_mesh_quality(0, 1); // reset to defaults
    let vertex_normals = compute_smooth_normals(&tris);
    eprintln!("Player mesh: {} tris, smooth normals computed", tris.len());

    let cy = 1.25; // center of stretched body
    let d = 3.0;   // camera distance (taller body needs more room)

    // Sheet 1: Flat (eye-level) — front, right side, back, left side
    let flat_views: Vec<([f32; 3], [f32; 3], &str)> = vec![
        ([0.0, cy, -d],  [0.0, cy, 0.0], "Front"),
        ([d, cy, 0.0],   [0.0, cy, 0.0], "Right"),
        ([0.0, cy, d],   [0.0, cy, 0.0], "Back"),
        ([-d, cy, 0.0],  [0.0, cy, 0.0], "Left"),
    ];
    let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vertex_normals, &flat_views, "Player Flat");
    save_png(&img, iw, ih, "debug/model_player_flat.png");

    // Sheet 2: Diagonal (3/4 elevated ~35°) — front-right, back-right, back-left, front-left
    let elev = 0.7; // camera raised above center
    let dd = d * 0.707; // diagonal distance component (d/sqrt(2))
    let diag_views: Vec<([f32; 3], [f32; 3], &str)> = vec![
        ([dd, cy + elev, -dd],  [0.0, cy, 0.0], "3/4 Front-R"),
        ([dd, cy + elev, dd],   [0.0, cy, 0.0], "3/4 Back-R"),
        ([-dd, cy + elev, dd],  [0.0, cy, 0.0], "3/4 Back-L"),
        ([-dd, cy + elev, -dd], [0.0, cy, 0.0], "3/4 Front-L"),
    ];
    let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vertex_normals, &diag_views, "Player Diagonal");
    save_png(&img, iw, ih, "debug/model_player_diagonal.png");

    // Sheet 3: Vertical (overhead ~65°) — 4 cardinal directions looking down
    let high = 1.8; // camera well above
    let vd = 1.2;   // closer horizontal distance (steep angle)
    let vert_views: Vec<([f32; 3], [f32; 3], &str)> = vec![
        ([0.0, cy + high, -vd],  [0.0, cy, 0.0], "Top Front"),
        ([vd, cy + high, 0.0],   [0.0, cy, 0.0], "Top Right"),
        ([0.0, cy + high, vd],   [0.0, cy, 0.0], "Top Back"),
        ([-vd, cy + high, 0.0],  [0.0, cy, 0.0], "Top Left"),
    ];
    let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vertex_normals, &vert_views, "Player Vertical");
    save_png(&img, iw, ih, "debug/model_player_vertical.png");

    // ── Player Normal map ──
    {
        let img = render_model_sheet_normals(&tris, cy, d, "Player Normal map");
        save_png(&img, IMG_W, IMG_H, "debug/model_player_normals.png");
    }

    // ── HEAD CLOSE-UP — 6 views for detailed comparison with ACU bust reference ──
    // Head Y range after transforms: chin ~1.82, crown ~2.23, center ~2.02
    {
        let hcy = 2.00; // head center Y (shows head + upper neck)
        let hcz = 0.03; // head center Z (cranium extends behind, shift target back)
        let hd = 0.65;  // camera distance (fits full head with margins)
        let hdd = hd * 0.707;
        let head_views: Vec<([f32; 3], [f32; 3], &str)> = vec![
            ([0.0, hcy, hcz - hd],         [0.0, hcy, hcz], "Front"),
            ([hdd, hcy, hcz - hdd],         [0.0, hcy, hcz], "3/4 Front-R"),
            ([hd, hcy, hcz],                [0.0, hcy, hcz], "Right"),
            ([hdd, hcy, hcz + hdd],          [0.0, hcy, hcz], "3/4 Back-R"),
            ([0.0, hcy, hcz + hd],           [0.0, hcy, hcz], "Back"),
            ([0.0, hcy + 0.25, hcz - hd * 0.8], [0.0, hcy - 0.05, hcz], "Below"),
        ];
        let head_panel_w: usize = 720;
        let head_panel_h: usize = 960;
        let head_n = head_views.len();
        let head_sheet_w = head_panel_w * head_n;
        let head_sheet_h = head_panel_h;

        let mut head_fb = raster::Framebuffer::new(head_panel_w, head_panel_h);
        let mut head_composite = vec![0xFF3A4455u32; head_sheet_w * head_sheet_h];

        let head_tri_label = format!("tris: {}", tris.len());
        for (vi, (eye, target, label)) in head_views.iter().enumerate() {
            head_fb.clear(0xFF445566);
            render_model_smooth(&mut head_fb, &tris, &vertex_normals, *eye, *target);

            for dy in 0..2_usize {
                for dx in 0..2_usize {
                    draw_label_scaled(&mut head_fb, 8 + dx, 8 + dy, label, 2);
                    draw_label_scaled(&mut head_fb, 8 + dx, 30 + dy, &head_tri_label, 2);
                }
            }

            let qx = vi * head_panel_w;
            for y in 0..head_panel_h {
                for x in 0..head_panel_w {
                    head_composite[y * head_sheet_w + (qx + x)] = head_fb.pixels[y * head_panel_w + x];
                }
            }
        }

        // Panel separators
        for pi in 1..head_n {
            let sx = pi * head_panel_w;
            for y in 0..head_sheet_h {
                for dx in 0..2_usize {
                    if sx + dx < head_sheet_w { head_composite[y * head_sheet_w + sx + dx] = 0xFFFFFFFF; }
                    if sx >= dx + 1 { head_composite[y * head_sheet_w + sx - 1 - dx] = 0xFFFFFFFF; }
                }
            }
        }

        eprintln!("Rendered head close-up: {} views, {}x{}", head_n, head_sheet_w, head_sheet_h);
        save_png(&head_composite, head_sheet_w, head_sheet_h, "debug/model_head.png");
    }

    // ── Female Player: same 3 sheets + head close-up ──
    {
        let mut female_player = make_player();
        female_player.is_female = true;
        tris.clear();
        mesh::set_mesh_quality(2, 3);
        render::gen_player_mesh(&female_player, &mut tris);
        mesh::set_mesh_quality(0, 1);
        let vn = compute_smooth_normals(&tris);
        eprintln!("Female player mesh: {} tris, smooth normals computed", tris.len());

        let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vn, &flat_views, "Female Flat");
        save_png(&img, iw, ih, "debug/model_player_female_flat.png");

        let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vn, &diag_views, "Female Diagonal");
        save_png(&img, iw, ih, "debug/model_player_female_diagonal.png");

        let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vn, &vert_views, "Female Vertical");
        save_png(&img, iw, ih, "debug/model_player_female_vertical.png");

        // Female head close-up
        let hcy = 2.00;
        let hcz = 0.03;
        let hd = 0.65;
        let hdd = hd * 0.707;
        let head_views: Vec<([f32; 3], [f32; 3], &str)> = vec![
            ([0.0, hcy, hcz - hd],         [0.0, hcy, hcz], "Front"),
            ([hdd, hcy, hcz - hdd],         [0.0, hcy, hcz], "3/4 Front-R"),
            ([hd, hcy, hcz],                [0.0, hcy, hcz], "Right"),
            ([hdd, hcy, hcz + hdd],          [0.0, hcy, hcz], "3/4 Back-R"),
            ([0.0, hcy, hcz + hd],           [0.0, hcy, hcz], "Back"),
            ([0.0, hcy + 0.25, hcz - hd * 0.8], [0.0, hcy - 0.05, hcz], "Below"),
        ];
        let head_panel_w: usize = 720;
        let head_panel_h: usize = 960;
        let head_n = head_views.len();
        let head_sheet_w = head_panel_w * head_n;
        let head_sheet_h = head_panel_h;

        let mut head_fb = raster::Framebuffer::new(head_panel_w, head_panel_h);
        let mut head_composite = vec![0xFF3A4455u32; head_sheet_w * head_sheet_h];

        let head_tri_label = format!("tris: {}", tris.len());
        for (vi, (eye, target, label)) in head_views.iter().enumerate() {
            head_fb.clear(0xFF445566);
            render_model_smooth(&mut head_fb, &tris, &vn, *eye, *target);
            for dy in 0..2_usize {
                for dx in 0..2_usize {
                    draw_label_scaled(&mut head_fb, 8 + dx, 8 + dy, label, 2);
                    draw_label_scaled(&mut head_fb, 8 + dx, 30 + dy, &head_tri_label, 2);
                }
            }
            let qx = vi * head_panel_w;
            for y in 0..head_panel_h {
                for x in 0..head_panel_w {
                    head_composite[y * head_sheet_w + (qx + x)] = head_fb.pixels[y * head_panel_w + x];
                }
            }
        }
        for pi in 1..head_n {
            let sx = pi * head_panel_w;
            for y in 0..head_sheet_h {
                for dx in 0..2_usize {
                    if sx + dx < head_sheet_w { head_composite[y * head_sheet_w + sx + dx] = 0xFFFFFFFF; }
                    if sx >= dx + 1 { head_composite[y * head_sheet_w + sx - 1 - dx] = 0xFFFFFFFF; }
                }
            }
        }
        eprintln!("Rendered female head close-up: {} views, {}x{}", head_n, head_sheet_w, head_sheet_h);
        save_png(&head_composite, head_sheet_w, head_sheet_h, "debug/model_head_female.png");
    }

    // ── Clothed Player (ACU outfit): 3 sheets ──
    {
        tris.clear();
        mesh::set_mesh_quality(2, 3);
        render::gen_clothed_player_body(&mut tris, false);
        mesh::set_mesh_quality(0, 1);
        let vn = compute_smooth_normals(&tris);
        eprintln!("Clothed player mesh: {} tris", tris.len());

        let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vn, &flat_views, "Clothed Flat");
        save_png(&img, iw, ih, "debug/model_clothed_player_flat.png");
        let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vn, &diag_views, "Clothed Diagonal");
        save_png(&img, iw, ih, "debug/model_clothed_player_diagonal.png");
        let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vn, &vert_views, "Clothed Vertical");
        save_png(&img, iw, ih, "debug/model_clothed_player_vertical.png");
    }

    // ── Clothed Female Player (ACU outfit): 3 sheets ──
    {
        tris.clear();
        mesh::set_mesh_quality(2, 3);
        render::gen_clothed_player_body(&mut tris, true);
        mesh::set_mesh_quality(0, 1);
        let vn = compute_smooth_normals(&tris);
        eprintln!("Clothed female player mesh: {} tris", tris.len());

        let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vn, &flat_views, "Clothed Female Flat");
        save_png(&img, iw, ih, "debug/model_clothed_female_flat.png");
        let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vn, &diag_views, "Clothed Female Diagonal");
        save_png(&img, iw, ih, "debug/model_clothed_female_diagonal.png");
        let (img, iw, ih) = render_8k_sheet_smooth(&tris, &vn, &vert_views, "Clothed Female Vertical");
        save_png(&img, iw, ih, "debug/model_clothed_female_vertical.png");
    }

    // ══════════════════════════════════════════════════════════════════════
    // FACE VARIATION GRID — 12 slider presets, front + 3/4 views per face
    // ══════════════════════════════════════════════════════════════════════
    {
        let presets: Vec<(&str, render::FaceSliders, u32)> = vec![
            ("Default",          render::FaceSliders::default_face(),       0xFFDEB887),
            ("Male Default",     render::FaceSliders::male_default(),       0xFFD2A87A),
            ("Female Default",   render::FaceSliders::female_default(),     0xFFE8C9A0),
            ("Square Jaw",       render::FaceSliders::preset_square_jaw(),  0xFFC89B6E),
            ("Narrow",           render::FaceSliders::preset_narrow(),      0xFFDDBC98),
            ("Round",            render::FaceSliders::preset_round(),       0xFFCCA882),
            ("Heavy Brow",       render::FaceSliders::preset_heavy_brow(),  0xFFBB9060),
            ("High Cheekbones",  render::FaceSliders::preset_high_cheekbones(), 0xFFA07850),
            ("Long Face",        render::FaceSliders::preset_long_face(),   0xFFDEB887),
            ("Wide",             render::FaceSliders::preset_wide(),        0xFFD2A87A),
            ("Delicate",         render::FaceSliders::preset_delicate(),    0xFFE8C9A0),
            ("Rugged",           render::FaceSliders::preset_rugged(),      0xFFC89B6E),
            ("Broad Nose",       render::FaceSliders::preset_broad_nose(),  0xFFBB9060),
            ("Sharp",            render::FaceSliders::preset_sharp(),       0xFFCCA882),
            ("Soft",             render::FaceSliders::preset_soft(),        0xFFDDBC98),
        ];

        // Each face gets 2 panels: front + 3/4 view
        let panel_w: usize = 360;
        let panel_h: usize = 480;
        let cols = 5;  // 5 faces per row
        let rows = (presets.len() + cols - 1) / cols;  // 3 rows
        let panels_per_face = 2;  // front + 3/4
        let sheet_w = panel_w * panels_per_face * cols;
        let sheet_h = panel_h * rows;

        let mut face_fb = raster::Framebuffer::new(panel_w, panel_h);
        let mut face_composite = vec![0xFF2A3040u32; sheet_w * sheet_h];

        let hcy = 2.00;
        let hcz = 0.03;
        let hd = 0.65;
        let hdd = hd * 0.707;
        let face_cam_views: [([f32; 3], [f32; 3]); 2] = [
            ([0.0, hcy, hcz - hd],         [0.0, hcy, hcz]),       // Front
            ([hdd, hcy, hcz - hdd],         [0.0, hcy, hcz]),       // 3/4 Front-R
        ];

        mesh::set_mesh_quality(1, 2); // moderate quality for grid
        for (pi, (name, sliders, skin)) in presets.iter().enumerate() {
            tris.clear();
            let is_female = name.contains("Female") || name.contains("Delicate") || name.contains("Soft");
            render::gen_head_standalone(&mut tris, sliders, *skin, is_female);

            let vn = compute_smooth_normals(&tris);

            let row = pi / cols;
            let col = pi % cols;

            for (vi, (eye, target)) in face_cam_views.iter().enumerate() {
                face_fb.clear(0xFF3A4A5A);
                render_model_smooth(&mut face_fb, &tris, &vn, *eye, *target);

                // Label
                draw_label(&mut face_fb, 4, 4, name);
                draw_label(&mut face_fb, 4, 14, &format!("tris:{}", tris.len()));

                let dest_x = (col * panels_per_face + vi) * panel_w;
                let dest_y = row * panel_h;
                for y in 0..panel_h {
                    for x in 0..panel_w {
                        let dx = dest_x + x;
                        let dy = dest_y + y;
                        if dx < sheet_w && dy < sheet_h {
                            face_composite[dy * sheet_w + dx] = face_fb.pixels[y * panel_w + x];
                        }
                    }
                }
            }
        }
        mesh::set_mesh_quality(0, 1); // reset

        // Grid lines
        for row in 1..rows {
            let sy = row * panel_h;
            for x in 0..sheet_w {
                if sy < sheet_h { face_composite[sy * sheet_w + x] = 0xFFFFFFFF; }
            }
        }
        for col in 1..(cols * panels_per_face) {
            let sx = col * panel_w;
            for y in 0..sheet_h {
                if sx < sheet_w { face_composite[y * sheet_w + sx] = 0xFFFFFFFF; }
            }
        }

        eprintln!("Rendered face variation grid: {} presets, {}x{}", presets.len(), sheet_w, sheet_h);
        save_png(&face_composite, sheet_w, sheet_h, "debug/model_face_variations.png");
    }

    let vehicle = make_vehicle(0xFFCC3333);
    tris.clear();
    render::gen_vehicle_mesh(&vehicle, &mut tris, false, false);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.7, 5.5, "Vehicle");
    save_png(&img, IMG_W, IMG_H, "debug/model_vehicle.png");

    let vehicle_interior = make_vehicle(0xFF3333CC);
    tris.clear();
    render::gen_vehicle_mesh(&vehicle_interior, &mut tris, true, false);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.7, 5.5, "Vehicle Interior");
    save_png(&img, IMG_W, IMG_H, "debug/model_vehicle_int.png");

    // Vehicle mid LOD
    tris.clear();
    render::gen_vehicle_mesh_mid(&vehicle, &mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.7, 7.0, "Vehicle Mid LOD");
    save_png(&img, IMG_W, IMG_H, "debug/model_vehicle_mid.png");

    let npc = make_npc(state::NpcJob::Collector);
    tris.clear();
    render::gen_npc_mesh(&npc, &mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 1.2, 4.5, "NPC Collector");
    save_png(&img, IMG_W, IMG_H, "debug/model_npc.png");

    // ── NPC LOD comparison: full, mid, low side by side ──
    {
        let lod_npc = make_npc(state::NpcJob::DeliveryCourier);
        // Mid-detail LOD
        tris.clear();
        render::gen_npc_mesh_mid(&lod_npc, &mut tris);
        let img = render_sheet(&mut gpu_ctx, &tris, 1.2, 4.5, "NPC Mid LOD");
        save_png(&img, IMG_W, IMG_H, "debug/model_npc_mid.png");
    }

    // ── NPC KO pose ──
    {
        let mut ko_npc = make_npc(state::NpcJob::Collector);
        ko_npc.state = state::NpcState::KnockedOut;
        tris.clear();
        render::gen_npc_mesh(&ko_npc, &mut tris);
        let img = render_sheet(&mut gpu_ctx, &tris, 0.5, 4.5, "NPC Knocked Out");
        save_png(&img, IMG_W, IMG_H, "debug/model_npc_ko.png");
    }

    // ── NPC ragdoll pose ──
    {
        let mut rag_npc = make_npc(state::NpcJob::Collector);
        rag_npc.ragdoll_active = true;
        // Simulate a tumbling ragdoll — body tilted ~45° with spread limbs
        rag_npc.ragdoll_points = [
            [0.0, 0.5, 0.0],       // hips
            [0.0, 1.2, -0.3],      // chest
            [0.1, 1.7, -0.5],      // head
            [-0.5, 0.9, -0.2],     // l_hand
            [0.5, 0.8, -0.4],      // r_hand
            [-0.2, 0.0, 0.3],      // l_foot
            [0.2, 0.0, 0.2],       // r_foot
        ];
        tris.clear();
        render::gen_npc_mesh(&rag_npc, &mut tris);
        let img = render_sheet(&mut gpu_ctx, &tris, 0.8, 4.5, "NPC Ragdoll");
        save_png(&img, IMG_W, IMG_H, "debug/model_npc_ragdoll.png");
    }

    // ── NPC Walking (mid-stride) ──
    {
        let mut walk_npc = make_npc(state::NpcJob::Collector);
        walk_npc.walk_phase = 1.5;
        tris.clear();
        render::gen_npc_mesh(&walk_npc, &mut tris);
        let img = render_sheet(&mut gpu_ctx, &tris, 1.2, 4.5, "NPC Walking");
        save_png(&img, IMG_W, IMG_H, "debug/model_npc_walk.png");
    }

    // ── NPC Attacking (mid-swing) ──
    {
        let mut atk_npc = make_npc(state::NpcJob::Collector);
        atk_npc.attack_phase = 0.5;
        tris.clear();
        render::gen_npc_mesh(&atk_npc, &mut tris);
        let img = render_sheet(&mut gpu_ctx, &tris, 1.2, 4.5, "NPC Attacking");
        save_png(&img, IMG_W, IMG_H, "debug/model_npc_attack.png");
    }

    // ── NPC Carrying ──
    {
        let mut carry_npc = make_npc(state::NpcJob::Collector);
        carry_npc.carrying_item = true;
        tris.clear();
        render::gen_npc_mesh(&carry_npc, &mut tris);
        let img = render_sheet(&mut gpu_ctx, &tris, 1.2, 4.5, "NPC Carrying");
        save_png(&img, IMG_W, IMG_H, "debug/model_npc_carry.png");
    }

    let bin = make_trash_bin();
    tris.clear();
    render::gen_trash_bin_mesh(&bin, &mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.4, 2.5, "Trash Bin");
    save_png(&img, IMG_W, IMG_H, "debug/model_trashbin.png");

    // ── World objects ──
    tris.clear(); gen_building(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 5.0, 20.0, "Building Pitched");
    save_png(&img, IMG_W, IMG_H, "debug/model_building.png");

    tris.clear(); gen_building_flat_roof(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 4.0, 18.0, "Building Flat");
    save_png(&img, IMG_W, IMG_H, "debug/model_building_flat.png");

    tris.clear(); gen_building_hip_roof(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 6.0, 22.0, "Building Hip");
    save_png(&img, IMG_W, IMG_H, "debug/model_building_hip.png");

    tris.clear(); gen_bridge(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 2.0, 28.0, "Bridge");
    save_png(&img, IMG_W, IMG_H, "debug/model_bridge.png");

    tris.clear(); gen_suburb_house(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 1.5, 12.0, "Suburb House");
    save_png(&img, IMG_W, IMG_H, "debug/model_suburb.png");

    tris.clear(); gen_market_stall(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 1.5, 8.0, "Market Stall");
    save_png(&img, IMG_W, IMG_H, "debug/model_stall.png");

    tris.clear(); gen_bus_stop(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 1.5, 8.0, "Bus Stop");
    save_png(&img, IMG_W, IMG_H, "debug/model_busstop.png");

    tris.clear(); gen_vending_machine(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.75, 3.5, "Vending Machine");
    save_png(&img, IMG_W, IMG_H, "debug/model_vending.png");

    tris.clear(); gen_phone_booth(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 1.1, 5.0, "Phone Booth");
    save_png(&img, IMG_W, IMG_H, "debug/model_phonebooth.png");

    tris.clear(); gen_fire_hydrant(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.3, 1.5, "Fire Hydrant");
    save_png(&img, IMG_W, IMG_H, "debug/model_hydrant.png");

    tris.clear(); gen_picnic_table(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.5, 4.0, "Picnic Table");
    save_png(&img, IMG_W, IMG_H, "debug/model_picnic.png");

    tris.clear(); gen_water_tower(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 3.0, 12.0, "Water Tower");
    save_png(&img, IMG_W, IMG_H, "debug/model_watertower.png");

    tris.clear(); gen_billboard(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 3.0, 12.0, "Billboard");
    save_png(&img, IMG_W, IMG_H, "debug/model_billboard.png");

    tris.clear(); gen_tree(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 2.0, 8.0, "Tree");
    save_png(&img, IMG_W, IMG_H, "debug/model_tree.png");

    tris.clear(); gen_wave_surface(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.5, 14.0, "Wave Surface");
    save_png(&img, IMG_W, IMG_H, "debug/model_wave.png");

    tris.clear(); gen_dumpster(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.5, 3.5, "Dumpster");
    save_png(&img, IMG_W, IMG_H, "debug/model_dumpster.png");

    tris.clear(); gen_street_light(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 2.0, 8.0, "Street Light");
    save_png(&img, IMG_W, IMG_H, "debug/model_streetlight.png");

    tris.clear(); gen_crane(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 7.0, 25.0, "Crane");
    save_png(&img, IMG_W, IMG_H, "debug/model_crane.png");

    tris.clear(); gen_warehouse(&mut tris);
    let img = render_sheet(&mut gpu_ctx, &tris, 3.0, 18.0, "Warehouse");
    save_png(&img, IMG_W, IMG_H, "debug/model_warehouse.png");

    // ── Primitives ──
    tris.clear();
    mesh::cylinder_tris(&mut tris, 0.0, 0.5, 0.0, 0.3, 1.0, 8, 0xFF3388CC);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.5, 3.0, "Cylinder");
    save_png(&img, IMG_W, IMG_H, "debug/model_cylinder.png");

    tris.clear();
    mesh::sphere_tris(&mut tris, 0.0, 0.5, 0.0, 0.5, 2, 0xFFCC4433);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.5, 2.5, "Sphere");
    save_png(&img, IMG_W, IMG_H, "debug/model_sphere.png");

    tris.clear();
    mesh::beveled_box_tris(&mut tris, 0.0, 0.5, 0.0, 1.0, 1.0, 1.0, 0.1, 0xFF44AA44);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.5, 3.0, "Beveled Box");
    save_png(&img, IMG_W, IMG_H, "debug/model_bevelbox.png");

    tris.clear();
    mesh::cone_tris(&mut tris, 0.0, 0.0, 0.0, 0.4, 1.0, 8, 0xFFCC8833);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.5, 3.0, "Cone");
    save_png(&img, IMG_W, IMG_H, "debug/model_cone.png");

    tris.clear();
    mesh::box_tris(&mut tris, 0.0, 0.5, 0.0, 1.0, 1.0, 1.0, 0xFF5577AA);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.5, 3.0, "Box");
    save_png(&img, IMG_W, IMG_H, "debug/model_box.png");

    tris.clear();
    mesh::pitched_roof_tris(&mut tris, 0.0, 0.0, 0.0, 4.0, 3.0, 1.5, 0xFF885544);
    let img = render_sheet(&mut gpu_ctx, &tris, 1.0, 8.0, "Pitched Roof");
    save_png(&img, IMG_W, IMG_H, "debug/model_pitchedroof.png");

    tris.clear();
    mesh::hip_roof_tris(&mut tris, 0.0, 0.0, 0.0, 4.0, 3.0, 1.5, 0xFF885544);
    let img = render_sheet(&mut gpu_ctx, &tris, 1.0, 8.0, "Hip Roof");
    save_png(&img, IMG_W, IMG_H, "debug/model_hiproof.png");

    tris.clear();
    let profile: [[f32;2]; 6] = [
        [0.0, 0.0], [0.3, 0.0], [0.25, 0.5],
        [0.35, 0.7], [0.2, 1.0], [0.0, 1.1],
    ];
    mesh::lathe_tris(&mut tris, 0.0, 0.0, 0.0, &profile, 8, 0xFFCC6644);
    let img = render_sheet(&mut gpu_ctx, &tris, 0.5, 3.0, "Lathe");
    save_png(&img, IMG_W, IMG_H, "debug/model_lathe.png");

    // Wall with holes standalone test
    tris.clear();
    let holes = vec![
        mesh::WallHole { x: 1.0, y: 1.0, w: 0.8, h: 1.2 },
        mesh::WallHole { x: 3.0, y: 1.0, w: 0.8, h: 1.2 },
        mesh::WallHole { x: 1.0, y: 3.5, w: 0.8, h: 1.2 },
        mesh::WallHole { x: 3.0, y: 3.5, w: 0.8, h: 1.2 },
    ];
    mesh::wall_with_holes_tris(&mut tris, -2.5, 0.0, 0.0, 5.0, 6.0, &holes, 0.15,
        0xFF887766, &[0xFF222244], 1.0, 1.0, false);
    let img = render_sheet(&mut gpu_ctx, &tris, 3.0, 10.0, "Wall Holes Z+");
    save_png(&img, IMG_W, IMG_H, "debug/model_wallholes.png");

    tris.clear();
    mesh::wall_with_holes_tris(&mut tris, 2.5, 0.0, 0.0, 5.0, 6.0, &holes, 0.15,
        0xFF887766, &[0xFF222244], -1.0, -1.0, false);
    let img = render_sheet(&mut gpu_ctx, &tris, 3.0, 10.0, "Wall Holes Z-");
    save_png(&img, IMG_W, IMG_H, "debug/model_wallholes_back.png");

    eprintln!("All model sheets saved to debug/model_*.png");
}
