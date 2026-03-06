// Model viewer: renders each entity type from 4 orthographic views
// Usage: cargo run --bin model_viewer
// Output: /tmp/clauding_models_*.png

use clauding::{state, render, raster, math, mesh};
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

fn clip_to_screen(c: [f32; 4], w: f32, h: f32) -> [f32; 3] {
    let inv_w = 1.0 / c[3];
    [
        (c[0] * inv_w + 1.0) * 0.5 * w,
        (1.0 - c[1] * inv_w) * 0.5 * h,
        c[2] * inv_w,
    ]
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

fn save_png(pixels: &[u32], w: usize, h: usize, path: &str) {
    use std::io::Write;

    // Build raw scanlines: filter_byte(0) + RGB per pixel, per row
    let row_bytes = 1 + w * 3;
    let mut raw = Vec::with_capacity(row_bytes * h);
    for y in 0..h {
        raw.push(0u8); // filter: None
        for x in 0..w {
            let c = pixels[y * w + x];
            raw.push(((c >> 16) & 0xFF) as u8);
            raw.push(((c >> 8) & 0xFF) as u8);
            raw.push((c & 0xFF) as u8);
        }
    }

    // Deflate using uncompressed (stored) blocks — valid zlib, just no compression
    let mut deflate = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 20);
    // Zlib header: CM=8 (deflate), CINFO=7 (32K window), FCHECK so header%31==0
    deflate.push(0x78);
    deflate.push(0x01);

    // Split into stored blocks (max 65535 bytes each)
    let mut offset = 0;
    while offset < raw.len() {
        let remaining = raw.len() - offset;
        let block_len = remaining.min(65535);
        let is_last = offset + block_len >= raw.len();
        deflate.push(if is_last { 1 } else { 0 }); // BFINAL + BTYPE=00 (stored)
        deflate.push((block_len & 0xFF) as u8);
        deflate.push(((block_len >> 8) & 0xFF) as u8);
        deflate.push((!block_len & 0xFF) as u8);
        deflate.push(((!block_len >> 8) & 0xFF) as u8);
        deflate.extend_from_slice(&raw[offset..offset + block_len]);
        offset += block_len;
    }

    // Adler-32 checksum
    let (mut s1, mut s2): (u32, u32) = (1, 0);
    for &b in &raw {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    let adler = (s2 << 16) | s1;
    deflate.extend_from_slice(&adler.to_be_bytes());

    // CRC-32 (used by PNG chunks)
    fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFFFFFF;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB88320 } else { crc >> 1 };
            }
        }
        !crc
    }

    fn write_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(data);
        let mut crc_data = Vec::with_capacity(4 + data.len());
        crc_data.extend_from_slice(tag);
        crc_data.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_data).to_be_bytes());
    }

    let mut png = Vec::with_capacity(deflate.len() + 100);
    // PNG signature
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    // IHDR
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.push(8);  // bit depth
    ihdr.push(2);  // color type: RGB
    ihdr.push(0);  // compression
    ihdr.push(0);  // filter
    ihdr.push(0);  // interlace
    write_chunk(&mut png, b"IHDR", &ihdr);

    // IDAT
    write_chunk(&mut png, b"IDAT", &deflate);

    // IEND
    write_chunk(&mut png, b"IEND", &[]);

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&png).unwrap();
}

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
        stuck_timer: 0.0, stuck_count: 0,
        detour_x: 0.0, detour_z: 0.0, detouring: false,
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
        ragdoll_prev: [[0.0; 3]; 7],
        ragdoll_timer: 0.0,
        wanted: false, bounty: 0.0, violation_timer: 0.0,
        police_target: None,
        wander_cooldown: 0.0,
    }
}

fn make_item(kind: state::ItemKind) -> state::Item {
    state::Item {
        x: 0.0, y: 0.0, z: 0.0,
        kind, active: true,
        spin_phase: 0.0,
        falling: false, vel_y: 0.0,
        claimed_by: None, skip_until: 0.0,
    }
}

fn make_trash_bin() -> state::TrashBin {
    state::TrashBin {
        x: 0.0, y: 0.0, z: 0.0,
        items_held: 3,
        carried_by: None,
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
        w, h, &win_holes, recess_depth, color, win_color, 1.0, 1.0, false);
    // Back face (z-)
    mesh::wall_with_holes_tris(tris,
        w * 0.5, 0.0, -d * 0.5,
        w, h, &win_holes, recess_depth, color, win_color, -1.0, -1.0, false);

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
        d, h, &side_holes, recess_depth, color, win_color, -1.0, 1.0, true);
    // Left face (x-) — swap_xz for YZ-plane wall
    mesh::wall_with_holes_tris(tris,
        d * 0.5, 0.0, -w * 0.5,
        d, h, &side_holes, recess_depth, color, win_color, 1.0, -1.0, true);

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
        w, h, &win_holes, 0.15, color, 0xFF222244, 1.0, 1.0, false);
    mesh::wall_with_holes_tris(tris, w * 0.5, 0.0, -d * 0.5,
        w, h, &win_holes, 0.15, color, 0xFF222244, -1.0, -1.0, false);

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
    // Rail bars
    mesh::cylinder_between(tris,
        [rail_x_l, deck_y + 0.8, -bridge_len * 0.5],
        [rail_x_l, deck_y + 0.8, bridge_len * 0.5],
        0.04, 4, 0xFF777766);
    mesh::cylinder_between(tris,
        [rail_x_r, deck_y + 0.8, -bridge_len * 0.5],
        [rail_x_r, deck_y + 0.8, bridge_len * 0.5],
        0.04, 4, 0xFF777766);
    // Railing posts
    for pi in 0..7 {
        let t = (pi as f32 + 0.5) / 7.0 - 0.5;
        let pz = t * bridge_len;
        mesh::cylinder_tris(tris, rail_x_l, deck_y + 0.4, pz, 0.03, 0.8, 4, 0xFF777766);
        mesh::cylinder_tris(tris, rail_x_r, deck_y + 0.4, pz, 0.03, 0.8, 4, 0xFF777766);
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
    mesh::beveled_box_tris(tris, 0.0, 5.5, 0.0, 3.0, 2.0, 0.15, 0.03, 0xFFDDDDCC);
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
    mesh::wave_surface_tris(tris, -5.0, 5.0, -3.0, 3.0, 0.0, 0.12, 0.5, 10, 6, 0xFF224466);
}

fn gen_dumpster(tris: &mut Vec<state::WorldTri>) {
    mesh::beveled_box_tris(tris, 0.0, 0.5, 0.0, 1.2, 1.0, 0.8, 0.05, 0xFF334488);
    mesh::box_tris(tris, 0.0, 1.05, 0.0, 1.25, 0.08, 0.82, 0xFF445599);
}

fn gen_street_light(tris: &mut Vec<state::WorldTri>) {
    // Pole
    mesh::cylinder_tris(tris, 0.0, 2.0, 0.0, 0.05, 4.0, 6, 0xFF666666);
    // Curved arm (cylinder between)
    mesh::cylinder_between(tris, [0.0, 4.0, 0.0], [0.8, 3.8, 0.0], 0.03, 4, 0xFF666666);
    // Lamp globe
    mesh::sphere_tris(tris, 0.8, 3.7, 0.0, 0.15, 1, 0xFFFFEE88);
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
    let mut tris: Vec<state::WorldTri> = Vec::with_capacity(8192);

    // ── Dynamic entities ──
    let player = make_player();
    tris.clear();
    render::gen_player_mesh(&player, &mut tris);
    let img = render_model_sheet(&tris, 0.7, 3.5, "Player");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_player.png");

    let vehicle = make_vehicle(0xFFCC3333);
    tris.clear();
    render::gen_vehicle_mesh(&vehicle, &mut tris, false);
    let img = render_model_sheet(&tris, 0.7, 7.0, "Vehicle");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_vehicle.png");

    let vehicle_interior = make_vehicle(0xFF3333CC);
    tris.clear();
    render::gen_vehicle_mesh(&vehicle_interior, &mut tris, true);
    let img = render_model_sheet(&tris, 0.7, 7.0, "Vehicle Interior");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_vehicle_int.png");

    let npc = make_npc(state::NpcJob::Collector);
    tris.clear();
    render::gen_npc_mesh(&npc, &mut tris);
    let img = render_model_sheet(&tris, 0.7, 3.5, "NPC Collector");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_npc.png");

    let bin = make_trash_bin();
    tris.clear();
    render::gen_trash_bin_mesh(&bin, &mut tris);
    let img = render_model_sheet(&tris, 0.4, 2.5, "Trash Bin");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_trashbin.png");

    // ── World objects ──
    tris.clear(); gen_building(&mut tris);
    let img = render_model_sheet(&tris, 5.0, 20.0, "Building Pitched");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_building.png");

    tris.clear(); gen_building_flat_roof(&mut tris);
    let img = render_model_sheet(&tris, 4.0, 18.0, "Building Flat");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_building_flat.png");

    tris.clear(); gen_building_hip_roof(&mut tris);
    let img = render_model_sheet(&tris, 6.0, 22.0, "Building Hip");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_building_hip.png");

    tris.clear(); gen_bridge(&mut tris);
    let img = render_model_sheet(&tris, 2.0, 28.0, "Bridge");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_bridge.png");

    tris.clear(); gen_suburb_house(&mut tris);
    let img = render_model_sheet(&tris, 1.5, 12.0, "Suburb House");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_suburb.png");

    tris.clear(); gen_market_stall(&mut tris);
    let img = render_model_sheet(&tris, 1.5, 8.0, "Market Stall");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_stall.png");

    tris.clear(); gen_bus_stop(&mut tris);
    let img = render_model_sheet(&tris, 1.5, 8.0, "Bus Stop");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_busstop.png");

    tris.clear(); gen_vending_machine(&mut tris);
    let img = render_model_sheet(&tris, 0.75, 3.5, "Vending Machine");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_vending.png");

    tris.clear(); gen_phone_booth(&mut tris);
    let img = render_model_sheet(&tris, 1.1, 5.0, "Phone Booth");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_phonebooth.png");

    tris.clear(); gen_fire_hydrant(&mut tris);
    let img = render_model_sheet(&tris, 0.3, 1.5, "Fire Hydrant");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_hydrant.png");

    tris.clear(); gen_picnic_table(&mut tris);
    let img = render_model_sheet(&tris, 0.5, 4.0, "Picnic Table");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_picnic.png");

    tris.clear(); gen_water_tower(&mut tris);
    let img = render_model_sheet(&tris, 3.0, 12.0, "Water Tower");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_watertower.png");

    tris.clear(); gen_billboard(&mut tris);
    let img = render_model_sheet(&tris, 3.0, 12.0, "Billboard");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_billboard.png");

    tris.clear(); gen_tree(&mut tris);
    let img = render_model_sheet(&tris, 2.0, 8.0, "Tree");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_tree.png");

    tris.clear(); gen_wave_surface(&mut tris);
    let img = render_model_sheet(&tris, 0.5, 14.0, "Wave Surface");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_wave.png");

    tris.clear(); gen_dumpster(&mut tris);
    let img = render_model_sheet(&tris, 0.5, 3.5, "Dumpster");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_dumpster.png");

    tris.clear(); gen_street_light(&mut tris);
    let img = render_model_sheet(&tris, 2.0, 8.0, "Street Light");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_streetlight.png");

    tris.clear(); gen_crane(&mut tris);
    let img = render_model_sheet(&tris, 7.0, 25.0, "Crane");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_crane.png");

    tris.clear(); gen_warehouse(&mut tris);
    let img = render_model_sheet(&tris, 3.0, 18.0, "Warehouse");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_warehouse.png");

    // ── Primitives ──
    tris.clear();
    mesh::cylinder_tris(&mut tris, 0.0, 0.5, 0.0, 0.3, 1.0, 8, 0xFF3388CC);
    let img = render_model_sheet(&tris, 0.5, 3.0, "Cylinder");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_cylinder.png");

    tris.clear();
    mesh::sphere_tris(&mut tris, 0.0, 0.5, 0.0, 0.5, 2, 0xFFCC4433);
    let img = render_model_sheet(&tris, 0.5, 2.5, "Sphere");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_sphere.png");

    tris.clear();
    mesh::beveled_box_tris(&mut tris, 0.0, 0.5, 0.0, 1.0, 1.0, 1.0, 0.1, 0xFF44AA44);
    let img = render_model_sheet(&tris, 0.5, 3.0, "Beveled Box");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_bevelbox.png");

    tris.clear();
    mesh::cone_tris(&mut tris, 0.0, 0.0, 0.0, 0.4, 1.0, 8, 0xFFCC8833);
    let img = render_model_sheet(&tris, 0.5, 3.0, "Cone");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_cone.png");

    tris.clear();
    mesh::box_tris(&mut tris, 0.0, 0.5, 0.0, 1.0, 1.0, 1.0, 0xFF5577AA);
    let img = render_model_sheet(&tris, 0.5, 3.0, "Box");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_box.png");

    tris.clear();
    mesh::pitched_roof_tris(&mut tris, 0.0, 0.0, 0.0, 4.0, 3.0, 1.5, 0xFF885544);
    let img = render_model_sheet(&tris, 1.0, 8.0, "Pitched Roof");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_pitchedroof.png");

    tris.clear();
    mesh::hip_roof_tris(&mut tris, 0.0, 0.0, 0.0, 4.0, 3.0, 1.5, 0xFF885544);
    let img = render_model_sheet(&tris, 1.0, 8.0, "Hip Roof");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_hiproof.png");

    tris.clear();
    let profile: [[f32;2]; 6] = [
        [0.0, 0.0], [0.3, 0.0], [0.25, 0.5],
        [0.35, 0.7], [0.2, 1.0], [0.0, 1.1],
    ];
    mesh::lathe_tris(&mut tris, 0.0, 0.0, 0.0, &profile, 8, 0xFFCC6644);
    let img = render_model_sheet(&tris, 0.5, 3.0, "Lathe");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_lathe.png");

    // Wall with holes standalone test
    tris.clear();
    let holes = vec![
        mesh::WallHole { x: 1.0, y: 1.0, w: 0.8, h: 1.2 },
        mesh::WallHole { x: 3.0, y: 1.0, w: 0.8, h: 1.2 },
        mesh::WallHole { x: 1.0, y: 3.5, w: 0.8, h: 1.2 },
        mesh::WallHole { x: 3.0, y: 3.5, w: 0.8, h: 1.2 },
    ];
    mesh::wall_with_holes_tris(&mut tris, -2.5, 0.0, 0.0, 5.0, 6.0, &holes, 0.15,
        0xFF887766, 0xFF222244, 1.0, 1.0, false);
    let img = render_model_sheet(&tris, 3.0, 10.0, "Wall Holes Z+");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_wallholes.png");

    tris.clear();
    mesh::wall_with_holes_tris(&mut tris, 2.5, 0.0, 0.0, 5.0, 6.0, &holes, 0.15,
        0xFF887766, 0xFF222244, -1.0, -1.0, false);
    let img = render_model_sheet(&tris, 3.0, 10.0, "Wall Holes Z-");
    save_png(&img, IMG_W, IMG_H, "/tmp/clauding_model_wallholes_back.png");

    eprintln!("All model sheets saved to /tmp/clauding_model_*.png");
}
