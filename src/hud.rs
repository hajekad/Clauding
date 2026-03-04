// HUD: health bar, stamina bar, money, score, minimap, vehicle prompt
// Draws directly to framebuffer pixels (no z-buffer, overlaid on top)

use crate::state::*;
use crate::raster::Framebuffer;

const BAR_X: usize = 20;
const BAR_W: usize = 200;
const BAR_H: usize = 16;
const BAR_GAP: usize = 6;

const HEALTH_Y: usize = 20;
const STAMINA_Y: usize = HEALTH_Y + BAR_H + BAR_GAP;

const HEALTH_COLOR: u32 = 0xFFCC2222;
const STAMINA_COLOR: u32 = 0xFF22CC22;
const BAR_BG: u32 = 0xFF333333;
const BAR_BORDER: u32 = 0xFF888888;

const MINIMAP_SIZE: usize = 140;
const MINIMAP_MARGIN: usize = 20;
const MINIMAP_BG: u32 = 0xCC224422;
const MINIMAP_ROAD: u32 = 0xFF555555;
const MINIMAP_PLAYER: u32 = 0xFFFFFFFF;
const MINIMAP_NPC: u32 = 0xFFFFDD33;
const MINIMAP_VEHICLE: u32 = 0xFF3388FF;
const MINIMAP_ITEM: u32 = 0xFFFF3333;

const MONEY_COLOR: u32 = 0xFFFFDD33;
const SCORE_COLOR: u32 = 0xFFFFFFFF;
const PROMPT_COLOR: u32 = 0xCCFFFFFF;

// 3x5 pixel font for digits 0-9 and $
const DIGIT_FONT: [[u8; 5]; 12] = [
    [0b111, 0b101, 0b101, 0b101, 0b111], // 0
    [0b010, 0b110, 0b010, 0b010, 0b111], // 1
    [0b111, 0b001, 0b111, 0b100, 0b111], // 2
    [0b111, 0b001, 0b111, 0b001, 0b111], // 3
    [0b101, 0b101, 0b111, 0b001, 0b001], // 4
    [0b111, 0b100, 0b111, 0b001, 0b111], // 5
    [0b111, 0b100, 0b111, 0b101, 0b111], // 6
    [0b111, 0b001, 0b010, 0b010, 0b010], // 7
    [0b111, 0b101, 0b111, 0b101, 0b111], // 8
    [0b111, 0b101, 0b111, 0b001, 0b111], // 9
    [0b010, 0b111, 0b110, 0b111, 0b010], // $ (index 10)
    [0b000, 0b000, 0b000, 0b000, 0b010], // . (index 11)
];

const TIME_COLOR: u32 = 0xFFFFFFFF;

pub fn sys_hud(fb: &mut Framebuffer, game: &GameState) {
    let p = &game.player;

    // Health bar
    draw_bar(fb, BAR_X, HEALTH_Y, BAR_W, BAR_H, p.health / 100.0, HEALTH_COLOR);
    // Stamina bar
    draw_bar(fb, BAR_X, STAMINA_Y, BAR_W, BAR_H, p.stamina / 100.0, STAMINA_COLOR);

    // Money ($ + number)
    let money_y = STAMINA_Y + BAR_H + BAR_GAP + 2;
    draw_char(fb, BAR_X, money_y, 10, 2, MONEY_COLOR); // $
    draw_number(fb, BAR_X + 10, money_y, p.money as u32, 2, MONEY_COLOR);

    // Score
    let score_y = money_y + 14;
    draw_number(fb, BAR_X, score_y, p.score, 2, SCORE_COLOR);

    // Time of day (top-right, HH:MM format)
    let hour = game.time_of_day as u32;
    let minute = ((game.time_of_day - hour as f32) * 60.0) as u32;
    let time_x = fb.w - 80;
    let time_y = 20;
    draw_number_padded(fb, time_x, time_y, hour, 2, 2, TIME_COLOR);
    // Colon
    let cx = time_x + 2 * (3 * 2 + 2) + 2;
    draw_dot_pair(fb, cx, time_y, 2, TIME_COLOR);
    let min_x = cx + 6;
    draw_number_padded(fb, min_x, time_y, minute, 2, 2, TIME_COLOR);

    // Minimap (bottom-right)
    draw_minimap(fb, game);

    // Vehicle enter prompt
    if p.in_vehicle.is_none() {
        // Check if near any vehicle
        let near_vehicle = game.world.vehicles.iter().any(|v| {
            let dx = p.x - v.x;
            let dz = p.z - v.z;
            dx * dx + dz * dz < VEHICLE_ENTER_DIST * VEHICLE_ENTER_DIST
        });
        if near_vehicle {
            // Draw "E" prompt at bottom center
            let cx = fb.w / 2;
            let cy = fb.h - 60;
            // Draw a small box with "E" hint
            draw_rect(fb, cx - 20, cy - 4, 40, 18, 0x88000000);
            // E = custom pattern
            let e_pattern: [u8; 5] = [0b111, 0b100, 0b111, 0b100, 0b111];
            draw_glyph(fb, cx - 4, cy, &e_pattern, 2, PROMPT_COLOR);
        }
    }
}

fn draw_bar(fb: &mut Framebuffer, x: usize, y: usize, w: usize, h: usize, fill: f32, color: u32) {
    // Border
    draw_rect(fb, x.wrapping_sub(1), y.wrapping_sub(1), w + 2, h + 2, BAR_BORDER);
    // Background
    draw_rect(fb, x, y, w, h, BAR_BG);
    // Fill
    let fw = ((w as f32) * fill.clamp(0.0, 1.0)) as usize;
    if fw > 0 {
        draw_rect(fb, x, y, fw, h, color);
    }
}

fn draw_rect(fb: &mut Framebuffer, x: usize, y: usize, w: usize, h: usize, color: u32) {
    let alpha = (color >> 24) & 0xFF;
    for dy in 0..h {
        let py = y + dy;
        if py >= fb.h { break; }
        for dx in 0..w {
            let px = x + dx;
            if px >= fb.w { break; }
            let idx = py * fb.w + px;
            if alpha >= 0xF0 {
                fb.pixels[idx] = color;
            } else {
                // Alpha blend
                fb.pixels[idx] = alpha_blend(fb.pixels[idx], color, alpha);
            }
        }
    }
}

fn alpha_blend(dst: u32, src: u32, alpha: u32) -> u32 {
    let a = alpha as f32 / 255.0;
    let inv = 1.0 - a;
    let r = (((src >> 16) & 0xFF) as f32 * a + ((dst >> 16) & 0xFF) as f32 * inv) as u32;
    let g = (((src >> 8) & 0xFF) as f32 * a + ((dst >> 8) & 0xFF) as f32 * inv) as u32;
    let b = ((src & 0xFF) as f32 * a + (dst & 0xFF) as f32 * inv) as u32;
    0xFF000000 | (r << 16) | (g << 8) | b
}

fn draw_char(fb: &mut Framebuffer, x: usize, y: usize, idx: usize, scale: usize, color: u32) {
    if idx >= DIGIT_FONT.len() { return; }
    draw_glyph(fb, x, y, &DIGIT_FONT[idx], scale, color);
}

fn draw_glyph(fb: &mut Framebuffer, x: usize, y: usize, glyph: &[u8; 5], scale: usize, color: u32) {
    for row in 0..5 {
        for col in 0..3 {
            if glyph[row] & (1 << (2 - col)) != 0 {
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

fn draw_number(fb: &mut Framebuffer, x: usize, y: usize, mut val: u32, scale: usize, color: u32) {
    if val == 0 {
        draw_char(fb, x, y, 0, scale, color);
        return;
    }
    // Extract digits
    let mut digits = [0u8; 10];
    let mut n = 0;
    while val > 0 {
        digits[n] = (val % 10) as u8;
        val /= 10;
        n += 1;
    }
    // Draw right-to-left digits left-to-right
    let char_w = 3 * scale + scale; // 3 pixel wide + 1 gap
    for i in 0..n {
        let d = digits[n - 1 - i] as usize;
        draw_char(fb, x + i * char_w, y, d, scale, color);
    }
}

fn draw_number_padded(fb: &mut Framebuffer, x: usize, y: usize, val: u32, min_digits: usize, scale: usize, color: u32) {
    let char_w = 3 * scale + scale;
    let mut digits = [0u8; 10];
    let mut n = 0;
    let mut v = val;
    loop {
        digits[n] = (v % 10) as u8;
        v /= 10;
        n += 1;
        if v == 0 { break; }
    }
    while n < min_digits { digits[n] = 0; n += 1; }
    for i in 0..n {
        let d = digits[n - 1 - i] as usize;
        draw_char(fb, x + i * char_w, y, d, scale, color);
    }
}

fn draw_dot_pair(fb: &mut Framebuffer, x: usize, y: usize, scale: usize, color: u32) {
    // Two dots stacked vertically (colon for time)
    for sy in 0..scale {
        for sx in 0..scale {
            let px = x + sx;
            let py1 = y + 1 * scale + sy;
            let py2 = y + 3 * scale + sy;
            if px < fb.w {
                if py1 < fb.h { fb.pixels[py1 * fb.w + px] = color; }
                if py2 < fb.h { fb.pixels[py2 * fb.w + px] = color; }
            }
        }
    }
}

fn draw_minimap(fb: &mut Framebuffer, game: &GameState) {
    let mx = fb.w - MINIMAP_SIZE - MINIMAP_MARGIN;
    let my = fb.h - MINIMAP_SIZE - MINIMAP_MARGIN;
    let size = MINIMAP_SIZE;

    // Background
    draw_rect(fb, mx, my, size, size, MINIMAP_BG);

    // Roads
    for &r in &game.road_positions {
        let map_r = world_to_minimap(r, size);
        let road_w = ((ROAD_WIDTH / WORLD_SIZE) * size as f32) as usize;
        let rw = road_w.max(1);
        // Horizontal road
        if map_r < size {
            draw_rect(fb, mx, my + map_r.saturating_sub(rw / 2), size, rw, MINIMAP_ROAD);
        }
        // Vertical road
        if map_r < size {
            draw_rect(fb, mx + map_r.saturating_sub(rw / 2), my, rw, size, MINIMAP_ROAD);
        }
    }

    // Items
    for item in &game.world.items {
        if !item.active { continue; }
        let ix = world_to_minimap(item.x, size);
        let iz = world_to_minimap(item.z, size);
        if ix < size && iz < size {
            draw_dot(fb, mx + ix, my + iz, 1, MINIMAP_ITEM);
        }
    }

    // Vehicles
    for v in &game.world.vehicles {
        let vx = world_to_minimap(v.x, size);
        let vz = world_to_minimap(v.z, size);
        if vx < size && vz < size {
            draw_dot(fb, mx + vx, my + vz, 2, MINIMAP_VEHICLE);
        }
    }

    // NPCs
    for npc in &game.world.npcs {
        let nx = world_to_minimap(npc.x, size);
        let nz = world_to_minimap(npc.z, size);
        if nx < size && nz < size {
            draw_dot(fb, mx + nx, my + nz, 1, MINIMAP_NPC);
        }
    }

    // Player (larger, white)
    let px = world_to_minimap(game.player.x, size);
    let pz = world_to_minimap(game.player.z, size);
    if px < size && pz < size {
        draw_dot(fb, mx + px, my + pz, 3, MINIMAP_PLAYER);
    }
}

fn world_to_minimap(coord: f32, size: usize) -> usize {
    ((coord + WORLD_HALF) / WORLD_SIZE * size as f32) as usize
}

fn draw_dot(fb: &mut Framebuffer, cx: usize, cy: usize, r: usize, color: u32) {
    for dy in 0..r * 2 + 1 {
        for dx in 0..r * 2 + 1 {
            let px = cx + dx - r;
            let py = cy + dy - r;
            if px < fb.w && py < fb.h {
                fb.pixels[py * fb.w + px] = color;
            }
        }
    }
}
