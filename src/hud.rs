// HUD: health bar, stamina bar, money, score, minimap, vehicle prompt
// Draws directly to framebuffer pixels (no z-buffer, overlaid on top)

use crate::state::*;
use crate::raster::Framebuffer;

const DAY_COLOR: u32 = 0xFFCCCCCC;

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
const MINIMAP_FIELD_ROAD: u32 = 0xFF665544;
const MINIMAP_PLAYER: u32 = 0xFFFFFFFF;
const MINIMAP_NPC_WORKING: u32 = 0xFFFFDD33;
const MINIMAP_NPC_SLEEPING: u32 = 0xFF555544;
const MINIMAP_NPC_DRIVING: u32 = 0xFF3388FF;
const MINIMAP_NPC_HOME: u32 = 0xFF886633;
const MINIMAP_VEHICLE: u32 = 0xFF3388FF;
const MINIMAP_ITEM: u32 = 0xFFFF3333;
const MINIMAP_BIN: u32 = 0xFF33AA33;

const MONEY_COLOR: u32 = 0xFFFFDD33;
const CARRYING_COLOR: u32 = 0xFFFFAA44;
const PROMPT_COLOR: u32 = 0xCCFFFFFF;
const JOB_COLOR: u32 = 0xFF44DDFF;
const MINIMAP_INTERACTIBLE: u32 = 0xFF44CCCC;

// 3x5 pixel font for digits 0-9, symbols, and A-Z
const FONT: [[u8; 5]; 42] = [
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
    // A-Z (indices 12-37)
    [0b010, 0b101, 0b111, 0b101, 0b101], // A 12
    [0b110, 0b101, 0b110, 0b101, 0b110], // B 13
    [0b111, 0b100, 0b100, 0b100, 0b111], // C 14
    [0b110, 0b101, 0b101, 0b101, 0b110], // D 15
    [0b111, 0b100, 0b111, 0b100, 0b111], // E 16
    [0b111, 0b100, 0b111, 0b100, 0b100], // F 17
    [0b111, 0b100, 0b101, 0b101, 0b111], // G 18
    [0b101, 0b101, 0b111, 0b101, 0b101], // H 19
    [0b111, 0b010, 0b010, 0b010, 0b111], // I 20
    [0b001, 0b001, 0b001, 0b101, 0b111], // J 21
    [0b101, 0b110, 0b100, 0b110, 0b101], // K 22
    [0b100, 0b100, 0b100, 0b100, 0b111], // L 23
    [0b101, 0b111, 0b111, 0b101, 0b101], // M 24
    [0b101, 0b111, 0b111, 0b111, 0b101], // N 25
    [0b111, 0b101, 0b101, 0b101, 0b111], // O 26
    [0b111, 0b101, 0b111, 0b100, 0b100], // P 27
    [0b111, 0b101, 0b101, 0b111, 0b011], // Q 28
    [0b111, 0b101, 0b110, 0b101, 0b101], // R 29
    [0b111, 0b100, 0b111, 0b001, 0b111], // S 30
    [0b111, 0b010, 0b010, 0b010, 0b010], // T 31
    [0b101, 0b101, 0b101, 0b101, 0b111], // U 32
    [0b101, 0b101, 0b101, 0b101, 0b010], // V 33
    [0b101, 0b101, 0b111, 0b111, 0b101], // W 34
    [0b101, 0b101, 0b010, 0b101, 0b101], // X 35
    [0b101, 0b101, 0b010, 0b010, 0b010], // Y 36
    [0b111, 0b001, 0b010, 0b100, 0b111], // Z 37
    // Symbols
    [0b000, 0b000, 0b000, 0b000, 0b000], // space 38
    [0b000, 0b000, 0b111, 0b000, 0b000], // - (dash) 39
    [0b000, 0b010, 0b000, 0b010, 0b000], // : (colon) 40
    [0b010, 0b001, 0b010, 0b100, 0b010], // > (arrow) 41
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
    draw_char_idx(fb, BAR_X, money_y, 10, 2, MONEY_COLOR); // $
    draw_number(fb, BAR_X + 10, money_y, p.money as u32, 2, MONEY_COLOR);

    // Carrying status
    let mut status_y = money_y + 18;
    if p.carrying_item {
        draw_text(fb, BAR_X, status_y, "CARRYING", 1, CARRYING_COLOR);
        status_y += 12;
    } else if p.carrying_bin.is_some() {
        draw_text(fb, BAR_X, status_y, "CARRYING BIN", 1, CARRYING_COLOR);
        status_y += 12;
    }

    // Active job display
    if p.active_job.job_type != PlayerJobType::None {
        let job_name = match p.active_job.job_type {
            PlayerJobType::None => "",
            PlayerJobType::GarbageCollector => "GARBAGE COLLECTOR",
            PlayerJobType::TaxiDriver => "TAXI DRIVER",
            PlayerJobType::DeliveryCourier => "DELIVERY COURIER",
            PlayerJobType::MailCarrier => "MAIL CARRIER",
            PlayerJobType::Paramedic => "PARAMEDIC",
            PlayerJobType::Firefighter => "FIREFIGHTER",
            PlayerJobType::PolicePatrol => "POLICE PATROL",
            PlayerJobType::StreetVendor => "STREET VENDOR",
            PlayerJobType::Mechanic => "MECHANIC",
            PlayerJobType::ConstructionWorker => "CONSTRUCTION",
            PlayerJobType::Fisherman => "FISHERMAN",
            PlayerJobType::Farmer => "FARMER",
            PlayerJobType::Lumberjack => "LUMBERJACK",
            PlayerJobType::Scavenger => "SCAVENGER",
        };
        draw_text(fb, BAR_X, status_y, "JOB:", 1, JOB_COLOR);
        draw_text(fb, BAR_X + 20, status_y, job_name, 1, JOB_COLOR);
        status_y += 12;
        // Progress
        let progress_w = 100;
        let done = p.active_job.items_done;
        let needed = p.active_job.items_needed.max(1);
        let fill = (done as f32 / needed as f32).min(1.0);
        draw_bar(fb, BAR_X, status_y, progress_w, 8, fill, JOB_COLOR);
        // Items counter
        draw_number(fb, BAR_X + progress_w + 6, status_y, done, 1, JOB_COLOR);
        draw_text(fb, BAR_X + progress_w + 6 + 16, status_y, "-", 1, JOB_COLOR);
        draw_number(fb, BAR_X + progress_w + 6 + 24, status_y, needed, 1, JOB_COLOR);
        status_y += 14;
        // Time remaining
        if p.active_job.time_remaining > 0.0 {
            let mins = (p.active_job.time_remaining / 60.0) as u32;
            let secs = (p.active_job.time_remaining % 60.0) as u32;
            draw_number(fb, BAR_X, status_y, mins, 1, JOB_COLOR);
            draw_text(fb, BAR_X + 8, status_y, ":", 1, JOB_COLOR);
            draw_number_padded(fb, BAR_X + 14, status_y, secs, 2, 1, JOB_COLOR);
        }
        let _ = status_y;
    }

    // Bank balance (if nonzero)
    if p.bank_balance > 0.0 {
        let bank_y = money_y + 12;
        draw_text(fb, BAR_X + 80, bank_y - 12, "BANK", 1, 0xFF88BBFF);
        draw_char_idx(fb, BAR_X + 112, bank_y - 12, 10, 1, 0xFF88BBFF);
        draw_number(fb, BAR_X + 118, bank_y - 12, p.bank_balance as u32, 1, 0xFF88BBFF);
    }

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

    // Day counter (below time)
    let day_y = time_y + 16;
    draw_text(fb, time_x, day_y, "DAY", 1, DAY_COLOR);
    draw_number(fb, time_x + 16, day_y, game.day_count, 1, DAY_COLOR);

    // Minimap (bottom-right)
    draw_minimap(fb, game);

    // Context-sensitive interaction prompt
    if p.in_vehicle.is_none() {
        let cx = fb.w / 2;
        let cy = fb.h - 60;
        let pickup_dist_sq = NPC_PICKUP_DIST * NPC_PICKUP_DIST;
        let bin_dist_sq = NPC_BIN_DIST * NPC_BIN_DIST;

        if p.carrying_item {
            // Near bin? Show "E DEPOSIT"
            let near_bin = game.world.trash_bins.iter().any(|b| {
                if b.carried_by.is_some() { return false; }
                let dx = p.x - b.x;
                let dz = p.z - b.z;
                dx * dx + dz * dz < bin_dist_sq
            });
            if near_bin {
                draw_rect(fb, cx - 50, cy - 4, 100, 14, 0x88000000);
                draw_text(fb, cx - 46, cy, "E DEPOSIT", 1, PROMPT_COLOR);
            }
        } else if p.carrying_bin.is_some() {
            // Always show "E SET DOWN"
            draw_rect(fb, cx - 50, cy - 4, 100, 14, 0x88000000);
            draw_text(fb, cx - 46, cy, "E SET DOWN", 1, PROMPT_COLOR);
        } else {
            // Not carrying anything — check what's nearby
            let near_item = game.world.items.iter().any(|it| {
                if !it.active || it.falling { return false; }
                let dx = p.x - it.x;
                let dz = p.z - it.z;
                dx * dx + dz * dz < pickup_dist_sq
            });
            let near_bin = game.world.trash_bins.iter().any(|b| {
                if b.carried_by.is_some() { return false; }
                let dx = p.x - b.x;
                let dz = p.z - b.z;
                dx * dx + dz * dz < bin_dist_sq
            });
            let near_vehicle = game.world.vehicles.iter().any(|v| {
                let dx = p.x - v.x;
                let dz = p.z - v.z;
                dx * dx + dz * dz < VEHICLE_ENTER_DIST * VEHICLE_ENTER_DIST
            });
            // Check interactibles
            let interact_dist_sq = INTERACT_DIST * INTERACT_DIST;
            let near_interactible = game.world.interactibles.iter().find(|i| {
                if i.cooldown > 0.0 { return false; }
                let dx = p.x - i.x;
                let dz = p.z - i.z;
                dx * dx + dz * dz < interact_dist_sq
            });

            if near_item {
                draw_rect(fb, cx - 50, cy - 4, 100, 14, 0x88000000);
                draw_text(fb, cx - 46, cy, "E PICK UP", 1, PROMPT_COLOR);
            } else if near_bin {
                draw_rect(fb, cx - 50, cy - 4, 100, 14, 0x88000000);
                draw_text(fb, cx - 46, cy, "E GRAB BIN", 1, PROMPT_COLOR);
            } else if near_vehicle {
                draw_rect(fb, cx - 50, cy - 4, 100, 14, 0x88000000);
                draw_text(fb, cx - 46, cy, "E ENTER", 1, PROMPT_COLOR);
            } else if let Some(inter) = near_interactible {
                let text = match inter.kind {
                    InteractibleKind::VendingMachine => "E BUY DRINK $2",
                    InteractibleKind::ParkBench => "E SIT",
                    InteractibleKind::Dumpster => "E SEARCH",
                    InteractibleKind::Atm => "E USE ATM",
                    InteractibleKind::PhoneBooth => "E GET JOB",
                    InteractibleKind::FireHydrant => "E ACTIVATE",
                    InteractibleKind::NewspaperStand => "E BUY PAPER $1",
                    InteractibleKind::Mailbox => "E DELIVER",
                    InteractibleKind::Payphone => "E CALL TAXI",
                };
                let tw = text.len() * 4 + 8;
                draw_rect(fb, cx - tw / 2, cy - 4, tw, 14, 0x88000000);
                draw_text(fb, cx - tw / 2 + 4, cy, text, 1, PROMPT_COLOR);
            }
        }
    }

    // Job selection menu overlay
    if game.player.job_menu_open {
        let menu_x = fb.w / 2 - 120;
        let menu_y = fb.h / 2 - 100;
        draw_rect(fb, menu_x, menu_y, 240, 220, 0xCC111122);
        draw_rect(fb, menu_x, menu_y, 240, 2, 0xFF44DDFF);
        draw_text(fb, menu_x + 8, menu_y + 8, "SELECT JOB", 2, JOB_COLOR);

        let jobs = [
            "GARBAGE COLLECTOR", "TAXI DRIVER", "DELIVERY COURIER",
            "MAIL CARRIER", "PARAMEDIC", "FIREFIGHTER", "POLICE PATROL",
            "STREET VENDOR", "MECHANIC", "CONSTRUCTION", "FISHERMAN",
            "FARMER", "LUMBERJACK", "SCAVENGER",
        ];
        let cursor = game.player.job_menu_cursor;
        for (i, name) in jobs.iter().enumerate() {
            let iy = menu_y + 30 + i * 13;
            if iy + 10 > menu_y + 220 { break; }
            let color = if i == cursor { 0xFFFFAA33 } else { 0xFFCCCCCC };
            if i == cursor {
                draw_rect(fb, menu_x + 2, iy - 1, 236, 12, 0xFF333355);
            }
            draw_text(fb, menu_x + 8, iy, name, 1, color);
        }
        draw_text(fb, menu_x + 8, menu_y + 205, "UP-DOWN  ENTER  ESC", 1, 0xFF888888);
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

pub fn draw_rect(fb: &mut Framebuffer, x: usize, y: usize, w: usize, h: usize, color: u32) {
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

fn draw_char_idx(fb: &mut Framebuffer, x: usize, y: usize, idx: usize, scale: usize, color: u32) {
    if idx >= FONT.len() { return; }
    draw_glyph(fb, x, y, &FONT[idx], scale, color);
}

fn ascii_to_font_idx(c: u8) -> Option<usize> {
    match c {
        b'0'..=b'9' => Some((c - b'0') as usize),
        b'$' => Some(10),
        b'.' => Some(11),
        b'A'..=b'Z' => Some(12 + (c - b'A') as usize),
        b'a'..=b'z' => Some(12 + (c - b'a') as usize),
        b' ' => Some(38),
        b'-' => Some(39),
        b':' => Some(40),
        b'>' => Some(41),
        b'[' => Some(14), // reuse C glyph shape for [
        b']' => Some(15), // reuse D glyph shape for ]
        b'?' => Some(28),  // reuse Q shape
        _ => Some(38), // space for unknown
    }
}

/// Draw a string using the pixel font. Handles ASCII A-Z, 0-9, and common symbols.
pub fn draw_text(fb: &mut Framebuffer, x: usize, y: usize, text: &str, scale: usize, color: u32) {
    let char_w = 3 * scale + scale;
    let mut cx = x;
    for &byte in text.as_bytes() {
        // Trim trailing spaces from fixed buffers
        if let Some(idx) = ascii_to_font_idx(byte) {
            draw_char_idx(fb, cx, y, idx, scale, color);
        }
        cx += char_w;
    }
}

/// Draw a string from a fixed byte buffer, stopping at trailing spaces
pub fn draw_text_bytes(fb: &mut Framebuffer, x: usize, y: usize, text: &[u8; 64], scale: usize, color: u32) {
    // Find actual length (trim trailing spaces)
    let mut len = 64;
    while len > 0 && text[len - 1] == b' ' { len -= 1; }
    let char_w = 3 * scale + scale;
    let mut cx = x;
    for i in 0..len {
        if let Some(idx) = ascii_to_font_idx(text[i]) {
            draw_char_idx(fb, cx, y, idx, scale, color);
        }
        cx += char_w;
    }
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
        draw_char_idx(fb, x, y, 0, scale, color);
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
        draw_char_idx(fb, x + i * char_w, y, d, scale, color);
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
        draw_char_idx(fb, x + i * char_w, y, d, scale, color);
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

/// Draw a thick line on the framebuffer using Bresenham's algorithm
fn draw_minimap_line(fb: &mut Framebuffer, x0: usize, y0: usize, x1: usize, y1: usize, thickness: usize, color: u32) {
    let (mut cx, mut cy) = (x0 as i32, y0 as i32);
    let (ex, ey) = (x1 as i32, y1 as i32);
    let dx = (ex - cx).abs();
    let dy = -(ey - cy).abs();
    let sx = if cx < ex { 1 } else { -1 };
    let sy = if cy < ey { 1 } else { -1 };
    let mut err = dx + dy;
    let half = thickness as i32 / 2;
    loop {
        // Draw a filled square at (cx, cy) for thickness
        for ty in -half..=(half) {
            for tx in -half..=(half) {
                let px = (cx + tx) as usize;
                let py = (cy + ty) as usize;
                if px < fb.w && py < fb.h {
                    fb.pixels[py * fb.w + px] = color;
                }
            }
        }
        if cx == ex && cy == ey { break; }
        let e2 = 2 * err;
        if e2 >= dy { err += dy; cx += sx; }
        if e2 <= dx { err += dx; cy += sy; }
    }
}

fn draw_minimap(fb: &mut Framebuffer, game: &GameState) {
    let mx = fb.w - MINIMAP_SIZE - MINIMAP_MARGIN;
    let my = fb.h - MINIMAP_SIZE - MINIMAP_MARGIN;
    let size = MINIMAP_SIZE;

    // Background
    draw_rect(fb, mx, my, size, size, MINIMAP_BG);

    // Roads (draw segments as lines)
    for seg in &game.road_network.segments {
        let x0 = world_to_minimap(seg.x0, size);
        let z0 = world_to_minimap(seg.z0, size);
        let x1 = world_to_minimap(seg.x1, size);
        let z1 = world_to_minimap(seg.z1, size);
        let (thickness, color) = match seg.tier {
            crate::state::RoadTier::CarRoad => (3usize, MINIMAP_ROAD),
            crate::state::RoadTier::FieldRoad => (1usize, MINIMAP_FIELD_ROAD),
        };
        draw_minimap_line(fb, mx + x0, my + z0, mx + x1, my + z1, thickness, color);
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

    // Trash bins
    for bin in &game.world.trash_bins {
        if bin.carried_by.is_some() { continue; }
        let bx = world_to_minimap(bin.x, size);
        let bz = world_to_minimap(bin.z, size);
        if bx < size && bz < size {
            draw_dot(fb, mx + bx, my + bz, 1, MINIMAP_BIN);
        }
    }

    // Interactibles
    for inter in &game.world.interactibles {
        let ix = world_to_minimap(inter.x, size);
        let iz = world_to_minimap(inter.z, size);
        if ix < size && iz < size {
            draw_dot(fb, mx + ix, my + iz, 1, MINIMAP_INTERACTIBLE);
        }
    }

    // NPCs (color-coded by job for working NPCs)
    for npc in &game.world.npcs {
        let nx = world_to_minimap(npc.x, size);
        let nz = world_to_minimap(npc.z, size);
        if nx < size && nz < size {
            let color = match npc.state {
                NpcState::Sleeping => MINIMAP_NPC_SLEEPING,
                NpcState::HomeTask => MINIMAP_NPC_HOME,
                NpcState::Driving => MINIMAP_NPC_DRIVING,
                NpcState::Working | NpcState::GoingToWork | NpcState::Interacting => {
                    match npc.job {
                        NpcJob::PolicePatrol => 0xFF4444FF,
                        NpcJob::Paramedic => 0xFFFF4444,
                        NpcJob::TaxiDriver => 0xFFFFFF44,
                        NpcJob::Firefighter => 0xFFFF6622,
                        NpcJob::StreetVendor => 0xFFFFAA44,
                        _ => MINIMAP_NPC_WORKING,
                    }
                }
                NpcState::GoingHome => MINIMAP_NPC_HOME,
            };
            draw_dot(fb, mx + nx, my + nz, 1, color);
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
