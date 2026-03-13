// Menu state machine, rendering, input handling

use crate::raster::Framebuffer;
use crate::input::{KeyBinds, ALL_ACTIONS, key_name};
use crate::hud::{draw_text, draw_text_bytes, draw_rect};

// Raw scancodes for menu navigation (never rebindable)
const KEY_ESC: usize = 1;
const KEY_UP: usize = 103;
const KEY_DOWN: usize = 108;
const KEY_LEFT: usize = 105;
const KEY_RIGHT: usize = 106;
const KEY_ENTER: usize = 28;
const KEY_BACKSPACE: usize = 14;

const SENSITIVITY_MIN: f32 = 0.1;
const SENSITIVITY_MAX: f32 = 3.0;
const SENSITIVITY_STEP: f32 = 0.1;

#[derive(Clone, Copy, PartialEq)]
pub enum MenuState {
    None,
    Main,
    NewWorld,
    Settings,
    Keybinds,
}

pub struct MenuData {
    pub state: MenuState,
    pub cursor: usize,
    pub rebinding: Option<usize>, // action index awaiting key press
    // New World menu
    pub seed_digits: [u8; 20],       // ASCII digit buffer for seed input
    pub seed_len: usize,             // number of digits entered
    pub regenerate_seed: Option<u64>, // set when GENERATE is pressed
}

impl MenuData {
    pub fn new() -> Self {
        MenuData {
            state: MenuState::None,
            cursor: 0,
            rebinding: None,
            seed_digits: [0; 20],
            seed_len: 0,
            regenerate_seed: None,
        }
    }
}

fn edge(keys: &[bool; 256], prev: &[bool; 256], sc: usize) -> bool {
    keys[sc] && !prev[sc]
}

/// Returns true if the game should quit.
/// `world_seed` is the current seed (used to pre-fill the New World input).
pub fn sys_menu_input(
    menu: &mut MenuData,
    keybinds: &mut KeyBinds,
    mouse_sensitivity: &mut f32,
    invert_x: &mut bool,
    invert_y: &mut bool,
    keys: &[bool; 256],
    prev_keys: &[bool; 256],
    world_seed: u64,
) -> bool {
    match menu.state {
        MenuState::None => {
            if edge(keys, prev_keys, KEY_ESC) {
                menu.state = MenuState::Main;
                menu.cursor = 0;
            }
        }
        MenuState::Main => {
            // RESUME(0), NEW WORLD(1), SETTINGS(2), QUIT(3)
            if edge(keys, prev_keys, KEY_ESC) {
                menu.state = MenuState::None;
                return false;
            }
            if edge(keys, prev_keys, KEY_UP) && menu.cursor > 0 {
                menu.cursor -= 1;
            }
            if edge(keys, prev_keys, KEY_DOWN) && menu.cursor < 3 {
                menu.cursor += 1;
            }
            if edge(keys, prev_keys, KEY_ENTER) {
                match menu.cursor {
                    0 => menu.state = MenuState::None,
                    1 => {
                        menu.state = MenuState::NewWorld;
                        menu.cursor = 0;
                        // Pre-fill seed input with current world seed
                        menu.seed_len = u64_to_digits(world_seed, &mut menu.seed_digits);
                    }
                    2 => { menu.state = MenuState::Settings; menu.cursor = 0; }
                    3 => return true,
                    _ => {}
                }
            }
        }
        MenuState::NewWorld => {
            // SEED(0), GENERATE(1), BACK(2)
            if edge(keys, prev_keys, KEY_ESC) {
                menu.state = MenuState::Main;
                menu.cursor = 1; // back to NEW WORLD item
                return false;
            }
            if edge(keys, prev_keys, KEY_UP) && menu.cursor > 0 {
                menu.cursor -= 1;
            }
            if edge(keys, prev_keys, KEY_DOWN) && menu.cursor < 2 {
                menu.cursor += 1;
            }
            // Digit input when cursor is on seed field (row 0)
            if menu.cursor == 0 {
                // Number keys (scancodes 2-11 = digits 1-9, 0)
                for sc in 2..=11 {
                    if edge(keys, prev_keys, sc) && menu.seed_len < 19 {
                        let digit = if sc == 11 { b'0' } else { b'0' + (sc as u8 - 1) };
                        menu.seed_digits[menu.seed_len] = digit;
                        menu.seed_len += 1;
                    }
                }
                // Backspace
                if edge(keys, prev_keys, KEY_BACKSPACE) && menu.seed_len > 0 {
                    menu.seed_len -= 1;
                }
            }
            if edge(keys, prev_keys, KEY_ENTER) {
                match menu.cursor {
                    0 => {} // Seed field — type digits with number keys
                    1 => {
                        // GENERATE: parse seed and signal regeneration
                        let seed = if menu.seed_len == 0 {
                            // Empty = random seed from system time
                            std::time::SystemTime::now()
                                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                                .map(|d| d.as_nanos() as u64)
                                .unwrap_or(42)
                        } else {
                            digits_to_u64(&menu.seed_digits, menu.seed_len)
                        };
                        menu.regenerate_seed = Some(seed);
                        menu.state = MenuState::None;
                    }
                    2 => {
                        menu.state = MenuState::Main;
                        menu.cursor = 1;
                    }
                    _ => {}
                }
            }
        }
        MenuState::Settings => {
            // Settings: Keybinds(0), Sensitivity(1), Invert X(2), Invert Y(3), Back(4)
            const LAST: usize = 4;
            if edge(keys, prev_keys, KEY_ESC) {
                menu.state = MenuState::Main;
                menu.cursor = 2; // back to SETTINGS item
                return false;
            }
            if edge(keys, prev_keys, KEY_UP) && menu.cursor > 0 {
                menu.cursor -= 1;
            }
            if edge(keys, prev_keys, KEY_DOWN) && menu.cursor < LAST {
                menu.cursor += 1;
            }
            // Sensitivity slider
            if menu.cursor == 1 {
                if edge(keys, prev_keys, KEY_LEFT) {
                    *mouse_sensitivity = (*mouse_sensitivity - SENSITIVITY_STEP).max(SENSITIVITY_MIN);
                }
                if edge(keys, prev_keys, KEY_RIGHT) {
                    *mouse_sensitivity = (*mouse_sensitivity + SENSITIVITY_STEP).min(SENSITIVITY_MAX);
                }
            }
            if edge(keys, prev_keys, KEY_ENTER) {
                match menu.cursor {
                    0 => { menu.state = MenuState::Keybinds; menu.cursor = 0; }
                    1 => {} // Sensitivity — Left/Right only
                    2 => { *invert_x = !*invert_x; }
                    3 => { *invert_y = !*invert_y; }
                    LAST => { menu.state = MenuState::Main; menu.cursor = 2; }
                    _ => {}
                }
            }
        }
        MenuState::Keybinds => {
            if let Some(action_idx) = menu.rebinding {
                if edge(keys, prev_keys, KEY_ESC) {
                    menu.rebinding = None;
                    return false;
                }
                for sc in 0..256 {
                    if sc == KEY_ESC { continue; }
                    if keys[sc] && !prev_keys[sc] {
                        keybinds.set_key(ALL_ACTIONS[action_idx], sc);
                        menu.rebinding = None;
                        break;
                    }
                }
            } else {
                let max_item = ALL_ACTIONS.len() + 1;
                if edge(keys, prev_keys, KEY_ESC) {
                    menu.state = MenuState::Settings;
                    menu.cursor = 0;
                    return false;
                }
                if edge(keys, prev_keys, KEY_UP) && menu.cursor > 0 {
                    menu.cursor -= 1;
                }
                if edge(keys, prev_keys, KEY_DOWN) && menu.cursor < max_item {
                    menu.cursor += 1;
                }
                if edge(keys, prev_keys, KEY_ENTER) {
                    if menu.cursor < ALL_ACTIONS.len() {
                        menu.rebinding = Some(menu.cursor);
                    } else if menu.cursor == ALL_ACTIONS.len() {
                        *keybinds = KeyBinds::default_binds();
                    } else {
                        menu.state = MenuState::Settings;
                        menu.cursor = 0;
                    }
                }
            }
        }
    }
    false
}

pub fn sys_menu_render(
    fb: &mut Framebuffer, menu: &MenuData, keybinds: &KeyBinds,
    sensitivity: f32, invert_x: bool, invert_y: bool,
) {
    if menu.state == MenuState::None { return; }

    for px in fb.pixels.iter_mut() {
        *px = dim_pixel(*px);
    }

    let cx = fb.w / 2;
    let cy = fb.h / 2;
    let scale = 3;
    let line_h = scale * 5 + scale * 3;
    let text_color = 0xFFCCCCCC;
    let hi_color = 0xFFFFFFFF;
    let title_color = 0xFFFFAA33;

    match menu.state {
        MenuState::Main => {
            let items = ["RESUME", "NEW WORLD", "SETTINGS", "QUIT"];
            let box_w = 260;
            let box_h = line_h * (items.len() + 1) + scale * 4;
            let bx = cx - box_w / 2;
            let by = cy - box_h / 2;
            draw_menu_box(fb, bx, by, box_w, box_h);

            draw_text(fb, cx - text_pw("PAUSED", scale) / 2, by + scale * 3, "PAUSED", scale, title_color);

            for (i, item) in items.iter().enumerate() {
                let y = by + line_h + scale * 3 + i * line_h;
                let c = if i == menu.cursor { hi_color } else { text_color };
                let buf = prefix_str(i == menu.cursor, item);
                draw_text_bytes(fb, cx - text_pw_buf(&buf, scale) / 2, y, &buf, scale, c);
            }
        }
        MenuState::NewWorld => {
            // Visual rows: title, seed input, hint, generate, back
            let box_w = 400;
            let box_h = line_h * 5 + scale * 4;
            let bx = cx - box_w / 2;
            let by = cy - box_h / 2;
            draw_menu_box(fb, bx, by, box_w, box_h);

            draw_text(fb, cx - text_pw("NEW WORLD", scale) / 2, by + scale * 3, "NEW WORLD", scale, title_color);

            let base_y = by + line_h + scale * 3;

            // SEED field (cursor 0)
            {
                let y = base_y;
                let c = if menu.cursor == 0 { hi_color } else { text_color };
                let buf = format_seed_line(menu.cursor == 0, &menu.seed_digits, menu.seed_len);
                draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
            }
            // Hint (not selectable)
            {
                let y = base_y + line_h;
                draw_text(fb, bx + scale * 4 + text_pw("  ", scale), y, "EMPTY - RANDOM", scale, 0xFF888888);
            }
            // GENERATE (cursor 1)
            {
                let y = base_y + 2 * line_h;
                let c = if menu.cursor == 1 { hi_color } else { text_color };
                let buf = prefix_str(menu.cursor == 1, "GENERATE");
                draw_text_bytes(fb, cx - text_pw_buf(&buf, scale) / 2, y, &buf, scale, c);
            }
            // BACK (cursor 2)
            {
                let y = base_y + 3 * line_h;
                let c = if menu.cursor == 2 { hi_color } else { text_color };
                let buf = prefix_str(menu.cursor == 2, "BACK");
                draw_text_bytes(fb, cx - text_pw_buf(&buf, scale) / 2, y, &buf, scale, c);
            }
        }
        MenuState::Settings => {
            let num_items = 5; // Keybinds, Sensitivity, Invert X, Invert Y, Back
            let box_w = 460;
            let box_h = line_h * (num_items + 1) + scale * 4;
            let bx = cx - box_w / 2;
            let by = cy - box_h / 2;
            draw_menu_box(fb, bx, by, box_w, box_h);

            draw_text(fb, cx - text_pw("SETTINGS", scale) / 2, by + scale * 3, "SETTINGS", scale, title_color);

            let mut row = 0;

            // Keybinds
            {
                let y = by + line_h + scale * 3 + row * line_h;
                let c = if menu.cursor == row { hi_color } else { text_color };
                let buf = prefix_str(menu.cursor == row, "KEYBINDS");
                draw_text_bytes(fb, cx - text_pw_buf(&buf, scale) / 2, y, &buf, scale, c);
                row += 1;
            }
            // Sensitivity
            {
                let y = by + line_h + scale * 3 + row * line_h;
                let c = if menu.cursor == row { hi_color } else { text_color };
                let buf = format_sensitivity(menu.cursor == row, sensitivity);
                draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
                if menu.cursor == row {
                    let slider_x = bx + scale * 4 + 24 * (3 * scale + scale);
                    let slider_w = 120;
                    let slider_y = y + scale;
                    draw_rect(fb, slider_x, slider_y, slider_w, scale * 3, 0xFF444444);
                    let fill = ((sensitivity - SENSITIVITY_MIN) / (SENSITIVITY_MAX - SENSITIVITY_MIN)).clamp(0.0, 1.0);
                    let fw = (fill * slider_w as f32) as usize;
                    if fw > 0 {
                        draw_rect(fb, slider_x, slider_y, fw, scale * 3, 0xFFFFAA33);
                    }
                }
                row += 1;
            }
            // Invert X
            {
                let y = by + line_h + scale * 3 + row * line_h;
                let c = if menu.cursor == row { hi_color } else { text_color };
                let buf = format_toggle(menu.cursor == row, "INVERT X", invert_x);
                draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
                row += 1;
            }
            // Invert Y
            {
                let y = by + line_h + scale * 3 + row * line_h;
                let c = if menu.cursor == row { hi_color } else { text_color };
                let buf = format_toggle(menu.cursor == row, "INVERT Y", invert_y);
                draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
                row += 1;
            }
            // Back
            {
                let y = by + line_h + scale * 3 + row * line_h;
                let c = if menu.cursor == row { hi_color } else { text_color };
                let buf = prefix_str(menu.cursor == row, "BACK");
                draw_text_bytes(fb, cx - text_pw_buf(&buf, scale) / 2, y, &buf, scale, c);
            }
        }
        MenuState::Keybinds => {
            let num_actions = ALL_ACTIONS.len();
            let total = num_actions + 2;
            let box_w = 500;
            let box_h = line_h * (total + 1) + scale * 4;
            let bx = cx - box_w / 2;
            let by = cy - box_h / 2;
            draw_menu_box(fb, bx, by, box_w, box_h);

            draw_text(fb, cx - text_pw("KEYBINDS", scale) / 2, by + scale * 3, "KEYBINDS", scale, title_color);

            for i in 0..num_actions {
                let y = by + line_h + scale * 3 + i * line_h;
                let action = ALL_ACTIONS[i];
                let c = if i == menu.cursor { hi_color } else { text_color };
                let sel = i == menu.cursor;

                if menu.rebinding == Some(i) {
                    let buf = concat3(if sel { "> " } else { "  " }, action.name(), "  PRESS KEY...");
                    draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, 0xFFFF6633);
                } else {
                    let kn = key_name(keybinds.key_for(action));
                    let buf = format_keybind_line(if sel { "> " } else { "  " }, action.name(), kn);
                    draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
                }
            }

            // Reset Defaults
            {
                let i = num_actions;
                let y = by + line_h + scale * 3 + i * line_h;
                let c = if menu.cursor == i { hi_color } else { text_color };
                let buf = prefix_str(menu.cursor == i, "RESET DEFAULTS");
                draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
            }
            // Back
            {
                let i = num_actions + 1;
                let y = by + line_h + scale * 3 + i * line_h;
                let c = if menu.cursor == i { hi_color } else { text_color };
                let buf = prefix_str(menu.cursor == i, "BACK");
                draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
            }
        }
        MenuState::None => {}
    }
}

fn draw_menu_box(fb: &mut Framebuffer, x: usize, y: usize, w: usize, h: usize) {
    draw_rect(fb, x.saturating_sub(2), y.saturating_sub(2), w + 4, h + 4, 0xFF888888);
    draw_rect(fb, x, y, w, h, 0xEE111111);
}

fn dim_pixel(c: u32) -> u32 {
    let r = ((c >> 16) & 0xFF) / 3;
    let g = ((c >> 8) & 0xFF) / 3;
    let b = (c & 0xFF) / 3;
    0xFF000000 | (r << 16) | (g << 8) | b
}

fn text_pw(s: &str, scale: usize) -> usize {
    let n = s.len();
    if n == 0 { return 0; }
    n * (3 * scale + scale) - scale
}

fn text_pw_buf(buf: &[u8; 64], scale: usize) -> usize {
    let mut len = 64;
    while len > 0 && buf[len - 1] == b' ' { len -= 1; }
    if len == 0 { return 0; }
    len * (3 * scale + scale) - scale
}

fn prefix_str(selected: bool, s: &str) -> [u8; 64] {
    let prefix = if selected { "> " } else { "  " };
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in prefix.as_bytes() { if i < 64 { buf[i] = c; i += 1; } }
    for &c in s.as_bytes() { if i < 64 { buf[i] = c; i += 1; } }
    buf
}

fn concat3(a: &str, b: &str, c: &str) -> [u8; 64] {
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &ch in a.as_bytes() { if i < 64 { buf[i] = ch; i += 1; } }
    for &ch in b.as_bytes() { if i < 64 { buf[i] = ch; i += 1; } }
    for &ch in c.as_bytes() { if i < 64 { buf[i] = ch; i += 1; } }
    buf
}

fn format_keybind_line(prefix: &str, action_name: &str, key: &str) -> [u8; 64] {
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in prefix.as_bytes() { if i < 64 { buf[i] = c; i += 1; } }
    for &c in action_name.as_bytes() { if i < 64 { buf[i] = c; i += 1; } }
    while i < 20 { if i < 64 { buf[i] = b' '; } i += 1; }
    if i < 64 { buf[i] = b'['; i += 1; }
    for &c in key.as_bytes() { if i < 64 { buf[i] = c; i += 1; } }
    if i < 64 { buf[i] = b']'; }
    buf
}

fn format_sensitivity(selected: bool, val: f32) -> [u8; 64] {
    let prefix = if selected { "> " } else { "  " };
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in prefix.as_bytes() { if i < 64 { buf[i] = c; i += 1; } }
    for &c in b"SENSITIVITY" { if i < 64 { buf[i] = c; i += 1; } }
    while i < 18 { if i < 64 { buf[i] = b' '; } i += 1; }
    let whole = val as u32;
    let frac = ((val - whole as f32) * 10.0 + 0.5) as u32;
    if i < 64 { buf[i] = b'0' + whole as u8; i += 1; }
    if i < 64 { buf[i] = b'.'; i += 1; }
    if i < 64 { buf[i] = b'0' + frac as u8; }
    buf
}

fn format_toggle(selected: bool, label: &str, value: bool) -> [u8; 64] {
    let prefix = if selected { "> " } else { "  " };
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in prefix.as_bytes() { if i < 64 { buf[i] = c; i += 1; } }
    for &c in label.as_bytes() { if i < 64 { buf[i] = c; i += 1; } }
    while i < 18 { if i < 64 { buf[i] = b' '; } i += 1; }
    let val_str = if value { b"ON" as &[u8] } else { b"OFF" };
    for &c in val_str { if i < 64 { buf[i] = c; i += 1; } }
    buf
}

/// Format the seed input line: "> SEED: 42_" or "  SEED: 42"
fn format_seed_line(selected: bool, digits: &[u8; 20], len: usize) -> [u8; 64] {
    let prefix = if selected { "> " } else { "  " };
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in prefix.as_bytes() { if i < 64 { buf[i] = c; i += 1; } }
    for &c in b"SEED: " { if i < 64 { buf[i] = c; i += 1; } }
    for j in 0..len {
        if i < 64 { buf[i] = digits[j]; i += 1; }
    }
    if selected && i < 64 {
        buf[i] = b'_'; // cursor
    }
    buf
}

/// Convert u64 to ASCII digit buffer, return digit count
fn u64_to_digits(val: u64, buf: &mut [u8; 20]) -> usize {
    if val == 0 {
        buf[0] = b'0';
        return 1;
    }
    let mut n = val;
    let mut tmp = [0u8; 20];
    let mut len = 0;
    while n > 0 {
        tmp[len] = b'0' + (n % 10) as u8;
        n /= 10;
        len += 1;
    }
    // Reverse into buf
    for i in 0..len {
        buf[i] = tmp[len - 1 - i];
    }
    len
}

/// Parse ASCII digit buffer to u64 (wrapping on overflow)
fn digits_to_u64(buf: &[u8; 20], len: usize) -> u64 {
    let mut val: u64 = 0;
    for i in 0..len {
        val = val.wrapping_mul(10).wrapping_add((buf[i] - b'0') as u64);
    }
    val
}
