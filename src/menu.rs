//! Menu system: title screen, loading, pause, settings, keybinds, new world.
//! Dark/gold aesthetic inspired by Tarkov, Warframe, and Dune.

use crate::color::{darken, lerp_color};
use crate::hud::{draw_rect, draw_text, draw_text_bytes};
use crate::input::{ALL_ACTIONS, KeyBinds, key_name};
use crate::raster::Framebuffer;

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

// ── AAA Dark/Gold color palette ──────────────────────────────────────────
const GOLD: u32 = 0xFFD4A843;
const GOLD_BRIGHT: u32 = 0xFFFFCC55;
const GOLD_DIM: u32 = 0xFF8A6E2F;
const TEXT_MAIN: u32 = 0xFFE0D8C8;
const TEXT_DIM: u32 = 0xFF7A7060;
const TEXT_SEL: u32 = 0xFFFFFFFF;
const BG_TOP: u32 = 0xFF050508;
const BG_BOT: u32 = 0xFF0C0E14;
const BG_PANEL: u32 = 0xEE101018;
const BORDER: u32 = 0xFF3A3020;
const BORDER_ACCENT: u32 = 0xFF6A5A30;

#[derive(Clone, Copy, PartialEq)]
pub enum MenuState {
    Title,    // startup main menu (before world loads)
    Loading,  // world generation in progress
    None,     // gameplay (no menu)
    Paused,   // ESC pause menu
    NewWorld, // seed input (from pause menu)
    Settings, // settings
    Keybinds, // keybind editor
}

pub struct MenuData {
    pub state: MenuState,
    pub cursor: usize,
    pub rebinding: Option<usize>,
    // New World seed input
    pub seed_digits: [u8; 20],
    pub seed_len: usize,
    pub regenerate_seed: Option<u64>,
    // Title/flow control
    pub start_new_game: bool,  // signal to main.rs: begin world gen
    pub came_from_title: bool, // Settings was opened from Title (not Pause)
    pub loading_seed: u64,     // seed being loaded (for display)
}

impl MenuData {
    pub fn new() -> Self {
        MenuData {
            state: MenuState::Title,
            cursor: 0,
            rebinding: None,
            seed_digits: [0; 20],
            seed_len: 0,
            regenerate_seed: None,
            start_new_game: false,
            came_from_title: false,
            loading_seed: 0,
        }
    }
}

fn edge(keys: &[bool; 256], prev: &[bool; 256], sc: usize) -> bool {
    keys[sc] && !prev[sc]
}

/// Returns true if the game should quit.
pub fn sys_menu_input(
    menu: &mut MenuData,
    keybinds: &mut KeyBinds,
    mouse_sensitivity: &mut f32,
    invert_x: &mut bool,
    invert_y: &mut bool,
    keys: &[bool; 256],
    prev_keys: &[bool; 256],
    world_seed: u64,
    player_model_index: &mut usize,
    character_count: usize,
) -> bool {
    match menu.state {
        MenuState::Title => {
            // NEW GAME(0), SETTINGS(1), QUIT(2)
            if edge(keys, prev_keys, KEY_UP) && menu.cursor > 0 {
                menu.cursor -= 1;
            }
            if edge(keys, prev_keys, KEY_DOWN) && menu.cursor < 2 {
                menu.cursor += 1;
            }
            if edge(keys, prev_keys, KEY_ENTER) {
                match menu.cursor {
                    0 => {
                        // NEW GAME: enter the title background world
                        menu.start_new_game = true;
                    }
                    1 => {
                        menu.state = MenuState::Settings;
                        menu.cursor = 0;
                        menu.came_from_title = true;
                    }
                    2 => return true,
                    _ => {}
                }
            }
        }
        MenuState::Loading => {
            // No input during loading — just a display state
        }
        MenuState::None => {
            if edge(keys, prev_keys, KEY_ESC) {
                menu.state = MenuState::Paused;
                menu.cursor = 0;
            }
        }
        MenuState::Paused => {
            // RESUME(0), NEW WORLD(1), CHARACTER(2), SETTINGS(3), QUIT(4)
            if edge(keys, prev_keys, KEY_ESC) {
                menu.state = MenuState::None;
                return false;
            }
            if edge(keys, prev_keys, KEY_UP) && menu.cursor > 0 {
                menu.cursor -= 1;
            }
            if edge(keys, prev_keys, KEY_DOWN) && menu.cursor < 4 {
                menu.cursor += 1;
            }
            if menu.cursor == 2 && character_count > 0 {
                // Left/Right to cycle character model
                if edge(keys, prev_keys, KEY_LEFT) {
                    *player_model_index = if *player_model_index == 0 {
                        character_count - 1
                    } else {
                        *player_model_index - 1
                    };
                }
                if edge(keys, prev_keys, KEY_RIGHT) {
                    *player_model_index = (*player_model_index + 1) % character_count;
                }
            }
            if edge(keys, prev_keys, KEY_ENTER) {
                match menu.cursor {
                    0 => menu.state = MenuState::None,
                    1 => {
                        menu.state = MenuState::NewWorld;
                        menu.cursor = 0;
                        menu.seed_len = u64_to_digits(world_seed, &mut menu.seed_digits);
                    }
                    2 => {
                        // Cycle character model on Enter too
                        if character_count > 0 {
                            *player_model_index = (*player_model_index + 1) % character_count;
                        }
                    }
                    3 => {
                        menu.state = MenuState::Settings;
                        menu.cursor = 0;
                        menu.came_from_title = false;
                    }
                    4 => return true,
                    _ => {}
                }
            }
        }
        MenuState::NewWorld => {
            // SEED(0), GENERATE(1), BACK(2)
            if edge(keys, prev_keys, KEY_ESC) {
                menu.state = MenuState::Paused;
                menu.cursor = 1;
                return false;
            }
            if edge(keys, prev_keys, KEY_UP) && menu.cursor > 0 {
                menu.cursor -= 1;
            }
            if edge(keys, prev_keys, KEY_DOWN) && menu.cursor < 2 {
                menu.cursor += 1;
            }
            if menu.cursor == 0 {
                for sc in 2..=11 {
                    if edge(keys, prev_keys, sc) && menu.seed_len < 19 {
                        let digit = if sc == 11 {
                            b'0'
                        } else {
                            b'0' + (sc as u8 - 1)
                        };
                        menu.seed_digits[menu.seed_len] = digit;
                        menu.seed_len += 1;
                    }
                }
                if edge(keys, prev_keys, KEY_BACKSPACE) && menu.seed_len > 0 {
                    menu.seed_len -= 1;
                }
            }
            if edge(keys, prev_keys, KEY_ENTER) {
                match menu.cursor {
                    0 => {}
                    1 => {
                        let seed = if menu.seed_len == 0 {
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
                        menu.state = MenuState::Paused;
                        menu.cursor = 1;
                    }
                    _ => {}
                }
            }
        }
        MenuState::Settings => {
            const LAST: usize = 4;
            if edge(keys, prev_keys, KEY_ESC) {
                if menu.came_from_title {
                    menu.state = MenuState::Title;
                    menu.cursor = 1;
                } else {
                    menu.state = MenuState::Paused;
                    menu.cursor = 3; // SETTINGS is now index 3 in Paused menu
                }
                return false;
            }
            if edge(keys, prev_keys, KEY_UP) && menu.cursor > 0 {
                menu.cursor -= 1;
            }
            if edge(keys, prev_keys, KEY_DOWN) && menu.cursor < LAST {
                menu.cursor += 1;
            }
            if menu.cursor == 1 {
                if edge(keys, prev_keys, KEY_LEFT) {
                    *mouse_sensitivity =
                        (*mouse_sensitivity - SENSITIVITY_STEP).max(SENSITIVITY_MIN);
                }
                if edge(keys, prev_keys, KEY_RIGHT) {
                    *mouse_sensitivity =
                        (*mouse_sensitivity + SENSITIVITY_STEP).min(SENSITIVITY_MAX);
                }
            }
            if edge(keys, prev_keys, KEY_ENTER) {
                match menu.cursor {
                    0 => {
                        menu.state = MenuState::Keybinds;
                        menu.cursor = 0;
                    }
                    1 => {}
                    2 => {
                        *invert_x = !*invert_x;
                    }
                    3 => {
                        *invert_y = !*invert_y;
                    }
                    LAST => {
                        if menu.came_from_title {
                            menu.state = MenuState::Title;
                            menu.cursor = 1;
                        } else {
                            menu.state = MenuState::Paused;
                            menu.cursor = 3; // SETTINGS is now index 3 in Paused menu
                        }
                    }
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
                    if sc == KEY_ESC {
                        continue;
                    }
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
    fb: &mut Framebuffer,
    menu: &MenuData,
    keybinds: &KeyBinds,
    sensitivity: f32,
    invert_x: bool,
    invert_y: bool,
    player_model_index: usize,
    character_names: &[String],
) {
    if menu.state == MenuState::None {
        return;
    }

    let cx = fb.w / 2;
    let cy = fb.h / 2;

    match menu.state {
        MenuState::Title => {
            // Game world renders underneath — apply vignette overlay
            draw_title_vignette(fb);

            // Decorative horizontal lines
            draw_hline(fb, 0, fb.h / 5, fb.w, BORDER);
            draw_hline(fb, 0, fb.h * 4 / 5, fb.w, BORDER);

            // Title: CLAUDING
            let title_scale = 8;
            let title = "CLAUDING";
            let tw = text_pw(title, title_scale);
            let title_y = fb.h * 25 / 100;
            draw_text(fb, cx - tw / 2, title_y, title, title_scale, GOLD);

            // Gold accent line under title
            let accent_y = title_y + title_scale * 5 + title_scale * 2;
            let accent_w = tw + title_scale * 4;
            draw_hline(fb, cx - accent_w / 2, accent_y, accent_w, GOLD_DIM);

            // Menu items
            let item_scale = 4;
            let line_h = item_scale * 5 + item_scale * 4;
            let items = ["NEW GAME", "SETTINGS", "QUIT"];
            let items_start_y = accent_y + line_h * 2;

            for (i, item) in items.iter().enumerate() {
                let y = items_start_y + i * line_h;
                let selected = i == menu.cursor;
                let color = if selected { TEXT_SEL } else { TEXT_DIM };
                let iw = text_pw(item, item_scale);
                draw_text(fb, cx - iw / 2, y, item, item_scale, color);

                if selected {
                    // Gold marker lines flanking selected item
                    let marker_w = iw / 3;
                    let my = y + item_scale * 5 + item_scale;
                    draw_hline(fb, cx - iw / 2, my, marker_w, GOLD);
                    draw_hline(fb, cx + iw / 2 - marker_w, my, marker_w, GOLD);
                }
            }

            // Footer hint
            let hint_scale = 2;
            let hint = "ARROWS TO NAVIGATE   ENTER TO SELECT";
            let hw = text_pw(hint, hint_scale);
            draw_text(
                fb,
                cx - hw / 2,
                fb.h - hint_scale * 5 - hint_scale * 6,
                hint,
                hint_scale,
                TEXT_DIM,
            );
        }
        MenuState::Loading => {
            draw_atmospheric_bg(fb);

            // LOADING WORLD
            let title_scale = 5;
            let title = "LOADING WORLD";
            let tw = text_pw(title, title_scale);
            let title_y = fb.h * 38 / 100;
            draw_text(fb, cx - tw / 2, title_y, title, title_scale, GOLD);

            // Seed display
            let seed_scale = 3;
            let mut seed_buf = [b' '; 64];
            let mut si = 0;
            for &c in b"SEED  " {
                if si < 64 {
                    seed_buf[si] = c;
                    si += 1;
                }
            }
            let mut digit_buf = [0u8; 20];
            let dlen = u64_to_digits(menu.loading_seed, &mut digit_buf);
            for j in 0..dlen {
                if si < 64 {
                    seed_buf[si] = digit_buf[j];
                    si += 1;
                }
            }
            let seed_w = text_pw_buf(&seed_buf, seed_scale);
            let seed_y = title_y + title_scale * 5 + title_scale * 4;
            draw_text_bytes(fb, cx - seed_w / 2, seed_y, &seed_buf, seed_scale, TEXT_DIM);

            // Progress bar (decorative — full bar since we render one frame then block)
            let bar_w = 400.min(fb.w * 2 / 5);
            let bar_h = 4;
            let bar_y = seed_y + seed_scale * 5 + seed_scale * 6;
            let bar_x = cx - bar_w / 2;
            draw_rect(fb, bar_x, bar_y, bar_w, bar_h, BORDER);
            // Animated-looking partial fill
            draw_rect(fb, bar_x, bar_y, bar_w / 3, bar_h, GOLD_DIM);

            // Hint
            let hint_scale = 2;
            let hint = "GENERATING TERRAIN...";
            let hw = text_pw(hint, hint_scale);
            let hint_y = bar_y + bar_h + hint_scale * 4;
            draw_text(fb, cx - hw / 2, hint_y, hint, hint_scale, TEXT_DIM);
        }
        MenuState::Paused => {
            // Darken 3D scene behind pause menu
            for px in fb.pixels.iter_mut() {
                *px = darken(*px, 0.15);
            }

            let scale = 3;
            let line_h = scale * 5 + scale * 4;

            let num_items = 5; // RESUME, NEW WORLD, CHARACTER, SETTINGS, QUIT
            let panel_w = 380;
            let panel_h = line_h * (num_items + 2) + scale * 4;
            let bx = cx - panel_w / 2;
            let by = cy - panel_h / 2;
            draw_panel(fb, bx, by, panel_w, panel_h);

            // Title
            let title = "PAUSED";
            let title_scale = 4;
            let tw = text_pw(title, title_scale);
            draw_text(fb, cx - tw / 2, by + scale * 4, title, title_scale, GOLD);

            // Separator
            let sep_y = by + title_scale * 5 + scale * 6;
            draw_hline(
                fb,
                bx + scale * 2,
                sep_y,
                panel_w - scale * 4,
                BORDER_ACCENT,
            );

            // Items
            let items_y = sep_y + scale * 3;
            let item_labels = ["RESUME", "NEW WORLD", "CHARACTER", "SETTINGS", "QUIT"];
            for (i, item) in item_labels.iter().enumerate() {
                let y = items_y + i * line_h;
                let selected = i == menu.cursor;
                let color = if selected { TEXT_SEL } else { TEXT_DIM };

                if i == 2 && !character_names.is_empty() {
                    // CHARACTER row: show model name with left/right arrows
                    let model_name = &character_names[player_model_index % character_names.len()];
                    let buf = format_character_line(selected, model_name);
                    draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, color);
                    if selected {
                        draw_hline(fb, bx + scale * 2, y - scale, scale * 2, GOLD);
                    }
                } else {
                    let iw = text_pw(item, scale);
                    draw_text(fb, cx - iw / 2, y, item, scale, color);
                    if selected {
                        let dash_x = cx - iw / 2 - scale * 4;
                        draw_hline(fb, dash_x, y + scale * 2, scale * 2, GOLD);
                    }
                }
            }
        }
        MenuState::NewWorld => {
            for px in fb.pixels.iter_mut() {
                *px = darken(*px, 0.15);
            }

            let scale = 3;
            let line_h = scale * 5 + scale * 4;

            let panel_w = 480;
            let panel_h = line_h * 6 + scale * 4;
            let bx = cx - panel_w / 2;
            let by = cy - panel_h / 2;
            draw_panel(fb, bx, by, panel_w, panel_h);

            let title_scale = 4;
            let title = "NEW WORLD";
            let tw = text_pw(title, title_scale);
            draw_text(fb, cx - tw / 2, by + scale * 4, title, title_scale, GOLD);

            let sep_y = by + title_scale * 5 + scale * 6;
            draw_hline(
                fb,
                bx + scale * 2,
                sep_y,
                panel_w - scale * 4,
                BORDER_ACCENT,
            );

            let items_y = sep_y + scale * 3;

            // SEED field (cursor 0)
            {
                let y = items_y;
                let selected = menu.cursor == 0;
                let color = if selected { TEXT_SEL } else { TEXT_MAIN };
                let buf = format_seed_line(selected, &menu.seed_digits, menu.seed_len);
                draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, color);
                if selected {
                    draw_hline(fb, bx + scale * 2, y - scale, scale * 2, GOLD);
                }
            }
            // Hint
            {
                let y = items_y + line_h;
                draw_text(
                    fb,
                    bx + scale * 4 + text_pw("  ", scale),
                    y,
                    "EMPTY - RANDOM",
                    2,
                    TEXT_DIM,
                );
            }
            // Separator before buttons
            let btn_sep_y = items_y + line_h * 2;
            draw_hline(fb, bx + scale * 2, btn_sep_y, panel_w - scale * 4, BORDER);
            // GENERATE (cursor 1)
            {
                let y = btn_sep_y + scale * 3;
                let selected = menu.cursor == 1;
                let color = if selected { TEXT_SEL } else { TEXT_DIM };
                let iw = text_pw("GENERATE", scale);
                draw_text(fb, cx - iw / 2, y, "GENERATE", scale, color);
                if selected {
                    draw_hline(fb, cx - iw / 2 - scale * 4, y + scale * 2, scale * 2, GOLD);
                }
            }
            // BACK (cursor 2)
            {
                let y = btn_sep_y + scale * 3 + line_h;
                let selected = menu.cursor == 2;
                let color = if selected { TEXT_SEL } else { TEXT_DIM };
                let iw = text_pw("BACK", scale);
                draw_text(fb, cx - iw / 2, y, "BACK", scale, color);
                if selected {
                    draw_hline(fb, cx - iw / 2 - scale * 4, y + scale * 2, scale * 2, GOLD);
                }
            }
        }
        MenuState::Settings => {
            // Game world renders underneath — darken or vignette depending on origin
            if menu.came_from_title {
                draw_title_vignette(fb);
            } else {
                for px in fb.pixels.iter_mut() {
                    *px = darken(*px, 0.15);
                }
            }

            let scale = 3;
            let line_h = scale * 5 + scale * 4;
            let num_items = 5;

            let panel_w = 480;
            let panel_h = line_h * (num_items + 2) + scale * 4;
            let bx = cx - panel_w / 2;
            let by = cy - panel_h / 2;
            draw_panel(fb, bx, by, panel_w, panel_h);

            let title_scale = 4;
            let title = "SETTINGS";
            let tw = text_pw(title, title_scale);
            draw_text(fb, cx - tw / 2, by + scale * 4, title, title_scale, GOLD);

            let sep_y = by + title_scale * 5 + scale * 6;
            draw_hline(
                fb,
                bx + scale * 2,
                sep_y,
                panel_w - scale * 4,
                BORDER_ACCENT,
            );

            let items_y = sep_y + scale * 3;
            let mut row = 0;

            // Keybinds
            {
                let y = items_y + row * line_h;
                let sel = menu.cursor == row;
                let c = if sel { TEXT_SEL } else { TEXT_DIM };
                let iw = text_pw("KEYBINDS", scale);
                draw_text(fb, cx - iw / 2, y, "KEYBINDS", scale, c);
                if sel {
                    draw_hline(fb, cx - iw / 2 - scale * 4, y + scale * 2, scale * 2, GOLD);
                }
                row += 1;
            }
            // Sensitivity
            {
                let y = items_y + row * line_h;
                let sel = menu.cursor == row;
                let c = if sel { TEXT_SEL } else { TEXT_MAIN };
                let buf = format_sensitivity(sel, sensitivity);
                draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
                if sel {
                    draw_hline(fb, bx + scale * 2, y - scale, scale * 2, GOLD);
                    // Slider
                    let slider_x = bx + scale * 4 + 24 * (3 * scale + scale);
                    let slider_w = 120;
                    let slider_y = y + scale;
                    draw_rect(fb, slider_x, slider_y, slider_w, scale * 2, BORDER);
                    let fill = ((sensitivity - SENSITIVITY_MIN)
                        / (SENSITIVITY_MAX - SENSITIVITY_MIN))
                        .clamp(0.0, 1.0);
                    let fw = (fill * slider_w as f32) as usize;
                    if fw > 0 {
                        draw_rect(fb, slider_x, slider_y, fw, scale * 2, GOLD);
                    }
                }
                row += 1;
            }
            // Invert X
            {
                let y = items_y + row * line_h;
                let sel = menu.cursor == row;
                let c = if sel { TEXT_SEL } else { TEXT_MAIN };
                let buf = format_toggle(sel, "INVERT X", invert_x);
                draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
                if sel {
                    draw_hline(fb, bx + scale * 2, y - scale, scale * 2, GOLD);
                }
                row += 1;
            }
            // Invert Y
            {
                let y = items_y + row * line_h;
                let sel = menu.cursor == row;
                let c = if sel { TEXT_SEL } else { TEXT_MAIN };
                let buf = format_toggle(sel, "INVERT Y", invert_y);
                draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
                if sel {
                    draw_hline(fb, bx + scale * 2, y - scale, scale * 2, GOLD);
                }
                row += 1;
            }
            // Back
            {
                let y = items_y + row * line_h;
                let sel = menu.cursor == row;
                let c = if sel { TEXT_SEL } else { TEXT_DIM };
                let iw = text_pw("BACK", scale);
                draw_text(fb, cx - iw / 2, y, "BACK", scale, c);
                if sel {
                    draw_hline(fb, cx - iw / 2 - scale * 4, y + scale * 2, scale * 2, GOLD);
                }
            }
        }
        MenuState::Keybinds => {
            if menu.came_from_title {
                draw_title_vignette(fb);
            } else {
                for px in fb.pixels.iter_mut() {
                    *px = darken(*px, 0.15);
                }
            }

            let scale = 3;
            let line_h = scale * 5 + scale * 4;
            let num_actions = ALL_ACTIONS.len();
            // Player actions + vehicle header + 4 vehicle rows + reset + back
            let total = num_actions + 1 + 4 + 2;

            let panel_w = 520;
            let panel_h = line_h * (total + 2) + scale * 4;
            let bx = cx - panel_w / 2;
            let by = cy - panel_h / 2;
            draw_panel(fb, bx, by, panel_w, panel_h);

            let title_scale = 4;
            let title = "KEYBINDS";
            let tw = text_pw(title, title_scale);
            draw_text(fb, cx - tw / 2, by + scale * 4, title, title_scale, GOLD);

            let sep_y = by + title_scale * 5 + scale * 6;
            draw_hline(
                fb,
                bx + scale * 2,
                sep_y,
                panel_w - scale * 4,
                BORDER_ACCENT,
            );

            let items_y = sep_y + scale * 3;

            // Rebindable player actions
            for i in 0..num_actions {
                let y = items_y + i * line_h;
                let action = ALL_ACTIONS[i];
                let sel = i == menu.cursor;
                let c = if sel { TEXT_SEL } else { TEXT_MAIN };

                if menu.rebinding == Some(i) {
                    let buf = concat3(
                        if sel { "> " } else { "  " },
                        action.name(),
                        "  PRESS KEY...",
                    );
                    draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, GOLD_BRIGHT);
                } else {
                    let kn = key_name(keybinds.key_for(action));
                    let buf = format_keybind_line(if sel { "> " } else { "  " }, action.name(), kn);
                    draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, c);
                }
                if sel && menu.rebinding.is_none() {
                    draw_hline(fb, bx + scale * 2, y - scale, scale * 2, GOLD);
                }
            }

            // Vehicle controls section (read-only, uses same keybinds contextually)
            let veh_y = items_y + num_actions * line_h;
            draw_hline(
                fb,
                bx + scale * 2,
                veh_y,
                panel_w - scale * 4,
                BORDER_ACCENT,
            );
            draw_text(
                fb,
                bx + scale * 4,
                veh_y + scale * 2,
                "VEHICLE",
                2,
                GOLD_DIM,
            );
            let veh_items_y = veh_y + scale * 2 + 2 * 5 + scale * 2;
            let veh_binds: [(&str, crate::input::Action); 4] = [
                ("Throttle", crate::input::Action::MoveForward),
                ("Brake", crate::input::Action::MoveBack),
                ("Steer", crate::input::Action::MoveLeft), // display both L/R
                ("Handbrake", crate::input::Action::Jump),
            ];
            for (vi, (label, action)) in veh_binds.iter().enumerate() {
                let y = veh_items_y + vi * line_h;
                if *label == "Steer" {
                    let lk = key_name(keybinds.key_for(crate::input::Action::MoveLeft));
                    let rk = key_name(keybinds.key_for(crate::input::Action::MoveRight));
                    let buf = format_vehicle_steer_line(label, lk, rk);
                    draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, TEXT_DIM);
                } else {
                    let kn = key_name(keybinds.key_for(*action));
                    let buf = format_keybind_line("  ", label, kn);
                    draw_text_bytes(fb, bx + scale * 4, y, &buf, scale, TEXT_DIM);
                }
            }

            // Reset Defaults
            {
                let i = num_actions;
                let y = veh_items_y + 4 * line_h;
                let sel = menu.cursor == i;
                let c = if sel { TEXT_SEL } else { TEXT_DIM };
                let iw = text_pw("RESET DEFAULTS", scale);
                draw_text(fb, bx + scale * 4, y, "RESET DEFAULTS", scale, c);
                if sel {
                    draw_hline(fb, bx + scale * 2, y - scale, scale * 2, GOLD);
                }
                let _ = iw;
            }
            // Back
            {
                let i = num_actions + 1;
                let y = veh_items_y + 5 * line_h;
                let sel = menu.cursor == i;
                let c = if sel { TEXT_SEL } else { TEXT_DIM };
                draw_text(fb, bx + scale * 4, y, "BACK", scale, c);
                if sel {
                    draw_hline(fb, bx + scale * 2, y - scale, scale * 2, GOLD);
                }
            }
        }
        MenuState::None => {}
    }
}

// ── Rendering primitives ─────────────────────────────────────────────────

fn draw_hline(fb: &mut Framebuffer, x: usize, y: usize, w: usize, color: u32) {
    if y >= fb.h {
        return;
    }
    let start = x.min(fb.w);
    let end = (x + w).min(fb.w);
    let row = y * fb.w;
    for px in start..end {
        fb.pixels[row + px] = color;
    }
}

/// Dark atmospheric gradient background with vignette
fn draw_atmospheric_bg(fb: &mut Framebuffer) {
    let w = fb.w;
    let h = fb.h;
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let max_dist = (cx * cx + cy * cy).sqrt();

    for y in 0..h {
        let t = y as f32 / h as f32;
        let base = lerp_color(BG_TOP, BG_BOT, t);
        let row = y * w;
        let dy = (y as f32 - cy) / max_dist;
        for x in 0..w {
            let dx = (x as f32 - cx) / max_dist;
            let dist = (dx * dx + dy * dy).sqrt();
            // Vignette: darken edges, corners darkest
            let vignette = 1.0 - (dist * 0.7).min(0.5);
            fb.pixels[row + x] = darken(base, vignette);
        }
    }
}

/// Cinematic vignette overlay on live game scene (title screen / title-origin menus).
/// Darkens the scene uniformly + stronger darkening at edges for readability.
fn draw_title_vignette(fb: &mut Framebuffer) {
    let w = fb.w;
    let h = fb.h;
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let max_dist = (cx * cx + cy * cy).sqrt();

    for y in 0..h {
        let row = y * w;
        let dy = (y as f32 - cy) / max_dist;
        for x in 0..w {
            let dx = (x as f32 - cx) / max_dist;
            let dist = (dx * dx + dy * dy).sqrt();
            // Base darken (0.45) + edge vignette (down to 0.15 at corners)
            let factor = 0.45 - (dist * 0.5).min(0.30);
            fb.pixels[row + x] = darken(fb.pixels[row + x], factor);
        }
    }
}

/// Panel with border and gold accent line at top
fn draw_panel(fb: &mut Framebuffer, x: usize, y: usize, w: usize, h: usize) {
    // Outer border
    draw_rect(
        fb,
        x.saturating_sub(1),
        y.saturating_sub(1),
        w + 2,
        h + 2,
        BORDER,
    );
    // Panel fill
    draw_rect(fb, x, y, w, h, BG_PANEL);
    // Gold accent at top
    draw_hline(fb, x, y, w, BORDER_ACCENT);
}

// ── Text helpers ─────────────────────────────────────────────────────────

fn text_pw(s: &str, scale: usize) -> usize {
    let n = s.len();
    if n == 0 {
        return 0;
    }
    n * (3 * scale + scale) - scale
}

fn text_pw_buf(buf: &[u8; 64], scale: usize) -> usize {
    let mut len = 64;
    while len > 0 && buf[len - 1] == b' ' {
        len -= 1;
    }
    if len == 0 {
        return 0;
    }
    len * (3 * scale + scale) - scale
}

fn concat3(a: &str, b: &str, c: &str) -> [u8; 64] {
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &ch in a.as_bytes() {
        if i < 64 {
            buf[i] = ch;
            i += 1;
        }
    }
    for &ch in b.as_bytes() {
        if i < 64 {
            buf[i] = ch;
            i += 1;
        }
    }
    for &ch in c.as_bytes() {
        if i < 64 {
            buf[i] = ch;
            i += 1;
        }
    }
    buf
}

fn format_keybind_line(prefix: &str, action_name: &str, key: &str) -> [u8; 64] {
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in prefix.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    for &c in action_name.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    while i < 20 {
        if i < 64 {
            buf[i] = b' ';
        }
        i += 1;
    }
    if i < 64 {
        buf[i] = b'[';
        i += 1;
    }
    for &c in key.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    if i < 64 {
        buf[i] = b']';
    }
    buf
}

fn format_vehicle_steer_line(label: &str, left_key: &str, right_key: &str) -> [u8; 64] {
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in b"  " {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    for &c in label.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    while i < 20 {
        if i < 64 {
            buf[i] = b' ';
        }
        i += 1;
    }
    if i < 64 {
        buf[i] = b'[';
        i += 1;
    }
    for &c in left_key.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    if i < 64 {
        buf[i] = b'/';
        i += 1;
    }
    for &c in right_key.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    if i < 64 {
        buf[i] = b']';
    }
    buf
}

fn format_sensitivity(selected: bool, val: f32) -> [u8; 64] {
    let prefix = if selected { "> " } else { "  " };
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in prefix.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    for &c in b"SENSITIVITY" {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    while i < 18 {
        if i < 64 {
            buf[i] = b' ';
        }
        i += 1;
    }
    let whole = val as u32;
    let frac = ((val - whole as f32) * 10.0 + 0.5) as u32;
    if i < 64 {
        buf[i] = b'0' + whole as u8;
        i += 1;
    }
    if i < 64 {
        buf[i] = b'.';
        i += 1;
    }
    if i < 64 {
        buf[i] = b'0' + frac as u8;
    }
    buf
}

fn format_toggle(selected: bool, label: &str, value: bool) -> [u8; 64] {
    let prefix = if selected { "> " } else { "  " };
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in prefix.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    for &c in label.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    while i < 18 {
        if i < 64 {
            buf[i] = b' ';
        }
        i += 1;
    }
    let val_str = if value { b"ON" as &[u8] } else { b"OFF" };
    for &c in val_str {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    buf
}

fn format_seed_line(selected: bool, digits: &[u8; 20], len: usize) -> [u8; 64] {
    let prefix = if selected { "> " } else { "  " };
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in prefix.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    for &c in b"SEED: " {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    for j in 0..len {
        if i < 64 {
            buf[i] = digits[j];
            i += 1;
        }
    }
    if selected && i < 64 {
        buf[i] = b'_';
    }
    buf
}

fn format_character_line(selected: bool, model_name: &str) -> [u8; 64] {
    let prefix = if selected { "> " } else { "  " };
    let mut buf = [b' '; 64];
    let mut i = 0;
    for &c in prefix.as_bytes() {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    for &c in b"CHARACTER" {
        if i < 64 {
            buf[i] = c;
            i += 1;
        }
    }
    while i < 18 {
        if i < 64 {
            buf[i] = b' ';
        }
        i += 1;
    }
    if i < 64 {
        buf[i] = b'<';
        i += 1;
    }
    if i < 64 {
        buf[i] = b' ';
        i += 1;
    }
    // Convert name to uppercase for display
    for &c in model_name.as_bytes() {
        if i < 58 {
            buf[i] = if c >= b'a' && c <= b'z' {
                c - 32
            } else if c == b'_' {
                b' '
            } else {
                c
            };
            i += 1;
        }
    }
    if i < 64 {
        buf[i] = b' ';
        i += 1;
    }
    if i < 64 {
        buf[i] = b'>';
    }
    buf
}

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
    for i in 0..len {
        buf[i] = tmp[len - 1 - i];
    }
    len
}

fn digits_to_u64(buf: &[u8; 20], len: usize) -> u64 {
    let mut val: u64 = 0;
    for i in 0..len {
        val = val.wrapping_mul(10).wrapping_add((buf[i] - b'0') as u64);
    }
    val
}
