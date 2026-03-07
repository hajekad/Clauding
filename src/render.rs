// sys_render: transform world + player geometry to screen, rasterize
// Near-plane clipping, backface/distance culling, day/night lighting

use crate::gpu::GpuVertex;
use crate::math::*;
use crate::mesh;
use crate::raster::*;
use crate::state::*;

const SKIN_COLOR: u32 = 0xFFDEB887;
const SHIRT_COLOR: u32 = 0xFF3355AA;
const PANTS_COLOR: u32 = 0xFF333355;

const VEHICLE_BODY_COLOR_DARKEN: f32 = 0.7;
const WINDSHIELD_COLOR: u32 = 0xFF88AACC;
const TIRE_COLOR: u32 = 0xFF222222;

const NEAR_W: f32 = 0.1;

// Trash bin colors
const BIN_COLOR: u32 = 0xFF226622;
const BIN_LID_COLOR: u32 = 0xFF338833;

// Interior colors
const DASHBOARD_COLOR: u32 = 0xFF333333;
const SEAT_COLOR: u32 = 0xFF444455;
const STEERING_COLOR: u32 = 0xFF222222;

// Carried item colors
const BAG_COLOR: u32 = 0xFF886644;

// ═══════════════════════════════════════════════════════════════════════════
// ACU-LEVEL CHARACTER SYSTEM — anatomically detailed body parts + period clothing
// Each body part modeled separately at high polycount with micro-geometry detail
// ═══════════════════════════════════════════════════════════════════════════

// --- Material palettes ---
const SKIN_TONES: [u32; 8] = [
    0xFFDEB887, 0xFFD2A87A, 0xFFC89B6E, 0xFFE8C9A0,
    0xFFBB9060, 0xFFA07850, 0xFFCCA882, 0xFFDDBC98,
];
const HAIR_COLORS: [u32; 8] = [
    0xFF332211, 0xFF443322, 0xFF221100, 0xFF554433,
    0xFF664422, 0xFF887755, 0xFF111111, 0xFFBBAA88, // black, silver/wig
];
const HAT_COLORS: [u32; 6] = [
    0xFF333333, 0xFF554433, 0xFF222222, 0xFF443344, 0xFF665544, 0xFF2A2A2A,
];
const COAT_COLORS: [u32; 10] = [
    0xFF443322, 0xFF333355, 0xFF554433, 0xFF444444, 0xFF553333,
    0xFF335544, 0xFF555544, 0xFF443355, 0xFF3A3A28, 0xFF4A3828,
];
const VEST_COLORS: [u32; 6] = [
    0xFF887755, 0xFF998866, 0xFF776655, 0xFF667744, 0xFF886644, 0xFF998877,
];
const SASH_COLORS: [u32; 4] = [0xFFAA2222, 0xFF882222, 0xFFCC3333, 0xFF993322];
const BOOT_BROWN: u32 = 0xFF3D2816;
const BOOT_BLACK: u32 = 0xFF1A1A1A;
const LEATHER_DARK: u32 = 0xFF3A2A1A;
const LEATHER_MED: u32 = 0xFF5A4A3A;
const SHIRT_LINEN: u32 = 0xFFDDD8CC;
const BUCKLE_BRASS: u32 = 0xFFCCBB66;
const BUCKLE_SILVER: u32 = 0xFFAAAAAAu32;
const STITCH_DARK: u32 = 0xFF222211;
const BUTTON_BRASS: u32 = 0xFFBBAA55;

/// Full ACU-style appearance descriptor — derived from seed, no struct storage.
struct NpcAppearance {
    skin: u32,
    hair: u32,
    hat_type: u8,    // 0=none,1=tricorn,2=top_hat,3=cap,4=bonnet,5=wide_brim,6=hood,7=powdered_wig
    hat_col: u32,
    coat_col: u32,
    vest_col: u32,
    has_coat: bool,
    has_cape: bool,       // shoulder cape/mantle drape
    has_sash: bool,       // red sash around waist
    has_cross_strap: bool,// diagonal leather chest strap
    has_bracers: bool,    // leather forearm bracers
    boot_type: u8,        // 0=buckled shoes, 1=mid-boot, 2=tall-boot
    boot_col: u32,
    sash_col: u32,
    face_age: u8,         // 0=young, 1=mid, 2=old (wrinkle density)
    is_female: bool,
}

fn npc_appearance(seed: u32) -> NpcAppearance {
    let s = seed;
    let coat_col = COAT_COLORS[(s / 13) as usize % COAT_COLORS.len()];
    let has_coat = s % 4 != 0;
    NpcAppearance {
        skin: SKIN_TONES[(s / 3) as usize % SKIN_TONES.len()],
        hair: HAIR_COLORS[(s / 5) as usize % HAIR_COLORS.len()],
        hat_type: (s / 7 % 8) as u8,
        hat_col: HAT_COLORS[(s / 11) as usize % HAT_COLORS.len()],
        coat_col,
        vest_col: VEST_COLORS[(s / 17) as usize % VEST_COLORS.len()],
        has_coat,
        has_cape: has_coat && s % 3 == 0,
        has_sash: s % 5 < 2,
        has_cross_strap: has_coat && s % 7 < 2,
        has_bracers: s % 4 < 2,
        boot_type: ((s / 19) % 3) as u8,
        boot_col: if s % 2 == 0 { BOOT_BROWN } else { BOOT_BLACK },
        sash_col: SASH_COLORS[(s / 23) as usize % SASH_COLORS.len()],
        face_age: ((s / 29) % 3) as u8,
        is_female: s % 5 == 0,
    }
}

/// Subtle pseudo-texture: vary color by position hash
fn fabric_vary(base: u32, x: f32, y: f32, seed: f32) -> u32 {
    let h = ((x * 127.1 + y * 311.7 + seed * 74.7).sin() * 43758.5).fract();
    darken(base, 0.93 + h * 0.14)
}

/// Seam line — very thin raised strip to simulate stitching
fn push_seam(tris: &mut Vec<WorldTri>, x: f32, y0: f32, y1: f32, z: f32, color: u32) {
    // Vertical seam line (thin box)
    push_box(tris, x, (y0+y1)*0.5, z, 0.004, (y1-y0).abs(), 0.003, color);
}

/// Horizontal seam
fn push_hseam(tris: &mut Vec<WorldTri>, x0: f32, x1: f32, y: f32, z: f32, color: u32) {
    push_box(tris, (x0+x1)*0.5, y, z, (x1-x0).abs(), 0.004, 0.003, color);
}

/// Linearly interpolate between two 3D points
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, a[2] + (b[2]-a[2])*t]
}

// Sky/fog/light colors for time of day
struct TimeColors {
    sky: u32,
    fog_r: f32, fog_g: f32, fog_b: f32,
    light_dir: Vec3,
    ambient: f32,
    sun_strength: f32,
}

fn time_colors(hour: f32) -> TimeColors {
    let (sky, amb, sun) = if hour < 5.0 {
        (lerp_color(0xFF0A0A20, 0xFF0A0A20, 0.0), 0.15, 0.0)
    } else if hour < 6.5 {
        let t = (hour - 5.0) / 1.5;
        (lerp_color(0xFF0A0A20, 0xFFDD8844, t), 0.15 + t * 0.3, t * 0.4)
    } else if hour < 8.0 {
        let t = (hour - 6.5) / 1.5;
        (lerp_color(0xFFDD8844, 0xFF87CEEB, t), 0.45 + t * 0.2, 0.4 + t * 0.25)
    } else if hour < 16.0 {
        (0xFF87CEEB, 0.65, 0.65)
    } else if hour < 18.0 {
        let t = (hour - 16.0) / 2.0;
        (lerp_color(0xFF87CEEB, 0xFFDD6633, t), 0.65 - t * 0.2, 0.65 - t * 0.25)
    } else if hour < 19.5 {
        let t = (hour - 18.0) / 1.5;
        (lerp_color(0xFFDD6633, 0xFF1A1A40, t), 0.45 - t * 0.3, 0.4 - t * 0.4)
    } else {
        (0xFF0A0A20, 0.15, 0.0)
    };

    let fr = ((sky >> 16) & 0xFF) as f32;
    let fg = ((sky >> 8) & 0xFF) as f32;
    let fb = (sky & 0xFF) as f32;

    let sun_angle = (hour - 6.0) / 12.0 * std::f32::consts::PI;
    let light_dir = if sun > 0.0 {
        let sy = sun_angle.sin().max(0.1);
        let sx = sun_angle.cos() * 0.5;
        let len = (sx * sx + sy * sy + 0.25).sqrt();
        [sx / len, sy / len, 0.5 / len]
    } else {
        [0.0, 1.0, 0.0]
    };

    TimeColors { sky, fog_r: fr, fog_g: fg, fog_b: fb, light_dir, ambient: amb, sun_strength: sun }
}

fn lerp_color(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let r = (((a >> 16) & 0xFF) as f32 * (1.0 - t) + ((b >> 16) & 0xFF) as f32 * t) as u32;
    let g = (((a >> 8) & 0xFF) as f32 * (1.0 - t) + ((b >> 8) & 0xFF) as f32 * t) as u32;
    let b_c = ((a & 0xFF) as f32 * (1.0 - t) + (b & 0xFF) as f32 * t) as u32;
    0xFF000000 | (r << 16) | (g << 8) | b_c
}

pub fn sky_color(hour: f32) -> u32 {
    time_colors(hour).sky
}

pub fn sys_render(
    fb: &mut Framebuffer, world: &WorldData, player: &Player, cam: &Camera,
    hour: f32, scratch: &mut Vec<WorldTri>,
) {
    let tc = time_colors(hour);
    let aspect = fb.w as f32 / fb.h as f32;
    let eye = v3(cam.x, cam.y, cam.z);
    let target = v3(cam.tx, cam.ty, cam.tz);
    let view = m4_look_at(eye, target, v3(0.0, 1.0, 0.0));
    let proj = m4_perspective(60.0_f32.to_radians(), aspect, 0.1, 200.0);
    let vp = m4_mul(&proj, &view);
    let fw = fb.w as f32;
    let fh = fb.h as f32;

    // Static world
    render_tris(fb, &vp, &world.static_tris, eye, &tc, fw, fh);

    // Dynamic entities: generate into scratch buffer, render once
    scratch.clear();
    for (vi, v) in world.vehicles.iter().enumerate() {
        let show_interior = player.in_vehicle == Some(vi);
        gen_vehicle_mesh(v, scratch, show_interior);
    }
    for npc in &world.npcs {
        if npc.state == NpcState::Sleeping { continue; }
        if npc.in_vehicle { continue; }
        gen_npc_mesh(npc, scratch);
    }
    for item in &world.items {
        if !item.active && !item.falling { continue; }
        gen_item_mesh(item, scratch);
    }
    // Trash bins (dynamic — can be carried/moved)
    for bin in &world.trash_bins {
        if bin.carried_by.is_some() { continue; } // rendered with NPC
        gen_trash_bin_mesh(bin, scratch);
    }
    if player.in_vehicle.is_none() {
        gen_player_mesh(player, scratch);
    }
    render_tris(fb, &vp, scratch, eye, &tc, fw, fh);
}

fn render_tris(fb: &mut Framebuffer, vp: &Mat4, tris: &[WorldTri], cam_pos: Vec3, tc: &TimeColors, fw: f32, fh: f32) {
    let fog_dist_sq = FOG_DIST * FOG_DIST;

    for tri in tris {
        let center = [
            (tri.v[0][0] + tri.v[1][0] + tri.v[2][0]) * 0.333,
            (tri.v[0][1] + tri.v[1][1] + tri.v[2][1]) * 0.333,
            (tri.v[0][2] + tri.v[1][2] + tri.v[2][2]) * 0.333,
        ];

        let dx = cam_pos[0] - center[0];
        let dy = cam_pos[1] - center[1];
        let dz = cam_pos[2] - center[2];
        let dist_sq = dx*dx + dy*dy + dz*dz;
        if dist_sq > fog_dist_sq { continue; }

        let dist = dist_sq.sqrt();

        // Compute final color before clipping (flat shading)
        let sun_lit = v3_dot(tri.normal, tc.light_dir).max(0.0) * tc.sun_strength;
        let intensity = sun_lit + tc.ambient;
        let fog = (dist / FOG_DIST).min(1.0);
        let color = shade_and_fog(tri.color, intensity, fog, tc);

        // Transform to clip space
        let c0 = m4_transform_no_div(vp, tri.v[0]);
        let c1 = m4_transform_no_div(vp, tri.v[1]);
        let c2 = m4_transform_no_div(vp, tri.v[2]);

        // Fast path: all vertices in front of near plane
        if c0[3] >= NEAR_W && c1[3] >= NEAR_W && c2[3] >= NEAR_W {
            let s0 = clip_to_screen(c0, fw, fh);
            let s1 = clip_to_screen(c1, fw, fh);
            let s2 = clip_to_screen(c2, fw, fh);

            // Quick off-screen reject
            if s0[0].max(s1[0]).max(s2[0]) < 0.0 { continue; }
            if s0[0].min(s1[0]).min(s2[0]) >= fw { continue; }
            if s0[1].max(s1[1]).max(s2[1]) < 0.0 { continue; }
            if s0[1].min(s1[1]).min(s2[1]) >= fh { continue; }

            draw_triangle(fb, &ScreenTri { v: [s0, s1, s2], color });
            continue;
        }

        // All behind near plane → skip
        if c0[3] < NEAR_W && c1[3] < NEAR_W && c2[3] < NEAR_W { continue; }

        // Near-plane clip (Sutherland-Hodgman against w=NEAR_W)
        let (clipped, nv) = clip_near(&[c0, c1, c2]);
        if nv < 3 { continue; }

        // Fan-triangulate clipped polygon
        let s0 = clip_to_screen(clipped[0], fw, fh);
        for i in 1..nv - 1 {
            let s1 = clip_to_screen(clipped[i], fw, fh);
            let s2 = clip_to_screen(clipped[i + 1], fw, fh);
            draw_triangle(fb, &ScreenTri { v: [s0, s1, s2], color });
        }
    }
}

/// Clip triangle against w=NEAR_W plane. Returns up to 4 vertices.
fn clip_near(verts: &[[f32; 4]; 3]) -> ([[f32; 4]; 4], usize) {
    let mut out = [[0.0f32; 4]; 4];
    let mut n = 0;

    for i in 0..3 {
        let cur = verts[i];
        let nxt = verts[(i + 1) % 3];
        let cur_in = cur[3] >= NEAR_W;
        let nxt_in = nxt[3] >= NEAR_W;

        if cur_in {
            out[n] = cur;
            n += 1;
        }
        if cur_in != nxt_in {
            let t = (NEAR_W - cur[3]) / (nxt[3] - cur[3]);
            out[n] = [
                cur[0] + t * (nxt[0] - cur[0]),
                cur[1] + t * (nxt[1] - cur[1]),
                cur[2] + t * (nxt[2] - cur[2]),
                NEAR_W,
            ];
            n += 1;
        }
    }
    (out, n)
}

#[inline(always)]
fn clip_to_screen(c: [f32; 4], w: f32, h: f32) -> [f32; 3] {
    let inv_w = 1.0 / c[3];
    [
        (c[0] * inv_w + 1.0) * 0.5 * w,
        (1.0 - c[1] * inv_w) * 0.5 * h,
        c[2] * inv_w,
    ]
}

fn shade_and_fog(color: u32, intensity: f32, fog: f32, tc: &TimeColors) -> u32 {
    let r = ((color >> 16) & 0xFF) as f32;
    let g = ((color >> 8) & 0xFF) as f32;
    let b = (color & 0xFF) as f32;
    let i = intensity.clamp(0.1, 1.3);
    let mix = fog * fog;
    let ro = ((r * i * (1.0 - mix) + tc.fog_r * mix) as u32).min(255);
    let go = ((g * i * (1.0 - mix) + tc.fog_g * mix) as u32).min(255);
    let bo = ((b * i * (1.0 - mix) + tc.fog_b * mix) as u32).min(255);
    0xFF000000 | (ro << 16) | (go << 8) | bo
}

// --- Mesh generators (push into shared scratch buffer) ---

// ═══════════════════════════════════════════════════════════════════════════
// CHARACTER BODY GENERATION — each body part separately modeled
// ═══════════════════════════════════════════════════════════════════════════

/// Generate the full anatomical head: skull, face, ears, hair/hat
fn gen_head(tris: &mut Vec<WorldTri>, app: &NpcAppearance, is_job_hat: Option<u32>) {
    let skin = app.skin;
    let hair = app.hair;
    let skin_dk = darken(skin, 0.92);
    let skin_shadow = darken(skin, 0.78);

    // ── SKULL — high-res lathe, 16 profile points × 14 segments ──
    let head_profile: [[f32; 2]; 16] = [
        [0.0,  -0.22],  // chin bottom
        [0.06, -0.21],  // chin front
        [0.09, -0.18],  // jawline front
        [0.11, -0.14],  // jaw angle
        [0.12, -0.10],  // lower cheek
        [0.135,-0.05],  // mid cheek (zygomatic)
        [0.14,  0.00],  // cheekbone peak / eye level
        [0.135, 0.04],  // upper cheek / temple
        [0.12,  0.08],  // temple
        [0.125, 0.12],  // parietal lower
        [0.13,  0.16],  // parietal peak
        [0.12,  0.20],  // upper parietal
        [0.10,  0.23],  // crown approach
        [0.07,  0.25],  // crown
        [0.03,  0.26],  // crown top
        [0.0,   0.265], // apex
    ];
    mesh::lathe_tris(tris, 0.0, 1.70, 0.0, &head_profile, 14, skin);

    // ── BROW / ORBITAL REGION ──
    // Supraorbital ridge (brow bone) — prominent shelf above eyes
    mesh::ellipsoid_tris(tris, 0.0, 1.77, -0.12, 0.12, 0.015, 0.035, 0, skin_dk);
    // Glabella (between brows)
    push_box(tris, 0.0, 1.775, -0.135, 0.03, 0.012, 0.02, skin_dk);
    // Individual eyebrow ridges
    mesh::ellipsoid_tris(tris, -0.055, 1.775, -0.13, 0.045, 0.008, 0.02, 0, darken(hair, 0.8));
    mesh::ellipsoid_tris(tris, 0.055, 1.775, -0.13, 0.045, 0.008, 0.02, 0, darken(hair, 0.8));

    // ── EYES — full orbital anatomy ──
    for &side in &[-1.0f32, 1.0] {
        let ex = side * 0.06;
        // Orbital socket (darkened recess)
        mesh::ellipsoid_tris(tris, ex, 1.74, -0.12, 0.035, 0.02, 0.015, 0, skin_shadow);
        // Eyeball — white sclera
        mesh::sphere_tris(tris, ex, 1.742, -0.125, 0.024, 1, 0xFFEEE8DD);
        // Iris — colored disc
        mesh::sphere_tris(tris, ex, 1.742, -0.148, 0.015, 0, 0xFF556644);
        // Pupil — dark center
        mesh::sphere_tris(tris, ex, 1.742, -0.152, 0.008, 0, 0xFF111100);
        // Upper eyelid — curved skin flap
        mesh::ellipsoid_tris(tris, ex, 1.755, -0.135, 0.032, 0.008, 0.018, 0, skin);
        // Lower eyelid — thinner
        mesh::ellipsoid_tris(tris, ex, 1.73, -0.135, 0.028, 0.005, 0.015, 0, skin);
        // Lacrimal caruncle (inner eye corner pink dot)
        mesh::sphere_tris(tris, ex - side * 0.03, 1.74, -0.135, 0.005, 0, 0xFFCC9988);
        // Crow's feet wrinkles (age-dependent)
        if app.face_age > 0 {
            for wi in 0..3 {
                let wy = 1.745 + wi as f32 * 0.008 - 0.008;
                push_box(tris, ex + side * 0.04, wy, -0.125, 0.012, 0.002, 0.002, skin_shadow);
            }
        }
    }

    // ── NOSE — full anatomy ──
    // Nasal bridge (dorsum)
    push_box(tris, 0.0, 1.735, -0.15, 0.018, 0.04, 0.02, skin);
    // Nasal bone ridge
    push_box(tris, 0.0, 1.755, -0.145, 0.012, 0.015, 0.015, darken(skin, 0.96));
    // Nose tip (lobule)
    mesh::sphere_tris(tris, 0.0, 1.70, -0.165, 0.022, 1, darken(skin, 0.97));
    // Alar wings (nostril sides)
    mesh::sphere_tris(tris, -0.015, 1.695, -0.158, 0.012, 0, darken(skin, 0.93));
    mesh::sphere_tris(tris, 0.015, 1.695, -0.158, 0.012, 0, darken(skin, 0.93));
    // Nostrils (dark openings)
    mesh::sphere_tris(tris, -0.008, 1.693, -0.165, 0.006, 0, darken(skin, 0.55));
    mesh::sphere_tris(tris, 0.008, 1.693, -0.165, 0.006, 0, darken(skin, 0.55));
    // Columella (between nostrils)
    push_box(tris, 0.0, 1.692, -0.162, 0.005, 0.005, 0.005, darken(skin, 0.9));
    // Septum crease
    push_box(tris, 0.0, 1.705, -0.155, 0.003, 0.015, 0.003, darken(skin, 0.88));

    // ── MOUTH — realistic lip anatomy ──
    // Philtrum (vertical groove above upper lip)
    push_box(tris, 0.0, 1.675, -0.155, 0.008, 0.015, 0.004, darken(skin, 0.88));
    // Upper lip — cupid's bow shape (3 segments)
    push_box(tris, 0.0, 1.668, -0.157, 0.05, 0.006, 0.008, darken(skin, 0.82));
    // Upper lip vermilion (red part)
    push_box(tris, 0.0, 1.664, -0.158, 0.04, 0.004, 0.008, 0xFFBB8877);
    // Lower lip — fuller
    mesh::ellipsoid_tris(tris, 0.0, 1.656, -0.155, 0.035, 0.007, 0.01, 0, 0xFFCC9988);
    // Oral commissures (mouth corners — dark dots)
    mesh::sphere_tris(tris, -0.032, 1.662, -0.153, 0.004, 0, darken(skin, 0.65));
    mesh::sphere_tris(tris, 0.032, 1.662, -0.153, 0.004, 0, darken(skin, 0.65));
    // Mentolabial sulcus (groove below lower lip)
    push_box(tris, 0.0, 1.648, -0.152, 0.035, 0.003, 0.003, darken(skin, 0.85));

    // ── CHIN ──
    mesh::sphere_tris(tris, 0.0, 1.635, -0.14, 0.028, 1, skin);
    // Mental protuberance (chin point)
    mesh::sphere_tris(tris, 0.0, 1.628, -0.148, 0.015, 0, darken(skin, 0.95));
    // Chin cleft (dimple) — optional based on seed
    if app.face_age != 1 {
        push_box(tris, 0.0, 1.63, -0.15, 0.004, 0.008, 0.003, darken(skin, 0.82));
    }

    // ── CHEEKBONES — zygomatic prominence ──
    mesh::ellipsoid_tris(tris, -0.1, 1.72, -0.08, 0.035, 0.015, 0.02, 0, darken(skin, 0.97));
    mesh::ellipsoid_tris(tris, 0.1, 1.72, -0.08, 0.035, 0.015, 0.02, 0, darken(skin, 0.97));

    // ── NASOLABIAL FOLDS (nose-to-mouth lines) ──
    for &side in &[-1.0f32, 1.0] {
        let x0 = side * 0.025;
        let x1 = side * 0.035;
        // Upper portion (near nose)
        push_box(tris, x0, 1.69, -0.155, 0.003, 0.01, 0.003, darken(skin, 0.83));
        // Lower portion (toward mouth)
        push_box(tris, x1, 1.67, -0.15, 0.003, 0.015, 0.003, darken(skin, 0.83));
        if app.face_age >= 2 {
            // Deeper folds for older faces
            push_box(tris, x1, 1.66, -0.148, 0.003, 0.01, 0.003, darken(skin, 0.78));
        }
    }

    // ── FOREHEAD wrinkles (age-dependent) ──
    if app.face_age >= 1 {
        for fi in 0..3 {
            let fy = 1.79 + fi as f32 * 0.012;
            push_box(tris, 0.0, fy, -0.13, 0.08, 0.002, 0.002, darken(skin, 0.87));
        }
    }

    // ── JAWLINE — defined mandibular ridge ──
    for &side in &[-1.0f32, 1.0] {
        // Mandibular angle (jaw corner below ear)
        mesh::ellipsoid_tris(tris, side * 0.11, 1.66, -0.02, 0.025, 0.012, 0.03, 0, darken(skin, 0.96));
        // Jaw body (chin to ear line)
        push_box(tris, side * 0.07, 1.65, -0.08, 0.05, 0.01, 0.06, darken(skin, 0.95));
        // Masseter muscle bulge
        mesh::ellipsoid_tris(tris, side * 0.105, 1.68, -0.04, 0.02, 0.025, 0.02, 0, darken(skin, 0.98));
    }

    // ── EARS — helix, antihelix, tragus, lobe ──
    for &side in &[-1.0f32, 1.0] {
        let ex = side * 0.145;
        // Helix (outer ear rim) — elongated ellipsoid
        mesh::ellipsoid_tris(tris, ex, 1.735, 0.005, 0.015, 0.035, 0.012, 0, skin);
        // Antihelix (inner ridge)
        mesh::ellipsoid_tris(tris, ex * 0.95, 1.735, 0.0, 0.01, 0.025, 0.008, 0, darken(skin, 0.92));
        // Tragus (small bump at ear canal)
        mesh::sphere_tris(tris, ex * 0.88, 1.73, -0.01, 0.006, 0, skin);
        // Antitragus
        mesh::sphere_tris(tris, ex * 0.9, 1.715, -0.005, 0.005, 0, skin);
        // Concha (deep cavity — dark)
        mesh::sphere_tris(tris, ex * 0.9, 1.73, 0.005, 0.012, 0, darken(skin, 0.65));
        // Ear lobe
        mesh::sphere_tris(tris, ex, 1.7, 0.0, 0.008, 0, darken(skin, 0.96));
    }

    // ── HAIR / HAT ──
    let hat = if let Some(jc) = is_job_hat {
        gen_job_hat(tris, jc);
        true
    } else {
        gen_hat(tris, app, hair)
    };

    // Hair visible around/below hat
    if hat && app.hat_type != 6 {
        // Nape hair
        mesh::ellipsoid_tris(tris, 0.0, 1.66, 0.1, 0.1, 0.06, 0.06, 0, hair);
        // Sideburns
        push_box(tris, -0.14, 1.71, -0.02, 0.015, 0.05, 0.03, hair);
        push_box(tris, 0.14, 1.71, -0.02, 0.015, 0.05, 0.03, hair);
        // Side hair below hat
        push_box(tris, -0.13, 1.73, 0.03, 0.02, 0.04, 0.06, hair);
        push_box(tris, 0.13, 1.73, 0.03, 0.02, 0.04, 0.06, hair);
    } else if !hat {
        // Full hair — volumetric
        mesh::sphere_tris(tris, 0.0, 1.85, 0.01, 0.15, 1, hair);
        // Side volume
        mesh::ellipsoid_tris(tris, -0.12, 1.76, 0.0, 0.04, 0.08, 0.08, 0, hair);
        mesh::ellipsoid_tris(tris, 0.12, 1.76, 0.0, 0.04, 0.08, 0.08, 0, hair);
        // Back volume
        mesh::ellipsoid_tris(tris, 0.0, 1.72, 0.1, 0.1, 0.06, 0.06, 0, darken(hair, 0.9));
        // Top detail strands
        push_box(tris, -0.04, 1.86, -0.04, 0.02, 0.01, 0.08, darken(hair, 0.9));
        push_box(tris, 0.04, 1.87, -0.02, 0.02, 0.01, 0.07, darken(hair, 1.1));
        // Sideburns
        push_box(tris, -0.14, 1.71, -0.025, 0.015, 0.06, 0.03, hair);
        push_box(tris, 0.14, 1.71, -0.025, 0.015, 0.06, 0.03, hair);
    }
}

/// Generate the anatomical neck with musculature — thicker, more visible muscles
fn gen_neck(tris: &mut Vec<WorldTri>, skin: u32) {
    // Base neck — thick, muscular (proportional to wide shoulders)
    mesh::tapered_cylinder_tris(tris, 0.0, 1.49, 0.0, 0.10, 0.085, 0.14, 10, skin);
    // Sternocleidomastoid muscles (prominent diagonal ridges, neck to collarbone)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.06, 1.47, -0.04, 0.028, 0.08, 0.022, 0, darken(skin, 0.93));
    }
    // Laryngeal prominence (Adam's apple)
    mesh::sphere_tris(tris, 0.0, 1.48, -0.09, 0.024, 0, darken(skin, 0.96));
    // Trapezius muscle base (wide, creates thick neck-to-shoulder slope)
    mesh::ellipsoid_tris(tris, 0.0, 1.44, 0.06, 0.14, 0.08, 0.07, 0, darken(skin, 0.95));
    // Suprasternal notch (hollow at base of throat)
    mesh::sphere_tris(tris, 0.0, 1.43, -0.08, 0.016, 0, darken(skin, 0.72));
    // Platysma muscle (thin sheet visible at sides)
    for &side in &[-1.0f32, 1.0] {
        push_box(tris, side * 0.07, 1.44, -0.05, 0.018, 0.05, 0.012, darken(skin, 0.92));
    }
    // Neck-to-trap slope (fills the visual gap between neck and shoulders)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.16, 1.44, 0.02, 0.10, 0.04, 0.06, 0, darken(skin, 0.95));
    }
}

/// Generate layered torso clothing: anatomical base → waistcoat → coat
/// V-taper silhouette: wide shoulders, narrow waist. Ellipsoid muscle groups visible under shirt.
fn gen_torso(tris: &mut Vec<WorldTri>, app: &NpcAppearance, vest_col: u32, swing: f32) {
    // ── ANATOMICAL BODY BASE — ellipsoid ribcage + abdomen for V-taper ──
    // Ribcage (wide barrel shape, shoulder width)
    mesh::ellipsoid_tris(tris, 0.0, 1.28, 0.0, 0.22, 0.20, 0.16, 1, SHIRT_LINEN);
    // Abdomen (narrower waist)
    mesh::ellipsoid_tris(tris, 0.0, 1.04, 0.0, 0.18, 0.14, 0.14, 1, SHIRT_LINEN);
    // Pectoral muscles (visible shape under shirt)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.09, 1.30, -0.13, 0.08, 0.06, 0.05, 0, darken(SHIRT_LINEN, 0.95));
    }
    // Trapezius / upper back (wide, connecting to shoulders)
    mesh::ellipsoid_tris(tris, 0.0, 1.38, 0.08, 0.20, 0.06, 0.10, 0, darken(SHIRT_LINEN, 0.92));
    // Latissimus dorsi (V-taper side muscles)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.17, 1.22, 0.05, 0.06, 0.15, 0.07, 0, darken(SHIRT_LINEN, 0.93));
    }
    // Serratus anterior (ribcage side definition)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.20, 1.18, -0.02, 0.04, 0.10, 0.05, 0, darken(SHIRT_LINEN, 0.94));
    }

    // Collar V-opening with folded fabric
    push_box(tris, -0.04, 1.42, -0.14, 0.05, 0.04, 0.04, SHIRT_LINEN);
    push_box(tris, 0.04, 1.42, -0.14, 0.05, 0.04, 0.04, SHIRT_LINEN);
    push_box(tris, -0.07, 1.425, -0.12, 0.015, 0.035, 0.02, darken(SHIRT_LINEN, 0.9));
    push_box(tris, 0.07, 1.425, -0.12, 0.015, 0.035, 0.02, darken(SHIRT_LINEN, 0.9));
    // Jabot ruffle at chest opening
    for ri in 0..4 {
        let ry = 1.35 - ri as f32 * 0.035;
        let rw = 0.04 - ri as f32 * 0.005;
        push_box(tris, 0.0, ry, -0.17, rw, 0.012, 0.01, darken(SHIRT_LINEN, 0.92 + ri as f32 * 0.02));
    }
    // Shirt fabric wrinkles
    push_box(tris, -0.03, 1.3, -0.15, 0.003, 0.04, 0.003, darken(SHIRT_LINEN, 0.85));
    push_box(tris, 0.025, 1.28, -0.15, 0.003, 0.035, 0.003, darken(SHIRT_LINEN, 0.85));

    // ── WAISTCOAT / VEST (wider, structured, over shirt) ──
    mesh::beveled_box_tris(tris, 0.0, 1.18, 0.0, 0.44, 0.54, 0.30, 0.03, vest_col);
    // V-neckline edges
    push_box(tris, -0.06, 1.34, -0.16, 0.04, 0.14, 0.005, darken(vest_col, 0.80));
    push_box(tris, 0.06, 1.34, -0.16, 0.04, 0.14, 0.005, darken(vest_col, 0.80));
    // Double-breasted buttons (2×4)
    for bi in 0..4 {
        let by = 1.34 - bi as f32 * 0.07;
        mesh::sphere_tris(tris, -0.04, by, -0.16, 0.009, 0, BUTTON_BRASS);
        mesh::sphere_tris(tris, 0.04, by, -0.16, 0.009, 0, BUTTON_BRASS);
    }
    // Pocket flaps + welts
    push_box(tris, -0.10, 1.10, -0.16, 0.05, 0.012, 0.01, darken(vest_col, 0.80));
    push_box(tris, 0.10, 1.10, -0.16, 0.05, 0.012, 0.01, darken(vest_col, 0.80));
    push_box(tris, -0.10, 1.092, -0.162, 0.05, 0.003, 0.005, darken(vest_col, 0.70));
    push_box(tris, 0.10, 1.092, -0.162, 0.05, 0.003, 0.005, darken(vest_col, 0.70));
    // Back adjustment strap
    push_box(tris, 0.0, 1.12, 0.155, 0.12, 0.02, 0.005, darken(vest_col, 0.78));
    mesh::sphere_tris(tris, -0.06, 1.12, 0.16, 0.006, 0, BUCKLE_BRASS);
    mesh::sphere_tris(tris, 0.06, 1.12, 0.16, 0.006, 0, BUCKLE_BRASS);
    // Seam lines
    push_seam(tris, -0.16, 0.92, 1.40, -0.13, STITCH_DARK);
    push_seam(tris, 0.16, 0.92, 1.40, -0.13, STITCH_DARK);
    push_seam(tris, 0.0, 0.92, 1.40, 0.155, STITCH_DARK);

    if app.has_coat {
        gen_coat(tris, app, swing);
    }
}

/// Generate the long coat — ACU-style with collar, lapels, tails, seams
fn gen_coat(tris: &mut Vec<WorldTri>, app: &NpcAppearance, swing: f32) {
    let coat = app.coat_col;
    let coat_dk = darken(coat, 0.82);
    let _coat_lt = darken(coat, 1.12);
    let lining = darken(coat, 1.2);

    // ── COAT BODY — organic shape using stacked ellipsoids for human silhouette ──
    // Upper chest / shoulder yoke (widest part)
    mesh::ellipsoid_tris(tris, 0.0, 1.34, 0.0, 0.30, 0.12, 0.20, 1, coat);
    // Mid chest
    mesh::ellipsoid_tris(tris, 0.0, 1.22, 0.0, 0.28, 0.14, 0.19, 1, coat);
    // Lower chest / ribs
    mesh::ellipsoid_tris(tris, 0.0, 1.08, 0.0, 0.26, 0.14, 0.18, 1, coat);
    // Waist area (narrower)
    mesh::ellipsoid_tris(tris, 0.0, 0.94, 0.0, 0.23, 0.10, 0.17, 0, coat);
    // Shoulder construction — wide structured pads
    mesh::beveled_box_tris(tris, 0.0, 1.40, 0.0, 0.64, 0.14, 0.38, 0.04, coat);
    // Shoulder joint covers (rounded transitions to arms)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.28, 1.40, 0.0, 0.08, 0.06, 0.10, 0, coat);
    }

    // ── STANDING COLLAR with fold ──
    // Collar stand (vertical part)
    mesh::cylinder_tris(tris, 0.0, 1.45, 0.0, 0.2, 0.05, 10, coat_dk);
    // Collar fold (folded-over part, darker)
    push_box(tris, 0.0, 1.47, 0.0, 0.38, 0.03, 0.28, darken(coat, 0.75));
    // Collar edge detail
    push_box(tris, 0.0, 1.475, 0.0, 0.39, 0.004, 0.29, darken(coat, 0.65));

    // ── WIDE LAPELS (triangular, following chest contour) ──
    // Left lapel
    push_box(tris, -0.1, 1.34, -0.165, 0.09, 0.18, 0.015, coat_dk);
    push_box(tris, -0.08, 1.34, -0.168, 0.07, 0.16, 0.005, darken(coat, 0.88));
    // Right lapel
    push_box(tris, 0.1, 1.34, -0.165, 0.09, 0.18, 0.015, coat_dk);
    push_box(tris, 0.08, 1.34, -0.168, 0.07, 0.16, 0.005, darken(coat, 0.88));
    // Lapel buttonhole (on left)
    push_box(tris, -0.06, 1.36, -0.17, 0.015, 0.006, 0.003, darken(coat, 0.5));

    // ── DOUBLE-BREASTED BUTTONS (2 columns × 4) ──
    for bi in 0..4 {
        let by = 1.3 - bi as f32 * 0.08;
        mesh::sphere_tris(tris, -0.06, by, -0.17, 0.01, 0, BUTTON_BRASS);
        mesh::sphere_tris(tris, 0.06, by, -0.17, 0.01, 0, BUTTON_BRASS);
        // Button thread detail
        push_box(tris, -0.06, by, -0.175, 0.003, 0.003, 0.002, darken(BUTTON_BRASS, 0.6));
        push_box(tris, 0.06, by, -0.175, 0.003, 0.003, 0.002, darken(BUTTON_BRASS, 0.6));
    }

    // ── COAT POCKET FLAPS ──
    for &side in &[-1.0f32, 1.0] {
        // Hip pocket flap
        push_box(tris, side * 0.14, 1.0, -0.165, 0.08, 0.018, 0.015, coat_dk);
        // Pocket flap edge stitch
        push_hseam(tris, side * 0.1, side * 0.18, 0.99, -0.168, STITCH_DARK);
        // Chest pocket (smaller)
        push_box(tris, side * 0.12, 1.25, -0.165, 0.04, 0.012, 0.01, coat_dk);
    }

    // ── COAT SEAM LINES (stitching detail) ──
    // Center back seam
    push_seam(tris, 0.0, 0.85, 1.42, 0.17, darken(coat, 0.7));
    // Side seams
    push_seam(tris, -0.24, 0.85, 1.38, 0.0, darken(coat, 0.75));
    push_seam(tris, 0.24, 0.85, 1.38, 0.0, darken(coat, 0.75));
    // Shoulder seams
    push_hseam(tris, -0.1, -0.27, 1.42, 0.0, darken(coat, 0.72));
    push_hseam(tris, 0.1, 0.27, 1.42, 0.0, darken(coat, 0.72));

    // ── EPAULETTES (shoulder tabs) ──
    for &side in &[-1.0f32, 1.0] {
        push_box(tris, side * 0.27, 1.43, 0.0, 0.06, 0.015, 0.09, coat_dk);
        push_box(tris, side * 0.27, 1.435, 0.0, 0.065, 0.005, 0.095, darken(coat, 0.65));
        mesh::sphere_tris(tris, side * 0.27, 1.44, -0.04, 0.006, 0, BUTTON_BRASS);
    }

    // ── COAT TAILS — 4 separate panels, sway with walk ──
    let ts = swing * 0.1; // tail sway
    // Back-left tail panel
    push_box(tris, -0.1, 0.55, 0.11 + ts, 0.13, 0.55, 0.08, coat);
    push_box(tris, -0.1, 0.55, 0.07 + ts, 0.12, 0.52, 0.005, lining); // lining
    push_seam(tris, -0.1, 0.3, 0.82, 0.15 + ts, darken(coat, 0.7)); // center seam
    // Back-right tail panel
    push_box(tris, 0.1, 0.55, 0.11 - ts, 0.13, 0.55, 0.08, coat);
    push_box(tris, 0.1, 0.55, 0.07 - ts, 0.12, 0.52, 0.005, lining);
    push_seam(tris, 0.1, 0.3, 0.82, 0.15 - ts, darken(coat, 0.7));
    // Back vent pleat (between the two back panels)
    push_box(tris, 0.0, 0.65, 0.1, 0.015, 0.35, 0.06, coat_dk);
    // Front-left skirt panel
    push_box(tris, -0.12, 0.62, -0.1, 0.12, 0.4, 0.07, coat);
    push_box(tris, -0.12, 0.62, -0.065, 0.11, 0.38, 0.005, lining);
    // Front-right skirt panel
    push_box(tris, 0.12, 0.62, -0.1, 0.12, 0.4, 0.07, coat);
    push_box(tris, 0.12, 0.62, -0.065, 0.11, 0.38, 0.005, lining);
    // Tail hem stitching
    push_hseam(tris, -0.17, -0.03, 0.28, 0.15 + ts, darken(coat, 0.65));
    push_hseam(tris, 0.03, 0.17, 0.28, 0.15 - ts, darken(coat, 0.65));
    push_hseam(tris, -0.18, -0.06, 0.42, -0.13, darken(coat, 0.65));
    push_hseam(tris, 0.06, 0.18, 0.42, -0.13, darken(coat, 0.65));

    // ── SHOULDER CAPE / MANTLE (if enabled) — ACU-style wide leather drape ──
    if app.has_cape {
        let cape_col = darken(coat, 0.88);
        let cape_edge = darken(coat, 0.7);
        let cape_dk = darken(coat, 0.75);
        // Front cape panels (wider, thicker leather) — drape over shoulders
        mesh::ellipsoid_tris(tris, -0.18, 1.28, -0.15, 0.15, 0.14, 0.06, 0, cape_col);
        mesh::ellipsoid_tris(tris, 0.18, 1.28, -0.15, 0.15, 0.14, 0.06, 0, cape_col);
        // Front cape edges (folded leather)
        push_box(tris, -0.18, 1.18, -0.17, 0.14, 0.006, 0.04, cape_dk);
        push_box(tris, 0.18, 1.18, -0.17, 0.14, 0.006, 0.04, cape_dk);
        // Back cape (larger mantle, extends lower)
        mesh::ellipsoid_tris(tris, 0.0, 1.22, 0.15, 0.30, 0.22, 0.05, 1, cape_col);
        // Back cape lower section
        push_box(tris, 0.0, 1.02, 0.16, 0.50, 0.16, 0.05, cape_col);
        // Cape edge hem stitching
        push_hseam(tris, -0.25, 0.25, 0.94, 0.17, cape_edge);
        push_hseam(tris, -0.28, -0.05, 1.17, -0.18, cape_edge);
        push_hseam(tris, 0.05, 0.28, 1.17, -0.18, cape_edge);
        // Cape shoulder seam
        push_hseam(tris, -0.15, -0.30, 1.39, -0.02, cape_edge);
        push_hseam(tris, 0.15, 0.30, 1.39, -0.02, cape_edge);
        // Cape collar overlap (wider)
        push_box(tris, 0.0, 1.42, 0.02, 0.40, 0.04, 0.28, cape_col);
        // Cape fold creases
        push_box(tris, -0.12, 1.25, 0.17, 0.003, 0.10, 0.003, cape_dk);
        push_box(tris, 0.12, 1.25, 0.17, 0.003, 0.10, 0.003, cape_dk);
    }
}

/// Generate hat/headwear with high detail
fn gen_hat(tris: &mut Vec<WorldTri>, app: &NpcAppearance, _hair: u32) -> bool {
    let hat_col = app.hat_col;
    match app.hat_type {
        1 => {
            // TRICORN — 3 upturned brim sections, crown, cockade
            mesh::cylinder_tris(tris, 0.0, 1.935, 0.0, 0.14, 0.1, 10, hat_col);
            // Brim disc
            mesh::cylinder_tris(tris, 0.0, 1.885, 0.0, 0.23, 0.015, 14, hat_col);
            // Three upturned brim flaps
            push_box(tris, 0.0, 1.91, -0.17, 0.15, 0.045, 0.05, darken(hat_col, 0.88));
            push_box(tris, -0.15, 1.91, 0.08, 0.06, 0.045, 0.13, darken(hat_col, 0.88));
            push_box(tris, 0.15, 1.91, 0.08, 0.06, 0.045, 0.13, darken(hat_col, 0.88));
            // Brim edge binding
            mesh::cylinder_tris(tris, 0.0, 1.885, 0.0, 0.235, 0.005, 14, darken(hat_col, 0.7));
            // Cockade (fabric rosette at front)
            mesh::sphere_tris(tris, 0.0, 1.935, -0.19, 0.025, 1, 0xFF888866);
            mesh::sphere_tris(tris, 0.0, 1.935, -0.195, 0.015, 0, 0xFFAAAA88);
            // Hat band
            mesh::cylinder_tris(tris, 0.0, 1.895, 0.0, 0.145, 0.015, 10, darken(hat_col, 0.75));
            true
        }
        2 => {
            // TOP HAT — tall crown, flat brim, band, silk texture
            mesh::cylinder_tris(tris, 0.0, 2.03, 0.0, 0.11, 0.24, 10, hat_col);
            // Top disc
            mesh::cylinder_tris(tris, 0.0, 2.155, 0.0, 0.11, 0.005, 10, darken(hat_col, 0.95));
            // Brim
            mesh::cylinder_tris(tris, 0.0, 1.905, 0.0, 0.18, 0.015, 12, hat_col);
            // Hat band (silk ribbon)
            mesh::cylinder_tris(tris, 0.0, 1.93, 0.0, 0.115, 0.02, 10, darken(hat_col, 0.6));
            // Brim edge
            mesh::cylinder_tris(tris, 0.0, 1.905, 0.0, 0.185, 0.004, 12, darken(hat_col, 0.7));
            true
        }
        3 => {
            // WORKER'S CAP — rounded with short visor
            mesh::sphere_tris(tris, 0.0, 1.9, -0.01, 0.155, 1, hat_col);
            // Crown button
            mesh::sphere_tris(tris, 0.0, 1.96, -0.01, 0.012, 0, darken(hat_col, 0.7));
            // Visor (short brim)
            push_box(tris, 0.0, 1.865, -0.16, 0.13, 0.008, 0.06, darken(hat_col, 0.82));
            // Visor edge
            push_box(tris, 0.0, 1.865, -0.19, 0.13, 0.005, 0.005, darken(hat_col, 0.6));
            true
        }
        4 => {
            // BONNET (female) — rounded, ribbon tie
            mesh::sphere_tris(tris, 0.0, 1.89, 0.03, 0.16, 1, hat_col);
            // Bonnet brim framing face
            push_box(tris, 0.0, 1.86, -0.14, 0.2, 0.05, 0.015, darken(hat_col, 0.88));
            // Ruffle edge
            for ri in 0..6 {
                let ra = (ri as f32 / 6.0) * std::f32::consts::PI - std::f32::consts::PI * 0.5;
                let rx = ra.sin() * 0.1;
                let ry = 1.86 + ra.cos() * 0.01;
                push_box(tris, rx, ry, -0.155, 0.02, 0.015, 0.01, darken(hat_col, 0.9));
            }
            // Chin ribbons
            push_box(tris, -0.07, 1.62, -0.06, 0.012, 0.2, 0.008, 0xFFDDCCBB);
            push_box(tris, 0.07, 1.62, -0.06, 0.012, 0.2, 0.008, 0xFFDDCCBB);
            true
        }
        5 => {
            // WIDE-BRIM — country/peasant hat
            mesh::cylinder_tris(tris, 0.0, 1.94, 0.0, 0.13, 0.1, 8, hat_col);
            mesh::cylinder_tris(tris, 0.0, 1.89, 0.0, 0.25, 0.015, 14, hat_col);
            mesh::cylinder_tris(tris, 0.0, 1.89, 0.0, 0.255, 0.005, 14, darken(hat_col, 0.7));
            mesh::cylinder_tris(tris, 0.0, 1.91, 0.0, 0.135, 0.018, 8, darken(hat_col, 0.72));
            true
        }
        6 => {
            // HOOD — assassin-style pointed hood with interior depth
            let hood_col = app.coat_col;
            // Hood main shape (exterior)
            let hood_profile: [[f32; 2]; 8] = [
                [0.16, -0.06], [0.18, 0.0], [0.19, 0.06], [0.18, 0.12],
                [0.15, 0.18], [0.10, 0.24], [0.04, 0.30], [0.0, 0.34],
            ];
            mesh::lathe_tris(tris, 0.0, 1.72, 0.02, &hood_profile, 10, hood_col);
            // Hood interior (darker)
            let inner_profile: [[f32; 2]; 5] = [
                [0.14, -0.04], [0.15, 0.02], [0.14, 0.08], [0.10, 0.14], [0.0, 0.18],
            ];
            mesh::lathe_tris(tris, 0.0, 1.72, 0.01, &inner_profile, 8, darken(hood_col, 0.6));
            // Hood peak (pointed tip)
            mesh::cone_tris(tris, 0.0, 2.1, 0.04, 0.04, 0.08, 6, hood_col);
            // Hood edge hem
            mesh::cylinder_tris(tris, 0.0, 1.68, 0.01, 0.17, 0.01, 10, darken(hood_col, 0.7));
            // Face shadow inside hood
            push_box(tris, 0.0, 1.76, -0.1, 0.14, 0.06, 0.01, darken(hood_col, 0.4));
            true
        }
        7 => {
            // POWDERED WIG — aristocratic style with side curls + ponytail
            let wig_col = 0xFFCCBBAA_u32; // off-white/grey
            // Main wig body
            mesh::sphere_tris(tris, 0.0, 1.86, 0.0, 0.16, 1, wig_col);
            // Top volume
            mesh::ellipsoid_tris(tris, 0.0, 1.9, -0.02, 0.12, 0.06, 0.1, 0, wig_col);
            // Side curls (left and right)
            for &side in &[-1.0f32, 1.0] {
                // 3 stacked curls per side
                for ci in 0..3 {
                    let cy = 1.72 - ci as f32 * 0.04;
                    let r = 0.025 - ci as f32 * 0.003;
                    mesh::sphere_tris(tris, side * 0.16, cy, 0.0, r, 0, darken(wig_col, 0.95));
                }
            }
            // Ponytail (ribbon-tied)
            mesh::cylinder_tris(tris, 0.0, 1.72, 0.12, 0.03, 0.15, 6, wig_col);
            // Ribbon tie
            push_box(tris, 0.0, 1.72, 0.12, 0.04, 0.015, 0.015, 0xFF222222);
            // Forehead hairline detail
            push_box(tris, 0.0, 1.82, -0.12, 0.1, 0.01, 0.02, darken(wig_col, 0.92));
            true
        }
        _ => false,
    }
}

fn gen_job_hat(tris: &mut Vec<WorldTri>, color: u32) {
    mesh::cylinder_tris(tris, 0.0, 1.935, 0.0, 0.17, 0.08, 8, color);
    push_box(tris, 0.0, 1.89, -0.13, 0.15, 0.01, 0.06, darken(color, 0.78));
    mesh::sphere_tris(tris, 0.0, 1.92, -0.17, 0.015, 0, BUCKLE_BRASS);
    // Hat band
    mesh::cylinder_tris(tris, 0.0, 1.9, 0.0, 0.175, 0.012, 8, darken(color, 0.6));
}

/// Belt system: leather belt, sash, cross-strap, pouches
fn gen_belt_system(tris: &mut Vec<WorldTri>, app: &NpcAppearance, pants_col: u32) {
    // ── PELVIS / HIP AREA — organic shape connecting torso to thighs ──
    mesh::ellipsoid_tris(tris, 0.0, 0.88, 0.0, 0.22, 0.08, 0.16, 0, pants_col);
    // Hip sockets (where thighs connect)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.14, 0.86, 0.0, 0.06, 0.06, 0.06, 0, pants_col);
    }
    mesh::cylinder_tris(tris, 0.0, 0.87, 0.0, 0.22, 0.06, 10, pants_col);

    // ── MAIN LEATHER BELT ──
    mesh::cylinder_tris(tris, 0.0, 0.87, 0.0, 0.215, 0.025, 12, LEATHER_DARK);
    // Belt edge stitching
    mesh::cylinder_tris(tris, 0.0, 0.885, 0.0, 0.218, 0.003, 12, STITCH_DARK);
    mesh::cylinder_tris(tris, 0.0, 0.855, 0.0, 0.218, 0.003, 12, STITCH_DARK);
    // Belt buckle — rectangular with prong detail
    push_box(tris, 0.0, 0.87, -0.22, 0.035, 0.022, 0.008, BUCKLE_BRASS);
    push_box(tris, 0.0, 0.87, -0.225, 0.025, 0.015, 0.003, darken(BUCKLE_BRASS, 0.7)); // inner
    push_box(tris, 0.0, 0.87, -0.228, 0.003, 0.015, 0.003, darken(BUCKLE_BRASS, 0.5)); // prong

    // ── BELT POUCHES ──
    // Right hip pouch
    push_box(tris, 0.19, 0.84, -0.02, 0.06, 0.07, 0.05, LEATHER_MED);
    push_box(tris, 0.19, 0.88, -0.02, 0.062, 0.01, 0.052, darken(LEATHER_MED, 0.82)); // flap
    mesh::sphere_tris(tris, 0.19, 0.875, -0.045, 0.005, 0, BUCKLE_BRASS); // flap button
    // Left hip pouch (smaller)
    push_box(tris, -0.2, 0.84, 0.02, 0.04, 0.05, 0.04, LEATHER_MED);
    push_box(tris, -0.2, 0.87, 0.02, 0.042, 0.008, 0.042, darken(LEATHER_MED, 0.82));
    // Back pouch
    push_box(tris, -0.08, 0.84, 0.16, 0.05, 0.06, 0.04, LEATHER_MED);
    push_box(tris, -0.08, 0.875, 0.16, 0.052, 0.008, 0.042, darken(LEATHER_MED, 0.82));

    // ── RED SASH / CUMMERBUND ──
    if app.has_sash {
        let sc = app.sash_col;
        // Wrapped around waist (slightly angled)
        mesh::cylinder_tris(tris, 0.0, 0.88, 0.0, 0.225, 0.04, 12, sc);
        // Sash tail hanging on left side
        push_box(tris, -0.22, 0.75, 0.0, 0.04, 0.2, 0.03, sc);
        push_box(tris, -0.23, 0.68, 0.0, 0.03, 0.06, 0.025, darken(sc, 0.9));
        // Fabric folds in sash
        push_box(tris, 0.05, 0.88, -0.18, 0.03, 0.025, 0.004, darken(sc, 0.8));
        push_box(tris, -0.08, 0.89, -0.15, 0.02, 0.02, 0.004, darken(sc, 0.75));
    }

    // ── CROSS-BODY LEATHER STRAP ──
    if app.has_cross_strap {
        let strap_col = LEATHER_MED;
        // Diagonal from right shoulder to left hip
        for si in 0..12 {
            let t = si as f32 / 11.0;
            let sx = 0.18 - t * 0.36;
            let sy = 0.87 + (1.0 - t) * 0.55;
            let sz = -0.13 - (t * (1.0 - t)) * 0.08;
            push_box(tris, sx, sy, sz, 0.04, 0.05, 0.012, strap_col);
        }
        // Strap buckle at chest
        push_box(tris, 0.0, 1.14, -0.165, 0.025, 0.02, 0.008, BUCKLE_BRASS);
        // Strap edge stitching
        for si in [2, 5, 8] {
            let t = si as f32 / 11.0;
            let sx = 0.18 - t * 0.36;
            let sy = 0.87 + (1.0 - t) * 0.55;
            let sz = -0.13 - (t * (1.0 - t)) * 0.08;
            push_box(tris, sx, sy, sz - 0.008, 0.042, 0.003, 0.002, STITCH_DARK);
        }
    }
}

/// Generate a single arm with joint-based positioning.
/// Shoulder → elbow → wrist chain creates visible angular bends at joints.
/// Each segment has distinct muscle definition (deltoid, bicep, tricep, forearm).
fn gen_arm(
    tris: &mut Vec<WorldTri>, side: f32, fwd: f32, bend: f32,
    sleeve_col: u32, skin: u32, app: &NpcAppearance,
) {
    let coat = app.coat_col;

    // ── JOINT POSITIONS — shoulder → elbow → wrist chain ──
    // Elbow offset creates visible angular bend between upper/lower arm
    let shoulder = [side * 0.30, 1.42, 0.0];
    let elbow = [side * 0.33, 1.08, fwd * 0.35];
    let wrist = [side * 0.31, 0.82, fwd * 0.15 - bend];

    // ── DELTOID — large 3-head muscle mass bridging shoulder to upper arm ──
    mesh::ellipsoid_tris(tris, shoulder[0], shoulder[1], shoulder[2],
        0.10, 0.08, 0.09, 1, sleeve_col);
    // Anterior deltoid head
    mesh::ellipsoid_tris(tris, shoulder[0], shoulder[1] - 0.04, shoulder[2] - 0.04,
        0.06, 0.05, 0.04, 0, darken(sleeve_col, 0.93));
    // Posterior deltoid head
    mesh::ellipsoid_tris(tris, shoulder[0], shoulder[1] - 0.04, shoulder[2] + 0.04,
        0.05, 0.05, 0.04, 0, darken(sleeve_col, 0.91));

    // ── UPPER ARM (shoulder → elbow) — angled segment with muscle bulges ──
    mesh::tapered_cylinder_between(tris, shoulder, elbow, 0.09, 0.065, 10, sleeve_col);
    // Bicep (anterior mass, peaks at 45% length)
    let bicep = lerp3(shoulder, elbow, 0.45);
    mesh::ellipsoid_tris(tris, bicep[0] - side * 0.01, bicep[1], bicep[2] - 0.03,
        0.045, 0.06, 0.04, 0, darken(sleeve_col, 0.94));
    // Tricep (posterior, longer muscle belly)
    let tricep = lerp3(shoulder, elbow, 0.50);
    mesh::ellipsoid_tris(tris, tricep[0], tricep[1], tricep[2] + 0.03,
        0.04, 0.07, 0.035, 0, darken(sleeve_col, 0.91));
    // Sleeve fabric compression wrinkles
    let sw1 = lerp3(shoulder, elbow, 0.30);
    push_box(tris, sw1[0] + side * 0.02, sw1[1], sw1[2], 0.004, 0.04, 0.02, darken(sleeve_col, 0.83));
    let sw2 = lerp3(shoulder, elbow, 0.65);
    push_box(tris, sw2[0] - side * 0.015, sw2[1], sw2[2], 0.004, 0.035, 0.018, darken(sleeve_col, 0.84));

    // ── ELBOW JOINT — distinct ball joint, visually separates upper/lower arm ──
    mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2], 0.064, 1, darken(sleeve_col, 0.92));
    // Olecranon process (bony posterior point)
    mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2] + 0.038, 0.025, 0, darken(sleeve_col, 0.84));
    // Medial epicondyle (inner bump)
    mesh::sphere_tris(tris, elbow[0] - side * 0.04, elbow[1], elbow[2], 0.017, 0, darken(sleeve_col, 0.86));

    // ── FOREARM (elbow → wrist) — angled segment, muscular ──
    mesh::tapered_cylinder_between(tris, elbow, wrist, 0.065, 0.048, 8, skin);
    // Brachioradialis (lateral forearm mass near elbow)
    let brach = lerp3(elbow, wrist, 0.25);
    mesh::ellipsoid_tris(tris, brach[0] + side * 0.015, brach[1], brach[2] - 0.015,
        0.035, 0.05, 0.03, 0, darken(skin, 0.96));
    // Flexor group (medial, palm side)
    let flex = lerp3(elbow, wrist, 0.30);
    mesh::ellipsoid_tris(tris, flex[0] - side * 0.01, flex[1] - 0.01, flex[2],
        0.03, 0.045, 0.025, 0, darken(skin, 0.94));
    // Extensor group (dorsal, back-of-hand side)
    let ext = lerp3(elbow, wrist, 0.35);
    mesh::ellipsoid_tris(tris, ext[0], ext[1] + 0.01, ext[2] + 0.015,
        0.025, 0.04, 0.025, 0, darken(skin, 0.95));
    // Forearm tendon ridges (visible near wrist)
    let tendon_area = lerp3(elbow, wrist, 0.70);
    push_box(tris, tendon_area[0] - side * 0.01, tendon_area[1], tendon_area[2] - 0.02,
        0.003, 0.04, 0.003, darken(skin, 0.88));
    push_box(tris, tendon_area[0] + side * 0.005, tendon_area[1], tendon_area[2] - 0.018,
        0.003, 0.035, 0.003, darken(skin, 0.89));

    // ── COAT CUFF — follows arm direction via tapered_cylinder_between ──
    if app.has_coat {
        let cuff_s = lerp3(elbow, wrist, 0.22);
        let cuff_e = lerp3(elbow, wrist, 0.42);
        let cuff_col = darken(coat, 0.78);
        let cuff_lining = darken(coat, 1.25);
        mesh::tapered_cylinder_between(tris, cuff_s, cuff_e, 0.074, 0.072, 8, cuff_col);
        mesh::tapered_cylinder_between(tris, cuff_s, cuff_e, 0.068, 0.066, 8, cuff_lining);
        // Cuff edge
        let cuff_edge_pos = lerp3(elbow, wrist, 0.41);
        mesh::cylinder_tris(tris, cuff_edge_pos[0], cuff_edge_pos[1], cuff_edge_pos[2],
            0.076, 0.005, 8, darken(cuff_col, 0.68));
        // Cuff buttons (3)
        for ci in 0..3 {
            let t = 0.25 + ci as f32 * 0.05;
            let bp = lerp3(elbow, wrist, t);
            mesh::sphere_tris(tris, bp[0] + side * 0.073, bp[1], bp[2] - 0.02, 0.007, 0, BUTTON_BRASS);
        }
    }

    // ── SHIRT RUFFLE CUFF (visible below coat cuff) ──
    let sc = lerp3(elbow, wrist, 0.62);
    mesh::cylinder_tris(tris, sc[0], sc[1], sc[2], 0.050, 0.028, 6, SHIRT_LINEN);
    mesh::cylinder_tris(tris, sc[0], sc[1] - 0.013, sc[2], 0.054, 0.010, 6, darken(SHIRT_LINEN, 0.92));

    // ── LEATHER BRACER / GAUNTLET ──
    if app.has_bracers {
        let br_s = lerp3(elbow, wrist, 0.38);
        let br_e = lerp3(elbow, wrist, 0.68);
        mesh::tapered_cylinder_between(tris, br_s, br_e, 0.062, 0.056, 8, LEATHER_DARK);
        // Edge ribs
        let br_top = lerp3(elbow, wrist, 0.39);
        let br_bot = lerp3(elbow, wrist, 0.67);
        mesh::cylinder_tris(tris, br_top[0], br_top[1], br_top[2], 0.065, 0.008, 8, darken(LEATHER_DARK, 0.78));
        mesh::cylinder_tris(tris, br_bot[0], br_bot[1], br_bot[2], 0.060, 0.008, 8, darken(LEATHER_DARK, 0.78));
        // Straps with buckles
        for si in 0..2 {
            let t = 0.46 + si as f32 * 0.12;
            let sp = lerp3(elbow, wrist, t);
            mesh::cylinder_tris(tris, sp[0], sp[1], sp[2], 0.066, 0.01, 8, LEATHER_MED);
            mesh::sphere_tris(tris, sp[0] - side * 0.063, sp[1], sp[2] - 0.02, 0.006, 0, BUCKLE_BRASS);
        }
        // Bracer stitching
        let stitch_mid = lerp3(elbow, wrist, 0.53);
        push_seam(tris, stitch_mid[0] + side * 0.058, br_s[1], br_e[1], stitch_mid[2], STITCH_DARK);
    }

    // ── WRIST JOINT — defined transition ──
    mesh::sphere_tris(tris, wrist[0], wrist[1], wrist[2], 0.042, 0, darken(skin, 0.95));
    // Ulnar styloid (bony bump)
    mesh::sphere_tris(tris, wrist[0] + side * 0.033, wrist[1], wrist[2] + 0.01, 0.014, 0, darken(skin, 0.90));

    // ── HAND ──
    gen_hand(tris, wrist[0], wrist[1] - 0.05, wrist[2] - 0.02, side, skin);
}

/// Generate a detailed hand — larger proportions, thicker fingers, muscle pads
fn gen_hand(tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32, side: f32, skin: u32) {
    let palm_dk = darken(skin, 0.93);
    let knuckle_dk = darken(skin, 0.85);
    let nail_col = darken(skin, 1.08);

    // ── PALM — wider, thicker ──
    push_box(tris, cx, cy, cz, 0.055, 0.05, 0.065, palm_dk);
    // Thenar eminence (thumb muscle pad)
    mesh::ellipsoid_tris(tris, cx + side * 0.02, cy - 0.01, cz + 0.01, 0.02, 0.02, 0.025, 0, darken(skin, 0.91));
    // Hypothenar eminence (pinky side pad)
    mesh::ellipsoid_tris(tris, cx - side * 0.02, cy - 0.01, cz + 0.01, 0.015, 0.018, 0.022, 0, darken(skin, 0.91));
    // Palm crease lines
    push_box(tris, cx, cy - 0.005, cz - 0.015, 0.04, 0.002, 0.003, darken(skin, 0.72));
    push_box(tris, cx, cy - 0.012, cz + 0.005, 0.035, 0.002, 0.003, darken(skin, 0.72));

    // ── KNUCKLE RIDGE (more prominent) ──
    push_box(tris, cx, cy + 0.025, cz - 0.035, 0.052, 0.012, 0.008, knuckle_dk);

    // ── 4 FINGERS — thicker, more visible joints ──
    for fi in 0..4 {
        let fx = cx + (fi as f32 - 1.5) * 0.013;
        let fz_base = cz - 0.04;
        let finger_len = 0.030 + (1.0 - (fi as f32 - 1.5).abs() * 0.35) * 0.012;

        // Proximal phalanx (thicker)
        push_box(tris, fx, cy + 0.005, fz_base - finger_len * 0.5, 0.009, 0.010, finger_len, skin);
        // Knuckle joint (ball)
        mesh::sphere_tris(tris, fx, cy + 0.005, fz_base - finger_len, 0.007, 0, knuckle_dk);
        // Distal phalanx
        let d_len = finger_len * 0.7;
        push_box(tris, fx, cy + 0.003, fz_base - finger_len - d_len * 0.5, 0.008, 0.008, d_len, skin);
        // Fingernail
        push_box(tris, fx, cy + 0.010, fz_base - finger_len - d_len + 0.003, 0.006, 0.003, 0.007, nail_col);
    }

    // ── THUMB — thicker, 2 segments ──
    let tx = cx + side * 0.032;
    let tz = cz + 0.005;
    // Metacarpal
    push_box(tris, tx, cy - 0.005, tz - 0.015, 0.013, 0.014, 0.025, skin);
    // CMC joint
    mesh::sphere_tris(tris, tx, cy - 0.005, tz - 0.030, 0.008, 0, knuckle_dk);
    // Distal thumb
    push_box(tris, tx, cy - 0.003, tz - 0.045, 0.011, 0.012, 0.022, skin);
    // Thumbnail
    push_box(tris, tx, cy + 0.006, tz - 0.055, 0.008, 0.003, 0.008, nail_col);
}

/// Generate a single leg with joint-based positioning.
/// Hip → knee → ankle chain creates visible angular bends.
/// Thick breeches with muscle definition, distinct knee joint.
fn gen_leg(
    tris: &mut Vec<WorldTri>, side: f32, fwd: f32, knee_bend: f32,
    pants_col: u32, app: &NpcAppearance,
) {
    // ── JOINT POSITIONS — hip → knee → ankle chain ──
    let lx = side * 0.14;
    let hip = [lx, 0.86, 0.0];
    let knee = [lx, 0.44, fwd * 0.5];
    let ankle = [lx, 0.08, fwd * 0.25 - knee_bend * 0.4];

    // ── THIGH (hip → knee) — puffy ACU breeches with muscle definition ──
    mesh::tapered_cylinder_between(tris, hip, knee, 0.14, 0.085, 12, pants_col);
    // Quadriceps (anterior thigh mass)
    let quad = lerp3(hip, knee, 0.40);
    mesh::ellipsoid_tris(tris, quad[0], quad[1], quad[2] - 0.04,
        0.06, 0.10, 0.05, 0, darken(pants_col, 0.95));
    // Vastus lateralis (outer thigh)
    let vlat = lerp3(hip, knee, 0.35);
    mesh::ellipsoid_tris(tris, vlat[0] + side * 0.04, vlat[1], vlat[2],
        0.04, 0.08, 0.04, 0, darken(pants_col, 0.93));
    // Hamstring (posterior thigh)
    let ham = lerp3(hip, knee, 0.50);
    mesh::ellipsoid_tris(tris, ham[0], ham[1], ham[2] + 0.035,
        0.05, 0.08, 0.04, 0, darken(pants_col, 0.91));
    // Adductor (inner thigh)
    let add = lerp3(hip, knee, 0.35);
    mesh::ellipsoid_tris(tris, add[0] - side * 0.03, add[1], add[2],
        0.04, 0.07, 0.04, 0, darken(pants_col, 0.93));
    // Breeches outer seam
    push_seam(tris, hip[0] + side * 0.11, knee[1], hip[1], (hip[2] + knee[2]) * 0.5, darken(pants_col, 0.74));
    // Breeches inner seam
    push_seam(tris, hip[0] - side * 0.04, knee[1], hip[1], (hip[2] + knee[2]) * 0.5, darken(pants_col, 0.74));
    // Fly buttons
    if side < 0.0 {
        for bi in 0..2 {
            let t = 0.15 + bi as f32 * 0.10;
            let bp = lerp3(hip, knee, t);
            mesh::sphere_tris(tris, bp[0] + 0.06, bp[1], bp[2] - 0.05, 0.006, 0, BUTTON_BRASS);
        }
    }

    // ── KNEE JOINT — distinct ball joint, visually separates thigh from calf ──
    mesh::sphere_tris(tris, knee[0], knee[1], knee[2], 0.090, 1, pants_col);
    // Patella (kneecap — prominent anterior bump)
    mesh::sphere_tris(tris, knee[0], knee[1], knee[2] - 0.055, 0.035, 0, darken(pants_col, 0.95));
    // Breeches knee band (gathered, with buttons)
    mesh::cylinder_tris(tris, knee[0], knee[1] - 0.01, knee[2], 0.088, 0.025, 10, darken(pants_col, 0.82));
    // 3 knee buttons (period detail)
    for bi in 0..3 {
        let ba = (bi as f32 / 3.0) * 0.6 - 0.3;
        mesh::sphere_tris(tris, knee[0] + side * 0.085, knee[1] - 0.01, knee[2] + ba * 0.06, 0.007, 0, BUCKLE_BRASS);
    }

    // ── BOOT / CALF (knee → ankle) ──
    gen_boot(tris, knee, ankle, side, app);
}

/// Generate boot + calf from knee to ankle joint positions.
/// Uses tapered_cylinder_between for proper leg angle. Includes calf muscles.
fn gen_boot(tris: &mut Vec<WorldTri>, knee: [f32; 3], ankle: [f32; 3], side: f32, app: &NpcAppearance) {
    let lx = knee[0];
    let bc = app.boot_col;
    let bc_dk = darken(bc, 0.8);
    let bc_lt = darken(bc, 1.15);

    // ── CALF SEGMENT (knee → ankle) — angled, with muscle definition ──
    mesh::tapered_cylinder_between(tris, knee, ankle, 0.085, 0.060, 10, bc);
    // Gastrocnemius (calf muscle, posterior — the "calf bulge")
    let calf_muscle = lerp3(knee, ankle, 0.25);
    mesh::ellipsoid_tris(tris, calf_muscle[0], calf_muscle[1], calf_muscle[2] + 0.03,
        0.04, 0.07, 0.04, 0, darken(bc, 0.94));
    // Soleus (deeper calf muscle)
    let soleus = lerp3(knee, ankle, 0.40);
    mesh::ellipsoid_tris(tris, soleus[0], soleus[1], soleus[2] + 0.02,
        0.035, 0.05, 0.03, 0, darken(bc, 0.92));
    // Tibialis anterior (shin, anterior)
    let shin = lerp3(knee, ankle, 0.30);
    mesh::ellipsoid_tris(tris, shin[0], shin[1], shin[2] - 0.025,
        0.03, 0.06, 0.025, 0, darken(bc, 0.96));
    // Shin ridge (visible through leather)
    let shin_ridge = lerp3(knee, ankle, 0.35);
    push_box(tris, shin_ridge[0], shin_ridge[1], shin_ridge[2] - 0.04,
        0.01, 0.12, 0.006, darken(bc, 0.87));

    // ── ANKLE JOINT ──
    mesh::sphere_tris(tris, ankle[0], ankle[1], ankle[2], 0.050, 0, darken(bc, 0.93));
    // Malleolus bumps (ankle bones)
    mesh::sphere_tris(tris, ankle[0] + side * 0.04, ankle[1] + 0.01, ankle[2], 0.016, 0, darken(bc, 0.87));
    mesh::sphere_tris(tris, ankle[0] - side * 0.035, ankle[1] + 0.015, ankle[2], 0.014, 0, darken(bc, 0.87));

    match app.boot_type {
        0 => {
            // BUCKLED SHOES — low shoes with stockings visible
            let stocking_top = lerp3(knee, ankle, 0.10);
            mesh::tapered_cylinder_between(tris, stocking_top, ankle, 0.068, 0.058, 7, 0xFFCCBBAA);
            // Garter
            mesh::cylinder_tris(tris, stocking_top[0], stocking_top[1], stocking_top[2],
                0.071, 0.014, 6, darken(0xFFCCBBAA_u32, 0.78));
            // Shoe body (wider)
            mesh::beveled_box_tris(tris, lx, 0.035, ankle[2] - 0.02, 0.10, 0.07, 0.16, 0.015, bc);
            // Shoe tongue
            push_box(tris, lx, 0.07, ankle[2] - 0.05, 0.05, 0.025, 0.04, bc_lt);
            // Shoe heel
            push_box(tris, lx, 0.018, ankle[2] + 0.04, 0.07, 0.035, 0.035, bc_dk);
            // Shoe sole
            push_box(tris, lx, 0.005, ankle[2] - 0.02, 0.09, 0.01, 0.16, darken(bc, 0.5));
            // Shoe buckle
            push_box(tris, lx, 0.055, ankle[2] - 0.085, 0.03, 0.022, 0.007, BUCKLE_BRASS);
        }
        1 => {
            // MID-CALF BOOTS
            let cuff_pos = lerp3(knee, ankle, 0.25);
            mesh::cylinder_tris(tris, cuff_pos[0], cuff_pos[1], cuff_pos[2],
                0.080, 0.030, 8, bc_lt);
            mesh::cylinder_tris(tris, cuff_pos[0], cuff_pos[1] + 0.015, cuff_pos[2],
                0.083, 0.005, 8, bc_dk);
            // Boot vamp
            mesh::beveled_box_tris(tris, lx, 0.035, ankle[2] - 0.02, 0.10, 0.07, 0.15, 0.015, bc);
            // Heel
            push_box(tris, lx, 0.018, ankle[2] + 0.04, 0.07, 0.035, 0.035, bc_dk);
            // Sole
            push_box(tris, lx, 0.005, ankle[2] - 0.02, 0.09, 0.01, 0.15, darken(bc, 0.5));
            // Strap with buckle
            let strap_pos = lerp3(knee, ankle, 0.50);
            mesh::cylinder_tris(tris, strap_pos[0], strap_pos[1], strap_pos[2],
                0.080, 0.012, 8, LEATHER_MED);
            mesh::sphere_tris(tris, strap_pos[0] - 0.077, strap_pos[1], strap_pos[2],
                0.007, 0, BUCKLE_BRASS);
            // Pull tab
            let pull = lerp3(knee, ankle, 0.22);
            push_box(tris, pull[0], pull[1], pull[2] + 0.06, 0.018, 0.03, 0.01, bc_dk);
            // Front seam
            push_seam(tris, lx, ankle[1], cuff_pos[1], (ankle[2] + cuff_pos[2]) * 0.5 - 0.05, STITCH_DARK);
        }
        _ => {
            // TALL KNEE-HIGH BOOTS — ACU assassin style
            let cuff_pos = lerp3(knee, ankle, 0.08);
            // Wide turned-down cuff
            mesh::cylinder_tris(tris, cuff_pos[0], cuff_pos[1], cuff_pos[2],
                0.090, 0.040, 8, bc_lt);
            mesh::cylinder_tris(tris, cuff_pos[0], cuff_pos[1] + 0.020, cuff_pos[2],
                0.093, 0.005, 8, bc_dk);
            // Cuff lining visible
            mesh::cylinder_tris(tris, cuff_pos[0], cuff_pos[1], cuff_pos[2],
                0.084, 0.035, 8, darken(bc, 1.3));
            // Boot vamp
            mesh::beveled_box_tris(tris, lx, 0.035, ankle[2] - 0.02, 0.10, 0.07, 0.15, 0.015, bc);
            // Toe cap
            push_box(tris, lx, 0.04, ankle[2] - 0.09, 0.09, 0.006, 0.012, bc_dk);
            // Heel
            push_box(tris, lx, 0.02, ankle[2] + 0.04, 0.07, 0.04, 0.035, bc_dk);
            // Sole
            push_box(tris, lx, 0.005, ankle[2] - 0.02, 0.09, 0.01, 0.15, darken(bc, 0.45));
            // Boot straps (3 horizontal with buckles)
            for si in 0..3 {
                let t = 0.30 + si as f32 * 0.18;
                let sp = lerp3(knee, ankle, t);
                mesh::cylinder_tris(tris, sp[0], sp[1], sp[2], 0.083, 0.012, 8, LEATHER_MED);
                mesh::sphere_tris(tris, sp[0] - 0.080, sp[1], sp[2] - 0.02, 0.006, 0, BUCKLE_BRASS);
            }
            // Boot seams (front and back)
            push_seam(tris, lx, ankle[1], cuff_pos[1], (ankle[2] + cuff_pos[2]) * 0.5 - 0.06, STITCH_DARK);
            push_seam(tris, lx, ankle[1], cuff_pos[1], (ankle[2] + cuff_pos[2]) * 0.5 + 0.06, STITCH_DARK);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN CHARACTER BODY — assembles all parts
// ═══════════════════════════════════════════════════════════════════════════

fn gen_character_body(
    tris: &mut Vec<WorldTri>,
    swing: f32,
    app: &NpcAppearance,
    shirt_col: u32,
    pants_col: u32,
    attack_phase: f32,
    carrying_item: bool,
    carrying_bin: bool,
    sitting: bool,
    is_job_hat: Option<u32>,
) {
    let skin = app.skin;
    let vest_col = app.vest_col;
    let arm_outer = if app.has_coat { app.coat_col } else { shirt_col };

    if sitting {
        gen_seated_body(tris, app, shirt_col, pants_col, is_job_hat);
        return;
    }

    // ── HEAD (skull + face + hair/hat) ──
    gen_head(tris, app, is_job_hat);

    // ── NECK ──
    gen_neck(tris, skin);

    // ── TORSO (undershirt + vest + coat) ──
    gen_torso(tris, app, vest_col, swing);

    // ── BELT / SASH / STRAPS / POUCHES ──
    gen_belt_system(tris, app, pants_col);

    // ── LEGS — wider stance, larger stride, more visible knee bend ──
    let l_fwd = -swing * 0.40;
    let r_fwd = swing * 0.40;
    let l_knee = if swing > 0.0 { swing * 0.22 } else { 0.0 };
    let r_knee = if swing < 0.0 { (-swing) * 0.22 } else { 0.0 };
    gen_leg(tris, -1.0, l_fwd, l_knee, pants_col, app);
    gen_leg(tris, 1.0, r_fwd, r_knee, pants_col, app);

    // ── ARMS — larger swing, more visible elbow articulation ──
    if attack_phase > 0.0 {
        gen_attack_arms(tris, attack_phase, arm_outer, skin, app);
    } else if carrying_item {
        gen_carry_arms(tris, arm_outer, skin, app);
        mesh::beveled_box_tris(tris, 0.0, 0.88, -0.50, 0.30, 0.30, 0.20, 0.02, BAG_COLOR);
        push_box(tris, 0.0, 1.1, -0.37, 0.02, 0.4, 0.02, LEATHER_MED);
    } else if carrying_bin {
        gen_carry_arms(tris, arm_outer, skin, app);
        mesh::cylinder_tris(tris, 0.0, 0.78, -0.55, 0.2, 0.55, 8, BIN_COLOR);
    } else {
        let l_arm_fwd = swing * 0.25;
        let r_arm_fwd = -swing * 0.25;
        let l_bend = 0.10 + swing.abs() * 0.14;
        let r_bend = 0.10 + swing.abs() * 0.14;
        gen_arm(tris, -1.0, l_arm_fwd, l_bend, arm_outer, skin, app);
        gen_arm(tris, 1.0, r_arm_fwd, r_bend, arm_outer, skin, app);
    }
}

fn gen_attack_arms(
    tris: &mut Vec<WorldTri>, attack_phase: f32,
    arm_outer: u32, skin: u32, app: &NpcAppearance,
) {
    let t = (attack_phase / ATTACK_ANIM_DURATION).clamp(0.0, 1.0);
    let extend = 1.0 - (1.0 - t) * (1.0 - t);

    // Right arm — punching forward, joint-based
    let r_shoulder = [0.32, 1.42, 0.0];
    let r_elbow = [0.34, 1.10, -0.15 - extend * 0.20];
    let r_wrist = [0.33, 0.92, -0.35 - extend * 0.35];
    mesh::ellipsoid_tris(tris, r_shoulder[0], r_shoulder[1], r_shoulder[2], 0.10, 0.07, 0.09, 1, arm_outer);
    mesh::tapered_cylinder_between(tris, r_shoulder, r_elbow, 0.09, 0.065, 10, arm_outer);
    mesh::sphere_tris(tris, r_elbow[0], r_elbow[1], r_elbow[2], 0.066, 1, arm_outer);
    mesh::tapered_cylinder_between(tris, r_elbow, r_wrist, 0.065, 0.048, 8, skin);
    // Fist
    mesh::sphere_tris(tris, r_wrist[0], r_wrist[1] - 0.03, r_wrist[2] - 0.03, 0.055, 1, skin);
    push_box(tris, r_wrist[0], r_wrist[1] - 0.03, r_wrist[2] - 0.06, 0.045, 0.04, 0.03, darken(skin, 0.85));

    // Left arm — guard position, joint-based
    let l_shoulder = [-0.30, 1.42, 0.0];
    let l_elbow = [-0.33, 1.10, -0.05];
    let l_wrist = [-0.32, 0.90, -0.14];
    mesh::ellipsoid_tris(tris, l_shoulder[0], l_shoulder[1], l_shoulder[2], 0.10, 0.07, 0.09, 1, arm_outer);
    mesh::tapered_cylinder_between(tris, l_shoulder, l_elbow, 0.09, 0.065, 10, arm_outer);
    mesh::sphere_tris(tris, l_elbow[0], l_elbow[1], l_elbow[2], 0.066, 1, arm_outer);
    mesh::tapered_cylinder_between(tris, l_elbow, l_wrist, 0.065, 0.048, 8, skin);
    mesh::sphere_tris(tris, l_wrist[0], l_wrist[1] - 0.03, l_wrist[2], 0.052, 0, skin);

    // Bracers visible during combat
    if app.has_bracers {
        let r_bracer = lerp3(r_elbow, r_wrist, 0.40);
        mesh::cylinder_tris(tris, r_bracer[0], r_bracer[1], r_bracer[2], 0.060, 0.12, 7, LEATHER_DARK);
        let l_bracer = lerp3(l_elbow, l_wrist, 0.40);
        mesh::cylinder_tris(tris, l_bracer[0], l_bracer[1], l_bracer[2], 0.060, 0.12, 7, LEATHER_DARK);
    }
}

fn gen_carry_arms(
    tris: &mut Vec<WorldTri>,
    arm_outer: u32, skin: u32, _app: &NpcAppearance,
) {
    // Both arms forward, holding object — joint-based
    for &side in &[-1.0f32, 1.0] {
        let shoulder = [side * 0.30, 1.42, 0.0];
        let elbow = [side * 0.28, 1.10, -0.22];
        let wrist = [side * 0.26, 0.88, -0.40];
        mesh::ellipsoid_tris(tris, shoulder[0], shoulder[1], shoulder[2], 0.10, 0.07, 0.09, 1, arm_outer);
        mesh::tapered_cylinder_between(tris, shoulder, elbow, 0.09, 0.065, 10, arm_outer);
        mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2], 0.066, 1, arm_outer);
        mesh::tapered_cylinder_between(tris, elbow, wrist, 0.065, 0.048, 8, skin);
        gen_hand(tris, wrist[0], wrist[1] - 0.05, wrist[2] - 0.02, side, skin);
    }
}

fn gen_seated_body(
    tris: &mut Vec<WorldTri>,
    app: &NpcAppearance, shirt_col: u32, pants_col: u32,
    is_job_hat: Option<u32>,
) {
    let skin = app.skin;
    let coat = app.coat_col;
    let arm_outer = if app.has_coat { coat } else { shirt_col };
    let vest = app.vest_col;

    // Head (offset down for seated height)
    let head_base = tris.len();
    gen_head(tris, app, is_job_hat);
    for tri in &mut tris[head_base..] {
        for v in &mut tri.v { v[1] -= 0.4; }
    }

    // Neck (thicker)
    mesh::tapered_cylinder_tris(tris, 0.0, 1.09, 0.0, 0.08, 0.07, 0.12, 8, skin);

    // Torso (wider, anatomical)
    mesh::ellipsoid_tris(tris, 0.0, 0.88, 0.0, 0.21, 0.20, 0.15, 1, SHIRT_LINEN);
    mesh::ellipsoid_tris(tris, 0.0, 0.68, 0.0, 0.17, 0.14, 0.13, 0, SHIRT_LINEN);
    mesh::beveled_box_tris(tris, 0.0, 0.78, 0.0, 0.44, 0.52, 0.30, 0.03, vest);
    if app.has_coat {
        mesh::beveled_box_tris(tris, 0.0, 0.78, 0.0, 0.56, 0.58, 0.36, 0.04, coat);
    }
    mesh::cylinder_tris(tris, 0.0, 0.52, 0.0, 0.22, 0.03, 10, LEATHER_DARK);

    // Seated thighs (horizontal, forward — joint-based)
    let l_hip = [-0.14, 0.44, 0.0];
    let l_knee_s = [-0.14, 0.42, -0.38];
    let r_hip = [0.14, 0.44, 0.0];
    let r_knee_s = [0.14, 0.42, -0.38];
    mesh::tapered_cylinder_between(tris, l_hip, l_knee_s, 0.10, 0.08, 8, pants_col);
    mesh::tapered_cylinder_between(tris, r_hip, r_knee_s, 0.10, 0.08, 8, pants_col);
    // Knee joints
    mesh::sphere_tris(tris, l_knee_s[0], l_knee_s[1], l_knee_s[2], 0.078, 0, pants_col);
    mesh::sphere_tris(tris, r_knee_s[0], r_knee_s[1], r_knee_s[2], 0.078, 0, pants_col);

    // Shins (hanging from knees — joint-based)
    let bc = app.boot_col;
    let l_ankle_s = [-0.14, 0.06, -0.40];
    let r_ankle_s = [0.14, 0.06, -0.40];
    if app.boot_type >= 2 {
        mesh::tapered_cylinder_between(tris, l_knee_s, l_ankle_s, 0.072, 0.056, 7, bc);
        mesh::tapered_cylinder_between(tris, r_knee_s, r_ankle_s, 0.072, 0.056, 7, bc);
    } else {
        mesh::tapered_cylinder_between(tris, l_knee_s, l_ankle_s, 0.066, 0.052, 6, 0xFFCCBBAA);
        mesh::tapered_cylinder_between(tris, r_knee_s, r_ankle_s, 0.066, 0.052, 6, 0xFFCCBBAA);
    }

    // Shoes (wider)
    mesh::beveled_box_tris(tris, -0.14, 0.02, -0.44, 0.10, 0.06, 0.14, 0.015, bc);
    mesh::beveled_box_tris(tris, 0.14, 0.02, -0.44, 0.10, 0.06, 0.14, 0.015, bc);

    // Arms resting on thighs — joint-based
    for &side in &[-1.0f32, 1.0] {
        let shoulder = [side * 0.32, 0.98, 0.0];
        let elbow = [side * 0.34, 0.64, -0.15];
        let wrist = [side * 0.30, 0.48, -0.30];
        mesh::ellipsoid_tris(tris, shoulder[0], shoulder[1], shoulder[2], 0.09, 0.06, 0.08, 0, arm_outer);
        mesh::tapered_cylinder_between(tris, shoulder, elbow, 0.07, 0.055, 7, arm_outer);
        mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2], 0.056, 0, arm_outer);
        mesh::tapered_cylinder_between(tris, elbow, wrist, 0.055, 0.042, 6, skin);
        gen_hand(tris, wrist[0], wrist[1] - 0.04, wrist[2] - 0.02, side, skin);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ANATOMICALLY ACCURATE NUDE MALE BODY — high-detail player character
// Reference: ACU head bust (deep orbital sockets, brow ridge, defined jaw)
// All major muscle groups modeled as individual ellipsoid masses over
// base skeletal forms. Skin-colored throughout — no clothing.
// ═══════════════════════════════════════════════════════════════════════════

/// Nude male torso — reference-accurate proportions from anatomy sculpture
/// Shoulders ~3x head width, massive pecs, aggressive V-taper, thick back muscles
fn gen_nude_torso(tris: &mut Vec<WorldTri>, skin: u32) {
    let sk = skin;
    let sk_dk = darken(sk, 0.96);
    let sk_shadow = darken(sk, 0.90);
    let sk_deep = darken(sk, 0.80);
    let sk_lt = darken(sk, 1.03);
    let nipple_col = darken(sk, 0.68);

    // ── BASE FORMS — wider ribcage, deeper front-to-back ──
    // Ribcage — wide barrel, projects forward (high subdiv for smooth surface)
    mesh::ellipsoid_tris(tris, 0.0, 1.26, -0.01, 0.26, 0.21, 0.18, 2, sk);
    // Abdomen — narrower waist for V-taper
    mesh::ellipsoid_tris(tris, 0.0, 1.02, 0.0, 0.20, 0.16, 0.15, 2, sk);
    // Waist bridge (fills gap between abdomen and pelvis — wide overlap)
    mesh::ellipsoid_tris(tris, 0.0, 0.94, 0.0, 0.22, 0.12, 0.16, 1, sk);
    // Pelvis — wide, continuous with hip sockets
    mesh::ellipsoid_tris(tris, 0.0, 0.87, 0.0, 0.24, 0.14, 0.18, 2, sk);
    // Hip-to-thigh transition (bridges pelvis to leg tops — smooth taper)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.12, 0.84, 0.0, 0.10, 0.08, 0.10, 1, sk);
    }
    // Groin front fill (prevents visible gap at front between legs)
    mesh::ellipsoid_tris(tris, 0.0, 0.84, -0.08, 0.12, 0.08, 0.08, 1, sk);
    // Groin back fill
    mesh::ellipsoid_tris(tris, 0.0, 0.84, 0.06, 0.10, 0.06, 0.06, 0, sk);
    // Inner groin (perineum area — fills gap between upper thighs)
    mesh::ellipsoid_tris(tris, 0.0, 0.82, 0.0, 0.06, 0.04, 0.08, 0, sk);
    // Pubic mound
    mesh::ellipsoid_tris(tris, 0.0, 0.86, -0.14, 0.08, 0.04, 0.04, 0, sk_dk);
    // Shoulder-to-torso bridge (armpit area — smooth transition to deltoids)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.24, 1.36, 0.0, 0.08, 0.08, 0.10, 1, sk);
        // Lower armpit fill (lat-to-arm transition)
        mesh::ellipsoid_tris(tris, side * 0.22, 1.30, 0.0, 0.06, 0.06, 0.08, 0, sk);
    }

    // ── CLAVICLES — wide S-curves defining shoulder width ──
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.10, 1.43, -0.07, 0.14, 0.015, 0.018, 0, sk_lt);
        mesh::ellipsoid_tris(tris, side * 0.22, 1.42, -0.04, 0.08, 0.013, 0.018, 0, sk_lt);
    }
    // Sternal notch depression
    mesh::sphere_tris(tris, 0.0, 1.44, -0.09, 0.015, 0, sk_deep);

    // ── STERNUM ──
    push_box(tris, 0.0, 1.33, -0.14, 0.018, 0.14, 0.010, sk_dk);
    push_box(tris, 0.0, 1.18, -0.13, 0.010, 0.02, 0.008, sk_dk);

    // ── PECTORALIS MAJOR — MASSIVE chest slabs (reference-accurate) ──
    for &side in &[-1.0f32, 1.0] {
        // Main pec mass — huge, dominates upper chest
        mesh::ellipsoid_tris(tris, side * 0.10, 1.29, -0.14, 0.14, 0.08, 0.08, 1, sk);
        // Clavicular head (upper pec shelf under collarbone)
        mesh::ellipsoid_tris(tris, side * 0.12, 1.37, -0.12, 0.10, 0.04, 0.06, 0, sk_dk);
        // Sternal head (lower, massive)
        mesh::ellipsoid_tris(tris, side * 0.10, 1.24, -0.15, 0.12, 0.05, 0.07, 1, sk);
        // Pec insertion (converges toward armpit)
        mesh::ellipsoid_tris(tris, side * 0.20, 1.32, -0.08, 0.06, 0.04, 0.05, 0, sk_dk);
        // Lower pec border (sharp defined line)
        mesh::ellipsoid_tris(tris, side * 0.11, 1.20, -0.11, 0.11, 0.010, 0.05, 0, sk_shadow);
        // Pec gap (dark cleft between pecs)
        push_box(tris, 0.0, 1.30, -0.16, 0.008, 0.10, 0.005, sk_shadow);
        // Areola + nipple (on pec surface)
        mesh::sphere_tris(tris, side * 0.11, 1.26, -0.20, 0.020, 0, darken(sk, 0.78));
        mesh::sphere_tris(tris, side * 0.11, 1.26, -0.21, 0.010, 0, nipple_col);
    }

    // ── RECTUS ABDOMINIS — defined 6-pack with wider bellies ──
    // Linea alba (deep midline groove)
    push_box(tris, 0.0, 1.08, -0.16, 0.006, 0.20, 0.006, sk_shadow);
    // Tendinous intersections (horizontal groove lines)
    for row in 0..3 {
        let ry = 1.18 - row as f32 * 0.07;
        push_box(tris, 0.0, ry, -0.16, 0.10, 0.004, 0.006, sk_shadow);
    }
    // Eight muscle bellies (4 per side for 8-pack)
    for &side in &[-1.0f32, 1.0] {
        for row in 0..4 {
            let ry = 1.18 - row as f32 * 0.07;
            let rw = 0.042 - row as f32 * 0.003;
            let rd = 0.030 - row as f32 * 0.002;
            mesh::ellipsoid_tris(tris, side * 0.045, ry, -0.15, rw, 0.030, rd, 0, sk);
        }
    }

    // ── NAVEL ──
    mesh::sphere_tris(tris, 0.0, 0.97, -0.17, 0.013, 0, sk_deep);
    mesh::sphere_tris(tris, 0.0, 0.97, -0.16, 0.019, 0, sk_shadow);

    // ── EXTERNAL OBLIQUES — thick side muscles creating V-taper ──
    for &side in &[-1.0f32, 1.0] {
        // Main oblique mass
        mesh::ellipsoid_tris(tris, side * 0.17, 1.12, -0.06, 0.06, 0.14, 0.08, 0, sk_dk);
        // Lower oblique (feeds into V-line)
        mesh::ellipsoid_tris(tris, side * 0.14, 1.00, -0.07, 0.05, 0.08, 0.06, 0, sk_dk);
        // V-line / inguinal ligament (Adonis belt)
        mesh::ellipsoid_tris(tris, side * 0.11, 0.93, -0.11, 0.08, 0.05, 0.04, 0, sk_shadow);
    }

    // ── SERRATUS ANTERIOR — prominent finger-like ribs ──
    for &side in &[-1.0f32, 1.0] {
        for fi in 0..5 {
            let sy = 1.24 - fi as f32 * 0.035;
            let sz = -0.05 + fi as f32 * 0.005;
            mesh::ellipsoid_tris(tris, side * 0.22, sy, sz, 0.030, 0.016, 0.025, 0, sk_dk);
        }
    }

    // ── ILIAC CREST — prominent hip bones ──
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.17, 0.91, -0.04, 0.07, 0.018, 0.05, 0, sk_lt);
        mesh::sphere_tris(tris, side * 0.19, 0.89, -0.09, 0.014, 0, sk_lt);
    }

    // ── LOWER ABDOMEN ──
    mesh::ellipsoid_tris(tris, 0.0, 0.92, -0.11, 0.14, 0.07, 0.07, 0, sk);

    // ── BACK — MASSIVE musculature (reference-accurate) ──
    // Upper trapezius (thick, from neck to shoulder tips) — HUGE diamond
    mesh::ellipsoid_tris(tris, 0.0, 1.40, 0.09, 0.26, 0.10, 0.10, 1, sk_dk);
    // Mid trapezius (between scapulae)
    mesh::ellipsoid_tris(tris, 0.0, 1.28, 0.12, 0.18, 0.08, 0.06, 0, sk_dk);
    // Lower trapezius (descending to mid-back)
    mesh::ellipsoid_tris(tris, 0.0, 1.16, 0.12, 0.12, 0.10, 0.05, 0, sk_dk);

    // Latissimus dorsi — WIDE V-taper wings (biggest change)
    for &side in &[-1.0f32, 1.0] {
        // Main lat body (extends from armpit to lower back)
        mesh::ellipsoid_tris(tris, side * 0.18, 1.16, 0.07, 0.08, 0.20, 0.08, 1, sk_dk);
        // Lat lateral flare (creates visible width from front)
        mesh::ellipsoid_tris(tris, side * 0.24, 1.22, 0.03, 0.05, 0.10, 0.06, 0, sk_dk);
        // Lat insertion near armpit/humerus
        mesh::ellipsoid_tris(tris, side * 0.22, 1.34, 0.02, 0.05, 0.06, 0.06, 0, sk_dk);
    }

    // Erector spinae (thick vertical columns flanking spine)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.05, 1.14, 0.14, 0.030, 0.24, 0.04, 0, sk);
    }
    // Spinal furrow (deep groove)
    push_box(tris, 0.0, 1.14, 0.14, 0.010, 0.26, 0.006, sk_shadow);

    // Scapulae (shoulder blades — visible bony shapes)
    for &side in &[-1.0f32, 1.0] {
        // Scapular body
        mesh::ellipsoid_tris(tris, side * 0.12, 1.30, 0.12, 0.07, 0.10, 0.018, 0, sk);
        // Scapular spine (horizontal ridge across blade)
        mesh::ellipsoid_tris(tris, side * 0.14, 1.36, 0.11, 0.07, 0.010, 0.020, 0, sk_lt);
        // Inferior angle (bottom point)
        mesh::sphere_tris(tris, side * 0.10, 1.20, 0.12, 0.012, 0, sk_lt);
    }

    // Infraspinatus (fills scapular fossa)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.14, 1.28, 0.10, 0.06, 0.06, 0.035, 0, sk_dk);
    }
    // Teres major (below scapula, connects to lat)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.18, 1.22, 0.07, 0.04, 0.05, 0.04, 0, sk_dk);
    }
    // Teres minor (above teres major)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.17, 1.28, 0.08, 0.03, 0.04, 0.03, 0, sk);
    }
    // Rhomboids (between spine and scapulae)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.06, 1.28, 0.12, 0.035, 0.08, 0.025, 0, sk);
    }

    // Lower back (lumbar — thick erector mass)
    mesh::ellipsoid_tris(tris, 0.0, 0.98, 0.11, 0.16, 0.12, 0.07, 0, sk);
    // Sacral triangle
    mesh::ellipsoid_tris(tris, 0.0, 0.88, 0.11, 0.08, 0.05, 0.03, 0, sk_shadow);
    // Thoracolumbar fascia (diamond shape at lower back)
    mesh::ellipsoid_tris(tris, 0.0, 1.00, 0.13, 0.10, 0.08, 0.015, 0, sk_lt);

    // ── RIB CAGE visible through skin (lean build) ──
    for &side in &[-1.0f32, 1.0] {
        for ri in 0..4 {
            let ry = 1.20 - ri as f32 * 0.032;
            let rz = -0.07 + ri as f32 * 0.008;
            mesh::ellipsoid_tris(tris, side * 0.19, ry, rz, 0.06, 0.005, 0.04, 0, sk_shadow);
        }
    }
    // Costal margin (lower rib edge — V-shape at bottom of ribcage)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.10, 1.10, -0.12, 0.08, 0.006, 0.03, 0, sk_shadow);
    }
}

/// Gluteal muscles — large, defined masses
fn gen_glutes(tris: &mut Vec<WorldTri>, skin: u32) {
    let sk_dk = darken(skin, 0.93);
    for &side in &[-1.0f32, 1.0] {
        // Gluteus maximus — large rounded mass
        mesh::ellipsoid_tris(tris, side * 0.10, 0.80, 0.10, 0.12, 0.10, 0.10, 1, skin);
        // Gluteus medius (upper lateral — creates hip width)
        mesh::ellipsoid_tris(tris, side * 0.16, 0.88, 0.06, 0.07, 0.05, 0.06, 0, sk_dk);
        // Gluteus minimus (deeper, visible at side)
        mesh::ellipsoid_tris(tris, side * 0.17, 0.85, 0.02, 0.04, 0.04, 0.04, 0, sk_dk);
    }
    // Gluteal cleft
    push_box(tris, 0.0, 0.80, 0.13, 0.008, 0.12, 0.006, darken(skin, 0.75));
    // Gluteal fold (sharp lower crease)
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.08, 0.73, 0.08, 0.10, 0.008, 0.06, 0, darken(skin, 0.82));
    }
    // Sacrum/coccyx transition
    mesh::ellipsoid_tris(tris, 0.0, 0.85, 0.13, 0.05, 0.04, 0.02, 0, darken(skin, 0.88));
}

/// Nude male arm — wider shoulders, thicker muscles matching anatomy references
fn gen_nude_arm(
    tris: &mut Vec<WorldTri>, side: f32, fwd: f32, bend: f32, skin: u32,
) {
    let sk = skin;
    let sk_dk = darken(sk, 0.96);
    let sk_shadow = darken(sk, 0.90);
    let sk_lt = darken(sk, 1.03);

    // ── JOINT POSITIONS — wider shoulder placement ──
    let shoulder = [side * 0.32, 1.42, 0.0];
    let elbow = [side * 0.36, 1.06, fwd * 0.35];
    let wrist = [side * 0.34, 0.80, fwd * 0.15 - bend];

    // ── DELTOID — MASSIVE 3-head shoulder cap (reference shows huge rounded delts) ──
    // Lateral deltoid (main visible bulk — creates shoulder width, high subdiv)
    mesh::ellipsoid_tris(tris, shoulder[0], shoulder[1] - 0.02, shoulder[2],
        0.12, 0.10, 0.10, 2, sk);
    // Anterior deltoid (front, thick)
    mesh::ellipsoid_tris(tris, shoulder[0] - side * 0.02, shoulder[1] - 0.05, shoulder[2] - 0.05,
        0.07, 0.07, 0.06, 1, sk_dk);
    // Posterior deltoid (back)
    mesh::ellipsoid_tris(tris, shoulder[0] - side * 0.01, shoulder[1] - 0.05, shoulder[2] + 0.05,
        0.06, 0.07, 0.06, 1, sk_dk);
    // Deltoid insertion V (converges on lateral humerus)
    let delt_ins = lerp3(shoulder, elbow, 0.40);
    mesh::ellipsoid_tris(tris, delt_ins[0] + side * 0.03, delt_ins[1], delt_ins[2],
        0.020, 0.05, 0.020, 0, sk);
    // Armpit fill (smooth transition from arm to torso)
    mesh::ellipsoid_tris(tris, shoulder[0] - side * 0.08, shoulder[1] - 0.10, shoulder[2],
        0.06, 0.06, 0.06, 0, sk);

    // ── UPPER ARM (shoulder → elbow) — thick, smooth ──
    mesh::tapered_cylinder_between(tris, shoulder, elbow, 0.10, 0.070, 12, sk);
    // Biceps brachii (large anterior mass — smooth)
    let bicep = lerp3(shoulder, elbow, 0.42);
    mesh::ellipsoid_tris(tris, bicep[0] - side * 0.01, bicep[1], bicep[2] - 0.04,
        0.055, 0.08, 0.050, 1, sk);
    // Bicep short head (medial)
    mesh::ellipsoid_tris(tris, bicep[0] - side * 0.03, bicep[1] + 0.01, bicep[2] - 0.03,
        0.030, 0.05, 0.028, 0, sk);
    // Triceps brachii (horseshoe shape, LARGER than bicep — smooth)
    let tricep = lerp3(shoulder, elbow, 0.48);
    mesh::ellipsoid_tris(tris, tricep[0], tricep[1], tricep[2] + 0.04,
        0.050, 0.10, 0.045, 1, sk_dk);
    // Tricep lateral head
    mesh::ellipsoid_tris(tris, tricep[0] + side * 0.03, tricep[1], tricep[2] + 0.02,
        0.032, 0.07, 0.032, 0, sk);
    // Tricep long head
    let tri_long = lerp3(shoulder, elbow, 0.35);
    mesh::ellipsoid_tris(tris, tri_long[0] - side * 0.01, tri_long[1], tri_long[2] + 0.04,
        0.035, 0.09, 0.032, 0, sk_dk);
    // Brachialis (between bicep/tricep, lateral view)
    let brach = lerp3(shoulder, elbow, 0.58);
    mesh::ellipsoid_tris(tris, brach[0] + side * 0.03, brach[1], brach[2],
        0.030, 0.06, 0.028, 0, sk_dk);
    // Cephalic vein (surface vein)
    let vein = lerp3(shoulder, elbow, 0.35);
    push_box(tris, vein[0] - side * 0.01, vein[1], vein[2] - 0.06,
        0.004, 0.08, 0.004, darken(sk, 0.88));
    // Bicipital groove
    let groove = lerp3(shoulder, elbow, 0.30);
    push_box(tris, groove[0] - side * 0.02, groove[1], groove[2] - 0.02,
        0.003, 0.06, 0.003, sk_shadow);

    // ── ELBOW JOINT — defined bony landmarks ──
    mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2], 0.066, 1, sk);
    // Olecranon (prominent elbow point)
    mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2] + 0.045, 0.028, 0, sk_lt);
    // Medial epicondyle
    mesh::sphere_tris(tris, elbow[0] - side * 0.045, elbow[1], elbow[2], 0.020, 0, sk_lt);
    // Lateral epicondyle
    mesh::sphere_tris(tris, elbow[0] + side * 0.035, elbow[1], elbow[2], 0.016, 0, sk_lt);
    // Cubital fossa (inner elbow)
    mesh::sphere_tris(tris, elbow[0] - side * 0.02, elbow[1], elbow[2] - 0.04, 0.014, 0, sk_shadow);

    // ── FOREARM (elbow → wrist) — muscular ──
    mesh::tapered_cylinder_between(tris, elbow, wrist, 0.068, 0.048, 8, sk);
    // Brachioradialis (largest forearm muscle)
    let brachrad = lerp3(elbow, wrist, 0.20);
    mesh::ellipsoid_tris(tris, brachrad[0] + side * 0.02, brachrad[1], brachrad[2] - 0.02,
        0.040, 0.065, 0.035, 0, sk);
    // Flexor group (palm side — medial mass)
    let flex = lerp3(elbow, wrist, 0.28);
    mesh::ellipsoid_tris(tris, flex[0] - side * 0.015, flex[1], flex[2] - 0.015,
        0.035, 0.060, 0.030, 0, sk_dk);
    // Extensor group (dorsal mass)
    let ext = lerp3(elbow, wrist, 0.30);
    mesh::ellipsoid_tris(tris, ext[0] + side * 0.01, ext[1], ext[2] + 0.02,
        0.030, 0.055, 0.028, 0, sk_dk);
    // Pronator teres (diagonal)
    let pron = lerp3(elbow, wrist, 0.15);
    mesh::ellipsoid_tris(tris, pron[0] - side * 0.02, pron[1], pron[2] - 0.02,
        0.025, 0.045, 0.020, 0, sk);
    // Forearm tendons near wrist
    let tendon = lerp3(elbow, wrist, 0.72);
    for ti in 0..3 {
        let tx = tendon[0] + (ti as f32 - 1.0) * 0.009;
        push_box(tris, tx, tendon[1], tendon[2] - 0.030,
            0.004, 0.05, 0.004, sk_shadow);
    }

    // ── WRIST — defined ──
    mesh::sphere_tris(tris, wrist[0], wrist[1], wrist[2], 0.042, 0, sk);
    mesh::sphere_tris(tris, wrist[0] + side * 0.035, wrist[1], wrist[2] + 0.01, 0.014, 0, sk_lt);
    mesh::sphere_tris(tris, wrist[0] - side * 0.028, wrist[1], wrist[2] - 0.01, 0.012, 0, sk_lt);

    // ── HAND ──
    gen_hand(tris, wrist[0], wrist[1] - 0.05, wrist[2] - 0.02, side, sk);
}

/// Bare foot with toes, arch, heel
fn gen_bare_foot(tris: &mut Vec<WorldTri>, ankle: [f32; 3], side: f32, skin: u32) {
    let sk = skin;
    let sk_dk = darken(sk, 0.93);
    let sk_lt = darken(sk, 1.04);
    let nail_col = darken(sk, 1.10);
    let lx = ankle[0];
    let az = ankle[2];

    // ── HEEL ──
    mesh::ellipsoid_tris(tris, lx, 0.03, az + 0.03, 0.04, 0.03, 0.04, 0, sk);
    mesh::ellipsoid_tris(tris, lx, 0.01, az + 0.03, 0.035, 0.01, 0.035, 0, sk_dk);

    // ── MIDFOOT ──
    mesh::ellipsoid_tris(tris, lx, 0.04, az - 0.02, 0.05, 0.025, 0.06, 0, sk);
    // Medial arch
    mesh::ellipsoid_tris(tris, lx - side * 0.02, 0.025, az, 0.015, 0.015, 0.04, 0, sk_lt);

    // ── FOREFOOT — ball of foot ──
    mesh::ellipsoid_tris(tris, lx, 0.02, az - 0.07, 0.055, 0.018, 0.03, 0, sk);
    mesh::ellipsoid_tris(tris, lx - side * 0.02, 0.01, az - 0.065, 0.025, 0.01, 0.02, 0, sk_dk);
    mesh::ellipsoid_tris(tris, lx + side * 0.02, 0.01, az - 0.065, 0.020, 0.01, 0.02, 0, sk_dk);

    // ── TOP OF FOOT + tendons ──
    push_box(tris, lx, 0.05, az - 0.03, 0.04, 0.01, 0.05, sk);
    for ti in 0..3 {
        let tx = lx + (ti as f32 - 1.0) * 0.012;
        push_box(tris, tx, 0.055, az - 0.03, 0.003, 0.005, 0.04, darken(sk, 0.87));
    }

    // ── TOES ──
    // Big toe
    let btx = lx - side * 0.025;
    mesh::ellipsoid_tris(tris, btx, 0.015, az - 0.10, 0.018, 0.013, 0.020, 0, sk);
    mesh::ellipsoid_tris(tris, btx, 0.015, az - 0.12, 0.016, 0.012, 0.016, 0, sk);
    push_box(tris, btx, 0.025, az - 0.13, 0.010, 0.005, 0.008, nail_col);
    // 4 smaller toes
    for ti in 0..4 {
        let tx = lx - side * 0.01 + (ti as f32 + 0.5) * side * 0.014;
        let toe_len = 0.015 - ti as f32 * 0.002;
        let toe_r = 0.010 - ti as f32 * 0.001;
        let tz = az - 0.095 - ti as f32 * 0.003;
        mesh::ellipsoid_tris(tris, tx, 0.012, tz, toe_r, 0.008, toe_len, 0, sk);
        push_box(tris, tx, 0.018, tz - toe_len + 0.003, 0.006, 0.003, 0.005, nail_col);
    }

    // ── SOLE ──
    push_box(tris, lx, 0.003, az - 0.03, 0.045, 0.003, 0.08, sk_dk);
}

/// Nude male leg — massive quads, hamstrings, calves matching anatomy sculpture references
fn gen_nude_leg(
    tris: &mut Vec<WorldTri>, side: f32, fwd: f32, knee_bend: f32, skin: u32,
) {
    let sk = skin;
    let sk_dk = darken(sk, 0.96);
    let sk_shadow = darken(sk, 0.90);
    let sk_lt = darken(sk, 1.03);

    // ── JOINT POSITIONS — wider stance, proportional to broad shoulders ──
    let lx = side * 0.15;
    let hip = [lx, 0.88, 0.0];
    let knee = [lx, 0.46, fwd * 0.5];
    let ankle = [lx, 0.08, fwd * 0.25 - knee_bend * 0.4];

    // ── HIP SOCKET — large smooth ball joint ──
    mesh::sphere_tris(tris, hip[0], hip[1], hip[2], 0.12, 2, sk);
    // Greater trochanter (prominent lateral bony landmark)
    mesh::sphere_tris(tris, hip[0] + side * 0.10, hip[1], hip[2], 0.028, 0, sk_lt);
    // Tensor fasciae latae (lateral hip muscle)
    mesh::ellipsoid_tris(tris, hip[0] + side * 0.08, hip[1] - 0.02, hip[2] - 0.03,
        0.04, 0.08, 0.04, 1, sk_dk);

    // ── THIGH (hip → knee) — MASSIVE, matching sculpture references ──
    mesh::tapered_cylinder_between(tris, hip, knee, 0.14, 0.090, 14, sk);

    // Rectus femoris (central quad — huge main mass)
    let rf = lerp3(hip, knee, 0.38);
    mesh::ellipsoid_tris(tris, rf[0], rf[1], rf[2] - 0.05,
        0.065, 0.14, 0.060, 1, sk);
    // Vastus lateralis (outer quad — creates outer thigh sweep)
    let vl = lerp3(hip, knee, 0.35);
    mesh::ellipsoid_tris(tris, vl[0] + side * 0.06, vl[1], vl[2] - 0.03,
        0.055, 0.14, 0.055, 1, sk_dk);
    // Vastus medialis (teardrop near knee — prominent on lean builds)
    let vm = lerp3(hip, knee, 0.68);
    mesh::ellipsoid_tris(tris, vm[0] - side * 0.05, vm[1], vm[2] - 0.04,
        0.045, 0.07, 0.040, 0, sk_dk);
    // Vastus intermedius (deep quad, adds mass between RF and VL)
    let vi = lerp3(hip, knee, 0.42);
    mesh::ellipsoid_tris(tris, vi[0] + side * 0.02, vi[1], vi[2] - 0.03,
        0.050, 0.10, 0.040, 0, sk);

    // Hamstrings — posterior thigh (LARGE group)
    let ham = lerp3(hip, knee, 0.48);
    mesh::ellipsoid_tris(tris, ham[0], ham[1], ham[2] + 0.05,
        0.065, 0.14, 0.055, 1, sk_dk);
    // Biceps femoris (lateral hamstring)
    let bf = lerp3(hip, knee, 0.52);
    mesh::ellipsoid_tris(tris, bf[0] + side * 0.05, bf[1], bf[2] + 0.04,
        0.040, 0.08, 0.040, 0, sk);
    // Semitendinosus (medial hamstring)
    let st = lerp3(hip, knee, 0.50);
    mesh::ellipsoid_tris(tris, st[0] - side * 0.04, st[1], st[2] + 0.04,
        0.035, 0.08, 0.035, 0, sk);
    // Semimembranosus (deep medial, adds inner bulk)
    let smem = lerp3(hip, knee, 0.55);
    mesh::ellipsoid_tris(tris, smem[0] - side * 0.03, smem[1], smem[2] + 0.03,
        0.030, 0.06, 0.030, 0, sk_dk);

    // Adductors (inner thigh — large muscle group)
    let add = lerp3(hip, knee, 0.28);
    mesh::ellipsoid_tris(tris, add[0] - side * 0.05, add[1], add[2],
        0.055, 0.14, 0.050, 1, sk_dk);
    // Adductor longus (more superficial, creates inner thigh line)
    let addl = lerp3(hip, knee, 0.35);
    mesh::ellipsoid_tris(tris, addl[0] - side * 0.06, addl[1], addl[2] - 0.02,
        0.030, 0.08, 0.030, 0, sk);
    // Gracilis (thin inner thigh muscle)
    let grac = lerp3(hip, knee, 0.50);
    mesh::ellipsoid_tris(tris, grac[0] - side * 0.07, grac[1], grac[2],
        0.015, 0.16, 0.015, 0, sk_shadow);

    // IT band (iliotibial tract — tight band on lateral thigh)
    let itb = lerp3(hip, knee, 0.50);
    mesh::ellipsoid_tris(tris, itb[0] + side * 0.09, itb[1], itb[2],
        0.012, 0.20, 0.018, 0, sk_shadow);
    // Sartorius (diagonal strap muscle from ASIS to medial knee)
    let sar = lerp3(hip, knee, 0.35);
    mesh::ellipsoid_tris(tris, sar[0] + side * 0.04, sar[1], sar[2] - 0.04,
        0.015, 0.12, 0.012, 0, sk_shadow);

    // ── KNEE JOINT — large, smooth ──
    mesh::sphere_tris(tris, knee[0], knee[1], knee[2], 0.085, 2, sk);
    // Patella (kneecap — prominent)
    mesh::sphere_tris(tris, knee[0], knee[1], knee[2] - 0.06, 0.035, 0, sk_lt);
    // Patellar tendon
    push_box(tris, knee[0], knee[1] - 0.06, knee[2] - 0.055, 0.018, 0.05, 0.010, sk_shadow);
    // Tibial tuberosity
    mesh::sphere_tris(tris, knee[0], knee[1] - 0.08, knee[2] - 0.05, 0.015, 0, sk_lt);
    // Popliteal fossa (back of knee depression)
    mesh::sphere_tris(tris, knee[0], knee[1], knee[2] + 0.06, 0.022, 0, sk_shadow);
    // Medial condyle
    mesh::sphere_tris(tris, knee[0] - side * 0.05, knee[1], knee[2], 0.020, 0, sk_lt);
    // Lateral condyle
    mesh::sphere_tris(tris, knee[0] + side * 0.04, knee[1], knee[2], 0.018, 0, sk_lt);

    // ── CALF (knee → ankle) — muscular, diamond-shaped ──
    mesh::tapered_cylinder_between(tris, knee, ankle, 0.085, 0.052, 12, sk);
    // Gastrocnemius medial head (larger, more prominent)
    let gast = lerp3(knee, ankle, 0.22);
    mesh::ellipsoid_tris(tris, gast[0] - side * 0.02, gast[1], gast[2] + 0.03,
        0.045, 0.09, 0.045, 1, sk);
    // Gastrocnemius lateral head
    mesh::ellipsoid_tris(tris, gast[0] + side * 0.02, gast[1], gast[2] + 0.025,
        0.038, 0.08, 0.038, 0, sk);
    // Soleus (wider than gastroc, lower)
    let sol = lerp3(knee, ankle, 0.42);
    mesh::ellipsoid_tris(tris, sol[0], sol[1], sol[2] + 0.025,
        0.045, 0.08, 0.038, 0, sk_dk);
    // Tibialis anterior (shin muscle — creates front calf shape)
    let ta = lerp3(knee, ankle, 0.28);
    mesh::ellipsoid_tris(tris, ta[0], ta[1], ta[2] - 0.04,
        0.032, 0.10, 0.030, 0, sk);
    // Peroneus longus (lateral compartment)
    let per = lerp3(knee, ankle, 0.32);
    mesh::ellipsoid_tris(tris, per[0] + side * 0.04, per[1], per[2],
        0.025, 0.08, 0.022, 0, sk_dk);
    // Peroneus brevis (lower lateral)
    let perb = lerp3(knee, ankle, 0.50);
    mesh::ellipsoid_tris(tris, perb[0] + side * 0.035, perb[1], perb[2],
        0.020, 0.05, 0.018, 0, sk_dk);
    // Shin bone (tibia ridge — visible subcutaneous)
    let shin = lerp3(knee, ankle, 0.30);
    push_box(tris, shin[0] - side * 0.01, shin[1], shin[2] - 0.05,
        0.010, 0.20, 0.008, sk_lt);
    // Achilles tendon (thick, prominent)
    let ach = lerp3(knee, ankle, 0.65);
    mesh::ellipsoid_tris(tris, ach[0], ach[1], ach[2] + 0.035,
        0.015, 0.10, 0.012, 0, sk_shadow);
    // Extensor digitorum longus (anterior-lateral)
    let edl = lerp3(knee, ankle, 0.35);
    mesh::ellipsoid_tris(tris, edl[0] + side * 0.02, edl[1], edl[2] - 0.03,
        0.020, 0.07, 0.018, 0, sk);

    // ── ANKLE — defined bony landmarks ──
    mesh::sphere_tris(tris, ankle[0], ankle[1], ankle[2], 0.048, 0, sk);
    // Medial malleolus (larger, lower)
    mesh::sphere_tris(tris, ankle[0] - side * 0.04, ankle[1] + 0.01, ankle[2], 0.018, 0, sk_lt);
    // Lateral malleolus (smaller, higher)
    mesh::sphere_tris(tris, ankle[0] + side * 0.042, ankle[1] + 0.005, ankle[2] + 0.005, 0.015, 0, sk_lt);

    // ── BARE FOOT ──
    gen_bare_foot(tris, ankle, side, sk);
}

/// Nude attack arm — punching forward with flexed muscles, proportional to wide shoulders
fn gen_nude_attack_arm(tris: &mut Vec<WorldTri>, side: f32, extend: f32, skin: u32) {
    let sk = skin;
    let sk_dk = darken(sk, 0.93);
    let sk_lt = darken(sk, 1.05);
    let shoulder = [side * 0.32, 1.42, 0.0];
    let elbow = [side * 0.36, 1.10, -0.15 - extend * 0.20];
    let wrist = [side * 0.34, 0.92, -0.35 - extend * 0.35];

    // Deltoid (massive, 3-head)
    mesh::ellipsoid_tris(tris, shoulder[0], shoulder[1] - 0.02, shoulder[2],
        0.12, 0.10, 0.10, 1, sk);
    mesh::ellipsoid_tris(tris, shoulder[0] - side * 0.02, shoulder[1] - 0.05, shoulder[2] - 0.05,
        0.07, 0.06, 0.05, 0, sk_dk);
    // Upper arm (thick)
    mesh::tapered_cylinder_between(tris, shoulder, elbow, 0.10, 0.070, 10, sk);
    // Bicep (flexed, large)
    let bicep = lerp3(shoulder, elbow, 0.40);
    mesh::ellipsoid_tris(tris, bicep[0], bicep[1], bicep[2] - 0.04,
        0.060, 0.07, 0.055, 1, sk);
    // Tricep (large horseshoe)
    let tricep = lerp3(shoulder, elbow, 0.48);
    mesh::ellipsoid_tris(tris, tricep[0], tricep[1], tricep[2] + 0.04,
        0.050, 0.09, 0.045, 1, sk_dk);
    // Brachialis
    let brach_u = lerp3(shoulder, elbow, 0.58);
    mesh::ellipsoid_tris(tris, brach_u[0] + side * 0.03, brach_u[1], brach_u[2],
        0.030, 0.06, 0.028, 0, sk_dk);
    // Elbow
    mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2], 0.066, 1, sk);
    mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2] + 0.045, 0.028, 0, sk_lt);
    // Forearm (thick)
    mesh::tapered_cylinder_between(tris, elbow, wrist, 0.068, 0.048, 8, sk);
    // Brachioradialis
    let brach = lerp3(elbow, wrist, 0.20);
    mesh::ellipsoid_tris(tris, brach[0] + side * 0.02, brach[1], brach[2] - 0.02,
        0.040, 0.065, 0.035, 0, sk);
    // Flexor group
    let flex = lerp3(elbow, wrist, 0.28);
    mesh::ellipsoid_tris(tris, flex[0] - side * 0.015, flex[1], flex[2] - 0.015,
        0.035, 0.060, 0.030, 0, sk_dk);
    // Fist (large)
    mesh::sphere_tris(tris, wrist[0], wrist[1] - 0.03, wrist[2] - 0.03, 0.055, 1, sk);
    push_box(tris, wrist[0], wrist[1] - 0.03, wrist[2] - 0.06, 0.045, 0.040, 0.030, darken(sk, 0.85));
}

/// Complete nude male player body with animation
fn gen_nude_player_body(
    tris: &mut Vec<WorldTri>,
    swing: f32,
    skin: u32,
    hair: u32,
    attack_phase: f32,
    carrying_item: bool,
    carrying_bin: bool,
    sitting: bool,
) {
    let head_app = NpcAppearance {
        skin, hair,
        hat_type: 0, hat_col: 0, coat_col: 0, vest_col: 0,
        has_coat: false, has_cape: false, has_sash: false,
        has_cross_strap: false, has_bracers: false,
        boot_type: 0, boot_col: 0, sash_col: 0,
        face_age: 0, is_female: false,
    };

    if sitting {
        // Seated nude body — head offset down 0.4
        let head_base = tris.len();
        gen_head(tris, &head_app, None);
        for tri in &mut tris[head_base..] {
            for v in &mut tri.v { v[1] -= 0.4; }
        }
        mesh::tapered_cylinder_tris(tris, 0.0, 1.09, 0.0, 0.08, 0.07, 0.12, 8, skin);
        // Seated torso (centered lower)
        let torso_base = tris.len();
        gen_nude_torso(tris, skin);
        for tri in &mut tris[torso_base..] {
            for v in &mut tri.v { v[1] -= 0.4; }
        }
        gen_glutes(tris, skin);
        // Horizontal thighs (matching wider stance and thicker proportions)
        for &side in &[-1.0f32, 1.0] {
            let hip_s = [side * 0.15, 0.44, 0.0];
            let knee_s = [side * 0.15, 0.42, -0.38];
            let ankle_s = [side * 0.15, 0.06, -0.40];
            mesh::tapered_cylinder_between(tris, hip_s, knee_s, 0.14, 0.090, 10, skin);
            mesh::sphere_tris(tris, knee_s[0], knee_s[1], knee_s[2], 0.085, 0, skin);
            mesh::tapered_cylinder_between(tris, knee_s, ankle_s, 0.085, 0.052, 8, skin);
            gen_bare_foot(tris, ankle_s, side, skin);
        }
        // Arms resting on thighs (matching wider shoulders)
        for &side in &[-1.0f32, 1.0] {
            let shoulder = [side * 0.32, 0.98, 0.0];
            let elbow = [side * 0.34, 0.64, -0.15];
            let wrist = [side * 0.32, 0.48, -0.30];
            mesh::ellipsoid_tris(tris, shoulder[0], shoulder[1], shoulder[2], 0.12, 0.10, 0.10, 1, skin);
            mesh::tapered_cylinder_between(tris, shoulder, elbow, 0.10, 0.070, 8, skin);
            mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2], 0.066, 0, skin);
            mesh::tapered_cylinder_between(tris, elbow, wrist, 0.068, 0.048, 7, skin);
            gen_hand(tris, wrist[0], wrist[1] - 0.04, wrist[2] - 0.02, side, skin);
        }
        return;
    }

    gen_head(tris, &head_app, None);
    gen_neck(tris, skin);
    gen_nude_torso(tris, skin);
    gen_glutes(tris, skin);

    // ── LEGS ──
    let l_fwd = -swing * 0.40;
    let r_fwd = swing * 0.40;
    let l_knee = if swing > 0.0 { swing * 0.22 } else { 0.0 };
    let r_knee = if swing < 0.0 { (-swing) * 0.22 } else { 0.0 };
    gen_nude_leg(tris, -1.0, l_fwd, l_knee, skin);
    gen_nude_leg(tris, 1.0, r_fwd, r_knee, skin);

    // ── ARMS ──
    if attack_phase > 0.0 {
        let t = (attack_phase / ATTACK_ANIM_DURATION).clamp(0.0, 1.0);
        let extend = 1.0 - (1.0 - t) * (1.0 - t);
        gen_nude_attack_arm(tris, 1.0, extend, skin);
        gen_nude_arm(tris, -1.0, -0.2, 0.3, skin);
    } else if carrying_item || carrying_bin {
        gen_nude_arm(tris, -1.0, -0.63, 0.30, skin);
        gen_nude_arm(tris, 1.0, -0.63, 0.30, skin);
        if carrying_item {
            mesh::beveled_box_tris(tris, 0.0, 0.88, -0.50, 0.30, 0.30, 0.20, 0.02, BAG_COLOR);
            push_box(tris, 0.0, 1.1, -0.37, 0.02, 0.4, 0.02, LEATHER_MED);
        } else {
            mesh::cylinder_tris(tris, 0.0, 0.78, -0.55, 0.2, 0.55, 8, BIN_COLOR);
        }
    } else {
        let l_arm_fwd = swing * 0.25;
        let r_arm_fwd = -swing * 0.25;
        let l_bend = 0.10 + swing.abs() * 0.14;
        let r_bend = 0.10 + swing.abs() * 0.14;
        gen_nude_arm(tris, -1.0, l_arm_fwd, l_bend, skin);
        gen_nude_arm(tris, 1.0, r_arm_fwd, r_bend, skin);
    }
}

pub fn gen_player_mesh(player: &Player, tris: &mut Vec<WorldTri>) {
    let base = tris.len();
    let skin = if player.hit_flash > 0.0 { 0xFFFF4444 } else { SKIN_COLOR };

    gen_nude_player_body(
        tris,
        player.walk_phase.sin() * 0.4,
        skin,
        0xFF332211,
        player.attack_phase,
        player.carrying_item,
        player.carrying_bin.is_some(),
        player.sitting,
    );

    let (sin_r, cos_r) = player.rot_y.sin_cos();
    for tri in &mut tris[base..] {
        for v in &mut tri.v {
            let rx = v[0] * cos_r + v[2] * sin_r;
            let rz = -v[0] * sin_r + v[2] * cos_r;
            v[0] = rx + player.x;
            v[1] += player.y;
            v[2] = rz + player.z;
        }
        let nx = tri.normal[0] * cos_r + tri.normal[2] * sin_r;
        let nz = -tri.normal[0] * sin_r + tri.normal[2] * cos_r;
        tri.normal[0] = nx;
        tri.normal[2] = nz;
    }
}

pub fn gen_vehicle_mesh(v: &Vehicle, tris: &mut Vec<WorldTri>, show_interior: bool) {
    let base = tris.len();
    let color = v.color;
    let cabin_color = darken(color, VEHICLE_BODY_COLOR_DARKEN);
    let trim_color = darken(color, 0.5);
    let undercarriage = 0xFF333333_u32;

    // Main body — beveled box chassis
    mesh::beveled_box_tris(tris, 0.0, 0.45, 0.0, 1.8, 0.6, 3.6, 0.08, color);

    // Undercarriage pan (visible from below)
    push_box(tris, 0.0, 0.12, 0.0, 1.6, 0.04, 3.4, undercarriage);

    // Wheel wells — dark recesses (4 arches)
    for &(wx, wz) in &[(-0.88f32, -1.1f32), (0.88, -1.1), (-0.88, 1.1), (0.88, 1.1)] {
        push_box(tris, wx, 0.3, wz, 0.12, 0.35, 0.5, undercarriage);
    }

    // Cabin — beveled with slightly tapered sides
    mesh::beveled_box_tris(tris, 0.0, 0.95, 0.2, 1.5, 0.5, 1.8, 0.06, cabin_color);

    // A-pillars (front windshield frame)
    push_box(tris, -0.68, 0.95, -0.65, 0.06, 0.45, 0.06, trim_color);
    push_box(tris, 0.68, 0.95, -0.65, 0.06, 0.45, 0.06, trim_color);
    // B-pillars (between front/rear doors)
    push_box(tris, -0.72, 0.95, 0.2, 0.06, 0.45, 0.06, trim_color);
    push_box(tris, 0.72, 0.95, 0.2, 0.06, 0.45, 0.06, trim_color);
    // C-pillars (rear)
    push_box(tris, -0.68, 0.95, 1.05, 0.06, 0.45, 0.06, trim_color);
    push_box(tris, 0.68, 0.95, 1.05, 0.06, 0.45, 0.06, trim_color);

    // Roof rack rails (thin bars on top)
    push_box(tris, -0.55, 1.22, 0.2, 0.03, 0.02, 1.4, trim_color);
    push_box(tris, 0.55, 1.22, 0.2, 0.03, 0.02, 1.4, trim_color);

    // Sloped hood (front slopes down with crease)
    push_box(tris, 0.0, 0.58, -1.5, 1.6, 0.18, 0.5, darken(color, 0.88));
    // Hood crease line
    push_box(tris, 0.0, 0.68, -1.4, 0.03, 0.01, 0.7, darken(color, 0.7));

    // Sloped trunk
    push_box(tris, 0.0, 0.6, 1.55, 1.6, 0.12, 0.4, darken(color, 0.88));

    // Front grille — dark slotted area
    push_box(tris, 0.0, 0.38, -1.82, 0.8, 0.18, 0.04, 0xFF222222);
    // Grille slats (3 horizontal bars)
    for i in 0..3 {
        let gy = 0.32 + i as f32 * 0.07;
        push_box(tris, 0.0, gy, -1.83, 0.7, 0.015, 0.02, 0xFF888888);
    }

    // Windshields
    push_box(tris, 0.0, 0.95, -0.7, 1.3, 0.35, 0.06, WINDSHIELD_COLOR);
    push_box(tris, 0.0, 0.95, 1.15, 1.3, 0.35, 0.06, WINDSHIELD_COLOR);

    // Side windows (front pair + rear pair)
    push_box(tris, -0.76, 0.95, -0.2, 0.04, 0.3, 0.6, WINDSHIELD_COLOR);
    push_box(tris, 0.76, 0.95, -0.2, 0.04, 0.3, 0.6, WINDSHIELD_COLOR);
    push_box(tris, -0.76, 0.95, 0.6, 0.04, 0.3, 0.5, WINDSHIELD_COLOR);
    push_box(tris, 0.76, 0.95, 0.6, 0.04, 0.3, 0.5, WINDSHIELD_COLOR);

    // Door handles (4 small cylinders on sides)
    for &(dx, dz) in &[(-0.91f32, -0.1f32), (0.91, -0.1), (-0.91, 0.65), (0.91, 0.65)] {
        mesh::cylinder_tris(tris, dx, 0.65, dz, 0.015, 0.08, 3, 0xFF888888);
    }

    // Side mirrors
    push_box(tris, -0.95, 0.85, -0.55, 0.08, 0.08, 0.12, trim_color);
    push_box(tris, 0.95, 0.85, -0.55, 0.08, 0.08, 0.12, trim_color);
    // Mirror glass face
    push_box(tris, -1.0, 0.85, -0.55, 0.02, 0.06, 0.1, 0xFF667788);
    push_box(tris, 1.0, 0.85, -0.55, 0.02, 0.06, 0.1, 0xFF667788);

    // Wheels — cylinder tire + hub + spokes
    let wheel_r = 0.22;
    let wheel_w = 0.18;
    for &(wx, wz) in &[(-0.85f32, -1.1f32), (0.85, -1.1), (-0.85, 1.1), (0.85, 1.1)] {
        // Tire — outer ring
        mesh::cylinder_tris(tris, wx, 0.22, wz, wheel_r, wheel_w, 8, TIRE_COLOR);
        // Hub cap (metallic center)
        mesh::cylinder_tris(tris, wx, 0.22, wz, wheel_r * 0.45, wheel_w + 0.02, 6, 0xFF999999);
        // Hub center bolt
        mesh::cylinder_tris(tris, wx, 0.22, wz, wheel_r * 0.12, wheel_w + 0.04, 4, 0xFFAAAAAA);
        // Brake disc visible behind spokes
        mesh::cylinder_tris(tris, wx, 0.22, wz, wheel_r * 0.7, 0.04, 6, 0xFF666666);
    }

    // Headlights (recessed with bezel)
    for &hx in &[-0.6f32, 0.6] {
        push_box(tris, hx, 0.45, -1.81, 0.3, 0.14, 0.04, 0xFF333333); // housing
        mesh::sphere_tris(tris, hx, 0.45, -1.83, 0.1, 0, 0xFFFFEE88); // bulb
    }
    // Turn signal indicators (small amber lights)
    push_box(tris, -0.85, 0.45, -1.79, 0.08, 0.05, 0.03, 0xFFFFAA22);
    push_box(tris, 0.85, 0.45, -1.79, 0.08, 0.05, 0.03, 0xFFFFAA22);

    // Tail lights (larger, with reflector housing)
    for &tx in &[-0.6f32, 0.6] {
        push_box(tris, tx, 0.45, 1.81, 0.28, 0.12, 0.04, 0xFF441111); // housing
        mesh::sphere_tris(tris, tx, 0.45, 1.83, 0.08, 0, 0xFFFF2222); // bulb
    }
    // Reverse lights (small white)
    push_box(tris, -0.25, 0.38, 1.82, 0.08, 0.06, 0.03, 0xFFDDDDDD);
    push_box(tris, 0.25, 0.38, 1.82, 0.08, 0.06, 0.03, 0xFFDDDDDD);

    // Bumpers (front + rear, with curve suggestion)
    mesh::beveled_box_tris(tris, 0.0, 0.22, -1.85, 1.7, 0.14, 0.1, 0.03, 0xFF444444);
    mesh::beveled_box_tris(tris, 0.0, 0.22, 1.85, 1.7, 0.14, 0.1, 0.03, 0xFF444444);

    // License plate area (rear)
    push_box(tris, 0.0, 0.35, 1.84, 0.35, 0.08, 0.02, 0xFFDDDDDD);
    // License plate frame
    push_box(tris, 0.0, 0.35, 1.85, 0.38, 0.01, 0.01, 0xFF333333);

    // Exhaust pipe (rear undercarriage)
    mesh::cylinder_tris(tris, -0.4, 0.14, 1.8, 0.03, 0.15, 4, 0xFF555555);

    // Antenna (thin cylinder on rear roof)
    mesh::cylinder_tris(tris, 0.3, 1.35, 0.8, 0.008, 0.3, 3, 0xFF222222);

    // Interior details (only for player's vehicle)
    if show_interior {
        // Dashboard
        mesh::beveled_box_tris(tris, 0.0, 0.78, -0.6, 1.3, 0.12, 0.35, 0.02, DASHBOARD_COLOR);
        // Instrument cluster (lighter area)
        push_box(tris, -0.3, 0.82, -0.72, 0.35, 0.08, 0.02, 0xFF222233);
        // Steering wheel (ring + column)
        mesh::cylinder_tris(tris, -0.3, 0.88, -0.45, 0.12, 0.02, 8, STEERING_COLOR);
        mesh::cylinder_tris(tris, -0.3, 0.82, -0.52, 0.025, 0.12, 4, STEERING_COLOR);
        // Center console
        push_box(tris, 0.0, 0.6, 0.0, 0.25, 0.2, 0.8, darken(DASHBOARD_COLOR, 0.8));
        // Gear shift knob
        mesh::sphere_tris(tris, 0.0, 0.72, -0.1, 0.03, 0, 0xFF222222);
        // Front seats (driver + passenger)
        for &sx in &[-0.35f32, 0.35] {
            // Seat base
            mesh::beveled_box_tris(tris, sx, 0.58, 0.0, 0.45, 0.12, 0.45, 0.02, SEAT_COLOR);
            // Seat back
            mesh::beveled_box_tris(tris, sx, 0.88, 0.22, 0.45, 0.45, 0.08, 0.02, SEAT_COLOR);
            // Headrest
            push_box(tris, sx, 1.12, 0.22, 0.2, 0.12, 0.06, SEAT_COLOR);
        }
        // Rear seat
        mesh::beveled_box_tris(tris, 0.0, 0.55, 0.6, 1.1, 0.1, 0.4, 0.02, SEAT_COLOR);
        mesh::beveled_box_tris(tris, 0.0, 0.82, 0.78, 1.1, 0.35, 0.06, 0.02, SEAT_COLOR);
        // Rearview mirror
        push_box(tris, 0.0, 1.1, -0.45, 0.2, 0.06, 0.02, 0xFF667788);
    }

    let (sin_r, cos_r) = v.rot_y.sin_cos();
    for tri in &mut tris[base..] {
        for vert in &mut tri.v {
            let rx = vert[0] * cos_r + vert[2] * sin_r;
            let rz = -vert[0] * sin_r + vert[2] * cos_r;
            vert[0] = rx + v.x;
            vert[1] += v.y;
            vert[2] = rz + v.z;
        }
        let nx = tri.normal[0] * cos_r + tri.normal[2] * sin_r;
        let nz = -tri.normal[0] * sin_r + tri.normal[2] * cos_r;
        tri.normal[0] = nx;
        tri.normal[2] = nz;
    }
}

fn job_shirt_color(npc: &Npc) -> u32 {
    match npc.job {
        NpcJob::PolicePatrol => 0xFF2233AA,
        NpcJob::Firefighter => 0xFFCC3322,
        NpcJob::Paramedic => 0xFFDDDDDD,
        NpcJob::ConstructionWorker => 0xFFDDAA22,
        NpcJob::StreetVendor => 0xFF885533,
        _ => npc.shirt_color,
    }
}

pub fn gen_npc_mesh(npc: &Npc, tris: &mut Vec<WorldTri>) {
    let shirt = if npc.hit_flash > 0.0 { 0xFFFF4444 } else { job_shirt_color(npc) };
    let app = npc_appearance(npc.brain_idx as u32);

    // Ragdoll rendering: sphere joints + cylinder limb segments
    if npc.ragdoll_active {
        let p = &npc.ragdoll_points;
        let arm_col = if app.has_coat { app.coat_col } else { shirt };
        // Torso with coat color
        mesh::cylinder_between(tris, p[0], p[1], 0.22, 6, arm_col);
        // Head
        mesh::lathe_tris(tris, p[2][0], p[2][1], p[2][2],
            &[[0.0, -0.15], [0.12, -0.08], [0.16, 0.02], [0.12, 0.12], [0.0, 0.16]], 6, app.skin);
        // Neck
        mesh::cylinder_between(tris, p[1], p[2], 0.07, 4, app.skin);
        // Arms (thicker)
        mesh::cylinder_between(tris, p[1], p[3], 0.07, 5, arm_col);
        mesh::cylinder_between(tris, p[1], p[4], 0.07, 5, arm_col);
        push_box(tris, p[3][0], p[3][1], p[3][2], 0.06, 0.05, 0.06, app.skin);
        push_box(tris, p[4][0], p[4][1], p[4][2], 0.06, 0.05, 0.06, app.skin);
        // Legs (thicker)
        mesh::cylinder_between(tris, p[0], p[5], 0.10, 6, npc.pants_color);
        mesh::cylinder_between(tris, p[0], p[6], 0.10, 6, npc.pants_color);
        mesh::beveled_box_tris(tris, p[5][0], p[5][1], p[5][2], 0.07, 0.05, 0.12, 0.01, BOOT_BROWN);
        mesh::beveled_box_tris(tris, p[6][0], p[6][1], p[6][2], 0.07, 0.05, 0.12, 0.01, BOOT_BROWN);
        return;
    }

    let base = tris.len();

    // KO pose — body flat on ground
    if npc.state == NpcState::KnockedOut {
        let arm_col = if app.has_coat { app.coat_col } else { shirt };
        mesh::cylinder_tris(tris, 0.0, 0.15, 0.0, 0.19, 0.65, 6, arm_col);
        mesh::lathe_tris(tris, 0.0, 0.15, -0.52,
            &[[0.0, -0.15], [0.12, -0.08], [0.16, 0.02], [0.12, 0.12], [0.0, 0.16]], 6, app.skin);
        mesh::cylinder_tris(tris, -0.45, 0.1, 0.15, 0.05, 0.55, 4, app.skin);
        mesh::cylinder_tris(tris, 0.45, 0.1, -0.08, 0.05, 0.55, 4, app.skin);
        mesh::cylinder_tris(tris, -0.14, 0.1, 0.45, 0.065, 0.6, 5, npc.pants_color);
        mesh::cylinder_tris(tris, 0.14, 0.1, 0.45, 0.065, 0.6, 5, npc.pants_color);
        mesh::beveled_box_tris(tris, -0.14, 0.03, 0.72, 0.07, 0.05, 0.12, 0.01, BOOT_BROWN);
        mesh::beveled_box_tris(tris, 0.14, 0.03, 0.72, 0.07, 0.05, 0.12, 0.01, BOOT_BROWN);

        let (sin_r, cos_r) = npc.rot_y.sin_cos();
        for tri in &mut tris[base..] {
            for v in &mut tri.v {
                let rx = v[0] * cos_r + v[2] * sin_r;
                let rz = -v[0] * sin_r + v[2] * cos_r;
                v[0] = rx + npc.x;
                v[1] += npc.y;
                v[2] = rz + npc.z;
            }
            let nx = tri.normal[0] * cos_r + tri.normal[2] * sin_r;
            let nz = -tri.normal[0] * sin_r + tri.normal[2] * cos_r;
            tri.normal[0] = nx;
            tri.normal[2] = nz;
        }
        return;
    }

    // Job-specific hat color
    let job_hat = match npc.job {
        NpcJob::PolicePatrol => Some(0xFF2233AA),
        NpcJob::Firefighter => Some(0xFFCC3322),
        NpcJob::Paramedic => Some(0xFFDDDDDD),
        NpcJob::ConstructionWorker => Some(0xFFDDAA22),
        NpcJob::MailCarrier => Some(0xFF3344CC),
        _ => None,
    };

    gen_character_body(
        tris,
        npc.walk_phase.sin() * 0.4,
        &app,
        shirt,
        npc.pants_color,
        npc.attack_phase,
        npc.carrying_item,
        npc.carrying_bin.is_some(),
        false,
        job_hat,
    );

    // Speech bubble (floating above head)
    if npc.interacting_with.is_some() {
        mesh::sphere_tris(tris, 0.0, 2.15, -0.15, 0.12, 0, 0xFFFFFFFF);
        mesh::sphere_tris(tris, 0.0, 2.0, -0.1, 0.04, 0, 0xFFFFFFFF);
    }

    let (sin_r, cos_r) = npc.rot_y.sin_cos();
    for tri in &mut tris[base..] {
        for v in &mut tri.v {
            let rx = v[0] * cos_r + v[2] * sin_r;
            let rz = -v[0] * sin_r + v[2] * cos_r;
            v[0] = rx + npc.x;
            v[1] += npc.y;
            v[2] = rz + npc.z;
        }
        let nx = tri.normal[0] * cos_r + tri.normal[2] * sin_r;
        let nz = -tri.normal[0] * sin_r + tri.normal[2] * cos_r;
        tri.normal[0] = nx;
        tri.normal[2] = nz;
    }
}

pub fn gen_item_mesh(item: &Item, tris: &mut Vec<WorldTri>) {
    let color = match item.kind {
        ItemKind::Health => 0xFFFF3333,
        ItemKind::Money => 0xFFFFDD33,
        ItemKind::Stamina => 0xFF33FF33,
        ItemKind::Food => 0xFFDD8833,
        ItemKind::Water => 0xFF3388FF,
    };
    let y = item.y + 0.8 + (item.spin_phase * 2.0).sin() * 0.2;

    // Type-specific shapes
    match item.kind {
        ItemKind::Money => {
            // Coin — flat cylinder spinning
            let base = tris.len();
            mesh::cylinder_tris(tris, 0.0, 0.0, 0.0, 0.2, 0.05, 8, color);
            // Rotate around Y
            let (sin_s, cos_s) = item.spin_phase.sin_cos();
            for tri in &mut tris[base..] {
                for v in &mut tri.v {
                    let rx = v[0] * cos_s + v[2] * sin_s;
                    let rz = -v[0] * sin_s + v[2] * cos_s;
                    v[0] = rx + item.x;
                    v[1] += y;
                    v[2] = rz + item.z;
                }
            }
        }
        ItemKind::Health => {
            // Cross shape — two intersecting boxes
            let base = tris.len();
            mesh::box_tris(tris, 0.0, 0.0, 0.0, 0.4, 0.12, 0.12, color);
            mesh::box_tris(tris, 0.0, 0.0, 0.0, 0.12, 0.4, 0.12, color);
            let (sin_s, cos_s) = item.spin_phase.sin_cos();
            for tri in &mut tris[base..] {
                for v in &mut tri.v {
                    let rx = v[0] * cos_s + v[2] * sin_s;
                    let rz = -v[0] * sin_s + v[2] * cos_s;
                    v[0] = rx + item.x;
                    v[1] += y;
                    v[2] = rz + item.z;
                }
            }
        }
        ItemKind::Water => {
            // Bottle — lathe profile
            let base = tris.len();
            let profile: [[f32;2]; 5] = [
                [0.0, -0.2], [0.1, -0.18], [0.1, 0.1], [0.05, 0.18], [0.0, 0.2],
            ];
            mesh::lathe_tris(tris, 0.0, 0.0, 0.0, &profile, 5, color);
            let (sin_s, cos_s) = item.spin_phase.sin_cos();
            for tri in &mut tris[base..] {
                for v in &mut tri.v {
                    let rx = v[0] * cos_s + v[2] * sin_s;
                    let rz = -v[0] * sin_s + v[2] * cos_s;
                    v[0] = rx + item.x;
                    v[1] += y;
                    v[2] = rz + item.z;
                }
            }
        }
        _ => {
            // Food, Stamina — sphere
            mesh::sphere_tris(tris, item.x, y, item.z, 0.2, 1, color);
        }
    }
}

pub fn gen_trash_bin_mesh(bin: &TrashBin, tris: &mut Vec<WorldTri>) {
    // Cylinder body with beveled rim + lid
    mesh::cylinder_tris(tris, bin.x, bin.y + 0.4, bin.z, 0.22, 0.8, 6, BIN_COLOR);
    // Rim at top
    mesh::cylinder_tris(tris, bin.x, bin.y + 0.82, bin.z, 0.25, 0.06, 6, BIN_LID_COLOR);
    // Lid
    mesh::cylinder_tris(tris, bin.x, bin.y + 0.87, bin.z, 0.24, 0.04, 6, BIN_LID_COLOR);
    // Overflow pile if more than half full
    if bin.items_held > 5 {
        mesh::sphere_tris(tris, bin.x, bin.y + 0.95, bin.z, 0.15, 0, BAG_COLOR);
    }
}

fn tri_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let e1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let e2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    let n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
    let l = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
    if l < 1e-10 { [0.0, 1.0, 0.0] } else { [n[0]/l, n[1]/l, n[2]/l] }
}

fn darken(color: u32, factor: f32) -> u32 {
    let r = (((color >> 16) & 0xFF) as f32 * factor) as u32;
    let g = (((color >> 8) & 0xFF) as f32 * factor) as u32;
    let b = ((color & 0xFF) as f32 * factor) as u32;
    0xFF000000 | (r << 16) | (g << 8) | b
}

fn push_box(tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32, w: f32, h: f32, d: f32, color: u32) {
    let (hw, hh, hd) = (w * 0.5, h * 0.5, d * 0.5);
    let c = [
        [cx-hw,cy-hh,cz+hd],[cx+hw,cy-hh,cz+hd],[cx+hw,cy+hh,cz+hd],[cx-hw,cy+hh,cz+hd],
        [cx-hw,cy-hh,cz-hd],[cx+hw,cy-hh,cz-hd],[cx+hw,cy+hh,cz-hd],[cx-hw,cy+hh,cz-hd],
    ];
    let faces: [([usize;4],[f32;3]);6] = [
        ([0,1,2,3],[0.0,0.0,1.0]),([5,4,7,6],[0.0,0.0,-1.0]),
        ([4,0,3,7],[-1.0,0.0,0.0]),([1,5,6,2],[1.0,0.0,0.0]),
        ([3,2,6,7],[0.0,1.0,0.0]),([4,5,1,0],[0.0,-1.0,0.0]),
    ];
    for (idx, normal) in faces {
        tris.push(WorldTri { v: [c[idx[0]],c[idx[1]],c[idx[2]]], normal, color });
        tris.push(WorldTri { v: [c[idx[0]],c[idx[2]],c[idx[3]]], normal, color });
    }
}

// --- GPU vertex generation (for Vulkan graphics pipeline) ---

/// Convert WorldTri slice to GpuVertex buffer with lighting + fog + distance cull
fn tris_to_gpu_verts(tris: &[WorldTri], cam_pos: Vec3, tc: &TimeColors, out: &mut Vec<GpuVertex>) {
    let fog_dist_sq = FOG_DIST * FOG_DIST;
    let inv_fog_dist_sq = 1.0 / fog_dist_sq;

    out.reserve(tris.len() * 3);

    for tri in tris {
        let cx = (tri.v[0][0] + tri.v[1][0] + tri.v[2][0]) * 0.333;
        let cy = (tri.v[0][1] + tri.v[1][1] + tri.v[2][1]) * 0.333;
        let cz = (tri.v[0][2] + tri.v[1][2] + tri.v[2][2]) * 0.333;
        let dx = cam_pos[0] - cx;
        let dy = cam_pos[1] - cy;
        let dz = cam_pos[2] - cz;
        let dist_sq = dx*dx + dy*dy + dz*dz;
        if dist_sq > fog_dist_sq { continue; }

        let sun_lit = v3_dot(tri.normal, tc.light_dir).max(0.0) * tc.sun_strength;
        let intensity = sun_lit + tc.ambient;
        // fog*fog = dist_sq / fog_dist_sq — avoids sqrt entirely
        let fog_sq = (dist_sq * inv_fog_dist_sq).min(1.0);
        let color = shade_and_fog_sq(tri.color, intensity, fog_sq, tc);

        for i in 0..3 {
            out.push(GpuVertex {
                pos: tri.v[i],
                color_packed: color,
                normal: tri.normal,
            });
        }
    }
}

/// Like shade_and_fog but takes fog_sq (= fog*fog) directly, avoiding sqrt
#[inline(always)]
fn shade_and_fog_sq(color: u32, intensity: f32, fog_sq: f32, tc: &TimeColors) -> u32 {
    let r = ((color >> 16) & 0xFF) as f32;
    let g = ((color >> 8) & 0xFF) as f32;
    let b = (color & 0xFF) as f32;
    let i = intensity.clamp(0.1, 1.3);
    let mix = fog_sq; // already squared
    let inv = 1.0 - mix;
    let ro = ((r * i * inv + tc.fog_r * mix) as u32).min(255);
    let go = ((g * i * inv + tc.fog_g * mix) as u32).min(255);
    let bo = ((b * i * inv + tc.fog_b * mix) as u32).min(255);
    0xFF000000 | (ro << 16) | (go << 8) | bo
}

/// Generate GPU vertices for static world geometry (call once, or when lighting changes significantly)
pub fn generate_static_gpu_vertices(
    world: &WorldData, cam_pos: Vec3, hour: f32, out: &mut Vec<GpuVertex>,
) {
    let tc = time_colors(hour);
    out.clear();
    tris_to_gpu_verts(&world.static_tris, cam_pos, &tc, out);
}

/// Generate GPU vertices for dynamic entities only (call each frame)
pub fn generate_dynamic_gpu_vertices(
    world: &WorldData, player: &Player, cam: &Camera,
    hour: f32, scratch: &mut Vec<WorldTri>, out: &mut Vec<GpuVertex>,
) {
    let tc = time_colors(hour);
    let eye = v3(cam.x, cam.y, cam.z);
    let fog_dist_sq = FOG_DIST * FOG_DIST;

    out.clear();
    scratch.clear();

    // Pre-cull entities by distance before generating mesh
    for (vi, v) in world.vehicles.iter().enumerate() {
        let dx = eye[0] - v.x;
        let dz = eye[2] - v.z;
        if dx*dx + dz*dz > fog_dist_sq { continue; }
        let show_interior = player.in_vehicle == Some(vi);
        gen_vehicle_mesh(v, scratch, show_interior);
    }
    for npc in &world.npcs {
        if npc.state == NpcState::Sleeping { continue; }
        if npc.in_vehicle { continue; }
        let dx = eye[0] - npc.x;
        let dz = eye[2] - npc.z;
        if dx*dx + dz*dz > fog_dist_sq { continue; }
        gen_npc_mesh(npc, scratch);
    }
    for item in &world.items {
        if !item.active && !item.falling { continue; }
        let dx = eye[0] - item.x;
        let dz = eye[2] - item.z;
        if dx*dx + dz*dz > fog_dist_sq { continue; }
        gen_item_mesh(item, scratch);
    }
    for bin in &world.trash_bins {
        if bin.carried_by.is_some() { continue; }
        let dx = eye[0] - bin.x;
        let dz = eye[2] - bin.z;
        if dx*dx + dz*dz > fog_dist_sq { continue; }
        gen_trash_bin_mesh(bin, scratch);
    }
    if player.in_vehicle.is_none() {
        gen_player_mesh(player, scratch);
    }
    tris_to_gpu_verts(scratch, eye, &tc, out);
}

/// Get sky color as float RGBA for GPU clear
pub fn sky_color_f32(hour: f32) -> [f32; 4] {
    let c = time_colors(hour).sky;
    [
        ((c >> 16) & 0xFF) as f32 / 255.0,
        ((c >> 8) & 0xFF) as f32 / 255.0,
        (c & 0xFF) as f32 / 255.0,
        1.0,
    ]
}
