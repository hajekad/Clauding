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

// Body proportion scaling: ~6.5 heads tall heroic proportions
const BODY_STRETCH: f32 = 1.25;  // moderate vertical stretch (less distortion)
const BODY_WIDEN: f32 = 1.0;     // no horizontal scale (chest was too massive)
const HEAD_SCALE: f32 = 0.62;    // Loomis proportions: deeper/wider head needs smaller scale
const HEAD_CY: f32 = 1.70;       // head's natural Y center
const NECK_TOP: f32 = 1.50;      // natural neck top Y

// ── Parameterized body proportions (from GLTF reference scans) ──
struct BodyProportions {
    body_stretch: f32,
    body_widen: f32,
    head_scale: f32,
    head_cy: f32,
    neck_top: f32,
    // Torso cross-sections
    shoulder_rx: f32,
    shoulder_deltoid_amp: f32,
    hip_rx: f32,
    waist_rx: f32,
    chest_rx: f32,
    chest_rz: f32,
    // Muscle definition (0.0=smooth, 1.0=full male muscle)
    muscle_def: f32,
    // Limb scaling
    arm_rx_scale: f32,
    leg_rx_scale: f32,
    shoulder_joint_x: f32,
    hip_joint_x: f32,
    // Neck
    neck_rx: f32,
    neck_rz: f32,
    has_adams_apple: bool,
    // Female-specific
    has_breasts: bool,
    breast_rx: f32,
    breast_ry: f32,
    breast_rz: f32,
    breast_y: f32,
    breast_z: f32,
    breast_x_off: f32,
    is_female: bool,
}

fn male_proportions() -> BodyProportions {
    BodyProportions {
        body_stretch: 1.25,
        body_widen: 1.0,
        head_scale: 0.62,
        head_cy: 1.70,
        neck_top: 1.50,
        shoulder_rx: 0.32,
        shoulder_deltoid_amp: 0.07,
        hip_rx: 0.18,
        waist_rx: 0.15,
        chest_rx: 0.22,
        chest_rz: 0.19,
        muscle_def: 1.0,
        arm_rx_scale: 1.0,
        leg_rx_scale: 1.0,
        shoulder_joint_x: 0.22,
        hip_joint_x: 0.12,
        neck_rx: 0.10,
        neck_rz: 0.085,
        has_adams_apple: true,
        has_breasts: false,
        breast_rx: 0.0, breast_ry: 0.0, breast_rz: 0.0,
        breast_y: 0.0, breast_z: 0.0, breast_x_off: 0.0,
        is_female: false,
    }
}

fn female_proportions() -> BodyProportions {
    BodyProportions {
        body_stretch: 1.25,
        body_widen: 1.0,
        head_scale: 0.57,
        head_cy: 1.70,
        neck_top: 1.50,
        shoulder_rx: 0.22,
        shoulder_deltoid_amp: 0.03,
        hip_rx: 0.20,
        waist_rx: 0.12,
        chest_rx: 0.17,
        chest_rz: 0.16,
        muscle_def: 0.3,
        arm_rx_scale: 0.90,
        leg_rx_scale: 0.90,
        shoulder_joint_x: 0.20,
        hip_joint_x: 0.13,
        neck_rx: 0.07,
        neck_rz: 0.065,
        has_adams_apple: false,
        has_breasts: true,
        breast_rx: 0.065,
        breast_ry: 0.055,
        breast_rz: 0.038,     // subtle forward projection
        breast_y: 1.26,
        breast_z: -0.18,      // closer to chest wall
        breast_x_off: 0.09,
        is_female: true,
    }
}

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
    let sk = skin;
    let sk_sh = darken(skin, 0.92);
    let sk_dk = darken(skin, 0.85);

    use std::f32::consts::{PI, TAU};
    let n = 48;

    // Loomis proportions: eyes 50%, brow 43% from top, nose bottom 68%, mouth 75%
    let hp = PI * 0.5;
    let re = 0.39; // right eye angular position
    let le = TAU - re;
    let fem = app.is_female;

    // Gender-specific parameters
    let jaw_w = if fem { 0.008 } else { 0.012 };
    let jawline = if fem { 0.010 } else { 0.015 };
    let gonial = if fem { 0.010 } else { 0.022 };
    let masseter = if fem { 0.005 } else { 0.012 };
    let brow_shelf = if fem { 0.008 } else { 0.020 };
    let brow_boss = if fem { 0.005 } else { 0.012 };
    let glabella = if fem { 0.003 } else { 0.008 };
    let supraorb = if fem { 0.004 } else { 0.010 };
    let cheek = if fem { 0.032 } else { 0.025 };
    let chin_proj = if fem { 0.035 } else { 0.048 };
    let chin_w = if fem { 0.05 } else { 0.06 };
    let nose_size = if fem { 0.028 } else { 0.035 };

    // ══════════════════════════════════════════════════════════════
    // SKULL LOFT — continuous head surface from chin to crown
    // ══════════════════════════════════════════════════════════════
    let rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
        // ── CHIN — forward-projecting mental protuberance ──
        (1.46, body_ring(0.0, -0.06, chin_w, 0.09, &[
            (0.0, 0.3, chin_proj),   // mental protuberance — forward projection
        ], n), sk),
        (1.49, body_ring(0.0, -0.04, 0.10, 0.11, &[
            (0.0, 0.3, chin_proj * 0.8),  // chin pad
        ], n), sk),
        // Chin pad upper surface
        (1.52, body_ring(0.0, -0.02, 0.13, 0.13, &[
            (0.0, 0.3, chin_proj * 0.5),
        ], n), sk),
        // Labiomental fold — crease between chin pad and lower lip
        (1.55, body_ring(0.0, -0.01, 0.15, 0.15, &[
            (0.0, 0.25, -0.008),     // labiomental sulcus — inward crease
            (hp, 0.25, jaw_w), (le, 0.25, jaw_w),
        ], n), sk),

        // ── JAW ──
        (1.58, body_ring(0.0, 0.01, 0.19, 0.17, &[
            (hp, 0.25, jawline), (le, 0.25, jawline),
            (0.0, 0.3, 0.008),       // lower face projection
        ], n), sk),
        // Jaw angle (gonion) — defined corner, not smooth curve
        (1.61, body_ring(0.0, 0.02, 0.20, 0.18, &[
            (hp, 0.18, gonial), (le, 0.18, gonial),  // narrow bump = sharper angle
            (hp - 0.3, 0.2, masseter), (le + 0.3, 0.2, masseter),
        ], n), sk),
        (1.63, body_ring(0.0, 0.02, 0.20, 0.18, &[
            (hp, 0.18, gonial * 0.8), (le, 0.18, gonial * 0.8),
            (hp - 0.3, 0.2, masseter), (le + 0.3, 0.2, masseter),
        ], n), sk),

        // ── MOUTH LEVEL ──
        (1.65, body_ring(0.0, 0.03, 0.195, 0.19, &[
            (0.0, 0.20, 0.008),      // oral projection
        ], n), sk),
        (1.67, body_ring(0.0, 0.03, 0.20, 0.20, &[
            (0.0, 0.25, 0.012),      // upper lip/maxilla projection
            (hp - 0.3, 0.2, masseter * 0.8), (le + 0.3, 0.2, masseter * 0.8),
        ], n), sk),

        // ── NOSE BASE ──
        (1.69, body_ring(0.0, 0.03, 0.205, 0.21, &[
            (0.0, 0.12, 0.014),      // piriform aperture
            (0.20, 0.10, -0.006), (TAU - 0.20, 0.10, -0.006), // nasolabial fold
        ], n), sk),
        // Nose mid — alar wings and tip projection
        (1.72, body_ring(0.0, 0.04, 0.21, 0.22, &[
            (0.0, 0.07, nose_size),   // nose tip projection
            (0.12, 0.06, 0.012), (TAU - 0.12, 0.06, 0.012), // alar
            (0.25, 0.10, -0.006), (TAU - 0.25, 0.10, -0.006), // nasolabial
        ], n), sk),
        // Nose bridge + cheekbones
        (1.75, body_ring(0.0, 0.04, 0.22, 0.23, &[
            (0.0, 0.06, nose_size * 0.7), // nasal bones
            (0.55, 0.2, cheek), (TAU - 0.55, 0.2, cheek), // zygomatic arch
            (0.30, 0.12, if fem { -0.008 } else { -0.005 }),
            (TAU - 0.30, 0.12, if fem { -0.008 } else { -0.005 }),
        ], n), sk),

        // ── EYE LEVEL — convex face, no concavities ──
        (1.77, body_ring(0.0, 0.05, 0.215, 0.24, &[
            (0.0, 0.07, 0.018),       // nasion / nose bridge
            (re, 0.08, 0.006), (le, 0.08, 0.006), // orbital rim
            (0.55, 0.15, cheek * 0.5), (TAU - 0.55, 0.15, cheek * 0.5), // zygomatic
        ], n), sk),
        (1.80, body_ring(0.0, 0.05, 0.21, 0.24, &[
            (0.0, 0.07, 0.014),       // upper nose bridge
        ], n), sk),

        // ── BROW RIDGE ──
        (1.82, body_ring(0.0, 0.06, 0.205, 0.25, &[
            (0.0, 0.45, brow_shelf),
            (0.0, 0.10, glabella),     // glabella
            (re, 0.12, supraorb), (le, 0.12, supraorb),
        ], n), sk),
        (1.84, body_ring(0.0, 0.06, 0.21, 0.25, &[
            (0.0, 0.5, brow_boss),
        ], n), sk),

        // ── FOREHEAD ──
        (1.87, body_ring(0.0, 0.07, 0.215, 0.26, &[
            (PI, 0.4, 0.010),
        ], n), sk),
        (1.91, body_ring(0.0, 0.08, 0.22, 0.27, &[
            (PI, 0.4, 0.015),
        ], n), sk),

        // ── CRANIAL VAULT ──
        (1.95, body_ring(0.0, 0.08, 0.22, 0.27, &[(PI, 0.4, 0.015)], n), sk),
        (1.99, body_ring(0.0, 0.08, 0.22, 0.27, &[(PI, 0.4, 0.015)], n), sk),

        // ── CROWN ──
        (2.03, body_ring(0.0, 0.07, 0.20, 0.24, &[], n), sk),
        (2.06, body_ring(0.0, 0.06, 0.17, 0.20, &[], n), sk),
        (2.08, body_ring(0.0, 0.05, 0.13, 0.16, &[], n), sk),
        (2.10, body_ring(0.0, 0.04, 0.09, 0.11, &[], n), sk),
        (2.12, body_ring(0.0, 0.03, 0.04, 0.05, &[], n), sk),
    ];
    mesh::loft_y_tris(tris, &rings);

    // ══════════════════════════════════════════════════════════════
    // EYE ASSEMBLY — protruding eyeball with overhanging lid shelves
    // ══════════════════════════════════════════════════════════════
    for &side in &[-1.0f32, 1.0] {
        let ex = side * 0.08;
        let ey = 1.775;
        let ez = -0.21;  // eye center well forward of face surface

        // Orbital rim — bony ridge frames the socket opening
        mesh::ellipsoid_tris(tris, ex, ey, ez + 0.030, 0.044, 0.028, 0.025, 2, sk);

        // Eyeball — large sphere protruding past the lid plane
        let eye_r = 0.024;
        mesh::sphere_tris(tris, ex, ey, ez, eye_r, 1, 0xFFEEEEEE); // sclera
        // Iris — colored disk on front of eyeball
        mesh::sphere_tris(tris, ex, ey, ez - 0.018, 0.013, 1, 0xFF445533);
        // Pupil — dark center
        mesh::sphere_tris(tris, ex, ey, ez - 0.022, 0.006, 0, 0xFF111100);

        // Upper eyelid — thick shelf that overhangs the eyeball top, creating shadow
        // Main lid body — sits above the eyeball and curves over it
        mesh::ellipsoid_tris(tris, ex, ey + 0.016, ez + 0.004, 0.038, 0.012, 0.030, 1, sk);
        // Lid crease fold — secondary fold above the lid
        mesh::ellipsoid_tris(tris, ex, ey + 0.022, ez + 0.010, 0.036, 0.005, 0.024, 1, darken(sk, 0.94));
        // Lash line — dark edge along lower margin of upper lid
        mesh::ellipsoid_tris(tris, ex, ey + 0.008, ez - 0.012, 0.035, 0.003, 0.022, 0, sk_dk);

        // Lower eyelid — defined fleshy rim below the eyeball
        mesh::ellipsoid_tris(tris, ex, ey - 0.014, ez + 0.002, 0.034, 0.008, 0.024, 1, sk);
        // Lower lid rim — slight thickness catching light
        mesh::ellipsoid_tris(tris, ex, ey - 0.009, ez - 0.010, 0.030, 0.003, 0.018, 0, darken(sk, 0.92));

        // Lacrimal caruncle — fleshy pink corner at medial canthus
        let caruncle_x = ex - side * 0.028;
        mesh::ellipsoid_tris(tris, caruncle_x, ey - 0.002, ez - 0.004,
            0.007, 0.007, 0.006, 0, 0xFFCC8888);
    }

    // ── EYEBROWS — on brow ridge ──
    for &side in &[-1.0f32, 1.0] {
        let brow_thick = if fem { 0.004 } else { 0.006 };
        mesh::ellipsoid_tris(tris, side * 0.075, 1.82, -0.205,
            0.040, brow_thick, 0.012, 0, darken(hair, 0.85));
        // Arch peak (lateral third is thinner)
        mesh::ellipsoid_tris(tris, side * 0.10, 1.825, -0.20,
            0.018, brow_thick * 0.6, 0.008, 0, darken(hair, 0.80));
    }

    // ══════════════════════════════════════════════════════════════
    // NOSE — bridge with plane changes, enclosed nostrils, philtrum ridges
    // ══════════════════════════════════════════════════════════════
    let ns = nose_size;
    // Bridge — 3 distinct planes: upper nasal bone, mid bridge, lateral walls
    // Upper bridge (nasal bone) — hard, flat plane
    mesh::ellipsoid_tris(tris, 0.0, 1.78, -0.230, 0.010, 0.018, 0.012, 0, sk);
    // Mid bridge — slightly wider, different angle catches different light
    mesh::ellipsoid_tris(tris, 0.0, 1.75, -0.240, 0.014, 0.030, 0.016, 1, sk);
    // Lateral nasal walls — angled planes flanking the bridge
    for &side in &[-1.0f32, 1.0] {
        // Upper lateral wall
        mesh::ellipsoid_tris(tris, side * 0.014, 1.75, -0.228,
            0.010, 0.028, 0.010, 0, sk);
        // Lower lateral wall — transitions to alar
        mesh::ellipsoid_tris(tris, side * 0.016, 1.72, -0.232,
            0.010, 0.020, 0.010, 0, sk);
    }
    // Nose tip — bifurcated dome (two lobules)
    mesh::sphere_tris(tris, 0.007, 1.715, -0.260, ns * 0.48, 1, darken(sk, 0.97));
    mesh::sphere_tris(tris, -0.007, 1.715, -0.260, ns * 0.48, 1, darken(sk, 0.97));
    // Supratip — slight ridge above the tip domes
    mesh::ellipsoid_tris(tris, 0.0, 1.725, -0.255, 0.010, 0.008, 0.010, 0, sk);
    // Alar wings — thick curved flaps that curl under to enclose nostrils
    for &side in &[-1.0f32, 1.0] {
        // Wing body — main fleshy flap
        mesh::ellipsoid_tris(tris, side * 0.020, 1.717, -0.245,
            0.016, 0.012, 0.014, 0, sk);
        // Wing outer edge — rolls outward
        mesh::ellipsoid_tris(tris, side * 0.024, 1.715, -0.238,
            0.008, 0.010, 0.008, 0, sk);
        // Wing curl (underside wrapping inward) — encloses nostril opening
        mesh::ellipsoid_tris(tris, side * 0.016, 1.708, -0.248,
            0.012, 0.006, 0.010, 0, darken(sk, 0.92));
        // Inner wing wall — completes the nostril enclosure
        mesh::ellipsoid_tris(tris, side * 0.010, 1.710, -0.245,
            0.006, 0.008, 0.008, 0, darken(sk, 0.93));
    }
    // Columella — strip between nostrils, angled down
    mesh::ellipsoid_tris(tris, 0.0, 1.706, -0.252, 0.005, 0.008, 0.006, 0, darken(sk, 0.95));
    // Nostril openings — teardrop-shaped dark cavities enclosed by alar wings
    for &side in &[-1.0f32, 1.0] {
        // Dark nostril interior — enclosed teardrop shape
        mesh::ellipsoid_tris(tris, side * 0.012, 1.704, -0.248,
            0.008, 0.005, 0.006, 0, darken(sk, 0.35));
        // Deeper interior shadow
        mesh::ellipsoid_tris(tris, side * 0.012, 1.706, -0.244,
            0.005, 0.004, 0.004, 0, darken(sk, 0.25));
    }
    // Philtrum — two prominent ridges from columella to Cupid's bow
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.006, 1.695, -0.218,
            0.003, 0.018, 0.004, 0, sk_sh);
    }
    // Philtrum groove — concavity between the two ridges
    mesh::ellipsoid_tris(tris, 0.0, 1.695, -0.214, 0.004, 0.016, 0.002, 0, darken(sk, 0.93));

    // ══════════════════════════════════════════════════════════════
    // MOUTH — clean vermilion border, Cupid's bow M-shape, defined commissures
    // ══════════════════════════════════════════════════════════════
    let lip_full = if fem { 1.3 } else { 1.0 };
    let lip_col = if fem { 0xFFCC8888 } else { 0xFFBB8877 };
    let lo_lip_col = if fem { 0xFFDD9999 } else { 0xFFCC9988 };

    // Orbicularis oris — tissue ring around mouth (skin-colored base)
    mesh::ellipsoid_tris(tris, 0.0, 1.667, -0.200, 0.048, 0.018 * lip_full, 0.016 * lip_full, 1, darken(sk, 0.95));

    // Upper lip — two halves forming clear Cupid's bow M-shape
    // Left peak of the bow
    mesh::ellipsoid_tris(tris, -0.014, 1.676, -0.210, 0.022, 0.005 * lip_full, 0.010 * lip_full, 0, lip_col);
    // Right peak of the bow
    mesh::ellipsoid_tris(tris, 0.014, 1.676, -0.210, 0.022, 0.005 * lip_full, 0.010 * lip_full, 0, lip_col);
    // Central tubercle — fills the dip of the M at center
    mesh::ellipsoid_tris(tris, 0.0, 1.674, -0.214, 0.008, 0.004 * lip_full, 0.007 * lip_full, 0, lip_col);
    // Vermilion border — sharp upper edge defining the lip-skin boundary
    mesh::ellipsoid_tris(tris, 0.0, 1.679, -0.207, 0.036, 0.002, 0.006, 0, darken(lip_col, 0.90));

    // Lower lip — fuller single form, slight central depression
    mesh::ellipsoid_tris(tris, 0.0, 1.656, -0.208, 0.040, 0.010 * lip_full, 0.013 * lip_full, 1, lo_lip_col);
    // Lower lip central softness
    mesh::ellipsoid_tris(tris, 0.0, 1.654, -0.214, 0.010, 0.003 * lip_full, 0.004, 0, darken(lo_lip_col, 0.93));
    // Lower vermilion border
    mesh::ellipsoid_tris(tris, 0.0, 1.651, -0.206, 0.036, 0.002, 0.005, 0, darken(lo_lip_col, 0.88));

    // Mouth crease — dark line between lips
    mesh::ellipsoid_tris(tris, 0.0, 1.666, -0.216, 0.036, 0.001, 0.003, 0, darken(sk, 0.50));

    // Commissures — mouth corners recede into cheek (continuous, not floating)
    for &side in &[-1.0f32, 1.0] {
        // Corner — connected to lip edges, receding into cheek
        mesh::ellipsoid_tris(tris, side * 0.036, 1.666, -0.198,
            0.008, 0.006, 0.006, 0, darken(sk, 0.75));
        // Nasolabial fold — defined crease from nose to mouth corner
        mesh::ellipsoid_tris(tris, side * 0.032, 1.69, -0.210,
            0.005, 0.020, 0.004, 0, sk_sh);
    }

    // ══════════════════════════════════════════════════════════════
    // EARS — anatomical landmarks: helix rim, antihelix Y-fork, concha bowl,
    //        tragus tab, lobe. All scaled up for visibility.
    // ══════════════════════════════════════════════════════════════
    for &side in &[-1.0f32, 1.0] {
        let ear_x = side * 0.22;
        let ear_z = 0.05;
        let s = side; // shorthand

        // ── HELIX — outer C-shaped rim fold, the most visible ear structure ──
        // Top arc — highest point of the ear
        mesh::ellipsoid_tris(tris, ear_x + s * 0.003, 1.80, ear_z,
            0.012 * s.abs(), 0.010, 0.016, 1, sk);
        // Upper front — helix curves forward from top
        mesh::ellipsoid_tris(tris, ear_x - s * 0.005, 1.79, ear_z - 0.012,
            0.010, 0.016, 0.010, 1, sk);
        // Mid helix — widest point, curling inward
        mesh::ellipsoid_tris(tris, ear_x + s * 0.008, 1.77, ear_z + 0.002,
            0.012, 0.022, 0.014, 1, sk);
        // Lower helix — curves down toward lobe
        mesh::ellipsoid_tris(tris, ear_x + s * 0.004, 1.74, ear_z - 0.004,
            0.010, 0.018, 0.012, 1, sk);
        // Helix root — where it attaches to head above ear canal
        mesh::ellipsoid_tris(tris, ear_x - s * 0.010, 1.77, ear_z - 0.015,
            0.008, 0.010, 0.008, 0, sk);

        // ── ANTIHELIX — Y-shaped inner ridge with clear fork ──
        // Main trunk — vertical ridge inside the helix
        mesh::ellipsoid_tris(tris, ear_x - s * 0.004, 1.76, ear_z + 0.002,
            0.008, 0.030, 0.010, 1, darken(sk, 0.96));
        // Superior crus (upper fork) — curves up and back
        mesh::ellipsoid_tris(tris, ear_x, 1.79, ear_z + 0.006,
            0.006, 0.014, 0.008, 0, darken(sk, 0.95));
        // Inferior crus (lower fork) — curves toward helix root
        mesh::ellipsoid_tris(tris, ear_x - s * 0.006, 1.79, ear_z - 0.006,
            0.006, 0.014, 0.008, 0, darken(sk, 0.95));
        // Fossa triangularis — depression between the two crura
        mesh::ellipsoid_tris(tris, ear_x - s * 0.002, 1.79, ear_z,
            0.004, 0.008, 0.004, 0, darken(sk, 0.80));

        // ── CONCHA — deep bowl leading to canal ──
        mesh::ellipsoid_tris(tris, ear_x - s * 0.012, 1.755, ear_z - 0.002,
            0.014, 0.018, 0.012, 0, darken(sk, 0.55));
        // Canal opening — dark center of concha
        mesh::sphere_tris(tris, ear_x - s * 0.016, 1.755, ear_z - 0.005,
            0.006, 0, darken(sk, 0.30));

        // ── TRAGUS — cartilage tab projecting in front of canal ──
        mesh::ellipsoid_tris(tris, ear_x - s * 0.020, 1.758, ear_z - 0.016,
            0.008, 0.010, 0.007, 0, sk);
        // Intertragic notch — gap between tragus and antitragus
        mesh::ellipsoid_tris(tris, ear_x - s * 0.014, 1.745, ear_z - 0.010,
            0.004, 0.005, 0.004, 0, darken(sk, 0.60));
        // Antitragus — bump opposite the tragus
        mesh::ellipsoid_tris(tris, ear_x - s * 0.006, 1.74, ear_z - 0.006,
            0.007, 0.008, 0.006, 0, sk);

        // ── LOBE — soft pendulous teardrop at bottom ──
        mesh::ellipsoid_tris(tris, ear_x + s * 0.002, 1.72, ear_z - 0.004,
            0.010, 0.016, 0.010, 1, darken(sk, 0.97));
    }

    // ── AGE-DEPENDENT DETAILS ──
    if app.face_age >= 1 {
        for fi in 0..3 {
            let fy = 1.85 + fi as f32 * 0.012;
            push_box(tris, 0.0, fy, -0.18, 0.08, 0.002, 0.002, sk_sh);
        }
    }
    if app.face_age > 0 {
        for &side in &[-1.0f32, 1.0] {
            for wi in 0..3 {
                let wy = 1.77 + wi as f32 * 0.007 - 0.007;
                push_box(tris, side * 0.09, wy, -0.16, 0.010, 0.002, 0.002, sk_sh);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════
    // HAIR — skull-hugging lofted mesh + sideburns
    // ══════════════════════════════════════════════════════════════
    let hat = if let Some(jc) = is_job_hat {
        gen_job_hat(tris, jc);
        true
    } else {
        gen_hat(tris, app, hair)
    };

    let hair_dk = darken(hair, 0.90);
    if hat && app.hat_type != 6 {
        // Hair peeking out from under hat
        mesh::ellipsoid_tris(tris, 0.0, 1.70, 0.14, 0.14, 0.06, 0.10, 0, hair);
        for &side in &[-1.0f32, 1.0] {
            push_box(tris, side * 0.17, 1.74, -0.02, 0.015, 0.05, 0.03, hair);
            push_box(tris, side * 0.16, 1.76, 0.04, 0.02, 0.04, 0.08, hair);
        }
    } else if !hat {
        // Full hair — flush against scalp at hairline, dome volume above.
        // Ring centers track skull centers to eliminate gap.
        let ho = 0.020; // hair sits just outside skull surface
        let hn = 32;
        use std::f32::consts::TAU;
        let hair_med = darken(hair, 0.95);
        // Skull ring centers for reference:
        //   y=1.87: cz=0.07, rx=0.215, rz=0.26
        //   y=1.91: cz=0.08, rx=0.22,  rz=0.27
        //   y=1.95: cz=0.08, rx=0.22,  rz=0.27
        //   y=1.99: cz=0.08, rx=0.22,  rz=0.27
        //   y=2.03: cz=0.07, rx=0.20,  rz=0.24
        let hair_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
            // Hairline base — flush with skull surface (no gap)
            (1.86, body_ring(0.0, 0.07, 0.215, 0.26, &[], hn), hair_dk),
            // Just above hairline — minimal volume added, sits on scalp
            (1.89, body_ring(0.0, 0.07, 0.22 + ho, 0.27 + ho, &[
                (PI, 0.5, 0.015),         // back volume
            ], hn), hair_dk),
            // Hairline to crown transition — volume grows
            (1.92, body_ring(0.0, 0.08, 0.23 + ho, 0.28 + ho, &[
                (PI, 0.5, 0.020),
                (hp + 0.3, 0.3, 0.008),   // part asymmetry
            ], hn), hair_med),
            // Cranial vault — full volume with strand-group bumps
            (1.96, body_ring(0.0, 0.08, 0.24 + ho, 0.29 + ho, &[
                (PI, 0.5, 0.025),         // back volume
                (hp, 0.30, 0.014),        // side volume R — strand group
                (PI + hp, 0.30, 0.014),   // side volume L — strand group
                (0.4, 0.20, 0.008),       // front-right strand group
                (TAU - 0.6, 0.20, 0.007), // front-left strand group
            ], hn), hair),
            // Mid vault — maximum width, strand texture
            (2.00, body_ring(0.0, 0.08, 0.24 + ho, 0.30 + ho, &[
                (PI, 0.5, 0.022),
                (hp, 0.25, 0.012),        // strand groups shifting position
                (PI + hp, 0.25, 0.012),   // (flow direction effect)
                (0.8, 0.20, 0.008),
                (TAU - 1.0, 0.20, 0.007),
            ], hn), hair),
            // Upper vault — volume holds, gentle taper begins
            (2.04, body_ring(0.0, 0.07, 0.23 + ho, 0.28 + ho, &[
                (PI, 0.4, 0.018),
                (hp, 0.25, 0.010),
                (PI + hp, 0.25, 0.010),
                (1.2, 0.20, 0.006),       // crown swirl groups
                (TAU - 1.4, 0.20, 0.006),
            ], hn), hair),
            // Crown — still has volume
            (2.08, body_ring(0.0, 0.06, 0.20 + ho, 0.24 + ho, &[
                (PI, 0.3, 0.012),
                (hp, 0.25, 0.008),
                (PI + hp, 0.25, 0.008),
            ], hn), hair),
            (2.11, body_ring(0.0, 0.05, 0.17 + ho, 0.20 + ho, &[
                (PI, 0.3, 0.008),
            ], hn), hair_med),
            // Crown dome tip — rounded, not flat
            (2.14, body_ring(0.0, 0.04, 0.12 + ho * 0.5, 0.14 + ho * 0.5, &[], hn), hair_med),
            (2.17, body_ring(0.0, 0.03, 0.06, 0.07, &[], hn), hair_dk),
        ];
        mesh::loft_y_tris(tris, &hair_rings);

        // Hairline coverage — fills any remaining transition at forehead
        mesh::ellipsoid_tris(tris, 0.0, 1.87, -0.19, 0.22, 0.04, 0.06, 1, hair);
        // Temple coverage — wraps from forehead around sides to ears
        for &side in &[-1.0f32, 1.0] {
            mesh::ellipsoid_tris(tris, side * 0.19, 1.87, 0.00, 0.08, 0.05, 0.16, 1, hair);
            mesh::ellipsoid_tris(tris, side * 0.21, 1.79, 0.04, 0.06, 0.10, 0.14, 1, hair);
        }
        // Back of head / nape — natural taper, not hard edge
        mesh::ellipsoid_tris(tris, 0.0, 1.84, 0.26, 0.20, 0.14, 0.10, 1, hair_dk);
        // Nape wisps — tapered hair flow at neckline
        mesh::ellipsoid_tris(tris, 0.0, 1.78, 0.22, 0.14, 0.08, 0.06, 0, hair_dk);
        // Nape wisp — hair at the back of the neck
        mesh::ellipsoid_tris(tris, 0.0, 1.76, 0.20, 0.14, 0.06, 0.08, 0, hair_dk);
        // Sideburns
        for &side in &[-1.0f32, 1.0] {
            push_box(tris, side * 0.20, 1.74, -0.01, 0.012, 0.05, 0.04, hair);
        }
    }
}

/// Neck — ring-lofted with SCM muscles and larynx. Continuous head-to-neck topology.
fn gen_neck(tris: &mut Vec<WorldTri>, skin: u32, props: &BodyProportions) {
    use std::f32::consts::PI;
    let nt = props.neck_top;
    let m = props.muscle_def;
    let rx = props.neck_rx;
    let rz = props.neck_rz;
    let n = 24;

    // Neck from torso top (1.48) through to neck_top — monotonically increasing Y
    let rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
        // Base — wide, blends with torso shoulder transition
        (1.48, body_ring(0.0, 0.0, rx * 1.10, rz * 1.10, &[
            (PI, 0.5, 0.025 * m),             // trapezius insertion
            (PI * 0.5, 0.25, 0.012 * m),      // SCM origin (right, behind ear)
            (PI * 1.5, 0.25, 0.012 * m),      // SCM origin (left)
        ], n), skin),
        // Mid-neck — SCM prominent, laryngeal prominence
        (1.49, body_ring(0.0, 0.0, rx * 1.02, rz * 1.02, &[
            (PI * 0.5, 0.28, 0.018 * m),      // SCM right — diagonal cord
            (PI * 1.5, 0.28, 0.018 * m),      // SCM left
            (0.0, 0.12, if props.has_adams_apple { 0.014 } else { 0.005 }), // larynx
            (PI, 0.4, 0.015 * m),             // nuchal muscles
        ], n), skin),
        // Upper neck — tapers, SCM fading
        (nt, body_ring(0.0, 0.0, rx * 0.95, rz * 0.95, &[
            (PI * 0.5, 0.25, 0.010 * m),      // SCM insertion
            (PI * 1.5, 0.25, 0.010 * m),
            (0.0, 0.10, if props.has_adams_apple { 0.008 } else { 0.003 }),
        ], n), skin),
    ];

    mesh::loft_y_tris(tris, &rings);
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
    let knuckle_dk = darken(skin, 0.88);
    let nail_col = darken(skin, 1.06);

    // ── PALM — lofted shape: wider at knuckles, narrower at wrist ──
    // Back of hand (dorsal surface — slightly convex)
    mesh::ellipsoid_tris(tris, cx, cy + 0.006, cz - 0.01, 0.040, 0.008, 0.045, 0, skin);
    // Palm pad (palmar surface — fleshy)
    mesh::ellipsoid_tris(tris, cx, cy - 0.006, cz - 0.01, 0.038, 0.010, 0.042, 0, palm_dk);
    // Thenar eminence (thumb muscle pad — prominent mound)
    mesh::ellipsoid_tris(tris, cx + side * 0.020, cy - 0.004, cz + 0.005,
        0.018, 0.012, 0.022, 0, darken(skin, 0.92));
    // Hypothenar (pinky side — smaller)
    mesh::ellipsoid_tris(tris, cx - side * 0.018, cy - 0.004, cz - 0.005,
        0.012, 0.010, 0.018, 0, darken(skin, 0.92));

    // ── 4 FINGERS — tapered, 3 phalanges each ──
    // Finger proportions: index longest, ring nearly same, middle longest, pinky shortest
    let finger_lengths = [0.032f32, 0.038, 0.035, 0.028]; // index, middle, ring, pinky
    let finger_radii  = [0.007f32, 0.0075, 0.007, 0.006]; // base radius, tapers
    let finger_spread = [0.012f32, 0.004, -0.004, -0.013]; // X offset from center

    for fi in 0..4 {
        let fx = cx + side * finger_spread[fi];
        let fz_base = cz - 0.050; // knuckle line
        let flen = finger_lengths[fi];
        let fr = finger_radii[fi];

        // Proximal phalanx (thickest)
        mesh::ellipsoid_tris(tris, fx, cy + 0.002, fz_base - flen * 0.3,
            fr, fr * 0.9, flen * 0.45, 0, skin);
        // PIP joint
        mesh::sphere_tris(tris, fx, cy + 0.001, fz_base - flen * 0.55, fr * 0.85, 0, knuckle_dk);
        // Middle phalanx (thinner)
        mesh::ellipsoid_tris(tris, fx, cy + 0.001, fz_base - flen * 0.72,
            fr * 0.8, fr * 0.75, flen * 0.28, 0, skin);
        // DIP joint
        mesh::sphere_tris(tris, fx, cy, fz_base - flen * 0.85, fr * 0.7, 0, knuckle_dk);
        // Distal phalanx (thinnest — fingertip)
        mesh::ellipsoid_tris(tris, fx, cy, fz_base - flen * 0.95,
            fr * 0.65, fr * 0.6, flen * 0.14, 0, skin);
        // Fingernail
        push_box(tris, fx, cy + 0.005, fz_base - flen + 0.003,
            fr * 0.6, 0.002, fr * 0.8, nail_col);
    }

    // ── THUMB — offset at angle, opposable position ──
    let tx = cx + side * 0.035;
    let tz = cz + 0.005;
    // Metacarpal (angled outward and forward)
    mesh::ellipsoid_tris(tris, tx, cy - 0.003, tz - 0.008,
        0.010, 0.011, 0.018, 0, skin);
    // MCP joint
    mesh::sphere_tris(tris, tx + side * 0.005, cy - 0.004, tz - 0.025, 0.008, 0, knuckle_dk);
    // Proximal phalanx (angled further out)
    mesh::ellipsoid_tris(tris, tx + side * 0.008, cy - 0.005, tz - 0.038,
        0.009, 0.010, 0.016, 0, skin);
    // IP joint
    mesh::sphere_tris(tris, tx + side * 0.008, cy - 0.005, tz - 0.050, 0.007, 0, knuckle_dk);
    // Distal phalanx (tip)
    mesh::ellipsoid_tris(tris, tx + side * 0.008, cy - 0.004, tz - 0.058,
        0.007, 0.008, 0.010, 0, skin);
    // Thumbnail
    push_box(tris, tx + side * 0.008, cy + 0.002, tz - 0.063,
        0.006, 0.002, 0.007, nail_col);
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

    // ── BODY (neck, torso, limbs) — generate then stretch for heroic proportions ──
    let body_base = tris.len();
    gen_neck(tris, skin, &male_proportions());
    gen_torso(tris, app, vest_col, swing);
    gen_belt_system(tris, app, pants_col);

    let l_fwd = -swing * 0.40;
    let r_fwd = swing * 0.40;
    let l_knee = if swing > 0.0 { swing * 0.22 } else { 0.0 };
    let r_knee = if swing < 0.0 { (-swing) * 0.22 } else { 0.0 };
    gen_leg(tris, -1.0, l_fwd, l_knee, pants_col, app);
    gen_leg(tris, 1.0, r_fwd, r_knee, pants_col, app);

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

    // Stretch body vertically and widen
    for tri in &mut tris[body_base..] {
        for v in &mut tri.v {
            v[0] *= BODY_WIDEN;
            v[1] *= BODY_STRETCH;
            v[2] *= BODY_WIDEN;
        }
    }

    // ── HEAD (separate, uniformly scaled, positioned on stretched neck) ──
    let head_base = tris.len();
    gen_head(tris, app, is_job_hat);
    for tri in &mut tris[head_base..] {
        for v in &mut tri.v {
            v[0] *= HEAD_SCALE;
            v[1] = HEAD_CY + (v[1] - HEAD_CY) * HEAD_SCALE;
            v[2] *= HEAD_SCALE;
        }
    }
    // Head sits on spine at skull base (y≈1.55), NOT at the chin.
    // Chin hangs below, overlapping upper neck.
    let skull_base = HEAD_CY + (1.55 - HEAD_CY) * HEAD_SCALE;
    let head_shift = NECK_TOP * BODY_STRETCH - skull_base;
    for tri in &mut tris[head_base..] {
        for v in &mut tri.v {
            v[1] += head_shift;
        }
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
// ANATOMICALLY ACCURATE NUDE MALE BODY — lofted cross-section surfaces
// Reference: ZBrush sculpt + ACU bust. Single continuous mesh per body part
// instead of overlapping ellipsoids. Muscle contour built into cross-sections.
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a cross-section contour as N (x, z) points.
/// theta=0: front (-Z), PI/2: right (+X), PI: back (+Z), 3PI/2: left (-X).
/// Bumps: (center_angle, angular_width_sigma, radial_amplitude).
fn body_ring(cx: f32, cz: f32, rx: f32, rz: f32, bumps: &[(f32, f32, f32)], n: usize) -> Vec<[f32; 2]> {
    use std::f32::consts::{PI, TAU};
    let step = TAU / n as f32;
    (0..n).map(|i| {
        let theta = i as f32 * step;
        let st = theta.sin();
        let ct = theta.cos();
        let base_x = cx + rx * st;
        let base_z = cz - rz * ct;

        let mut dr = 0.0f32;
        for &(center, width, amp) in bumps {
            let mut diff = theta - center;
            if diff > PI { diff -= TAU; }
            if diff < -PI { diff += TAU; }
            dr += amp * (-0.5 * (diff / width).powi(2)).exp();
        }

        [base_x + dr * st, base_z - dr * ct]
    }).collect()
}

/// Nude male torso — single continuous lofted surface with V-taper.
/// Cross-section rings define the body contour at each height; muscle shape
/// is built into the ring profiles via Gaussian bumps. No overlapping ellipsoids.
fn gen_nude_torso(tris: &mut Vec<WorldTri>, skin: u32, props: &BodyProportions) {
    let sk = skin;
    let m = props.muscle_def;
    let sk_shadow = darken(sk, 1.0 - 0.03 * m);
    let sk_deep = darken(sk, 1.0 - 0.07 * m);
    let nipple_col = darken(sk, 0.78);

    use std::f32::consts::PI;
    let hp = PI * 0.5;
    let n = 32;

    // Scale factors for each torso zone (1.0 for male baseline)
    let sh = props.hip_rx / 0.18;       // hip scale
    let sw = props.waist_rx / 0.15;     // waist scale
    let sc = props.chest_rx / 0.22;     // chest scale
    let ss = props.shoulder_rx / 0.32;  // shoulder scale

    // Interpolate scale factor by Y height
    let s = |y: f32| -> f32 {
        if y <= 0.92 { sh }
        else if y <= 1.00 { sh + (sw - sh) * (y - 0.92) / 0.08 }
        else if y <= 1.24 { sw + (sc - sw) * (y - 1.00) / 0.24 }
        else if y <= 1.42 { sc + (ss - sc) * (y - 1.24) / 0.18 }
        else { ss }
    };

    // Breast bump profile: given a Y height, compute the breast bump amplitude
    // Uses Gaussian falloff centered at breast_y with wide spread for natural shape
    let breast_bump = |y: f32| -> f32 {
        if !props.has_breasts { return 0.0; }
        let by = props.breast_y; // center of breast volume (~1.26)
        let dy = (y - by) / 0.10; // wider spread for softer transition
        let amp = props.breast_rz * 1.3; // moderate forward projection (less pointy)
        amp * (-0.5 * dy * dy).exp()
    };

    // Helper: scale rx/rz by zone factor, bump amps by muscle_def
    // Adds breast bumps as structural (not scaled by muscle_def)
    let sr = |y: f32, rx: f32, rz: f32, bumps: &[(f32, f32, f32)]| -> (f32, f32, Vec<(f32, f32, f32)>) {
        let sf = s(y);
        let mut sb: Vec<(f32, f32, f32)> = bumps.iter()
            .map(|&(a, w, amp)| (a, w, amp * m))
            .collect();
        // Add breast bumps — wide angular spread for natural rounded shape
        let bb = breast_bump(y);
        if bb > 0.001 {
            sb.push((0.30, 0.55, bb));   // right breast (very wide, soft curve)
            sb.push((-0.30, 0.55, bb));  // left breast
        }
        (rx * sf, rz * sf, sb)
    };

    // Build each ring: structural bumps use s(y), muscle bumps use m
    let ring = |y: f32, rx: f32, rz: f32, bumps: &[(f32, f32, f32)]| -> (f32, Vec<[f32; 2]>, u32) {
        let (rx2, rz2, sb) = sr(y, rx, rz, bumps);
        (y, body_ring(0.0, 0.0, rx2, rz2, &sb, n), sk)
    };

    // ── LOFTED BODY SURFACE — continuous mesh from crotch to neck ──
    let rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
        (0.82, body_ring(0.0, 0.0, 0.04, 0.04, &[], n), sk),

        ring(0.86, 0.10, 0.08, &[
            (PI, 0.5, 0.025),
        ]),
        ring(0.89, 0.14, 0.11, &[
            (PI, 0.5, 0.04),
        ]),
        // Pelvis — hip width, glute bulge
        {
            let sf = s(0.92);
            (0.92, body_ring(0.0, 0.0, 0.18 * sf, 0.14 * sf, &[
                (hp, 0.35, 0.025 * sf),           // hip bone (structural — ASIS)
                (PI + hp, 0.35, 0.025 * sf),
                (PI, 0.5, 0.06 * sf),             // glute (structural)
                (PI - 0.4, 0.3, 0.03 * m),        // glute lobe (muscle)
                (PI + 0.4, 0.3, 0.03 * m),
            ], n), sk)
        },
        // Upper pelvis / V-line / iliac crest
        {
            let sf = s(0.96);
            (0.96, body_ring(0.0, 0.0, 0.17 * sf, 0.13 * sf, &[
                (hp, 0.35, 0.02 * sf),
                (PI + hp, 0.35, 0.02 * sf),
                (PI, 0.4, 0.04 * sf),
                (0.4, 0.3, -0.012 * m),           // V-line indent
                (-0.4, 0.3, -0.012 * m),
            ], n), sk)
        },
        // Waist — narrowest point
        ring(1.00, 0.15, 0.13, &[
            (0.8, 0.3, 0.022),      // oblique
            (-0.8, 0.3, 0.022),
            (PI, 0.5, 0.04),        // lumbar curve
            (PI - 0.3, 0.2, 0.015), // erector spinae
            (PI + 0.3, 0.2, 0.015),
        ]),
        // Lower abs / navel level
        ring(1.04, 0.155, 0.14, &[
            (0.0, 0.4, 0.025),     // rectus abdominis
            (0.3, 0.15, -0.008),   // linea alba indent L
            (-0.3, 0.15, -0.008),  // linea alba indent R
            (0.8, 0.3, 0.025),     // oblique
            (-0.8, 0.3, 0.025),
            (PI, 0.5, 0.035),
            (PI - 0.3, 0.2, 0.015),
            (PI + 0.3, 0.2, 0.015),
        ]),
        // Mid-abs
        ring(1.08, 0.17, 0.15, &[
            (0.0, 0.35, 0.03),     // abs (upper row)
            (0.3, 0.12, -0.010),   // tendinous intersection
            (-0.3, 0.12, -0.010),
            (0.8, 0.3, 0.025),     // oblique
            (-0.8, 0.3, 0.025),
            (PI, 0.5, 0.04),
            (PI - 0.3, 0.2, 0.015),
            (PI + 0.3, 0.2, 0.015),
        ]),
        // Lower ribs / serratus
        ring(1.12, 0.19, 0.16, &[
            (0.0, 0.4, 0.02),      // upper abs
            (0.3, 0.12, -0.008),   // tendinous intersection
            (-0.3, 0.12, -0.008),
            (0.7, 0.2, 0.02),      // serratus anterior
            (-0.7, 0.2, 0.02),
            (hp, 0.35, 0.035),     // lateral body wall (serratus wrapping)
            (PI + hp, 0.35, 0.035),
            (PI - 0.6, 0.35, 0.035), // lat emerging
            (PI + 0.6, 0.35, 0.035),
            (PI, 0.4, 0.04),
            (PI - 0.25, 0.15, 0.012),
            (PI + 0.25, 0.15, 0.012),
        ]),
        // Lower pec line / serratus peak
        ring(1.16, 0.20, 0.17, &[
            (0.4, 0.35, 0.035),    // pec lower shelf (sternal head)
            (-0.4, 0.35, 0.035),
            (0.0, 0.10, -0.015),   // sternal groove — deeper, narrower
            (0.7, 0.2, 0.025),     // serratus anterior
            (-0.7, 0.2, 0.025),
            (hp, 0.35, 0.040),     // lateral body wall
            (PI + hp, 0.35, 0.040),
            (PI - 0.5, 0.35, 0.040), // lat (growing)
            (PI + 0.5, 0.35, 0.040),
            (PI, 0.15, -0.008),    // spine groove
            (PI - 0.25, 0.15, 0.012), // erector spinae
            (PI + 0.25, 0.15, 0.012),
        ]),
        // Mid pec — clavicular head distinct from sternal head
        ring(1.21, 0.21, 0.20, &[
            (0.4, 0.35, 0.038),    // pec sternal head bulk
            (-0.4, 0.35, 0.038),
            (0.0, 0.10, -0.015),   // sternal groove — linear depression
            (hp, 0.35, 0.040),     // lateral body wall — pec wrapping
            (PI + hp, 0.35, 0.040),
            (PI - 0.5, 0.3, 0.030), // lat (peak)
            (PI + 0.5, 0.3, 0.030),
            (PI, 0.15, -0.008),    // spine groove
            (PI - 0.25, 0.12, 0.012), // erector spinae
            (PI + 0.25, 0.12, 0.012),
        ]),
        // Pec shelf / chest peak
        ring(1.26, 0.22, 0.21, &[
            (0.4, 0.40, 0.038),    // pec peak
            (-0.4, 0.40, 0.038),
            (0.0, 0.10, -0.015),   // sternal groove
            (hp, 0.35, 0.040),     // lateral body wall — pec to arm
            (PI + hp, 0.35, 0.040),
            (PI - 0.5, 0.3, 0.025), // lat insertion
            (PI + 0.5, 0.3, 0.025),
            (PI, 0.15, -0.008),    // spine groove
            (PI - 0.6, 0.25, 0.018), // scapular medial border
            (PI + 0.6, 0.25, 0.018),
        ]),
        // Upper pec → clavicular head
        ring(1.32, 0.22, 0.19, &[
            (0.35, 0.35, 0.028),   // clavicular pec head — raised upper pec
            (-0.35, 0.35, 0.028),
            (0.0, 0.10, -0.010),   // sternal notch approach
            (hp, 0.30, 0.035),     // lateral body wall — narrowing toward shoulder
            (PI + hp, 0.30, 0.035),
            (PI, 0.15, -0.008),    // spine groove
            (PI - 0.5, 0.25, 0.022), // rhomboid / scapular
            (PI + 0.5, 0.25, 0.022),
        ]),
        // Shoulder approach — deltoid begins as distinct lateral mass
        // Base rx = ribcage width; narrow deltoid bumps (width 0.35) carry shoulder width.
        // Total target: ~0.28 per side for male (0.56m bi-deltoid)
        {
            let sf = s(1.36);
            let da = props.shoulder_deltoid_amp;
            (1.36, body_ring(0.0, 0.0, 0.20 * sf, 0.19 * sf, &[
                (hp, 0.35, da + 0.02 * sf),   // deltoid emerging (narrow bump)
                (PI + hp, 0.35, da + 0.02 * sf),
                (0.5, 0.3, 0.020 * m),    // anterior deltoid
                (-0.5, 0.3, 0.020 * m),
                (PI - 0.5, 0.3, 0.025 * m), // posterior deltoid
                (PI + 0.5, 0.3, 0.025 * m),
                (PI, 0.5, 0.05 * sf),     // upper back mass
                (0.3, 0.15, 0.010 * sf),  // clavicle ridge (anterior)
                (-0.3, 0.15, 0.010 * sf),
            ], n), sk)
        },
        // Mid-shoulder — deltoid growing
        {
            let sf = s(1.39);
            let da = props.shoulder_deltoid_amp;
            (1.39, body_ring(0.0, 0.0, 0.18 * sf, 0.18 * sf, &[
                (hp, 0.35, da + 0.04 * sf),
                (PI + hp, 0.35, da + 0.04 * sf),
                (0.5, 0.3, 0.015 * m),
                (-0.5, 0.3, 0.015 * m),
                (PI - 0.5, 0.3, 0.025 * m),
                (PI + 0.5, 0.3, 0.025 * m),
                (PI, 0.5, 0.055 * sf),     // trapezius
            ], n), sk)
        },
        // Shoulder peak — deltoid cap at maximum, ribcage narrowing
        // Front/back trapezius: rx=0.16 to neck_rx=0.10 → ~45° slope
        {
            let sf = s(1.42);
            let da = props.shoulder_deltoid_amp;
            (1.42, body_ring(0.0, 0.0, 0.16 * sf, 0.17 * sf, &[
                (hp, 0.35, da + 0.05 * sf),   // deltoid cap — concentrated lateral
                (PI + hp, 0.35, da + 0.05 * sf),
                (0.0, 0.4, 0.015 * sf),   // pec projection (front)
                (PI, 0.5, 0.055 * sf),     // trapezius
            ], n), sk)
        },
        // Shoulder-to-neck — deltoid fading, trapezius slope
        {
            let sf = s(1.43);
            let da = props.shoulder_deltoid_amp;
            (1.43, body_ring(0.0, 0.0, 0.14 * sf, 0.15 * sf, &[
                (hp, 0.30, da * 0.3 + 0.02 * sf), (PI + hp, 0.30, da * 0.3 + 0.02 * sf),
                (PI, 0.5, 0.050 * sf),     // trapezius
            ], n), sk)
        },
        // Trapezius mid-slope
        {
            let sf = s(1.44);
            (1.44, body_ring(0.0, 0.0, 0.13 * sf, 0.13 * sf, &[
                (hp, 0.25, 0.010 * sf), (PI + hp, 0.25, 0.010 * sf),
                (PI, 0.5, 0.045 * sf),     // trapezius slope
            ], n), sk)
        },
        // Upper trapezius — approaching neck
        {
            let sf = s(1.45);
            (1.45, body_ring(0.0, 0.0, 0.12 * sf, 0.11 * sf, &[
                (PI, 0.5, 0.040 * sf),     // upper trap
            ], n), sk)
        },
        // Neck base — trapezius wraps around
        {
            let sf = s(1.46);
            (1.46, body_ring(0.0, 0.0, 0.12 * sf, 0.10 * sf, &[
                (PI, 0.5, 0.04 * sf),      // upper trap / nuchal
                (PI - 0.4, 0.3, 0.012 * sf),
                (PI + 0.4, 0.3, 0.012 * sf),
            ], n), sk)
        },
        // Neck transition
        (1.48, body_ring(0.0, 0.0, props.neck_rx, props.neck_rz, &[
            (PI, 0.5, 0.035 * m),      // trap insertion
        ], n), sk),
    ];

    mesh::loft_y_tris(tris, &rings);

    // ── MINIMAL SURFACE DETAIL ──
    // Nipples — lower-outer quadrant of each pectoral, slightly recessed
    let nip_y = if props.has_breasts { 1.23 } else { 1.20 };
    for &side in &[-1.0f32, 1.0] {
        let nx = side * 0.12 * sc; // outer placement
        mesh::sphere_tris(tris, nx, nip_y, -0.22 * sc, 0.004, 0, nipple_col);
        // Areola — subtle ring around nipple
        mesh::ellipsoid_tris(tris, nx, nip_y, -0.215 * sc, 0.012, 0.012, 0.003, 0, darken(nipple_col, 0.95));
    }
    // Navel (tiny indent)
    mesh::sphere_tris(tris, 0.0, 1.02, -0.17 * sw, 0.008, 0, sk_deep);
}

/// Gluteal muscles — now just a subtle cleft line, bulk is in the torso ring profile
fn gen_glutes(tris: &mut Vec<WorldTri>, skin: u32, props: &BodyProportions) {
    let _hip_s = props.hip_rx / 0.18;
    // No overlaid geometry — glute shape is built into the torso loft's lower rings
}

/// Nude arm — single continuous loft from shoulder to wrist, no sphere joints
fn gen_nude_arm(
    tris: &mut Vec<WorldTri>, side: f32, fwd: f32, bend: f32, skin: u32,
    props: &BodyProportions,
) {
    let sk = skin;
    let a = props.arm_rx_scale;
    let m = props.muscle_def;

    use std::f32::consts::PI;
    let hp = PI * 0.5;
    let n = 24;

    // ── JOINT POSITIONS ──
    let shoulder = [side * props.shoulder_joint_x, 1.42, 0.0];
    let elbow = [side * (props.shoulder_joint_x + 0.10), 1.06, fwd * 0.35];
    let wrist = [side * (props.shoulder_joint_x + 0.06), 0.80, fwd * 0.15 - bend];

    // ── SINGLE CONTINUOUS ARM LOFT (shoulder → elbow → wrist) ──
    // Upper arm rings are wide at shoulder to overlap with torso volume.
    // Moderate inward bumps at PI+hp push arm inner surface into torso.
    // Top cap seals arm from above. Z-buffer resolves overlap with torso.
    let arm_heights: Vec<(f32, f32, f32, Vec<(f32, f32, f32)>)> = vec![
        // Shoulder cap — arm center at shoulder_joint_x=0.22
        // Outer edge ≈ 0.22+rx+bump ≈ 0.34, protrudes ~0.06 past torso shoulder
        (1.44, 0.11, 0.12, vec![
            (hp, 0.35, 0.030 * m),             // deltoid cap lateral
            (PI + hp, 0.35, 0.035),            // inward overlap (structural)
        ]),
        (1.42, 0.10, 0.10, vec![
            (hp, 0.35, 0.030 * m),             // deltoid cap
            (0.5, 0.30, 0.020 * m),            // anterior delt
            (PI - 0.5, 0.30, 0.016 * m),       // posterior delt
            (PI + hp, 0.35, 0.035),            // inward overlap
        ]),
        (1.39, 0.092, 0.088, vec![
            (hp, 0.35, 0.032 * m),             // deltoid peak
            (0.5, 0.35, 0.022 * m),            // anterior deltoid
            (PI - 0.5, 0.35, 0.018 * m),       // posterior deltoid
            (PI + hp, 0.40, 0.040),            // inward overlap
        ]),
        (1.36, 0.085, 0.080, vec![
            (hp, 0.35, 0.025 * m),             // deltoid insertion
            (0.5, 0.30, 0.016 * m),            // anterior tail
            (PI - 0.5, 0.30, 0.012 * m),       // posterior tail
            (PI + hp, 0.45, 0.045),            // pec wrap inward
        ]),
        (1.32, 0.080, 0.074, vec![
            (hp, 0.20, -0.005 * m),            // deltoid-bicep groove (lateral)
            (0.0, 0.35, 0.018 * m),            // bicep emerging
            (PI, 0.4, 0.015 * m),              // tricep long head
            (PI + hp, 0.45, 0.048),            // pec/lat wrap inward
        ]),
        (1.26, 0.076, 0.070, vec![
            (0.0, 0.4, 0.026 * m),             // bicep — stronger peak
            (PI, 0.45, 0.020 * m),             // tricep
            (hp, 0.25, 0.013 * m),             // brachialis
            (PI + hp, 0.45, 0.045),            // pec/lat wrap
        ]),
        (1.20, 0.074, 0.068, vec![
            (0.0, 0.4, 0.030 * m),             // bicep peak — strongest
            (PI, 0.50, 0.026 * m),             // tricep — horseshoe shape
            (PI - 0.4, 0.25, 0.013 * m),       // tricep lateral head
            (PI + 0.4, 0.25, 0.013 * m),       // tricep medial head
            (hp, 0.25, 0.015 * m),             // brachialis
            (PI + hp, 0.45, 0.040),            // inner overlap
        ]),
        (1.14, 0.064, 0.058, vec![
            (0.0, 0.35, 0.020 * m),            // bicep taper
            (PI, 0.45, 0.018 * m),             // tricep taper
            (PI - 0.4, 0.20, 0.009 * m),       // tricep lateral
            (PI + 0.4, 0.20, 0.009 * m),       // tricep medial
            (PI + hp, 0.40, 0.028),            // inner taper
        ]),
        // Elbow — olecranon point + narrowest transition
        (1.08, 0.055, 0.052, vec![
            (PI, 0.20, 0.014 * m),             // olecranon — bony protrusion
        ]),
        (1.06, 0.052, 0.050, vec![
            (PI, 0.18, 0.012 * m),             // olecranon point
            (0.0, 0.25, -0.004),               // cubital fossa (front concavity)
        ]),
        (1.04, 0.055, 0.052, vec![]),
        // Forearm — widens for muscle belly, tapers, cross-section flattens toward wrist
        (1.00, 0.058, 0.054, vec![
            (hp - 0.3, 0.35, 0.024 * m),       // brachioradialis (outer bulge)
            (PI + hp + 0.3, 0.3, 0.018 * m),   // flexor group
            (hp + 0.3, 0.3, 0.016 * m),        // extensor group
            (PI + hp, 0.15, 0.006),            // ulnar border (bony ridge)
        ]),
        (0.95, 0.054, 0.050, vec![
            (hp - 0.3, 0.35, 0.022 * m),       // brachioradialis taper
            (PI + hp + 0.3, 0.3, 0.014 * m),   // flexor group
            (PI + hp, 0.12, 0.006),            // ulnar border
        ]),
        (0.90, 0.048, 0.044, vec![
            (hp, 0.3, 0.012 * m),              // extensor carpi
            (PI + hp, 0.20, 0.008),            // ulnar border (more prominent)
        ]),
        (0.84, 0.044, 0.036, vec![             // flatter cross-section toward wrist
            (PI + hp, 0.15, 0.006),            // ulnar border
        ]),
        // Wrist — oval, flatter than circular (radius/ulna crossing)
        (0.80, 0.038, 0.030, vec![]),
    ];

    let shoulder_y = 1.44; // top of arm loft (overlaps torso)
    let elbow_y = 1.06;
    let wrist_y = 0.80;
    let arm_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = arm_heights.iter().map(|&(ref y, rx, rz, ref bumps)| {
        // Interpolate center position along shoulder→elbow→wrist path
        // Upper arm uses cubic ease-in: arm center stays near body even longer,
        // moves outward quickly only near elbow. Maximizes torso overlap.
        let (cx, cz) = if *y >= elbow_y {
            let t_lin = ((shoulder_y - *y) / (shoulder_y - elbow_y)).clamp(0.0, 1.0);
            let t = t_lin * t_lin * t_lin; // cubic — stays near shoulder even longer
            (shoulder[0] * (1.0 - t) + elbow[0] * t, shoulder[2] * (1.0 - t) + elbow[2] * t)
        } else {
            let t = (elbow_y - *y) / (elbow_y - wrist_y);
            (elbow[0] * (1.0 - t) + wrist[0] * t, elbow[2] * (1.0 - t) + wrist[2] * t)
        };
        (*y, limb_ring(cx, cz, rx * a, rz * a, side, bumps, n), sk)
    }).collect();

    mesh::loft_y_tris(tris, &arm_rings);

    // ── HAND ──
    gen_hand(tris, wrist[0], wrist[1] - 0.05, wrist[2] - 0.02, side, sk);
}

/// Bare foot with proper arch, heel, ball, and toes
fn gen_bare_foot(tris: &mut Vec<WorldTri>, ankle: [f32; 3], side: f32, skin: u32) {
    let sk = skin;
    let sk_dk = darken(sk, 0.96);
    let nail_col = darken(sk, 1.06);
    let lx = ankle[0];
    let az = ankle[2];

    // ── HEEL — calcaneus, rounded posterior ──
    mesh::ellipsoid_tris(tris, lx, 0.028, az + 0.035, 0.035, 0.028, 0.035, 0, sk);
    // Achilles insertion (slight bump at back)
    mesh::ellipsoid_tris(tris, lx, 0.045, az + 0.04, 0.020, 0.015, 0.018, 0, sk_dk);

    // ── MIDFOOT — arch structure ──
    // Dorsum (top of foot — convex ridge)
    mesh::ellipsoid_tris(tris, lx, 0.042, az - 0.02, 0.038, 0.018, 0.055, 0, sk);
    // Lateral border (outer edge — touches ground)
    mesh::ellipsoid_tris(tris, lx + side * 0.020, 0.015, az - 0.01, 0.020, 0.015, 0.050, 0, sk);
    // Medial arch (inner — doesn't touch ground, concave underneath)
    mesh::ellipsoid_tris(tris, lx - side * 0.015, 0.030, az - 0.005, 0.018, 0.020, 0.045, 0, sk);

    // ── FOREFOOT — metatarsal heads (ball of foot) ──
    // Ball of foot — wide transverse arch
    mesh::ellipsoid_tris(tris, lx, 0.018, az - 0.065, 0.042, 0.016, 0.025, 0, sk);
    // 1st metatarsal head (big toe side — prominent)
    mesh::ellipsoid_tris(tris, lx - side * 0.020, 0.015, az - 0.068, 0.016, 0.013, 0.016, 0, sk_dk);
    // 5th metatarsal head (pinky side)
    mesh::ellipsoid_tris(tris, lx + side * 0.025, 0.013, az - 0.060, 0.012, 0.010, 0.014, 0, sk_dk);

    // ── EXTENSOR TENDONS (top of foot, subtle ridges) ──
    for ti in 0..4 {
        let tx = lx + (ti as f32 - 1.5) * side * 0.010;
        mesh::ellipsoid_tris(tris, tx, 0.048, az - 0.030, 0.003, 0.004, 0.035, 0, darken(sk, 0.97));
    }

    // ── TOES — hallux (big toe) + 4 lesser toes ──
    // Big toe — 2 phalanges, wider and thicker
    let btx = lx - side * 0.022;
    // Proximal phalanx
    mesh::ellipsoid_tris(tris, btx, 0.013, az - 0.088, 0.014, 0.011, 0.018, 0, sk);
    // Distal phalanx
    mesh::ellipsoid_tris(tris, btx, 0.012, az - 0.108, 0.012, 0.010, 0.014, 0, sk);
    // Toenail
    push_box(tris, btx, 0.020, az - 0.118, 0.008, 0.003, 0.006, nail_col);

    // 4 lesser toes — progressively shorter and thinner
    for ti in 0..4 {
        let tx = lx - side * 0.008 + (ti as f32 + 0.5) * side * 0.012;
        let toe_len = 0.014 - ti as f32 * 0.002;
        let toe_r = 0.008 - ti as f32 * 0.001;
        let tz = az - 0.082 + ti as f32 * 0.004; // each toe slightly shorter reach
        // Single phalanx (small toes read as one unit at game scale)
        mesh::ellipsoid_tris(tris, tx, 0.010, tz - toe_len * 0.5, toe_r, 0.006, toe_len, 0, sk);
        // Toenail
        push_box(tris, tx, 0.015, tz - toe_len + 0.002, 0.005, 0.002, 0.004, nail_col);
    }

    // ── SOLE — flat pad for ground contact ──
    mesh::ellipsoid_tris(tris, lx, 0.004, az - 0.02, 0.038, 0.004, 0.065, 0, sk_dk);
}

/// Generate a limb cross-section ring, mirroring bumps for left-side limbs.
/// For right limb (side>0), bumps are used as-is. For left (side<0), lateral bumps are mirrored.
fn limb_ring(cx: f32, cz: f32, rx: f32, rz: f32, side: f32, bumps: &[(f32, f32, f32)], n: usize) -> Vec<[f32; 2]> {
    use std::f32::consts::TAU;
    if side < 0.0 {
        let mirrored: Vec<(f32, f32, f32)> = bumps.iter()
            .map(|&(c, w, a)| (TAU - c, w, a))
            .collect();
        body_ring(cx, cz, rx, rz, &mirrored, n)
    } else {
        body_ring(cx, cz, rx, rz, bumps, n)
    }
}

/// Nude leg — single continuous loft from hip to ankle, no sphere joints
fn gen_nude_leg(
    tris: &mut Vec<WorldTri>, side: f32, fwd: f32, knee_bend: f32, skin: u32,
    props: &BodyProportions,
) {
    let sk = skin;
    let l = props.leg_rx_scale;
    let m = props.muscle_def;

    use std::f32::consts::PI;
    let hp = PI * 0.5;
    let n = 24;

    let lx = side * props.hip_joint_x;
    let hip = [lx, 0.92, 0.0];
    let knee = [lx, 0.48, fwd * 0.5];
    let ankle = [lx, 0.08, fwd * 0.25 - knee_bend * 0.4];

    // ── SINGLE CONTINUOUS LEG LOFT (hip → knee → ankle) ──
    let leg_heights: Vec<(f32, f32, f32, Vec<(f32, f32, f32)>)> = vec![
        // Hip top — wide to overlap with torso
        (0.92, 0.090, 0.082, vec![
            (PI, 0.5, 0.015 * m),              // glute transition
        ]),
        (0.88, 0.094, 0.084, vec![
            (PI, 0.5, 0.020 * m),              // glute-ham
            (PI + hp, 0.4, 0.016 * m),         // adductors
        ]),
        (0.84, 0.096, 0.086, vec![
            (0.0, 0.4, 0.022 * m),             // quads
            (PI, 0.5, 0.025 * m),              // hamstrings
            (PI + hp, 0.4, 0.020 * m),         // adductors
        ]),
        (0.78, 0.098, 0.088, vec![
            (0.0, 0.4, 0.028 * m),             // quads peak
            (hp - 0.3, 0.35, 0.020 * m),       // VL
            (PI, 0.5, 0.028 * m),              // hamstrings
            (PI + hp, 0.4, 0.022 * m),         // adductors
        ]),
        (0.70, 0.094, 0.084, vec![
            (0.0, 0.4, 0.030 * m),             // rectus femoris (peak)
            (hp - 0.3, 0.35, 0.024 * m),       // vastus lateralis
            (PI + hp + 0.3, 0.3, 0.018 * m),   // vastus medialis
            (PI, 0.5, 0.028 * m),              // hamstrings (peak)
            (PI + hp, 0.4, 0.020 * m),         // adductors
        ]),
        (0.62, 0.084, 0.076, vec![
            (0.0, 0.4, 0.028 * m),             // rectus femoris
            (hp - 0.3, 0.35, 0.020 * m),       // vastus lateralis
            (PI + hp + 0.3, 0.3, 0.016 * m),   // vastus medialis
            (PI, 0.5, 0.024 * m),              // hamstrings
        ]),
        (0.54, 0.070, 0.064, vec![
            (0.0, 0.35, 0.022 * m),            // lower quad
            (PI + hp + 0.3, 0.3, 0.014 * m),   // VM teardrop
        ]),
        // Knee — patella on front, popliteal fossa on back
        (0.50, 0.058, 0.055, vec![
            (0.0, 0.20, 0.018 * m),            // patella — oval raised form
            (PI, 0.25, -0.006 * m),            // popliteal fossa (concavity)
        ]),
        (0.48, 0.055, 0.052, vec![
            (0.0, 0.18, 0.016 * m),            // patella
            (PI, 0.20, -0.005 * m),            // popliteal fossa
            (PI - 0.4, 0.15, 0.006 * m),       // hamstring tendon (medial)
            (PI + 0.4, 0.15, 0.006 * m),       // hamstring tendon (lateral)
        ]),
        (0.46, 0.058, 0.055, vec![
            (0.0, 0.15, 0.006),                // tibial tuberosity
        ]),
        // Calf — gastrocnemius heart shape, tibial ridge on front
        (0.42, 0.060, 0.056, vec![
            (PI, 0.5, 0.020 * m),              // soleus
            (0.0, 0.10, 0.006),                // tibial ridge (shin bone)
        ]),
        (0.36, 0.064, 0.058, vec![
            (PI - 0.3, 0.30, 0.028 * m),       // gastrocnemius medial (larger, lower)
            (PI + 0.3, 0.30, 0.020 * m),       // gastrocnemius lateral (smaller)
            (0.0, 0.30, 0.016 * m),            // tibialis anterior
            (0.0, 0.08, 0.006),                // tibial ridge
        ]),
        (0.30, 0.060, 0.054, vec![
            (PI - 0.3, 0.30, 0.030 * m),       // gastrocnemius medial (peak)
            (PI + 0.3, 0.30, 0.024 * m),       // gastrocnemius lateral
            (0.0, 0.30, 0.018 * m),            // tibialis anterior (peak)
            (0.0, 0.08, 0.006),                // tibial ridge
        ]),
        (0.22, 0.048, 0.044, vec![
            (PI, 0.4, 0.020 * m),              // soleus taper
            (0.0, 0.25, 0.012 * m),            // tibialis anterior
            (0.0, 0.08, 0.005),                // tibial ridge
        ]),
        (0.14, 0.040, 0.036, vec![
            (PI, 0.15, 0.012),                 // Achilles tendon — visible ridge
            (0.0, 0.08, 0.004),                // tibial ridge (fading)
        ]),
        // Ankle — malleoli (ankle bones) + Achilles
        (0.08, 0.034, 0.032, vec![
            (hp, 0.15, 0.006),                 // lateral malleolus
            (PI + hp, 0.15, 0.008),            // medial malleolus (slightly larger)
            (PI, 0.12, 0.008),                 // Achilles insertion
        ]),
    ];

    let hip_y = 0.92;
    let knee_y = 0.48;
    let ankle_y = 0.08;
    let knee_cz = fwd * 0.5;
    let ankle_cz = fwd * 0.25 - knee_bend * 0.4;

    let leg_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = leg_heights.iter().map(|&(ref y, rx, rz, ref bumps)| {
        let (cz,) = if *y >= knee_y {
            let t = (hip_y - *y) / (hip_y - knee_y);
            (knee_cz * t,)
        } else {
            let t = (knee_y - *y) / (knee_y - ankle_y);
            (knee_cz * (1.0 - t) + ankle_cz * t,)
        };
        (*y, limb_ring(lx, cz, rx * l, rz * l, side, bumps, n), sk)
    }).collect();

    mesh::loft_y_tris(tris, &leg_rings);

    // ── BARE FOOT ──
    gen_bare_foot(tris, ankle, side, sk);
}

/// Nude attack arm — single continuous loft, no sphere joints
fn gen_nude_attack_arm(tris: &mut Vec<WorldTri>, side: f32, extend: f32, skin: u32, props: &BodyProportions) {
    let sk = skin;
    let a = props.arm_rx_scale;
    let m = props.muscle_def;

    use std::f32::consts::PI;
    let hp = PI * 0.5;
    let n = 24;

    let sx = props.shoulder_joint_x;
    let shoulder = [side * sx, 1.42, 0.0];
    let elbow = [side * (sx + 0.10), 1.10, -0.15 - extend * 0.20];
    let wrist = [side * (sx + 0.06), 0.92, -0.35 - extend * 0.35];

    // ── SINGLE CONTINUOUS ATTACK ARM LOFT ──
    let arm_heights: Vec<(f32, f32, f32, Vec<(f32, f32, f32)>)> = vec![
        (1.42, 0.10, 0.10, vec![
            (hp, 0.35, 0.035 * m), (0.5, 0.30, 0.022 * m), (PI - 0.5, 0.30, 0.018 * m),
            (PI + hp, 0.35, 0.035),            // inward overlap
        ]),
        (1.38, 0.092, 0.088, vec![
            (hp, 0.35, 0.038 * m), (0.5, 0.30, 0.024 * m), (PI - 0.5, 0.30, 0.020 * m),
            (PI + hp, 0.40, 0.040),            // pec/lat wrap
        ]),
        (1.32, 0.080, 0.074, vec![
            (hp, 0.30, 0.022 * m),
            (PI + hp, 0.45, 0.045),            // pec/lat wrap
        ]),
        (1.28, 0.076, 0.070, vec![
            (0.0, 0.4, 0.026 * m), (PI, 0.45, 0.020 * m),
            (PI + hp, 0.45, 0.045),            // pec/lat wrap
        ]),
        (1.24, 0.074, 0.068, vec![
            (0.0, 0.4, 0.030 * m), (PI, 0.45, 0.024 * m),
            (PI + hp, 0.40, 0.038),            // inner taper
        ]),
        (1.18, 0.058, 0.054, vec![
            (0.0, 0.35, 0.016 * m), (PI, 0.4, 0.014 * m),
            (PI + hp, 0.35, 0.028),            // inner taper
        ]),
        // Elbow — continuous through
        (1.12, 0.048, 0.046, vec![(PI, 0.3, 0.008 * m)]),
        (1.10, 0.046, 0.044, vec![]),
        (1.08, 0.048, 0.046, vec![]),
        // Forearm
        (1.02, 0.052, 0.048, vec![
            (hp - 0.3, 0.35, 0.016 * m), (PI + hp + 0.3, 0.3, 0.012 * m),
        ]),
        (0.96, 0.044, 0.040, vec![
            (hp - 0.3, 0.35, 0.010 * m),
        ]),
        (0.92, 0.036, 0.034, vec![]),
    ];

    let shoulder_y = 1.42;
    let elbow_y = 1.10;
    let wrist_y = 0.92;
    let arm_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = arm_heights.iter().map(|&(ref y, rx, rz, ref bumps)| {
        let (cx, cz) = if *y >= elbow_y {
            let t_lin = (shoulder_y - *y) / (shoulder_y - elbow_y);
            let t = t_lin * t_lin * t_lin; // cubic — stays near shoulder longer
            (shoulder[0] * (1.0 - t) + elbow[0] * t, shoulder[2] * (1.0 - t) + elbow[2] * t)
        } else {
            let t = (elbow_y - *y) / (elbow_y - wrist_y);
            (elbow[0] * (1.0 - t) + wrist[0] * t, elbow[2] * (1.0 - t) + wrist[2] * t)
        };
        (*y, limb_ring(cx, cz, rx * a, rz * a, side, bumps, n), sk)
    }).collect();

    mesh::loft_y_tris(tris, &arm_rings);

    // ── FIST ──
    push_box(tris, wrist[0], wrist[1] - 0.03, wrist[2] - 0.04, 0.040 * a, 0.035, 0.025 * a, darken(sk, 0.95));
}

/// Complete nude player body with animation — male or female via BodyProportions
fn gen_nude_player_body(
    tris: &mut Vec<WorldTri>,
    swing: f32,
    skin: u32,
    hair: u32,
    attack_phase: f32,
    carrying_item: bool,
    carrying_bin: bool,
    sitting: bool,
    is_female: bool,
) {
    let props = if is_female { female_proportions() } else { male_proportions() };
    let head_app = NpcAppearance {
        skin, hair,
        hat_type: 0, hat_col: 0, coat_col: 0, vest_col: 0,
        has_coat: false, has_cape: false, has_sash: false,
        has_cross_strap: false, has_bracers: false,
        boot_type: 0, boot_col: 0, sash_col: 0,
        face_age: 0, is_female,
    };

    if sitting {
        // Seated nude body
        let body_base = tris.len();
        mesh::tapered_cylinder_tris(tris, 0.0, 1.09, 0.0, 0.08, 0.07, 0.12, 8, skin);
        let torso_base = tris.len();
        gen_nude_torso(tris, skin, &props);
        for tri in &mut tris[torso_base..] {
            for v in &mut tri.v { v[1] -= 0.4; }
        }
        gen_glutes(tris, skin, &props);
        for &side in &[-1.0f32, 1.0] {
            let hip_x = side * (props.hip_joint_x + 0.03);
            let hip_s = [hip_x, 0.48, 0.0];
            let knee_s = [hip_x, 0.46, -0.42];
            let ankle_s = [hip_x, 0.06, -0.44];
            let l = props.leg_rx_scale;
            mesh::tapered_cylinder_between(tris, hip_s, knee_s, 0.14 * l, 0.090 * l, 10, skin);
            mesh::sphere_tris(tris, knee_s[0], knee_s[1], knee_s[2], 0.085 * l, 0, skin);
            mesh::tapered_cylinder_between(tris, knee_s, ankle_s, 0.085 * l, 0.052 * l, 8, skin);
            gen_bare_foot(tris, ankle_s, side, skin);
        }
        for &side in &[-1.0f32, 1.0] {
            let sh_x = side * props.shoulder_joint_x;
            let shoulder = [sh_x, 0.98, 0.0];
            let elbow = [side * (props.shoulder_joint_x + 0.06), 0.64, -0.15];
            let wrist = [sh_x, 0.48, -0.30];
            let aa = props.arm_rx_scale;
            mesh::ellipsoid_tris(tris, shoulder[0], shoulder[1], shoulder[2], 0.12 * aa, 0.10, 0.10 * aa, 1, skin);
            mesh::tapered_cylinder_between(tris, shoulder, elbow, 0.10 * aa, 0.070 * aa, 8, skin);
            mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2], 0.066 * aa, 0, skin);
            mesh::tapered_cylinder_between(tris, elbow, wrist, 0.068 * aa, 0.048 * aa, 7, skin);
            gen_hand(tris, wrist[0], wrist[1] - 0.04, wrist[2] - 0.02, side, skin);
        }
        // Stretch body
        let bs = props.body_stretch;
        let bw = props.body_widen;
        for tri in &mut tris[body_base..] {
            for v in &mut tri.v {
                v[0] *= bw;
                v[1] *= bs;
                v[2] *= bw;
            }
        }
        // Head (scaled and positioned on seated neck)
        let hs = props.head_scale;
        let hcy = props.head_cy;
        let head_base = tris.len();
        gen_head(tris, &head_app, None);
        for tri in &mut tris[head_base..] {
            for v in &mut tri.v {
                v[0] *= hs;
                v[1] = hcy + (v[1] - hcy) * hs;
                v[2] *= hs;
            }
        }
        let skull_base = hcy + (1.55 - hcy) * hs;
        let sit_neck_top = 1.15 * bs;
        let head_shift = sit_neck_top - skull_base;
        for tri in &mut tris[head_base..] {
            for v in &mut tri.v {
                v[1] += head_shift;
            }
        }
        return;
    }

    // ── BODY (neck, torso, limbs) — generated at natural coords, then stretched ──
    let body_base = tris.len();
    gen_neck(tris, skin, &props);
    gen_nude_torso(tris, skin, &props);
    gen_glutes(tris, skin, &props);

    // Legs
    let l_fwd = -swing * 0.40;
    let r_fwd = swing * 0.40;
    let l_knee = if swing > 0.0 { swing * 0.22 } else { 0.0 };
    let r_knee = if swing < 0.0 { (-swing) * 0.22 } else { 0.0 };
    gen_nude_leg(tris, -1.0, l_fwd, l_knee, skin, &props);
    gen_nude_leg(tris, 1.0, r_fwd, r_knee, skin, &props);

    // Arms
    if attack_phase > 0.0 {
        let t = (attack_phase / ATTACK_ANIM_DURATION).clamp(0.0, 1.0);
        let extend = 1.0 - (1.0 - t) * (1.0 - t);
        gen_nude_attack_arm(tris, 1.0, extend, skin, &props);
        gen_nude_arm(tris, -1.0, -0.2, 0.3, skin, &props);
    } else if carrying_item || carrying_bin {
        gen_nude_arm(tris, -1.0, -0.63, 0.30, skin, &props);
        gen_nude_arm(tris, 1.0, -0.63, 0.30, skin, &props);
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
        gen_nude_arm(tris, -1.0, l_arm_fwd, l_bend, skin, &props);
        gen_nude_arm(tris, 1.0, r_arm_fwd, r_bend, skin, &props);
    }

    // Stretch body vertically (taller) and slightly widen (muscular mass)
    let bs = props.body_stretch;
    let bw = props.body_widen;
    for tri in &mut tris[body_base..] {
        for v in &mut tri.v {
            v[0] *= bw;
            v[1] *= bs;
            v[2] *= bw;
        }
    }

    // ── HEAD (generated separately, uniformly scaled, positioned on stretched neck) ──
    let hs = props.head_scale;
    let hcy = props.head_cy;
    let head_base = tris.len();
    gen_head(tris, &head_app, None);
    for tri in &mut tris[head_base..] {
        for v in &mut tri.v {
            v[0] *= hs;
            v[1] = hcy + (v[1] - hcy) * hs;
            v[2] *= hs;
        }
    }
    let skull_base = hcy + (1.55 - hcy) * hs;
    let head_shift = props.neck_top * bs - skull_base;
    for tri in &mut tris[head_base..] {
        for v in &mut tri.v {
            v[1] += head_shift;
        }
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
        player.is_female,
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

    // Speech bubble (floating above stretched head)
    if npc.interacting_with.is_some() {
        mesh::sphere_tris(tris, 0.0, 2.85, -0.15, 0.12, 0, 0xFFFFFFFF);
        mesh::sphere_tris(tris, 0.0, 2.70, -0.1, 0.04, 0, 0xFFFFFFFF);
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
