// sys_render: transform world + player geometry to screen, rasterize
// Near-plane clipping, backface/distance culling, day/night lighting

use crate::gpu::GpuVertex;
use crate::math::*;
use crate::mesh;
use crate::raster::*;
use crate::state::*;

const SKIN_COLOR: u32 = 0xFFDEB887;

const VEHICLE_BODY_COLOR_DARKEN: f32 = 0.7;
const WINDSHIELD_COLOR: u32 = 0xFF88AACC;
const TIRE_COLOR: u32 = 0xFF222222;

const NEAR_W: f32 = 0.1;

// Body proportion scaling: ~6.5 heads tall heroic proportions
const BODY_STRETCH: f32 = 1.25;  // moderate vertical stretch (less distortion)

// ── Parameterized body proportions (from GLTF reference scans) ──
#[allow(dead_code)]
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

// ── Slider helper: linear interpolation from slider 0.0–1.0 to parameter range ──
#[inline(always)]
fn sl(lo: f32, hi: f32, t: f32) -> f32 { lo + (hi - lo) * t }

/// Character face creation sliders. All values 0.0–1.0 where 0.5 is average.
#[derive(Clone)]
pub struct FaceSliders {
    pub skull_width: f32,       // overall head width
    pub skull_depth: f32,       // front-to-back cranium depth
    pub jaw_width: f32,         // mandible width
    pub jaw_definition: f32,    // gonial angle sharpness
    pub chin_projection: f32,   // mental protuberance forward
    pub chin_width: f32,        // chin width
    pub brow_ridge: f32,        // supraorbital torus
    pub cheekbone: f32,         // zygomatic arch prominence
    pub nose_size: f32,         // overall nose scale
    pub nose_bridge: f32,       // nasal bridge width
    pub lip_fullness: f32,      // lip volume
    pub eye_spacing: f32,       // interpupillary distance
    pub eye_size: f32,          // orbital opening size
    pub eye_depth: f32,         // orbital socket depth
    pub forehead_height: f32,   // forehead vertical extent
    pub ear_size: f32,          // ear scale
    pub masseter: f32,          // jaw muscle mass
}

impl FaceSliders {
    pub fn default_face() -> Self {
        FaceSliders {
            skull_width: 0.50, skull_depth: 0.50,
            jaw_width: 0.50, jaw_definition: 0.50,
            chin_projection: 0.50, chin_width: 0.50,
            brow_ridge: 0.50, cheekbone: 0.50,
            nose_size: 0.50, nose_bridge: 0.50,
            lip_fullness: 0.50,
            eye_spacing: 0.50, eye_size: 0.50, eye_depth: 0.50,
            forehead_height: 0.50, ear_size: 0.50, masseter: 0.50,
        }
    }
    pub fn male_default() -> Self {
        FaceSliders {
            skull_width: 0.58, skull_depth: 0.52,
            jaw_width: 0.65, jaw_definition: 0.72,
            chin_projection: 0.68, chin_width: 0.62,
            brow_ridge: 0.78, cheekbone: 0.40,
            nose_size: 0.65, nose_bridge: 0.60,
            lip_fullness: 0.35,
            eye_spacing: 0.50, eye_size: 0.42, eye_depth: 0.65,
            forehead_height: 0.48, ear_size: 0.52, masseter: 0.68,
        }
    }
    pub fn female_default() -> Self {
        FaceSliders {
            skull_width: 0.42, skull_depth: 0.46,
            jaw_width: 0.22, jaw_definition: 0.20,
            chin_projection: 0.35, chin_width: 0.30,
            brow_ridge: 0.12, cheekbone: 0.78,
            nose_size: 0.30, nose_bridge: 0.35,
            lip_fullness: 0.72,
            eye_spacing: 0.50, eye_size: 0.62, eye_depth: 0.32,
            forehead_height: 0.58, ear_size: 0.42, masseter: 0.12,
        }
    }

    // ── Character creation presets ──
    pub fn preset_square_jaw() -> Self {
        FaceSliders { jaw_width: 0.85, jaw_definition: 0.90, chin_width: 0.80,
            chin_projection: 0.65, masseter: 0.85, brow_ridge: 0.60,
            ..Self::male_default() }
    }
    pub fn preset_narrow() -> Self {
        FaceSliders { skull_width: 0.25, jaw_width: 0.25, chin_width: 0.25,
            nose_size: 0.35, cheekbone: 0.65, lip_fullness: 0.45,
            ..Self::default_face() }
    }
    pub fn preset_round() -> Self {
        FaceSliders { skull_width: 0.70, jaw_width: 0.55, chin_projection: 0.30,
            cheekbone: 0.55, lip_fullness: 0.60, brow_ridge: 0.35,
            ..Self::default_face() }
    }
    pub fn preset_heavy_brow() -> Self {
        FaceSliders { brow_ridge: 0.95, eye_depth: 0.80, forehead_height: 0.35,
            jaw_definition: 0.70, nose_bridge: 0.70, masseter: 0.65,
            ..Self::male_default() }
    }
    pub fn preset_high_cheekbones() -> Self {
        FaceSliders { cheekbone: 0.90, jaw_width: 0.35, chin_projection: 0.55,
            skull_width: 0.42, nose_size: 0.40, lip_fullness: 0.55,
            ..Self::default_face() }
    }
    pub fn preset_long_face() -> Self {
        FaceSliders { skull_depth: 0.65, forehead_height: 0.80, chin_projection: 0.70,
            skull_width: 0.38, jaw_width: 0.40, nose_size: 0.55,
            ..Self::default_face() }
    }
    pub fn preset_wide() -> Self {
        FaceSliders { skull_width: 0.80, jaw_width: 0.75, nose_size: 0.70,
            cheekbone: 0.65, chin_width: 0.75, lip_fullness: 0.55,
            ..Self::default_face() }
    }
    pub fn preset_delicate() -> Self {
        FaceSliders { skull_width: 0.35, jaw_width: 0.20, brow_ridge: 0.10,
            nose_size: 0.25, lip_fullness: 0.70, eye_size: 0.65,
            chin_projection: 0.35, chin_width: 0.30, masseter: 0.10,
            ..Self::female_default() }
    }
    pub fn preset_rugged() -> Self {
        FaceSliders { jaw_definition: 0.85, masseter: 0.90, brow_ridge: 0.85,
            chin_projection: 0.80, nose_size: 0.70, skull_width: 0.60,
            forehead_height: 0.40, eye_depth: 0.70,
            ..Self::male_default() }
    }
    pub fn preset_broad_nose() -> Self {
        FaceSliders { nose_size: 0.85, nose_bridge: 0.80, lip_fullness: 0.70,
            cheekbone: 0.60, jaw_width: 0.55, skull_width: 0.55,
            ..Self::default_face() }
    }
    pub fn preset_sharp() -> Self {
        FaceSliders { jaw_definition: 0.80, chin_width: 0.25, chin_projection: 0.75,
            cheekbone: 0.85, skull_width: 0.35, nose_bridge: 0.35,
            nose_size: 0.45, brow_ridge: 0.55, ear_size: 0.40,
            ..Self::default_face() }
    }
    pub fn preset_soft() -> Self {
        FaceSliders { jaw_width: 0.45, jaw_definition: 0.20, brow_ridge: 0.25,
            chin_projection: 0.30, masseter: 0.15, cheekbone: 0.60,
            lip_fullness: 0.65, nose_size: 0.40, eye_size: 0.60,
            ..Self::female_default() }
    }

    /// Generate pseudo-random face variation from seed (adds ±0.15 jitter to base)
    pub fn randomized(base: &FaceSliders, seed: u32) -> Self {
        fn vary(base_val: f32, s: u32, idx: u32) -> f32 {
            let h = (s.wrapping_mul(idx.wrapping_mul(2654435761))) as f32 / u32::MAX as f32;
            (base_val + (h - 0.5) * 0.30).clamp(0.0, 1.0)
        }
        FaceSliders {
            skull_width: vary(base.skull_width, seed, 1),
            skull_depth: vary(base.skull_depth, seed, 2),
            jaw_width: vary(base.jaw_width, seed, 3),
            jaw_definition: vary(base.jaw_definition, seed, 4),
            chin_projection: vary(base.chin_projection, seed, 5),
            chin_width: vary(base.chin_width, seed, 6),
            brow_ridge: vary(base.brow_ridge, seed, 7),
            cheekbone: vary(base.cheekbone, seed, 8),
            nose_size: vary(base.nose_size, seed, 9),
            nose_bridge: vary(base.nose_bridge, seed, 10),
            lip_fullness: vary(base.lip_fullness, seed, 11),
            eye_spacing: vary(base.eye_spacing, seed, 12),
            eye_size: vary(base.eye_size, seed, 13),
            eye_depth: vary(base.eye_depth, seed, 14),
            forehead_height: vary(base.forehead_height, seed, 15),
            ear_size: vary(base.ear_size, seed, 16),
            masseter: vary(base.masseter, seed, 17),
        }
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
const STITCH_DARK: u32 = 0xFF222211;
const BUTTON_BRASS: u32 = 0xFFBBAA55;

/// Full ACU-style appearance descriptor — derived from seed, no struct storage.
#[derive(Clone)]
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
    face: FaceSliders,
}

fn npc_appearance(seed: u32) -> NpcAppearance {
    let s = seed;
    let coat_col = COAT_COLORS[(s / 13) as usize % COAT_COLORS.len()];
    let has_coat = s % 4 != 0;
    let is_female = s % 5 == 0;
    let face_base = if is_female { FaceSliders::female_default() } else { FaceSliders::male_default() };
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
        is_female,
        face: FaceSliders::randomized(&face_base, s),
    }
}

fn player_appearance(is_female: bool) -> NpcAppearance {
    let face = if is_female { FaceSliders::female_default() } else { FaceSliders::male_default() };
    NpcAppearance {
        skin: SKIN_COLOR,
        hair: 0xFF332211,
        hat_type: 6,           // hood
        hat_col: 0xFF2A2A3A,   // dark blue-grey
        coat_col: 0xFF2A3044,  // dark navy (Arno's coat)
        vest_col: 0xFF998866,  // tan/beige waistcoat
        has_coat: true,
        has_cape: true,        // shoulder mantle
        has_sash: true,        // red sash
        has_cross_strap: true, // leather chest strap
        has_bracers: true,     // leather bracers
        boot_type: 2,          // tall boots
        boot_col: BOOT_BROWN,
        sash_col: 0xFFAA2222,  // red sash
        face_age: 0,
        is_female,
        face,
    }
}

const PLAYER_SHIRT: u32 = SHIRT_LINEN;   // cream/off-white undershirt
const PLAYER_PANTS: u32 = 0xFF3A3A44;    // dark grey-blue breeches

/// Seam line — very thin raised strip to simulate stitching
fn push_seam(tris: &mut Vec<WorldTri>, x: f32, y0: f32, y1: f32, z: f32, color: u32) {
    // Vertical seam line (thin box)
    push_box(tris, x, (y0+y1)*0.5, z, 0.004, (y1-y0).abs(), 0.003, color);
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
    let (sky, amb, sun) = if hour < 4.5 {
        (0xFF0C0C24, 0.12, 0.0)
    } else if hour < 5.5 {
        let t = (hour - 4.5) / 1.0;
        (lerp_color(0xFF0C0C24, 0xFF553366, t), 0.12 + t * 0.15, t * 0.1)
    } else if hour < 6.5 {
        let t = (hour - 5.5) / 1.0;
        (lerp_color(0xFF553366, 0xFFEE9944, t), 0.27 + t * 0.2, 0.1 + t * 0.35)
    } else if hour < 8.0 {
        let t = (hour - 6.5) / 1.5;
        (lerp_color(0xFFEE9944, 0xFF88CCEE, t), 0.47 + t * 0.18, 0.45 + t * 0.2)
    } else if hour < 16.0 {
        (0xFF88CCEE, 0.65, 0.65)
    } else if hour < 17.5 {
        let t = (hour - 16.0) / 1.5;
        (lerp_color(0xFF88CCEE, 0xFFEEAA55, t), 0.65 - t * 0.1, 0.65 - t * 0.15)
    } else if hour < 18.5 {
        let t = (hour - 17.5) / 1.0;
        (lerp_color(0xFFEEAA55, 0xFFCC4422, t), 0.55 - t * 0.15, 0.50 - t * 0.2)
    } else if hour < 19.5 {
        let t = (hour - 18.5) / 1.0;
        (lerp_color(0xFFCC4422, 0xFF332244, t), 0.40 - t * 0.2, 0.30 - t * 0.25)
    } else if hour < 20.5 {
        let t = (hour - 19.5) / 1.0;
        (lerp_color(0xFF332244, 0xFF0C0C24, t), 0.20 - t * 0.08, 0.05 - t * 0.05)
    } else {
        (0xFF0C0C24, 0.12, 0.0)
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
    let _sk_dk = darken(skin, 0.85);
    let _hair_dk = darken(hair, 0.90);
    let _hair_med = darken(hair, 0.95);

    use std::f32::consts::{PI, TAU};
    let n = 48;

    // Loomis proportions: eyes 50%, brow 43% from top, nose bottom 68%, mouth 75%
    let hp = PI * 0.5;
    let re = 0.39; // right eye angular position
    let le = TAU - re;
    let f = &app.face;

    // Map face sliders (0.0–1.0) to concrete geometric parameters
    // Wider ranges than v1 for more dramatic differentiation
    let skw = sl(0.85, 1.15, f.skull_width);
    let skd = sl(0.85, 1.15, f.skull_depth);
    let jaw_w = sl(0.004, 0.020, f.jaw_width);
    let jawline = sl(0.004, 0.024, f.jaw_width);
    let gonial = sl(0.003, 0.035, f.jaw_definition);
    let masseter = sl(0.002, 0.020, f.masseter);
    let brow_shelf = sl(0.002, 0.035, f.brow_ridge);
    let brow_boss = sl(0.002, 0.020, f.brow_ridge);
    let glabella = sl(0.001, 0.015, f.brow_ridge);
    let supraorb = sl(0.002, 0.018, f.brow_ridge);
    let cheek = sl(0.008, 0.050, f.cheekbone);
    let chin_proj = sl(0.010, 0.070, f.chin_projection);
    let chin_w = sl(0.045, 0.080, f.chin_width);  // min 0.045 prevents spike chin
    let nose_size = sl(0.018, 0.050, f.nose_size);
    let nose_br = sl(0.008, 0.020, f.nose_bridge);
    let lip_full = sl(0.6, 1.6, f.lip_fullness);
    let eye_x = sl(0.060, 0.100, f.eye_spacing);
    let eye_r = sl(0.018, 0.034, f.eye_size);
    let eye_z = sl(-0.220, -0.270, f.eye_depth);
    let fh_off = sl(-0.03, 0.03, f.forehead_height);
    let ear_s = sl(0.70, 1.30, f.ear_size);
    let fem = app.is_female;

    // Orbital socket depth — negative displacement at eye positions
    // Creates concavity that eyeballs sit inside
    let orb_depth = sl(-0.012, -0.035, f.eye_depth);

    // ══════════════════════════════════════════════════════════════
    // SKULL LOFT — anatomically structured head surface
    // Key features: orbital sockets, prominent brow, zygomatic arch,
    // nasal bridge, occipital curve, frontal eminences
    // ══════════════════════════════════════════════════════════════
    let rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
        // ── CHIN — forward-projecting mental protuberance ──
        (1.46, body_ring(0.0, -0.06 * skd, chin_w, 0.09 * skd, &[
            (0.0, 0.3, chin_proj),
        ], n), sk),
        (1.49, body_ring(0.0, -0.04 * skd, (chin_w + 0.10 * skw) * 0.5, 0.11 * skd, &[
            (0.0, 0.3, chin_proj * 0.7),
        ], n), sk),
        (1.52, body_ring(0.0, -0.02 * skd, 0.13 * skw, 0.13 * skd, &[
            (0.0, 0.3, chin_proj * 0.4),
        ], n), sk),
        // Labiomental fold
        (1.55, body_ring(0.0, 0.0, 0.15 * skw, 0.15 * skd, &[
            (0.0, 0.20, -0.022),
            (hp, 0.25, jaw_w), (le, 0.25, jaw_w),
        ], n), sk),

        // ── JAW — mandible with defined gonial angle ──
        (1.58, body_ring(0.0, 0.01 * skd, 0.19 * skw, 0.18 * skd, &[
            (hp, 0.25, jawline), (le, 0.25, jawline),
            (0.0, 0.3, 0.012),
        ], n), sk),
        (1.61, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.19 * skd, &[
            (hp, 0.15, gonial), (le, 0.15, gonial),
            (hp - 0.3, 0.2, masseter), (le + 0.3, 0.2, masseter),
        ], n), sk),
        (1.63, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.19 * skd, &[
            (hp, 0.15, gonial * 0.7), (le, 0.15, gonial * 0.7),
            (hp - 0.3, 0.2, masseter * 0.9), (le + 0.3, 0.2, masseter * 0.9),
        ], n), sk),

        // ── MOUTH LEVEL — face center shifted forward (cz=0.01) ──
        (1.65, body_ring(0.0, 0.01 * skd, 0.195 * skw, 0.21 * skd, &[
            (0.0, 0.20, 0.012),
            (0.20, 0.08, -0.004), (TAU - 0.20, 0.08, -0.004),
        ], n), sk),
        (1.67, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.23 * skd, &[
            (0.0, 0.25, 0.016),
            (hp - 0.3, 0.2, masseter * 0.7), (le + 0.3, 0.2, masseter * 0.7),
            (0.22, 0.08, -0.008), (TAU - 0.22, 0.08, -0.008),
        ], n), sk),

        // ── NOSE BASE — piriform aperture ──
        (1.69, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.24 * skd, &[
            (0.0, 0.10, 0.020),
            (0.22, 0.08, -0.008), (TAU - 0.22, 0.08, -0.008),
        ], n), sk),
        // Nose mid — tip + alar projection
        (1.72, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.25 * skd, &[
            (0.0, 0.06, nose_size * 1.2),
            (0.12, 0.05, nose_br * 0.8), (TAU - 0.12, 0.05, nose_br * 0.8),
            (0.25, 0.08, -0.008), (TAU - 0.25, 0.08, -0.008),
        ], n), sk),

        // ── INFRAORBITAL / CHEEKBONE — zygomatic arch, orbital socket begins ──
        (1.74, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.26 * skd, &[
            (0.0, 0.06, nose_size * 0.8),
            (0.55, 0.18, cheek * 1.1), (TAU - 0.55, 0.18, cheek * 1.1),
            (re, 0.12, orb_depth * 0.5), (le, 0.12, orb_depth * 0.5),
        ], n), sk),

        // ── ORBITAL SOCKET — deepest recess at eye level ──
        (1.77, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.27 * skd, &[
            (0.0, 0.06, nose_br * 1.1),
            (re, 0.12, orb_depth), (le, 0.12, orb_depth),
            (0.55, 0.15, cheek * 0.6), (TAU - 0.55, 0.15, cheek * 0.6),
        ], n), sk),
        // Upper orbital — socket recovering toward brow
        (1.80, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.27 * skd, &[
            (0.0, 0.06, nose_br * 0.8),
            (re, 0.12, orb_depth * 0.4), (le, 0.12, orb_depth * 0.4),
        ], n), sk),

        // ── BROW RIDGE — supraorbital torus, projects forward over eye sockets ──
        (1.82, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.28 * skd, &[
            (0.0, 0.50, brow_shelf),
            (0.0, 0.10, glabella),
            (re, 0.10, supraorb), (le, 0.10, supraorb),
        ], n), sk),
        (1.84, body_ring(0.0, 0.02 * skd, 0.20 * skw, 0.28 * skd, &[
            (0.0, 0.50, brow_boss * 0.8),
        ], n), sk),

        // ── FOREHEAD — frontal eminences ──
        (1.87 + fh_off, body_ring(0.0, 0.04 * skd, 0.21 * skw, 0.27 * skd, &[
            (0.25, 0.15, 0.008), (TAU - 0.25, 0.15, 0.008),
            (PI, 0.4, 0.010),
        ], n), sk),

        // ── CRANIAL VAULT ──
        (1.91 + fh_off, body_ring(0.0, 0.07 * skd, 0.21 * skw, 0.27 * skd, &[
            (PI, 0.35, 0.025),
            (hp, 0.3, 0.008), (le, 0.3, 0.008),
        ], n), sk),
        (1.95 + fh_off, body_ring(0.0, 0.08 * skd, 0.21 * skw, 0.28 * skd, &[
            (PI, 0.35, 0.030),
        ], n), sk),
        (1.99 + fh_off, body_ring(0.0, 0.08 * skd, 0.21 * skw, 0.27 * skd, &[
            (PI, 0.35, 0.025),
        ], n), sk),

        // ── CROWN ──
        (2.03 + fh_off, body_ring(0.0, 0.06 * skd, 0.19 * skw, 0.24 * skd, &[], n), sk),
        (2.06 + fh_off, body_ring(0.0, 0.05 * skd, 0.16 * skw, 0.20 * skd, &[], n), sk),
        (2.08 + fh_off, body_ring(0.0, 0.04 * skd, 0.12 * skw, 0.16 * skd, &[], n), sk),
        (2.10 + fh_off, body_ring(0.0, 0.03 * skd, 0.08 * skw, 0.11 * skd, &[], n), sk),
        (2.12 + fh_off, body_ring(0.0, 0.02 * skd, 0.04 * skw, 0.05 * skd, &[], n), sk),
    ];
    mesh::loft_y_tris(tris, &rings);

    // ══════════════════════════════════════════════════════════════
    // EYE ASSEMBLY — parameterized by eye_spacing, eye_size, eye_depth sliders
    // ══════════════════════════════════════════════════════════════
    for &side in &[-1.0f32, 1.0] {
        let ex = side * eye_x;
        let ey = 1.775;
        let ez = eye_z;

        // Eyeball
        mesh::sphere_tris(tris, ex, ey, ez, eye_r, 2, 0xFFEEEEEE);
        // Iris — scaled with eye size
        mesh::sphere_tris(tris, ex, ey, ez - eye_r * 0.77, eye_r * 0.54, 1, 0xFF445533);
        // Pupil
        mesh::sphere_tris(tris, ex, ey, ez - eye_r * 0.92, eye_r * 0.23, 0, 0xFF111100);

        // Upper eyelid — center slightly forward of eye so lid front covers eyeball
        let lid_w = eye_r * 1.10;
        mesh::ellipsoid_tris(tris, ex, ey + eye_r * 0.50, ez - 0.002, lid_w, eye_r * 0.44, eye_r * 0.92, 1, sk);

        // Lower eyelid
        mesh::ellipsoid_tris(tris, ex, ey - eye_r * 0.58, ez - 0.002, lid_w * 0.90, eye_r * 0.28, eye_r * 0.80, 1, sk);
    }

    // ── EYEBROWS — follow brow ridge curvature ──
    for &side in &[-1.0f32, 1.0] {
        let brow_thick = sl(0.003, 0.007, f.brow_ridge);
        // Inner brow — narrower to stay on skull surface
        mesh::ellipsoid_tris(tris, side * eye_x, 1.82, -0.280,
            0.028, brow_thick, 0.010, 0, darken(hair, 0.85));
        // Brow tail — small, hugging skull surface
        mesh::ellipsoid_tris(tris, side * (eye_x + 0.014), 1.823, -0.258,
            0.010, brow_thick * 0.5, 0.006, 0, darken(hair, 0.80));
    }

    // ══════════════════════════════════════════════════════════════
    // NOSE — anatomically defined: bridge with plane changes, defined
    // alar wings, visible columella, enclosed nostrils
    // ══════════════════════════════════════════════════════════════
    let ns = nose_size;
    let nb = nose_br;
    // Nasal bone — tall ridge forming the bridge
    mesh::ellipsoid_tris(tris, 0.0, 1.76, -0.290, nb * 1.2, 0.040, nb * 1.0, 1, sk);
    // Dorsum — continuous bridge from nasion to tip
    mesh::ellipsoid_tris(tris, 0.0, 1.74, -0.285, nb * 1.0, 0.030, nb * 0.8, 1, sk);
    // Lateral walls — defined planes flanking the bridge
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * nb * 1.2, 1.73, -0.268,
            nb * 0.7, 0.038, nb * 0.6, 0, sk);
    }
    // Nose tip — defined ball
    mesh::ellipsoid_tris(tris, 0.0, 1.715, -0.300, ns * 0.50, 0.016, ns * 0.50, 1, darken(sk, 0.96));
    // Alar wings — fleshy lateral flaps
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * ns * 0.70, 1.712, -0.278,
            ns * 0.55, 0.018, ns * 0.45, 1, sk);
        // Alar crease
        mesh::ellipsoid_tris(tris, side * ns * 0.80, 1.710, -0.265,
            ns * 0.22, 0.012, ns * 0.16, 0, darken(sk, 0.88));
        // Wing underside
        mesh::ellipsoid_tris(tris, side * ns * 0.50, 1.705, -0.285,
            ns * 0.42, 0.008, ns * 0.32, 0, darken(sk, 0.88));
    }
    // Columella — central pillar between nostrils
    mesh::ellipsoid_tris(tris, 0.0, 1.706, -0.292, 0.008, 0.013, 0.012, 0, darken(sk, 0.93));
    // Nostrils — dark openings
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * ns * 0.35, 1.703, -0.282,
            ns * 0.26, 0.008, 0.010, 0, darken(sk, 0.25));
    }
    // ── PHILTRUM — two visible ridges from columella to Cupid's bow ──
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.007, 1.695, -0.255,
            0.005, 0.022, 0.008, 0, sk_sh);
    }
    // Philtrum groove
    mesh::ellipsoid_tris(tris, 0.0, 1.695, -0.257,
        0.004, 0.020, 0.004, 0, darken(sk, 0.88));

    // ══════════════════════════════════════════════════════════════
    // MOUTH — Cupid's bow upper lip, full lower lip, vermilion borders
    // ══════════════════════════════════════════════════════════════
    let lip_col = if fem { 0xFFCC8888 } else { 0xFFBB8877 };
    let lo_lip_col = if fem { 0xFFDD9999 } else { 0xFFCC9988 };
    let lf = lip_full;

    // Upper lip — two halves for Cupid's bow
    mesh::ellipsoid_tris(tris, -0.013, 1.676, -0.248, 0.028, 0.006 * lf, 0.012 * lf, 0, lip_col);
    mesh::ellipsoid_tris(tris,  0.013, 1.676, -0.248, 0.028, 0.006 * lf, 0.012 * lf, 0, lip_col);
    // Cupid's bow peak
    mesh::ellipsoid_tris(tris, 0.0, 1.680, -0.251, 0.008, 0.004 * lf, 0.006 * lf, 0, lip_col);

    // Lower lip
    mesh::ellipsoid_tris(tris, 0.0, 1.654, -0.245, 0.040, 0.012 * lf, 0.014 * lf, 1, lo_lip_col);
    // Vermilion border
    mesh::ellipsoid_tris(tris, 0.0, 1.665, -0.251, 0.042, 0.002, 0.004, 0, darken(lip_col, 0.80));

    // Mouth line
    push_box(tris, 0.0, 1.666, -0.253, 0.036, 0.001, 0.003, darken(sk, 0.40));
    // Commissures
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.035, 1.664, -0.247,
            0.006, 0.004, 0.004, 0, darken(sk, 0.55));
    }

    // Nasolabial folds
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * 0.034, 1.69, -0.250,
            0.005, 0.025, 0.005, 0, darken(sk, 0.86));
    }

    // ══════════════════════════════════════════════════════════════
    // EARS — scaled by ear_size slider
    // ══════════════════════════════════════════════════════════════
    let es = ear_s;
    for &side in &[-1.0f32, 1.0] {
        let ear_base_x = side * 0.21 * skw;
        let ez = 0.05 * skd;
        let s = side;

        // Helix
        mesh::ellipsoid_tris(tris, ear_base_x + s * 0.010 * es, 1.81, ez,
            0.014 * es, 0.012 * es, 0.020 * es, 1, sk);
        mesh::ellipsoid_tris(tris, ear_base_x, 1.79, ez - 0.020 * es,
            0.012 * es, 0.018 * es, 0.014 * es, 1, sk);
        mesh::ellipsoid_tris(tris, ear_base_x + s * 0.016 * es, 1.76, ez,
            0.014 * es, 0.028 * es, 0.018 * es, 1, sk);
        mesh::ellipsoid_tris(tris, ear_base_x + s * 0.010 * es, 1.73, ez - 0.005,
            0.012 * es, 0.024 * es, 0.014 * es, 1, sk);

        // Antihelix
        mesh::ellipsoid_tris(tris, ear_base_x, 1.76, ez + 0.002,
            0.010 * es, 0.036 * es, 0.012 * es, 1, darken(sk, 0.95));
        mesh::ellipsoid_tris(tris, ear_base_x + s * 0.004 * es, 1.80, ez + 0.008,
            0.008 * es, 0.016 * es, 0.010 * es, 0, darken(sk, 0.94));
        mesh::ellipsoid_tris(tris, ear_base_x - s * 0.004 * es, 1.80, ez - 0.008,
            0.008 * es, 0.016 * es, 0.010 * es, 0, darken(sk, 0.94));

        // Concha
        mesh::ellipsoid_tris(tris, ear_base_x - s * 0.010, 1.755, ez - 0.004,
            0.018 * es, 0.022 * es, 0.016 * es, 0, darken(sk, 0.50));
        mesh::sphere_tris(tris, ear_base_x - s * 0.016, 1.755, ez - 0.008,
            0.008 * es, 0, darken(sk, 0.25));

        // Tragus
        mesh::ellipsoid_tris(tris, ear_base_x - s * 0.024, 1.76, ez - 0.020,
            0.010 * es, 0.012 * es, 0.008 * es, 0, sk);

        // Lobe
        mesh::ellipsoid_tris(tris, ear_base_x + s * 0.004, 1.71, ez - 0.006,
            0.012 * es, 0.020 * es, 0.012 * es, 1, darken(sk, 0.97));
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

    if hat && app.hat_type != 6 {
        // Hair peeking out from under hat
        mesh::ellipsoid_tris(tris, 0.0, 1.70, 0.14, 0.14, 0.06, 0.10, 0, hair);
        for &side in &[-1.0f32, 1.0] {
            push_box(tris, side * 0.17, 1.74, -0.02, 0.015, 0.05, 0.03, hair);
            push_box(tris, side * 0.16, 1.76, 0.04, 0.02, 0.04, 0.08, hair);
        }
    }
}

/// Neck — ring-lofted with SCM muscles and larynx. Continuous head-to-neck topology.
fn gen_neck(tris: &mut Vec<WorldTri>, skin: u32, props: &BodyProportions, n: usize) {
    use std::f32::consts::PI;
    let nt = props.neck_top;
    let m = props.muscle_def;
    let rx = props.neck_rx;
    let rz = props.neck_rz;

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
fn gen_nude_torso(tris: &mut Vec<WorldTri>, skin: u32, props: &BodyProportions, n: usize) {
    let sk = skin;
    let m = props.muscle_def;
    let _sk_shadow = darken(sk, 1.0 - 0.03 * m);
    let sk_deep = darken(sk, 1.0 - 0.07 * m);
    let nipple_col = darken(sk, 0.78);

    use std::f32::consts::PI;
    let hp = PI * 0.5;

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

/// Nude arm — single continuous loft from shoulder to wrist, no sphere joints
fn gen_nude_arm(
    tris: &mut Vec<WorldTri>, side: f32, fwd: f32, bend: f32, skin: u32,
    props: &BodyProportions, n: usize,
) {
    let sk = skin;
    let a = props.arm_rx_scale;
    let m = props.muscle_def;

    use std::f32::consts::PI;
    let hp = PI * 0.5;

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
    props: &BodyProportions, n: usize,
) {
    let sk = skin;
    let l = props.leg_rx_scale;
    let m = props.muscle_def;

    use std::f32::consts::PI;
    let hp = PI * 0.5;

    let lx = side * props.hip_joint_x;
    let _hip = [lx, 0.92, 0.0];
    let _knee = [lx, 0.48, fwd * 0.5];
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
fn gen_nude_attack_arm(tris: &mut Vec<WorldTri>, side: f32, extend: f32, skin: u32, props: &BodyProportions, n: usize) {
    let sk = skin;
    let a = props.arm_rx_scale;
    let m = props.muscle_def;

    use std::f32::consts::PI;
    let hp = PI * 0.5;

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

/// ACU-style clothing over the detailed player body. Generated in natural (pre-stretch)
/// coordinates so it transforms identically with the body. All major surfaces use the
/// same loft technique as the nude body — offset body rings for smooth coverage.
fn gen_player_clothing(
    tris: &mut Vec<WorldTri>,
    props: &BodyProportions,
    app: &NpcAppearance,
    shirt_col: u32,
    pants_col: u32,
    swing: f32,
    attack_phase: f32,
    carrying_item: bool,
    carrying_bin: bool,
) {
    use std::f32::consts::PI;
    let coat = app.coat_col;
    let vest = app.vest_col;
    let n = 16;

    // Scale factors matching gen_nude_torso
    let sh = props.hip_rx / 0.18;
    let sw = props.waist_rx / 0.15;
    let sc = props.chest_rx / 0.22;
    let ss = props.shoulder_rx / 0.32;
    let s = |y: f32| -> f32 {
        if y <= 0.92 { sh }
        else if y <= 1.00 { sh + (sw - sh) * (y - 0.92) / 0.08 }
        else if y <= 1.24 { sw + (sc - sw) * (y - 1.00) / 0.24 }
        else if y <= 1.42 { sc + (ss - sc) * (y - 1.24) / 0.18 }
        else { ss }
    };

    let hp = PI * 0.5;
    let co = 0.035; // clothing offset over body

    // Breast coverage bump — matches nude body breast profile so coat covers chest
    let breast_bump = |y: f32| -> f32 {
        if !props.has_breasts { return 0.0; }
        let dy = (y - props.breast_y) / 0.10;
        (props.breast_rz * 1.3 + 0.015) * (-0.5 * dy * dy).exp() // body breast amp + margin
    };

    // ── COAT/VEST SHELL — lofted tube matching body contour + offset ──
    // Ring radii derived from nude torso ring definitions. Muscle bumps attenuated
    // to 50% for smooth cloth surface. No shoulder pad or armpit filler ellipsoids.
    if app.has_coat {
        let coat_ring = |y: f32, rx: f32, rz: f32, bumps: &[(f32, f32, f32)]| -> (f32, Vec<[f32; 2]>, u32) {
            let sf = s(y);
            let ab: Vec<(f32, f32, f32)> = bumps.iter()
                .map(|&(a, w, amp)| (a, w, amp * 0.5))
                .collect();
            (y, body_ring(0.0, 0.0, rx * sf + co, rz * sf + co, &ab, n), coat)
        };
        let coat_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
            coat_ring(0.88, 0.14, 0.12, &[(PI, 0.5, 0.04)]),
            coat_ring(0.92, 0.18, 0.14, &[
                (hp, 0.40, 0.05), (PI + hp, 0.40, 0.05), (PI, 0.5, 0.06),
            ]),
            coat_ring(0.96, 0.17, 0.13, &[
                (hp, 0.40, 0.04), (PI + hp, 0.40, 0.04), (PI, 0.4, 0.04),
            ]),
            coat_ring(1.00, 0.16, 0.13, &[
                (hp, 0.40, 0.04), (PI + hp, 0.40, 0.04), (PI, 0.5, 0.04),
            ]),
            coat_ring(1.04, 0.17, 0.14, &[
                (hp, 0.40, 0.05), (PI + hp, 0.40, 0.05), (PI, 0.5, 0.035),
            ]),
            // Armhole zone (Y=1.08-1.32): bypass 0.5 attenuation for full lateral bumps.
            // Breast coverage bumps (theta=±0.30) match nude body breast profile.
            {
                let sf = s(1.08); let bb = breast_bump(1.08);
                let mut b = vec![(hp, 0.45, 0.04), (PI + hp, 0.45, 0.04), (PI, 0.5, 0.02 * sf)];
                if bb > 0.001 { b.push((0.30, 0.55, bb)); b.push((-0.30, 0.55, bb)); }
                (1.08, body_ring(0.0, 0.0, 0.19 * sf + co, 0.16 * sf + co, &b, n), coat)
            },
            {
                let sf = s(1.12); let bb = breast_bump(1.12);
                let mut b = vec![(hp, 0.45, 0.05), (PI + hp, 0.45, 0.05), (PI, 0.4, 0.02 * sf)];
                if bb > 0.001 { b.push((0.30, 0.55, bb)); b.push((-0.30, 0.55, bb)); }
                (1.12, body_ring(0.0, 0.0, 0.21 * sf + co, 0.17 * sf + co, &b, n), coat)
            },
            {
                let sf = s(1.16); let bb = breast_bump(1.16);
                let mut b = vec![(hp, 0.45, 0.06), (PI + hp, 0.45, 0.06)];
                if bb > 0.001 { b.push((0.30, 0.55, bb)); b.push((-0.30, 0.55, bb)); }
                (1.16, body_ring(0.0, 0.0, 0.22 * sf + co, 0.18 * sf + co, &b, n), coat)
            },
            {
                let sf = s(1.21); let bb = breast_bump(1.21);
                let mut b = vec![(hp, 0.45, 0.07), (PI + hp, 0.45, 0.07)];
                if bb > 0.001 { b.push((0.30, 0.55, bb)); b.push((-0.30, 0.55, bb)); }
                (1.21, body_ring(0.0, 0.0, 0.23 * sf + co, 0.20 * sf + co, &b, n), coat)
            },
            {
                let sf = s(1.26); let bb = breast_bump(1.26);
                let mut b = vec![(hp, 0.45, 0.07), (PI + hp, 0.45, 0.07)];
                if bb > 0.001 { b.push((0.30, 0.55, bb)); b.push((-0.30, 0.55, bb)); }
                (1.26, body_ring(0.0, 0.0, 0.24 * sf + co, 0.21 * sf + co, &b, n), coat)
            },
            {
                let sf = s(1.32); let bb = breast_bump(1.32);
                let mut b = vec![(hp, 0.45, 0.08), (PI + hp, 0.45, 0.08), (PI, 0.5, 0.02 * sf)];
                if bb > 0.001 { b.push((0.30, 0.55, bb)); b.push((-0.30, 0.55, bb)); }
                (1.32, body_ring(0.0, 0.0, 0.24 * sf + co, 0.20 * sf + co, &b, n), coat)
            },
            {
                let sf = s(1.36); let da = props.shoulder_deltoid_amp * 0.5;
                (1.36, body_ring(0.0, 0.0, 0.22 * sf + co, 0.20 * sf + co, &[
                    (hp, 0.45, da + 0.05), (PI + hp, 0.45, da + 0.05), (PI, 0.5, 0.030 * sf),
                ], n), coat)
            },
            {
                let sf = s(1.39); let da = props.shoulder_deltoid_amp * 0.5;
                (1.39, body_ring(0.0, 0.0, 0.20 * sf + co, 0.19 * sf + co, &[
                    (hp, 0.45, da + 0.06), (PI + hp, 0.45, da + 0.06), (PI, 0.5, 0.035 * sf),
                ], n), coat)
            },
            {
                let sf = s(1.42); let da = props.shoulder_deltoid_amp * 0.5;
                (1.42, body_ring(0.0, 0.0, 0.18 * sf + co, 0.18 * sf + co, &[
                    (hp, 0.45, da + 0.07), (PI + hp, 0.45, da + 0.07), (PI, 0.5, 0.035 * sf),
                ], n), coat)
            },
            {
                let sf = s(1.44);
                (1.44, body_ring(0.0, 0.0, 0.14 * sf + co, 0.14 * sf + co, &[
                    (hp, 0.35, 0.03), (PI + hp, 0.35, 0.03), (PI, 0.5, 0.028 * sf),
                ], n), darken(coat, 0.92))
            },
            (1.48, body_ring(0.0, 0.0, props.neck_rx + co, props.neck_rz + co, &[], n),
                darken(coat, 0.88)),
        ];
        mesh::loft_y_tris(tris, &coat_rings);

        // Lapels
        let lapel_z = -(0.21 * sc + co + 0.015);
        push_box(tris, -0.07, 1.34, lapel_z, 0.030, 0.08, 0.012, darken(coat, 0.92));
        push_box(tris, 0.07, 1.34, lapel_z, 0.030, 0.08, 0.012, darken(coat, 0.92));
        // Buttons
        for row in 0..4 {
            let by = 1.30 - row as f32 * 0.08;
            let bz = -(0.20 * s(by) + co + 0.01);
            for &bx in &[-0.04f32, 0.04] {
                mesh::sphere_tris(tris, bx, by, bz, 0.007, 0, BUTTON_BRASS);
            }
        }
        // Back seam
        push_seam(tris, 0.0, 0.88, 1.40, 0.14 * sh + co + 0.005, STITCH_DARK);

        // ── COAT TAILS — 3D lofted curtains ──
        let tail_sway = swing * 0.08;
        let back_z = 0.14 * sh + co;
        for &tx in &[-0.06f32, 0.06] {
            let tail_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
                (0.88, body_ring(tx, back_z, 0.08, 0.015, &[], 8), coat),
                (0.74, body_ring(tx, back_z + 0.02 + tail_sway * 0.3, 0.09, 0.012, &[], 8), coat),
                (0.58, body_ring(tx, back_z + 0.04 + tail_sway * 0.6, 0.10, 0.010, &[], 8), coat),
                (0.42, body_ring(tx, back_z + 0.06 + tail_sway, 0.11, 0.008, &[], 8),
                    darken(coat, 0.95)),
            ];
            mesh::loft_y_tris(tris, &tail_rings);
        }
        // Front skirt panels
        let front_z = -(0.14 * sh + co);
        for &tx in &[-0.09f32, 0.09] {
            let skirt_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
                (0.88, body_ring(tx, front_z, 0.06, 0.012, &[], 8), coat),
                (0.76, body_ring(tx, front_z - 0.01 - tail_sway * 0.15, 0.07, 0.010, &[], 8), coat),
                (0.64, body_ring(tx, front_z - 0.02 - tail_sway * 0.3, 0.07, 0.008, &[], 8),
                    darken(coat, 0.96)),
            ];
            mesh::loft_y_tris(tris, &skirt_rings);
        }

        // ── CAPE ──
        if app.has_cape {
            let cape_col = darken(coat, 0.88);
            let cape_z = 0.21 * sc + co;
            let cape_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
                (1.44, body_ring(0.0, cape_z * 0.5, 0.16, 0.02, &[], 8), cape_col),
                (1.32, body_ring(0.0, cape_z, 0.20, 0.03, &[], 8), cape_col),
                (1.16, body_ring(0.0, cape_z + 0.01, 0.22, 0.03, &[], 8), cape_col),
                (1.00, body_ring(0.0, cape_z + 0.02, 0.22, 0.03, &[], 8), cape_col),
                (0.86, body_ring(0.0, cape_z + 0.03, 0.24, 0.02, &[], 8),
                    darken(cape_col, 0.95)),
            ];
            mesh::loft_y_tris(tris, &cape_rings);
        }
    } else {
        // No coat — vest/shirt shell (thinner offset)
        let vco = 0.025;
        let vest_ring = |y: f32, rx: f32, rz: f32| -> (f32, Vec<[f32; 2]>, u32) {
            let sf = s(y);
            (y, body_ring(0.0, 0.0, rx * sf + vco, rz * sf + vco, &[], n), vest)
        };
        let vest_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
            vest_ring(0.88, 0.14, 0.12),
            vest_ring(0.92, 0.18, 0.14),
            vest_ring(1.00, 0.15, 0.13),
            vest_ring(1.08, 0.17, 0.15),
            vest_ring(1.16, 0.20, 0.17),
            vest_ring(1.26, 0.22, 0.21),
            vest_ring(1.34, 0.22, 0.19),
            vest_ring(1.40, 0.18, 0.18),
            (1.44, body_ring(0.0, 0.0, props.neck_rx + 0.04, props.neck_rz + 0.03, &[], n), shirt_col),
        ];
        mesh::loft_y_tris(tris, &vest_rings);
    }

    // Vest details (visible with or without coat via V-neckline)
    let vest_front_z = -(0.21 * sc + co + 0.005);
    push_box(tris, -0.04, 1.34, vest_front_z, 0.008, 0.10, 0.008, darken(vest, 0.90));
    push_box(tris, 0.04, 1.34, vest_front_z, 0.008, 0.10, 0.008, darken(vest, 0.90));
    for row in 0..4 {
        let by = 1.28 - row as f32 * 0.06;
        for &bx in &[-0.025f32, 0.025] {
            mesh::sphere_tris(tris, bx, by, vest_front_z - 0.005, 0.005, 0, BUTTON_BRASS);
        }
    }
    // Shirt collar / jabot
    let collar_z = -(0.20 * s(1.42) + co + 0.01);
    push_box(tris, -0.035, 1.44, collar_z, 0.008, 0.018, 0.008, shirt_col);
    push_box(tris, 0.035, 1.44, collar_z, 0.008, 0.018, 0.008, shirt_col);
    for i in 0..4 {
        let ry = 1.37 - i as f32 * 0.035;
        push_box(tris, 0.0, ry, vest_front_z - 0.005,
            0.020 + i as f32 * 0.002, 0.010, 0.006, shirt_col);
    }

    // ── BELT SYSTEM ──
    let belt_y = 0.88;
    let belt_r = 0.18 * sh + co + 0.01;
    mesh::cylinder_tris(tris, 0.0, belt_y, 0.0, belt_r, 0.025, 12, LEATHER_DARK);
    mesh::cylinder_tris(tris, 0.0, belt_y + 0.013, 0.0, belt_r + 0.003, 0.003, 12, STITCH_DARK);
    mesh::cylinder_tris(tris, 0.0, belt_y - 0.013, 0.0, belt_r + 0.003, 0.003, 12, STITCH_DARK);
    push_box(tris, 0.0, belt_y, -(belt_r + 0.005), 0.025, 0.018, 0.008, BUCKLE_BRASS);
    push_box(tris, 0.0, belt_y, -(belt_r + 0.010), 0.004, 0.014, 0.004, BUCKLE_BRASS);
    // Pouches
    push_box(tris, 0.17, 0.84, -(belt_r * 0.6), 0.05, 0.06, 0.04, LEATHER_MED);
    push_box(tris, 0.17, 0.88, -(belt_r * 0.6), 0.05, 0.008, 0.04, darken(LEATHER_MED, 0.90));
    push_box(tris, -0.18, 0.84, -(belt_r * 0.4), 0.04, 0.05, 0.04, LEATHER_MED);
    if app.has_sash {
        mesh::cylinder_tris(tris, 0.0, belt_y + 0.005, 0.0, belt_r + 0.005, 0.020, 12, app.sash_col);
        push_box(tris, -0.20, 0.75, 0.0, 0.035, 0.14, 0.020, app.sash_col);
        push_box(tris, -0.20, 0.58, 0.0, 0.030, 0.08, 0.015, darken(app.sash_col, 0.92));
    }
    if app.has_cross_strap {
        let strap_front = 0.21 * sc + co;
        for i in 0..12 {
            let t = i as f32 / 11.0;
            let strap_x = 0.16 * (1.0 - 2.0 * t);
            let sy = 1.42 * (1.0 - t) + belt_y * t;
            let sz = -(t * (1.0 - t)) * 0.06 - strap_front;
            push_box(tris, strap_x, sy, sz, 0.018, 0.025, 0.008, LEATHER_MED);
        }
        push_box(tris, 0.0, 1.14, -(strap_front + 0.03), 0.015, 0.015, 0.010, BUCKLE_BRASS);
    }

    // ── PANTS + BOOTS — lofted tubes following leg joint interpolation ──
    let l_fwd = -swing * 0.40;
    let r_fwd = swing * 0.40;
    let l_knee_b = if swing > 0.0 { swing * 0.22 } else { 0.0 };
    let r_knee_b = if swing < 0.0 { (-swing) * 0.22 } else { 0.0 };
    let l = props.leg_rx_scale;
    let bc = app.boot_col;

    for &(side, fwd, knee_bend) in &[(-1.0f32, l_fwd, l_knee_b), (1.0, r_fwd, r_knee_b)] {
        let lx = side * props.hip_joint_x;
        let hip_y = 0.92;
        let knee_y = 0.48;
        let ankle_y = 0.08;
        let knee_cz = fwd * 0.5;
        let ankle_cz = fwd * 0.25 - knee_bend * 0.4;

        // Pants loft (hip → knee) — same interpolation as gen_nude_leg
        // Inner-thigh bumps push rings past body centerline for overlap coverage
        let pco = 0.030;
        let pants_data: Vec<(f32, f32, f32, Vec<(f32, f32, f32)>)> = vec![
            (0.92, 0.100, 0.095, vec![
                (PI + hp, 0.70, 0.085), (PI, 0.60, 0.060), (0.0, 0.60, 0.055),
            ]),
            (0.88, 0.102, 0.097, vec![
                (PI + hp, 0.70, 0.080), (PI, 0.60, 0.055), (0.0, 0.60, 0.050),
            ]),
            (0.84, 0.104, 0.097, vec![
                (PI + hp, 0.65, 0.065), (PI, 0.55, 0.045), (0.0, 0.55, 0.040),
            ]),
            (0.78, 0.104, 0.097, vec![
                (PI + hp, 0.60, 0.050), (PI, 0.50, 0.035), (0.0, 0.50, 0.030),
            ]),
            (0.70, 0.100, 0.093, vec![
                (PI + hp, 0.55, 0.030), (PI, 0.45, 0.020), (0.0, 0.45, 0.015),
            ]),
            (0.62, 0.088, 0.082, vec![
                (PI + hp, 0.50, 0.015),
            ]),
            (0.54, 0.074, 0.068, vec![]),
            (0.48, 0.058, 0.055, vec![]),
        ];
        let pants_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = pants_data.iter().map(|(y, rx, rz, bumps)| {
            let cz = if *y >= knee_y {
                let t = (hip_y - *y) / (hip_y - knee_y);
                knee_cz * t
            } else { knee_cz };
            (*y, limb_ring(lx, cz, (*rx + pco) * l, (*rz + pco) * l, side, bumps, 12), pants_col)
        }).collect();
        mesh::loft_y_tris(tris, &pants_rings);

        // Knee band
        let knee_pt = [lx, knee_y, knee_cz];
        mesh::cylinder_tris(tris, knee_pt[0], knee_pt[1] - 0.02, knee_pt[2],
            (0.060 + pco) * l, 0.020, 8, darken(pants_col, 0.84));
        for ki in 0..3 {
            let kbx = knee_pt[0] + side * (0.035 + ki as f32 * 0.012) * l;
            mesh::sphere_tris(tris, kbx, knee_pt[1] - 0.02, knee_pt[2] - (0.050 + pco) * l,
                0.005, 0, BUCKLE_BRASS);
        }

        // Boot loft (knee → ankle) — same interpolation as gen_nude_leg lower half
        let bco = 0.018;
        let boot_data: [(f32, f32, f32); 7] = [
            (0.48, 0.057, 0.054),
            (0.42, 0.062, 0.058),
            (0.36, 0.066, 0.060),
            (0.30, 0.062, 0.056),
            (0.22, 0.050, 0.046),
            (0.14, 0.042, 0.038),
            (0.08, 0.036, 0.034),
        ];
        let boot_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = boot_data.iter().map(|&(y, rx, rz)| {
            let cz = if y >= knee_y { knee_cz }
            else {
                let t = (knee_y - y) / (knee_y - ankle_y);
                knee_cz * (1.0 - t) + ankle_cz * t
            };
            (y, limb_ring(lx, cz, (rx + bco) * l, (rz + bco) * l, side, &[], 10), bc)
        }).collect();
        mesh::loft_y_tris(tris, &boot_rings);

        // Boot cuff details
        if app.boot_type >= 2 {
            let cuff_t = 0.08;
            let cuff_cz = knee_cz * (1.0 - cuff_t) + ankle_cz * cuff_t;
            let cuff_y = knee_y * (1.0 - cuff_t) + ankle_y * cuff_t;
            mesh::cylinder_tris(tris, lx, cuff_y, cuff_cz,
                (0.068 + bco) * l, 0.022, 8, darken(bc, 0.90));
            for si in 0..3 {
                let st = 0.30 + si as f32 * 0.18;
                let sy = knee_y * (1.0 - st) + ankle_y * st;
                let sz = knee_cz * (1.0 - st) + ankle_cz * st;
                let sr = ((0.062 * (1.0 - st) + 0.036 * st) + bco + 0.005) * l;
                mesh::cylinder_tris(tris, lx, sy, sz, sr, 0.007, 8, darken(bc, 0.85));
                push_box(tris, lx + side * sr, sy, sz, 0.007, 0.008, 0.005, BUCKLE_BRASS);
            }
        } else if app.boot_type == 1 {
            let cuff_t = 0.20;
            let cuff_y = knee_y * (1.0 - cuff_t) + ankle_y * cuff_t;
            let cuff_cz = knee_cz * (1.0 - cuff_t) + ankle_cz * cuff_t;
            mesh::cylinder_tris(tris, lx, cuff_y, cuff_cz,
                (0.062 + bco) * l, 0.018, 8, darken(bc, 0.90));
        } else {
            let st_y0 = knee_y * 0.90 + ankle_y * 0.10;
            let st_y1 = knee_y * 0.50 + ankle_y * 0.50;
            let st_cz0 = knee_cz * 0.90 + ankle_cz * 0.10;
            let st_cz1 = knee_cz * 0.50 + ankle_cz * 0.50;
            mesh::tapered_cylinder_between(tris,
                [lx, st_y0, st_cz0], [lx, st_y1, st_cz1],
                (0.058 + bco) * l, (0.050 + bco) * l, 8, 0xFFCCBBAA);
            mesh::cylinder_tris(tris, lx, st_y0, st_cz0, (0.060 + bco) * l, 0.008, 8, LEATHER_DARK);
        }

        // Boot foot
        let ankle_pt = [lx, ankle_y, ankle_cz];
        mesh::beveled_box_tris(tris, ankle_pt[0], ankle_pt[1] - 0.035, ankle_pt[2] - 0.04,
            0.10 * l, 0.07, 0.15 * l, 0.015, bc);
        push_box(tris, ankle_pt[0], ankle_pt[1] - 0.07, ankle_pt[2] - 0.04,
            0.09 * l, 0.005, 0.14 * l, darken(bc, 0.65));
        push_box(tris, ankle_pt[0], ankle_pt[1] - 0.05, ankle_pt[2] + 0.06 * l,
            0.04 * l, 0.020, 0.03 * l, darken(bc, 0.75));
        if app.boot_type == 0 {
            push_box(tris, ankle_pt[0], ankle_pt[1] - 0.02, ankle_pt[2] - 0.10 * l,
                0.02, 0.015, 0.008, BUCKLE_BRASS);
        }
    }

    // ── SLEEVES — lofted tubes following arm joint interpolation ──
    // Same cubic ease-in as gen_nude_arm. Inner bumps (PI+hp) push sleeve into
    // torso volume for seamless armhole coverage — no filler ellipsoids needed.
    let sleeve_col = if app.has_coat { coat } else { shirt_col };
    let sx = props.shoulder_joint_x;
    let a = props.arm_rx_scale;
    let aco = 0.030;

    struct ArmPose { side: f32, shoulder: [f32; 3], elbow: [f32; 3], wrist: [f32; 3] }
    let arm_poses: Vec<ArmPose> = if attack_phase > 0.0 {
        let t = (attack_phase / ATTACK_ANIM_DURATION).clamp(0.0, 1.0);
        let extend = 1.0 - (1.0 - t) * (1.0 - t);
        vec![
            ArmPose { side: -1.0,
                shoulder: [-sx, 1.42, 0.0],
                elbow: [-(sx + 0.10), 1.06, -0.2 * 0.35],
                wrist: [-(sx + 0.06), 0.80, -0.2 * 0.15 - 0.3],
            },
            ArmPose { side: 1.0,
                shoulder: [sx, 1.42, 0.0],
                elbow: [sx + 0.10, 1.10, -0.15 - extend * 0.20],
                wrist: [sx + 0.06, 0.92, -0.35 - extend * 0.35],
            },
        ]
    } else if carrying_item || carrying_bin {
        vec![
            ArmPose { side: -1.0,
                shoulder: [-sx, 1.42, 0.0],
                elbow: [-(sx + 0.10), 1.06, -0.63 * 0.35],
                wrist: [-(sx + 0.06), 0.80, -0.63 * 0.15 - 0.30],
            },
            ArmPose { side: 1.0,
                shoulder: [sx, 1.42, 0.0],
                elbow: [sx + 0.10, 1.06, -0.63 * 0.35],
                wrist: [sx + 0.06, 0.80, -0.63 * 0.15 - 0.30],
            },
        ]
    } else {
        let l_arm_fwd = swing * 0.25;
        let r_arm_fwd = -swing * 0.25;
        let bend = 0.10 + swing.abs() * 0.14;
        vec![
            ArmPose { side: -1.0,
                shoulder: [-sx, 1.42, 0.0],
                elbow: [-(sx + 0.10), 1.06, l_arm_fwd * 0.35],
                wrist: [-(sx + 0.06), 0.80, l_arm_fwd * 0.15 - bend],
            },
            ArmPose { side: 1.0,
                shoulder: [sx, 1.42, 0.0],
                elbow: [sx + 0.10, 1.06, r_arm_fwd * 0.35],
                wrist: [sx + 0.06, 0.80, r_arm_fwd * 0.15 - bend],
            },
        ]
    };

    for arm in &arm_poses {
        let shoulder = arm.shoulder;
        let elbow = arm.elbow;
        let wrist = arm.wrist;
        let shoulder_y = shoulder[1];
        let elbow_y = elbow[1];

        // Lofted sleeve (above shoulder → elbow) with integrated inner overlap bumps.
        // Sleeve cap extends ABOVE Y=1.44 to close the armhole where coat narrows to collar.
        // PI+hp pushes inward, 0/PI push front/back to fill armhole gap from all angles.
        let sleeve_data: Vec<(f32, f32, Vec<(f32, f32, f32)>)> = vec![
            // Sleeve cap rings above coat shoulder — seal armhole from top
            (1.48, 0.09, vec![
                (PI + hp, 0.80, 0.140),
                (0.0, 0.60, 0.080), (PI, 0.60, 0.080),
            ]),
            (1.46, 0.11, vec![
                (hp, 0.35, 0.020), (PI + hp, 0.80, 0.130),
                (0.0, 0.60, 0.075), (PI, 0.60, 0.075),
            ]),
            // Upper sleeve — puffy military cut, very wide inner overlap
            (1.44, 0.13, vec![
                (hp, 0.35, 0.030), (PI + hp, 0.80, 0.120),
                (0.0, 0.60, 0.070), (PI, 0.60, 0.070),
            ]),
            (1.42, 0.13, vec![
                (hp, 0.35, 0.030), (PI + hp, 0.80, 0.130),
                (0.0, 0.60, 0.075), (PI, 0.60, 0.075),
            ]),
            (1.39, 0.12, vec![
                (hp, 0.35, 0.025), (PI + hp, 0.80, 0.130),
                (0.0, 0.60, 0.070), (PI, 0.60, 0.070),
            ]),
            (1.36, 0.11, vec![
                (hp, 0.35, 0.020), (PI + hp, 0.75, 0.120),
                (0.0, 0.55, 0.060), (PI, 0.55, 0.060),
            ]),
            (1.32, 0.10, vec![
                (PI + hp, 0.70, 0.110),
                (0.0, 0.50, 0.050), (PI, 0.50, 0.050),
            ]),
            (1.26, 0.095, vec![
                (PI + hp, 0.65, 0.090),
                (0.0, 0.45, 0.035), (PI, 0.45, 0.035),
            ]),
            (1.18, 0.082, vec![
                (PI + hp, 0.60, 0.065),
                (0.0, 0.40, 0.025), (PI, 0.40, 0.025),
            ]),
            (elbow_y, 0.056, vec![]),
        ];
        let sleeve_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = sleeve_data.iter().map(|(y, rx, bumps)| {
            let (cx, cz) = if *y >= elbow_y {
                let t_lin = (shoulder_y - *y) / (shoulder_y - elbow_y);
                let t = t_lin * t_lin * t_lin;
                (shoulder[0] * (1.0 - t) + elbow[0] * t,
                 shoulder[2] * (1.0 - t) + elbow[2] * t)
            } else {
                (elbow[0], elbow[2])
            };
            let r = (*rx + aco) * a;
            (*y, limb_ring(cx, cz, r, r, arm.side, bumps, 10), sleeve_col)
        }).collect();
        mesh::loft_y_tris(tris, &sleeve_rings);

        // Elbow joint
        mesh::sphere_tris(tris, elbow[0], elbow[1], elbow[2],
            (0.056 + aco) * a, 0, darken(sleeve_col, 0.92));

        // Coat cuff (elbow → mid-forearm)
        if app.has_coat {
            let lerp = |t: f32| -> [f32; 3] {
                [elbow[0] * (1.0 - t) + wrist[0] * t,
                 elbow[1] * (1.0 - t) + wrist[1] * t,
                 elbow[2] * (1.0 - t) + wrist[2] * t]
            };
            let cs = 0.20;
            let ce = 0.42;
            mesh::tapered_cylinder_between(tris, lerp(cs), lerp(ce),
                (0.056 + aco) * a, (0.054 + aco) * a, 8, darken(coat, 0.78));
            let ce_p = lerp(ce);
            mesh::cylinder_tris(tris, ce_p[0], ce_p[1], ce_p[2],
                (0.050 + aco) * a, 0.007, 8, darken(coat, 1.25));
            for bi in 0..3 {
                let bt = cs + (ce - cs) * (bi as f32 + 0.5) / 3.0;
                let bp = lerp(bt);
                mesh::sphere_tris(tris, bp[0] + arm.side * (0.052 + aco) * a, bp[1], bp[2],
                    0.005, 0, BUTTON_BRASS);
            }
        }

        // Shirt ruffle cuff
        let rt = 0.62;
        let rp = [
            elbow[0] * (1.0 - rt) + wrist[0] * rt,
            elbow[1] * (1.0 - rt) + wrist[1] * rt,
            elbow[2] * (1.0 - rt) + wrist[2] * rt,
        ];
        mesh::cylinder_tris(tris, rp[0], rp[1], rp[2], (0.042 + aco) * a, 0.010, 8, SHIRT_LINEN);

        // Bracer
        if app.has_bracers {
            let lerp = |t: f32| -> [f32; 3] {
                [elbow[0] * (1.0 - t) + wrist[0] * t,
                 elbow[1] * (1.0 - t) + wrist[1] * t,
                 elbow[2] * (1.0 - t) + wrist[2] * t]
            };
            let brs = 0.38;
            let bre = 0.68;
            mesh::tapered_cylinder_between(tris, lerp(brs), lerp(bre),
                (0.052 + aco) * a, (0.046 + aco) * a, 8, LEATHER_DARK);
            let brs_p = lerp(brs);
            let bre_p = lerp(bre);
            mesh::cylinder_tris(tris, brs_p[0], brs_p[1], brs_p[2], (0.054 + aco) * a, 0.007, 8, LEATHER_DARK);
            mesh::cylinder_tris(tris, bre_p[0], bre_p[1], bre_p[2], (0.048 + aco) * a, 0.007, 8, LEATHER_DARK);
            let mb_p = lerp((brs + bre) * 0.5);
            mesh::cylinder_tris(tris, mb_p[0], mb_p[1], mb_p[2], (0.054 + aco) * a, 0.005, 8, darken(LEATHER_DARK, 0.85));
            push_box(tris, mb_p[0] + arm.side * (0.054 + aco) * a, mb_p[1], mb_p[2],
                0.007, 0.008, 0.005, BUCKLE_BRASS);
        }
    }
}

/// Complete player body with animation — male or female via BodyProportions.
/// When clothing is Some, adds ACU-style clothing over the body before the stretch.
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
    clothing: Option<(&NpcAppearance, u32, u32)>, // (appearance, shirt_col, pants_col)
    job_hat: Option<u32>,
) {
    let props = if is_female { female_proportions() } else { male_proportions() };
    let head_app = if let Some((app, _, _)) = clothing {
        app.clone()
    } else {
        let face = if is_female { FaceSliders::female_default() } else { FaceSliders::male_default() };
        NpcAppearance {
            skin, hair,
            hat_type: 0, hat_col: 0, coat_col: 0, vest_col: 0,
            has_coat: false, has_cape: false, has_sash: false,
            has_cross_strap: false, has_bracers: false,
            boot_type: 0, boot_col: 0, sash_col: 0,
            face_age: 0, is_female, face,
        }
    };

    if sitting {
        // Seated nude body
        let body_base = tris.len();
        mesh::tapered_cylinder_tris(tris, 0.0, 1.09, 0.0, 0.08, 0.07, 0.12, 8, skin);
        let torso_base = tris.len();
        gen_nude_torso(tris, skin, &props, 32);
        for tri in &mut tris[torso_base..] {
            for v in &mut tri.v { v[1] -= 0.4; }
        }
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
        gen_head(tris, &head_app, job_hat);
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
    gen_neck(tris, skin, &props, 24);
    gen_nude_torso(tris, skin, &props, 32);

    // Legs
    let l_fwd = -swing * 0.40;
    let r_fwd = swing * 0.40;
    let l_knee = if swing > 0.0 { swing * 0.22 } else { 0.0 };
    let r_knee = if swing < 0.0 { (-swing) * 0.22 } else { 0.0 };
    gen_nude_leg(tris, -1.0, l_fwd, l_knee, skin, &props, 24);
    gen_nude_leg(tris, 1.0, r_fwd, r_knee, skin, &props, 24);

    // Arms
    if attack_phase > 0.0 {
        let t = (attack_phase / ATTACK_ANIM_DURATION).clamp(0.0, 1.0);
        let extend = 1.0 - (1.0 - t) * (1.0 - t);
        gen_nude_attack_arm(tris, 1.0, extend, skin, &props, 24);
        gen_nude_arm(tris, -1.0, -0.2, 0.3, skin, &props, 24);
    } else if carrying_item || carrying_bin {
        gen_nude_arm(tris, -1.0, -0.63, 0.30, skin, &props, 24);
        gen_nude_arm(tris, 1.0, -0.63, 0.30, skin, &props, 24);
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
        gen_nude_arm(tris, -1.0, l_arm_fwd, l_bend, skin, &props, 24);
        gen_nude_arm(tris, 1.0, r_arm_fwd, r_bend, skin, &props, 24);
    }

    // ── CLOTHING (added before stretch so it transforms with the body) ──
    if let Some((app, shirt_col, pants_col)) = clothing {
        gen_player_clothing(
            tris, &props, app, shirt_col, pants_col,
            swing, attack_phase, carrying_item, carrying_bin,
        );
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
    gen_head(tris, &head_app, job_hat);
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
    let app = player_appearance(player.is_female);

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
        Some((&app, PLAYER_SHIRT, PLAYER_PANTS)),
        None,
    );

    let rot = terrain_rot3x3(player.terrain_normal, player.rot_y);
    for tri in &mut tris[base..] {
        for v in &mut tri.v {
            let rv = rot3x3_apply(&rot, *v);
            v[0] = rv[0] + player.x;
            v[1] = rv[1] + player.y;
            v[2] = rv[2] + player.z;
        }
        tri.normal = rot3x3_apply(&rot, tri.normal);
    }
}

/// Generate a clothed player body (for model_viewer debug renders).
/// Uses the detailed nude body with ACU-style clothing layered on top.
pub fn gen_clothed_player_body(tris: &mut Vec<WorldTri>, is_female: bool) {
    let app = player_appearance(is_female);
    gen_nude_player_body(
        tris, 0.0, SKIN_COLOR, 0xFF332211,
        0.0, false, false, false, is_female,
        Some((&app, PLAYER_SHIRT, PLAYER_PANTS)),
        None,
    );
}

/// Generate a standalone head + neck with face sliders, scaled and positioned.
/// Used by model_viewer for rendering face variations.
pub fn gen_head_standalone(tris: &mut Vec<WorldTri>, face: &FaceSliders, skin: u32, hair: u32, is_female: bool) {
    let props = if is_female { female_proportions() } else { male_proportions() };
    let app = NpcAppearance {
        skin, hair,
        hat_type: 0, hat_col: 0, coat_col: 0, vest_col: 0,
        has_coat: false, has_cape: false, has_sash: false,
        has_cross_strap: false, has_bracers: false,
        boot_type: 0, boot_col: 0, sash_col: 0,
        face_age: 0, is_female, face: face.clone(),
    };
    let head_base = tris.len();
    gen_head(tris, &app, None);
    gen_neck(tris, skin, &props, 24);
    // Scale + position (same transforms as gen_nude_player_body)
    let hs = props.head_scale;
    let hcy = props.head_cy;
    for tri in &mut tris[head_base..] {
        for v in &mut tri.v {
            v[0] *= hs;
            v[1] = hcy + (v[1] - hcy) * hs;
            v[2] *= hs;
        }
    }
    let skull_base = hcy + (1.55 - hcy) * hs;
    let head_shift = props.neck_top * props.body_stretch - skull_base;
    for tri in &mut tris[head_base..] {
        for v in &mut tri.v {
            v[1] += head_shift;
        }
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
        mesh::sphere_tris(tris, hx, 0.45, -1.83, 0.1, 0, 0x00FFEE88); // emissive
    }
    // Turn signal indicators (small amber lights)
    push_box(tris, -0.85, 0.45, -1.79, 0.08, 0.05, 0.03, 0x00FFAA22); // emissive
    push_box(tris, 0.85, 0.45, -1.79, 0.08, 0.05, 0.03, 0x00FFAA22); // emissive

    // Tail lights (larger, with reflector housing)
    for &tx in &[-0.6f32, 0.6] {
        push_box(tris, tx, 0.45, 1.81, 0.28, 0.12, 0.04, 0xFF441111); // housing
        mesh::sphere_tris(tris, tx, 0.45, 1.83, 0.08, 0, 0x00FF2222); // emissive
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

    let rot = terrain_rot3x3(v.terrain_normal, v.rot_y);
    for tri in &mut tris[base..] {
        for vert in &mut tri.v {
            let rv = rot3x3_apply(&rot, *vert);
            vert[0] = rv[0] + v.x;
            vert[1] = rv[1] + v.y;
            vert[2] = rv[2] + v.z;
        }
        tri.normal = rot3x3_apply(&rot, tri.normal);
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

    // Ragdoll rendering: actual character model oriented by ragdoll joints
    if npc.ragdoll_active {
        let base = tris.len();
        let p = &npc.ragdoll_points;
        // hips=0, chest=1, head=2, l_hand=3, r_hand=4, l_foot=5, r_foot=6

        // Compute body orientation from ragdoll joint positions
        // "Up" axis: hips → head (spine direction)
        let ux = p[2][0] - p[0][0];
        let uy = p[2][1] - p[0][1];
        let uz = p[2][2] - p[0][2];
        let ulen = (ux * ux + uy * uy + uz * uz).sqrt().max(0.01);
        let up = [ux / ulen, uy / ulen, uz / ulen];

        // "Right" axis: left_foot → right_foot, orthogonalized against up
        let rx = p[6][0] - p[5][0];
        let ry = p[6][1] - p[5][1];
        let rz = p[6][2] - p[5][2];
        let dot_ru = rx * up[0] + ry * up[1] + rz * up[2];
        let rx = rx - dot_ru * up[0];
        let ry = ry - dot_ru * up[1];
        let rz = rz - dot_ru * up[2];
        let rlen = (rx * rx + ry * ry + rz * rz).sqrt().max(0.01);
        let right = [rx / rlen, ry / rlen, rz / rlen];

        // "Forward" = cross(right, up)
        let fwd = [
            right[1] * up[2] - right[2] * up[1],
            right[2] * up[0] - right[0] * up[2],
            right[0] * up[1] - right[1] * up[0],
        ];

        let job_hat = match npc.job {
            NpcJob::PolicePatrol => Some(0xFF2233AA),
            NpcJob::Firefighter => Some(0xFFCC3322),
            NpcJob::Paramedic => Some(0xFFDDDDDD),
            NpcJob::ConstructionWorker => Some(0xFFDDAA22),
            NpcJob::MailCarrier => Some(0xFF3344CC),
            _ => None,
        };

        gen_nude_player_body(
            tris, 0.0, app.skin, app.hair, 0.0, false, false, false,
            app.is_female,
            Some((&app, shirt, npc.pants_color)),
            job_hat,
        );

        // Transform from local space (standing, feet at y=0) to ragdoll orientation.
        // Map local origin to the midpoint between ragdoll feet.
        let foot_mid = [
            (p[5][0] + p[6][0]) * 0.5,
            (p[5][1] + p[6][1]) * 0.5,
            (p[5][2] + p[6][2]) * 0.5,
        ];
        for tri in &mut tris[base..] {
            for v in &mut tri.v {
                let lx = v[0];
                let ly = v[1];
                let lz = v[2];
                v[0] = right[0] * lx + up[0] * ly + fwd[0] * lz + foot_mid[0];
                v[1] = right[1] * lx + up[1] * ly + fwd[1] * lz + foot_mid[1];
                v[2] = right[2] * lx + up[2] * ly + fwd[2] * lz + foot_mid[2];
            }
            let nx = tri.normal[0];
            let ny = tri.normal[1];
            let nz = tri.normal[2];
            tri.normal[0] = right[0] * nx + up[0] * ny + fwd[0] * nz;
            tri.normal[1] = right[1] * nx + up[1] * ny + fwd[1] * nz;
            tri.normal[2] = right[2] * nx + up[2] * ny + fwd[2] * nz;
        }
        return;
    }

    let base = tris.len();

    // KO pose — actual character model lying face-down on the ground
    if npc.state == NpcState::KnockedOut {
        let job_hat = match npc.job {
            NpcJob::PolicePatrol => Some(0xFF2233AA),
            NpcJob::Firefighter => Some(0xFFCC3322),
            NpcJob::Paramedic => Some(0xFFDDDDDD),
            NpcJob::ConstructionWorker => Some(0xFFDDAA22),
            NpcJob::MailCarrier => Some(0xFF3344CC),
            _ => None,
        };

        gen_nude_player_body(
            tris, 0.0, app.skin, app.hair, 0.0, false, false, false,
            app.is_female,
            Some((&app, shirt, npc.pants_color)),
            job_hat,
        );

        // Rotate 90° around X to lie face-down: (x, y, z) → (x, z, -y)
        // Then offset Y so body rests on ground surface (y ≈ 0.15)
        let (sin_r, cos_r) = npc.rot_y.sin_cos();
        for tri in &mut tris[base..] {
            for v in &mut tri.v {
                // First: rotate -90° around X axis (face-down)
                let lx = v[0];
                let ly = v[2];       // y' = old_z
                let lz = -v[1];      // z' = -old_y
                // Then: rotate around Y by heading + translate to world
                let rx = lx * cos_r + lz * sin_r;
                let rz = -lx * sin_r + lz * cos_r;
                v[0] = rx + npc.x;
                v[1] = ly + npc.y + 0.15; // slight offset above ground
                v[2] = rz + npc.z;
            }
            let nx = tri.normal[0];
            let ny = tri.normal[2];
            let nz = -tri.normal[1];
            let rnx = nx * cos_r + nz * sin_r;
            let rnz = -nx * sin_r + nz * cos_r;
            tri.normal = [rnx, ny, rnz];
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

    gen_nude_player_body(
        tris,
        npc.walk_phase.sin() * 0.4,
        app.skin,
        app.hair,
        npc.attack_phase,
        npc.carrying_item,
        npc.carrying_bin.is_some(),
        false,
        app.is_female,
        Some((&app, shirt, npc.pants_color)),
        job_hat,
    );

    // Speech bubble (floating above stretched head)
    if npc.interacting_with.is_some() {
        mesh::sphere_tris(tris, 0.0, 2.85, -0.15, 0.12, 0, 0xFFFFFFFF);
        mesh::sphere_tris(tris, 0.0, 2.70, -0.1, 0.04, 0, 0xFFFFFFFF);
    }

    let rot = terrain_rot3x3(npc.terrain_normal, npc.rot_y);
    for tri in &mut tris[base..] {
        for v in &mut tri.v {
            let rv = rot3x3_apply(&rot, *v);
            v[0] = rv[0] + npc.x;
            v[1] = rv[1] + npc.y;
            v[2] = rv[2] + npc.z;
        }
        tri.normal = rot3x3_apply(&rot, tri.normal);
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
    let base = tris.len();
    // Generate at origin, transform after
    mesh::cylinder_tris(tris, 0.0, 0.4, 0.0, 0.22, 0.8, 6, BIN_COLOR);
    mesh::cylinder_tris(tris, 0.0, 0.82, 0.0, 0.25, 0.06, 6, BIN_LID_COLOR);
    mesh::cylinder_tris(tris, 0.0, 0.87, 0.0, 0.24, 0.04, 6, BIN_LID_COLOR);
    if bin.items_held > 5 {
        mesh::sphere_tris(tris, 0.0, 0.95, 0.0, 0.15, 0, BAG_COLOR);
    }
    // Terrain-aligned transform
    let rot = terrain_rot3x3(bin.terrain_normal, 0.0);
    for tri in &mut tris[base..] {
        for v in &mut tri.v {
            let rv = rot3x3_apply(&rot, *v);
            v[0] = rv[0] + bin.x;
            v[1] = rv[1] + bin.y;
            v[2] = rv[2] + bin.z;
        }
        tri.normal = rot3x3_apply(&rot, tri.normal);
    }
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

/// Generate GPU vertices for static world geometry (upload once, never regenerate —
/// GPU shader handles lighting/fog dynamically via push constants)
pub fn generate_static_gpu_vertices(world: &WorldData, out: &mut Vec<GpuVertex>) {
    out.clear();
    out.reserve(world.static_tris.len() * 3);
    for tri in &world.static_tris {
        for i in 0..3 {
            out.push(GpuVertex {
                pos: tri.v[i],
                color_packed: tri.color,
                normal: tri.normal,
            });
        }
    }
}

/// Frustum + distance check. Returns Some(dist_sq) if visible, None if culled.
#[inline]
fn view_cull(eye: Vec3, fwd_x: f32, fwd_z: f32, ex: f32, ez: f32, fog_dist_sq: f32) -> Option<f32> {
    let dx = ex - eye[0];
    let dz = ez - eye[2];
    let dist_sq = dx * dx + dz * dz;
    if dist_sq > fog_dist_sq { return None; }
    if dist_sq < 400.0 { return Some(dist_sq); } // <20m always visible
    let inv_dist = 1.0 / dist_sq.sqrt();
    let dot = fwd_x * dx * inv_dist + fwd_z * dz * inv_dist;
    if dot > 0.259 { Some(dist_sq) } else { None } // cos(75°)
}

/// Medium-detail NPC: simplified body with correct clothing colors (~200 tris)
pub fn gen_npc_mesh_mid(npc: &Npc, tris: &mut Vec<WorldTri>) {
    let shirt = if npc.hit_flash > 0.0 { 0xFFFF4444 } else { job_shirt_color(npc) };
    let app = npc_appearance(npc.brain_idx as u32);
    let arm_col = if app.has_coat { app.coat_col } else { shirt };
    let base = tris.len();

    // Head — small sphere (lowered to reduce gap)
    mesh::sphere_tris(tris, 0.0, 1.52, 0.0, 0.14, 0, app.skin);
    // Neck — short cylinder connecting torso to head
    mesh::cylinder_tris(tris, 0.0, 1.30, 0.0, 0.06, 0.30, 4, app.skin);
    // Torso — tapered cylinder (shirt/coat color)
    mesh::tapered_cylinder_tris(tris, 0.0, 0.85, 0.0, 0.22, 0.16, 0.70, 5, arm_col);
    // Arms — two thin cylinders
    let swing = npc.walk_phase.sin() * 0.2;
    mesh::cylinder_tris(tris, -0.24, 1.10, swing * 0.1, 0.05, 0.45, 4, arm_col);
    mesh::cylinder_tris(tris,  0.24, 1.10, -swing * 0.1, 0.05, 0.45, 4, arm_col);
    // Legs — two cylinders with walk animation
    mesh::cylinder_tris(tris, -0.09, 0.30, -swing * 0.15, 0.06, 0.45, 4, npc.pants_color);
    mesh::cylinder_tris(tris,  0.09, 0.30,  swing * 0.15, 0.06, 0.45, 4, npc.pants_color);
    // Boots
    push_box(tris, -0.09, 0.04, -swing * 0.15, 0.05, 0.05, 0.08, BOOT_BROWN);
    push_box(tris,  0.09, 0.04,  swing * 0.15, 0.05, 0.05, 0.08, BOOT_BROWN);

    // Apply body stretch + world transform
    let rot = terrain_rot3x3(npc.terrain_normal, npc.rot_y);
    for tri in &mut tris[base..] {
        for v in &mut tri.v {
            v[1] *= BODY_STRETCH;
            let rv = rot3x3_apply(&rot, *v);
            v[0] = rv[0] + npc.x;
            v[1] = rv[1] + npc.y;
            v[2] = rv[2] + npc.z;
        }
        tri.normal = rot3x3_apply(&rot, tri.normal);
    }
}

/// Low-detail NPC: 3 colored boxes (~36 tris vs ~14K full detail)
fn gen_npc_mesh_lod(npc: &Npc, tris: &mut Vec<WorldTri>) {
    let app = npc_appearance(npc.brain_idx as u32);
    let body_col = if app.has_coat { app.coat_col } else { 0xFF3355AA };
    let base = tris.len();
    push_box(tris, 0.0, 0.75, 0.0, 0.20, 0.55, 0.12, body_col);
    push_box(tris, 0.0, 0.25, 0.0, 0.13, 0.25, 0.10, npc.pants_color);
    push_box(tris, 0.0, 1.55, 0.0, 0.10, 0.12, 0.10, app.skin);
    let rot = terrain_rot3x3(npc.terrain_normal, npc.rot_y);
    for tri in &mut tris[base..] {
        for v in &mut tri.v {
            let rv = rot3x3_apply(&rot, *v);
            v[0] = rv[0] + npc.x;
            v[1] = rv[1] + npc.y;
            v[2] = rv[2] + npc.z;
        }
        tri.normal = rot3x3_apply(&rot, tri.normal);
    }
}

// LOD distance thresholds (squared)
const LOD_NPC_FULL_SQ: f32 = 625.0;    // < 25m: full detail
const LOD_NPC_MID_SQ: f32 = 6400.0;    // 25-80m: medium detail (~200 tris)
const LOD_NPC_LOW_SQ: f32 = 40000.0;   // 80-200m: low detail boxes
const LOD_VEH_FULL_SQ: f32 = 2500.0;   // < 50m: full detail vehicle
const LOD_VEH_DIST_SQ: f32 = 40000.0;  // > 200m: skip vehicles

/// Low-detail vehicle mesh: 2 colored boxes (body + cabin), ~24 tris
fn gen_vehicle_mesh_lod(v: &Vehicle, tris: &mut Vec<WorldTri>) {
    let base = tris.len();
    push_box(tris, 0.0, 0.35, 0.0, 1.8, 0.5, 3.6, v.color);
    push_box(tris, 0.0, 0.95, 0.2, 1.4, 0.45, 1.8, darken(v.color, 0.85));
    let rot = terrain_rot3x3(v.terrain_normal, v.rot_y);
    for tri in &mut tris[base..] {
        for vert in &mut tri.v {
            let rv = rot3x3_apply(&rot, *vert);
            vert[0] = rv[0] + v.x;
            vert[1] = rv[1] + v.y;
            vert[2] = rv[2] + v.z;
        }
        tri.normal = rot3x3_apply(&rot, tri.normal);
    }
}

/// Generate GPU vertices for dynamic entities only (call each frame)
pub fn generate_dynamic_gpu_vertices(
    world: &WorldData, player: &Player, cam: &Camera,
    scratch: &mut Vec<WorldTri>, out: &mut Vec<GpuVertex>,
) {
    let eye = v3(cam.x, cam.y, cam.z);
    let fog_dist_sq = FOG_DIST * FOG_DIST;

    let fdx = cam.tx - cam.x;
    let fdz = cam.tz - cam.z;
    let flen = (fdx * fdx + fdz * fdz).sqrt().max(0.001);
    let fwd_x = fdx / flen;
    let fwd_z = fdz / flen;

    out.clear();
    scratch.clear();

    // Vehicles: frustum + distance LOD
    for (vi, v) in world.vehicles.iter().enumerate() {
        let dist_sq = match view_cull(eye, fwd_x, fwd_z, v.x, v.z, fog_dist_sq) {
            Some(d) => d,
            None => continue,
        };
        if dist_sq > LOD_VEH_DIST_SQ { continue; }
        if dist_sq < LOD_VEH_FULL_SQ {
            let show_interior = player.in_vehicle == Some(vi);
            gen_vehicle_mesh(v, scratch, show_interior);
        } else {
            gen_vehicle_mesh_lod(v, scratch);
        }
    }
    // NPCs: frustum + distance-based LOD
    for npc in &world.npcs {
        if npc.state == NpcState::Sleeping { continue; }
        if npc.in_vehicle { continue; }
        let dist_sq = match view_cull(eye, fwd_x, fwd_z, npc.x, npc.z, fog_dist_sq) {
            Some(d) => d,
            None => continue,
        };
        if dist_sq < LOD_NPC_FULL_SQ {
            gen_npc_mesh(npc, scratch);
        } else if dist_sq < LOD_NPC_MID_SQ {
            gen_npc_mesh_mid(npc, scratch);
        } else if dist_sq < LOD_NPC_LOW_SQ {
            gen_npc_mesh_lod(npc, scratch);
        }
    }
    for item in &world.items {
        if !item.active && !item.falling { continue; }
        if view_cull(eye, fwd_x, fwd_z, item.x, item.z, fog_dist_sq).is_none() { continue; }
        gen_item_mesh(item, scratch);
    }
    for bin in &world.trash_bins {
        if bin.carried_by.is_some() { continue; }
        if view_cull(eye, fwd_x, fwd_z, bin.x, bin.z, fog_dist_sq).is_none() { continue; }
        gen_trash_bin_mesh(bin, scratch);
    }
    if player.in_vehicle.is_none() {
        gen_player_mesh(player, scratch);
    }
    // Convert to GPU format (raw material colors — GPU shader does lighting)
    out.reserve(scratch.len() * 3);
    for tri in scratch.iter() {
        for i in 0..3 {
            out.push(GpuVertex {
                pos: tri.v[i],
                color_packed: tri.color,
                normal: tri.normal,
            });
        }
    }
}

/// Build GPU push constants from current frame state
pub fn gpu_push_constants(hour: f32, eye: Vec3, target: Vec3, vp: &Mat4) -> crate::gpu::GpuPushConstants {
    let tc = time_colors(hour);
    let fog_dist_sq = FOG_DIST * FOG_DIST;
    let fdx = target[0] - eye[0];
    let fdz = target[2] - eye[2];
    let flen = (fdx * fdx + fdz * fdz).sqrt().max(0.001);
    crate::gpu::GpuPushConstants {
        vp: *vp,
        light_dir_ambient: [tc.light_dir[0], tc.light_dir[1], tc.light_dir[2], tc.ambient],
        sun_fog_params: [tc.sun_strength, 1.0 / fog_dist_sq, fdx / flen, fdz / flen],
        fog_color: [tc.fog_r / 255.0, tc.fog_g / 255.0, tc.fog_b / 255.0, 0.0],
        eye_pos: [eye[0], eye[1], eye[2], 0.0],
    }
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
