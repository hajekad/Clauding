// sys_render: transform world + player geometry to screen, rasterize
// Near-plane clipping, backface/distance culling, day/night lighting
// G-fixes: sculpted face geometry, detailed bracers

use crate::anatomy;
use crate::gpu::GpuVertex;
use crate::math::*;
use crate::mesh;
use crate::raster::*;
use crate::state::*;
use crate::color::{lerp_color, darken};
use std::sync::atomic::{AtomicBool, Ordering};

/// Debug flag: when true, player renders without clothing
pub static NUDE_MODE: AtomicBool = AtomicBool::new(false);

const VEHICLE_BODY_COLOR_DARKEN: f32 = 0.7;
const WINDSHIELD_COLOR: u32 = 0xFF88AACC;
const TIRE_COLOR: u32 = 0xFF222222;

const NEAR_W: f32 = 0.1;

// Body proportion scaling: ~6.5 heads tall heroic proportions
const BODY_STRETCH: f32 = 1.25;  // moderate vertical stretch (less distortion)

const FOG_DIST_SQ: f32 = FOG_DIST * FOG_DIST;

fn job_hat_color(job: NpcJob) -> Option<u32> {
    match job {
        NpcJob::PolicePatrol => Some(0xFF2233AA),
        NpcJob::Firefighter => Some(0xFFCC3322),
        NpcJob::Paramedic => Some(0xFFDDDDDD),
        NpcJob::ConstructionWorker => Some(0xFFDDAA22),
        NpcJob::MailCarrier => Some(0xFF3344CC),
        _ => None,
    }
}

/// Terrain-aligned rotation + world-space translation for mesh triangles.
fn place_mesh(
    tris: &mut Vec<WorldTri>, base: usize,
    terrain_normal: Vec3, tilt_max: f32, rot_y: f32,
    x: f32, y: f32, z: f32,
) {
    let rot = terrain_rot3x3(clamp_normal_tilt(terrain_normal, tilt_max), rot_y);
    for tri in &mut tris[base..] {
        for v in &mut tri.v {
            let rv = rot3x3_apply(&rot, *v);
            v[0] = rv[0] + x;
            v[1] = rv[1] + y;
            v[2] = rv[2] + z;
        }
        tri.normal = rot3x3_apply(&rot, tri.normal);
    }
}

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
    breast_rz: f32,
    breast_y: f32,
}

fn male_proportions() -> BodyProportions {
    BodyProportions {
        body_stretch: 1.25,
        body_widen: 1.0,
        head_scale: 0.70,
        head_cy: 1.70,
        neck_top: 1.49,
        shoulder_rx: 0.36,
        shoulder_deltoid_amp: 0.10,
        hip_rx: 0.28,
        waist_rx: 0.20,
        chest_rx: 0.24,
        muscle_def: 1.0,
        arm_rx_scale: 1.0,
        leg_rx_scale: 1.0,
        shoulder_joint_x: 0.24,
        hip_joint_x: 0.12,
        neck_rx: 0.15,     // F5: thicker neck (was 0.12)
        neck_rz: 0.13,     // F5: thicker neck (was 0.10)
        has_adams_apple: true,
        has_breasts: false,
        breast_rz: 0.0,
        breast_y: 0.0,
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
        muscle_def: 0.3,
        arm_rx_scale: 0.90,
        leg_rx_scale: 0.90,
        shoulder_joint_x: 0.20,
        hip_joint_x: 0.13,
        neck_rx: 0.11,     // F5: thicker neck (was 0.09)
        neck_rz: 0.10,     // F5: thicker neck (was 0.08)
        has_adams_apple: false,
        has_breasts: true,
        breast_rz: 0.038,     // subtle forward projection
        breast_y: 1.26,
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
const SKIN_TONES: [u32; 12] = [
    0xFFF5D6B8, // very light / pale
    0xFFE8C9A0, // light warm
    0xFFDEB887, // light tan
    0xFFD2A87A, // medium light
    0xFFCCA882, // golden
    0xFFC89B6E, // medium
    0xFFBB9060, // olive
    0xFFA07850, // medium dark
    0xFF8B6540, // brown
    0xFF704D30, // dark brown
    0xFF5A3D28, // deep brown
    0xFFDDBC98, // warm beige
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
const BELT_COLORS: [u32; 5] = [
    0xFF3A2A1A, 0xFF2A1A0A, 0xFF4A3A2A, 0xFF332211, 0xFF1A1A1A, // brown/tan/black leather variants
];
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
    belt_col: u32,        // belt leather color — varies per NPC
    face_age: u8,         // 0=young, 1=mid, 2=old (wrinkle density)
    is_female: bool,
    face: FaceSliders,
}

fn npc_appearance(seed: u32) -> NpcAppearance {
    let s = seed;
    let coat_col = COAT_COLORS[(s / 13) as usize % COAT_COLORS.len()];
    let has_coat = s % 4 != 0;
    let is_female = s % 2 == 0;
    let face_base = if is_female { FaceSliders::female_default() } else { FaceSliders::male_default() };
    NpcAppearance {
        skin: SKIN_TONES[(s / 3) as usize % SKIN_TONES.len()],
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
        belt_col: BELT_COLORS[(s / 31) as usize % BELT_COLORS.len()],
        face_age: ((s / 29) % 3) as u8,
        is_female,
        face: FaceSliders::randomized(&face_base, s),
    }
}

fn player_appearance(is_female: bool) -> NpcAppearance {
    // Distinctive protagonist face — sharp jaw, defined brow, memorable profile
    let face = if is_female {
        FaceSliders {
            jaw_definition: 0.55, chin_projection: 0.50, cheekbone: 0.82,
            brow_ridge: 0.28, eye_size: 0.58, lip_fullness: 0.65,
            skull_width: 0.44, nose_size: 0.32,
            ..FaceSliders::female_default()
        }
    } else {
        FaceSliders {
            jaw_definition: 0.80, chin_projection: 0.72, cheekbone: 0.52,
            brow_ridge: 0.70, eye_depth: 0.60, nose_bridge: 0.55,
            masseter: 0.62, forehead_height: 0.42,
            ..FaceSliders::male_default()
        }
    };
    NpcAppearance {
        skin: SKIN_TONES[0], // use palette instead of constant
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
        belt_col: LEATHER_DARK,
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

/// Compute fog-specific color that tracks sky but biases warm during sunset/sunrise.
/// During golden hour transitions, the sky color is a desaturated blend of blue and
/// orange that appears whitish. This biases the fog toward warm amber so distant
/// objects fade into a cohesive warm haze rather than a pale disconnect.
fn fog_color_for_hour(sky: u32, hour: f32) -> (f32, f32, f32) {
    let sr = ((sky >> 16) & 0xFF) as f32;
    let sg = ((sky >> 8) & 0xFF) as f32;
    let sb = (sky & 0xFF) as f32;

    // Sunset warm bias (hours 16-19.5): push fog toward warm amber
    let sunset_bias = if hour >= 16.0 && hour < 17.5 {
        (hour - 16.0) / 1.5 * 0.45 // ramp up to 45% warm bias by peak sunset
    } else if hour >= 17.5 && hour < 18.5 {
        0.45 - (hour - 17.5) * 0.20 // ease off as sky catches up to deep red
    } else if hour >= 18.5 && hour < 19.5 {
        0.25 - (hour - 18.5) * 0.25 // fade out during dusk
    } else {
        0.0
    };

    // Sunrise warm bias (hours 5.5-8.0): push fog toward warm gold
    let sunrise_bias = if hour >= 5.5 && hour < 6.5 {
        (hour - 5.5) / 1.0 * 0.40 // ramp up during sunrise
    } else if hour >= 6.5 && hour < 8.0 {
        0.40 - (hour - 6.5) / 1.5 * 0.40 // fade out as morning brightens
    } else {
        0.0
    };

    let bias = (sunset_bias + sunrise_bias).min(0.5);
    if bias > 0.001 {
        // Warm target: amber tones scaled to current sky brightness
        let brightness = (sr + sg + sb) / 3.0;
        let warm_r = (brightness * 1.35).min(255.0);
        let warm_g = brightness * 0.75;
        let warm_b = brightness * 0.35;
        (
            sr + (warm_r - sr) * bias,
            sg + (warm_g - sg) * bias,
            sb + (warm_b - sb) * bias,
        )
    } else {
        (sr, sg, sb)
    }
}

fn time_colors(hour: f32) -> TimeColors {
    // Smooth easing helper: cubic ease-in-out for natural transitions
    let ease = |t: f32| -> f32 {
        let t = t.clamp(0.0, 1.0);
        if t < 0.5 { 4.0 * t * t * t } else { 1.0 - (-2.0 * t + 2.0).powi(3) / 2.0 }
    };

    let (sky, amb, sun) = if hour < 3.0 {
        // Deep night: moonlight ambient so terrain/buildings are faintly visible
        let t = hour / 3.0;
        let night_amb = 0.18 + t * 0.02;
        (lerp_color(0xFF101028, 0xFF141430, t), night_amb, 0.0)
    } else if hour < 4.5 {
        // Pre-dawn: sky begins to lighten, ambient slowly rises
        let t = (hour - 3.0) / 1.5;
        let te = ease(t);
        (lerp_color(0xFF141430, 0xFF1A1535, te), 0.20 + te * 0.06, 0.0)
    } else if hour < 5.5 {
        // Civil twilight begins: purple/pink horizon glow, first hint of sun
        let t = (hour - 4.5) / 1.0;
        let te = ease(t);
        (lerp_color(0xFF1A1535, 0xFF553366, te), 0.26 + te * 0.10, te * 0.08)
    } else if hour < 6.5 {
        // Sunrise: orange/gold horizon, rapid ambient increase
        let t = (hour - 5.5) / 1.0;
        let te = ease(t);
        (lerp_color(0xFF553366, 0xFFEE9944, te), 0.36 + te * 0.16, 0.08 + te * 0.30)
    } else if hour < 8.0 {
        // Golden hour -> morning: warm light transitions to blue sky
        let t = (hour - 6.5) / 1.5;
        let te = ease(t);
        (lerp_color(0xFFEE9944, 0xFF88CCEE, te), 0.52 + te * 0.16, 0.38 + te * 0.22)
    } else if hour < 10.0 {
        // Morning: brightening toward midday
        let t = (hour - 8.0) / 2.0;
        (lerp_color(0xFF88CCEE, 0xFF99DDFF, t), 0.68 + t * 0.07, 0.60 + t * 0.08)
    } else if hour < 14.0 {
        // Midday plateau: sun near zenith, brightest period with subtle arc
        let t = (hour - 10.0) / 4.0; // 0 at 10, 1 at 14
        // Sine arc peaks at solar noon (12:00)
        let noon_t = (hour - 10.0) / 4.0 * std::f32::consts::PI;
        let noon_boost = noon_t.sin() * 0.05;
        (lerp_color(0xFF99DDFF, 0xFF88CCEE, t), 0.75 + noon_boost, 0.68 + noon_boost)
    } else if hour < 16.0 {
        // Afternoon: gradual decline from midday
        let t = (hour - 14.0) / 2.0;
        (lerp_color(0xFF88CCEE, 0xFF88CCEE, t), 0.75 - t * 0.05, 0.68 - t * 0.10)
    } else if hour < 17.5 {
        // Late afternoon -> golden hour: warm tones return
        let t = (hour - 16.0) / 1.5;
        let te = ease(t);
        (lerp_color(0xFF88CCEE, 0xFFEEAA55, te), 0.70 - te * 0.10, 0.58 - te * 0.15)
    } else if hour < 18.5 {
        // Sunset: orange to deep red
        let t = (hour - 17.5) / 1.0;
        let te = ease(t);
        (lerp_color(0xFFEEAA55, 0xFFCC4422, te), 0.60 - te * 0.14, 0.43 - te * 0.18)
    } else if hour < 19.5 {
        // Dusk: red to purple, sun below horizon
        let t = (hour - 18.5) / 1.0;
        let te = ease(t);
        (lerp_color(0xFFCC4422, 0xFF332244, te), 0.46 - te * 0.18, 0.25 - te * 0.18)
    } else if hour < 21.0 {
        // Late dusk -> night: purple fading to deep blue, ambient settles to moonlight
        let t = (hour - 19.5) / 1.5;
        let te = ease(t);
        (lerp_color(0xFF332244, 0xFF141430, te), 0.28 - te * 0.08, 0.07 - te * 0.07)
    } else {
        // Night: moonlight ambient so terrain/buildings are faintly visible
        let t = (hour - 21.0) / 3.0;
        let night_amb = 0.20 - t * 0.02;
        (lerp_color(0xFF141430, 0xFF101028, t), night_amb, 0.0)
    };

    // Fog color: tracks sky but biases warm during sunset/sunrise transitions
    // to prevent desaturated "white fog against orange sky" disconnect.
    let (fr, fg, fb) = fog_color_for_hour(sky, hour);

    // --- Sun direction: full arc across the sky ---
    // Sun rises in the east (negative X), arcs overhead, sets in the west (positive X).
    // The z-component shifts from south to overhead to north, giving parallax on building faces.
    // sun_angle: 0 at 6:00 (horizon east), PI/2 at 12:00 (zenith), PI at 18:00 (horizon west)
    let sun_angle = (hour - 6.0) / 12.0 * std::f32::consts::PI;
    let light_dir = if sun > 0.0 {
        // Y = height (sin of elevation), X = east-west, Z = slight south bias
        let elevation = sun_angle.sin().max(0.05);
        let east_west = sun_angle.cos(); // full range: -1 (east) to +1 (west)
        // Slight southward bias that shifts through the day
        let south_bias = 0.3 * (sun_angle * 0.5).sin();
        let len = (east_west * east_west + elevation * elevation + south_bias * south_bias).sqrt();
        [east_west / len, elevation / len, south_bias / len]
    } else {
        // Moonlight direction: high and slightly offset for faint directional fill
        // Rotates slowly through the night for subtle variation
        let moon_angle = ((hour + 6.0) % 24.0) / 12.0 * std::f32::consts::PI;
        let my = 0.7 + 0.3 * moon_angle.sin();
        let mx = moon_angle.cos() * 0.3;
        let len = (mx * mx + my * my + 0.04).sqrt();
        [mx / len, my / len, 0.2 / len]
    };

    TimeColors { sky, fog_r: fr, fog_g: fg, fog_b: fb, light_dir, ambient: amb, sun_strength: sun }
}

pub fn sky_color(hour: f32) -> u32 {
    time_colors(hour).sky
}

pub fn sys_render(
    fb: &mut Framebuffer, world: &WorldData, player: &Player, cam: &Camera,
    hour: f32, scratch: &mut Vec<WorldTri>,
    character_models: &[Vec<WorldTri>],
    animation_data: Option<&crate::skeleton_anim::AnimationData>,
    car_models: &[Vec<WorldTri>],
) {
    let tc = time_colors(hour);
    let aspect = fb.w as f32 / fb.h as f32;
    let eye = v3(cam.x, cam.y, cam.z);
    let target = v3(cam.tx, cam.ty, cam.tz);
    let view = m4_look_at(eye, target, v3(0.0, 1.0, 0.0));
    let proj = m4_perspective(60.0_f32.to_radians(), aspect, 0.1, WORLD_SIZE * 2.0);
    let vp = m4_mul(&proj, &view);
    let fw = fb.w as f32;
    let fh = fb.h as f32;

    // Static world
    render_tris(fb, &vp, &world.static_tris, eye, &tc, fw, fh);

    // Dynamic entities: generate into scratch buffer, render once
    scratch.clear();
    let is_night = hour < 6.0 || hour > 20.0;
    for (vi, v) in world.vehicles.iter().enumerate() {
        let show_interior = player.in_vehicle == Some(vi);
        if !car_models.is_empty() {
            gen_vehicle_mesh_gltf(v, vi, car_models, scratch, show_interior, is_night);
        } else {
            gen_vehicle_mesh(v, scratch, show_interior, is_night);
        }
    }
    for npc in &world.npcs {
        if npc.state == NpcState::Sleeping { continue; }
        if npc.in_vehicle { continue; }
        if !character_models.is_empty() {
            gen_npc_mesh_gltf(npc, character_models, scratch, animation_data);
        } else {
            gen_npc_mesh(npc, scratch);
        }
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
        if !character_models.is_empty() {
            gen_player_mesh_gltf(player, character_models, scratch, animation_data);
        } else {
            gen_player_mesh(player, scratch);
        }
    }
    // Night-time street light glow halos (dynamic — only rendered at night)
    if is_night {
        for sl in &world.street_lights {
            let dx = sl.x - eye[0];
            let dz = sl.z - eye[2];
            if dx * dx + dz * dz > 150.0 * 150.0 { continue; } // distance cull
            let gy = sl.ground_y;
            mesh::glow_halo(scratch, sl.x, gy + 5.2, sl.z, 0.2, 1.5, 8, 0x00FFDD88);
        }
    }
    render_tris(fb, &vp, scratch, eye, &tc, fw, fh);
}

fn render_tris(fb: &mut Framebuffer, vp: &Mat4, tris: &[WorldTri], cam_pos: Vec3, tc: &TimeColors, fw: f32, fh: f32) {
    let fog_dist_sq = FOG_DIST_SQ;

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
        // Half-Lambert wrap diffuse: (dot*0.5+0.5)^2 gives softer shadow falloff
        // so building faces perpendicular to the sun still receive some light
        let dot_nl = v3_dot(tri.normal, tc.light_dir);
        let wrap = (dot_nl * 0.5 + 0.5) * (dot_nl * 0.5 + 0.5);
        let sun_lit = wrap * tc.sun_strength;
        // Faint directional moonlight at night — provides surface differentiation
        let moon_lit = if tc.sun_strength < 0.01 {
            dot_nl.max(0.0) * 0.04
        } else { 0.0 };
        let intensity = sun_lit + moon_lit + tc.ambient;
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
pub fn clip_to_screen(c: [f32; 4], w: f32, h: f32) -> [f32; 3] {
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
    // Smoothstep fog curve: zero derivative at both endpoints gives gentle fade-in
    // near the camera and smooth fade-out at the fog boundary, preventing the abrupt
    // pop-in/pop-out of distant buildings. Matches the GPU shader's SmoothStep.
    let t = fog.clamp(0.0, 1.0);
    let mix = t * t * (3.0 - 2.0 * t);
    // At night (low sun_strength), add cool blue moonlight tint so surfaces stay visible.
    // This gives roads, vehicles, river etc. a faint blue-silver appearance instead of pure black.
    let moonlight = (1.0 - tc.sun_strength.min(1.0)) * 0.06;
    let mr = r * i + moonlight * r * 0.5;  // red suppressed under moonlight
    let mg = g * i + moonlight * g * 0.7;  // green slightly present
    let mb = b * i + moonlight * b * 1.2;  // blue boosted for cool moonlit look
    // Round to nearest (not truncate) for closer parity with GPU float pipeline
    let ro = ((mr * (1.0 - mix) + tc.fog_r * mix + 0.5) as u32).min(255);
    let go = ((mg * (1.0 - mix) + tc.fog_g * mix + 0.5) as u32).min(255);
    let bo = ((mb * (1.0 - mix) + tc.fog_b * mix + 0.5) as u32).min(255);
    0xFF000000 | (ro << 16) | (go << 8) | bo
}

// --- Mesh generators (push into shared scratch buffer) ---

// ═══════════════════════════════════════════════════════════════════════════
// CHARACTER BODY GENERATION — each body part separately modeled
// ═══════════════════════════════════════════════════════════════════════════

/// Generate the full anatomical head: skull, face, ears, hat
fn gen_head(tris: &mut Vec<WorldTri>, app: &NpcAppearance, is_job_hat: Option<u32>) {
    let skin = app.skin;
    let sk = skin;
    let sk_sh = darken(skin, 0.92);

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
    let brow_shelf = sl(0.008, 0.045, f.brow_ridge);
    let brow_boss = sl(0.005, 0.028, f.brow_ridge);
    let glabella = sl(0.003, 0.020, f.brow_ridge);
    let supraorb = sl(0.005, 0.025, f.brow_ridge);
    let cheek = sl(0.014, 0.060, f.cheekbone);
    let chin_proj = sl(0.020, 0.080, f.chin_projection); // stronger chin
    let chin_w = sl(0.045, 0.080, f.chin_width);  // min 0.045 prevents spike chin
    let nose_size = sl(0.024, 0.060, f.nose_size);
    let nose_br = sl(0.012, 0.028, f.nose_bridge);
    let lip_full = sl(0.6, 1.6, f.lip_fullness);
    let eye_x = sl(0.060, 0.100, f.eye_spacing);
    let eye_r = sl(0.024, 0.040, f.eye_size); // larger eyes
    let eye_z = sl(-0.210, -0.240, f.eye_depth);
    let fh_off = sl(-0.03, 0.03, f.forehead_height);
    let ear_s = sl(1.00, 1.80, f.ear_size); // larger ears to be visible after head scaling
    let fem = app.is_female;

    // Orbital socket depth — negative displacement at eye positions
    // Creates concavity that eyeballs sit inside
    // Deep sockets ensure eyes visibly sit in sculpted recesses
    let orb_depth = sl(-0.008, -0.018, f.eye_depth); // shallower sockets to avoid black void

    // ══════════════════════════════════════════════════════════════
    // SKULL LOFT — anatomically structured head surface
    // Key features: orbital sockets, prominent brow, zygomatic arch,
    // nasal bridge, occipital curve, frontal eminences
    // ══════════════════════════════════════════════════════════════
    // Skull depth ratio: rz ≈ 1.25x rx (real human skull proportions)
    let rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
        // ── CHIN — forward-projecting ──
        (1.46, body_ring(0.0, -0.04 * skd, chin_w, 0.08 * skd, &[
            (0.0, 0.3, chin_proj),
        ], n), sk),
        (1.49, body_ring(0.0, -0.02 * skd, (chin_w + 0.10 * skw) * 0.5, 0.10 * skd, &[
            (0.0, 0.3, chin_proj * 0.6),
        ], n), sk),
        (1.52, body_ring(0.0, -0.01 * skd, 0.13 * skw, 0.12 * skd, &[
            (0.0, 0.3, chin_proj * 0.3),
        ], n), sk),
        // Labiomental fold
        (1.55, body_ring(0.0, 0.0, 0.15 * skw, 0.16 * skd, &[
            (0.0, 0.20, -0.015),
            (hp, 0.25, jaw_w), (le, 0.25, jaw_w),
        ], n), sk),

        // ── JAW ──
        (1.58, body_ring(0.0, 0.01 * skd, 0.19 * skw, 0.19 * skd, &[
            (hp, 0.25, jawline), (le, 0.25, jawline),
            (0.0, 0.3, 0.012),
        ], n), sk),
        (1.61, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.20 * skd, &[
            (hp, 0.15, gonial), (le, 0.15, gonial),
            (hp - 0.3, 0.2, masseter), (le + 0.3, 0.2, masseter),
        ], n), sk),
        (1.63, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.21 * skd, &[
            (hp, 0.15, gonial * 0.7), (le, 0.15, gonial * 0.7),
            (hp - 0.3, 0.2, masseter * 0.9), (le + 0.3, 0.2, masseter * 0.9),
        ], n), sk),

        // ── MOUTH LEVEL ──
        (1.65, body_ring(0.0, 0.01 * skd, 0.195 * skw, 0.22 * skd, &[
            (0.0, 0.25, 0.020),
        ], n), sk),
        (1.67, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.23 * skd, &[
            (0.0, 0.28, 0.025),
            (hp - 0.3, 0.2, masseter * 0.7), (le + 0.3, 0.2, masseter * 0.7),
        ], n), sk),

        // ── NOSE BASE ──
        (1.69, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.23 * skd, &[
            (0.0, 0.12, 0.030),
        ], n), sk),
        (1.72, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.24 * skd, &[
            (0.0, 0.08, nose_size * 1.2),
            (0.12, 0.06, nose_br), (TAU - 0.12, 0.06, nose_br),
        ], n), sk),

        // ── CHEEKBONE / ORBITAL ──
        (1.74, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.24 * skd, &[
            (0.0, 0.06, nose_size * 0.6),
            (0.55, 0.18, cheek * 1.1), (TAU - 0.55, 0.18, cheek * 1.1),
            (re, 0.12, orb_depth * 0.5), (le, 0.12, orb_depth * 0.5),
        ], n), sk),
        (1.77, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.25 * skd, &[
            (0.0, 0.06, nose_br * 0.8),
            (re, 0.12, orb_depth), (le, 0.12, orb_depth),
            (0.55, 0.15, cheek * 0.6), (TAU - 0.55, 0.15, cheek * 0.6),
        ], n), sk),
        (1.80, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.25 * skd, &[
            (re, 0.12, orb_depth * 0.3), (le, 0.12, orb_depth * 0.3),
        ], n), sk),

        // ── BROW RIDGE — prominent forward shelf ──
        (1.82, body_ring(0.0, 0.0, 0.20 * skw, 0.25 * skd, &[
            (0.0, 0.50, brow_shelf),
            (0.0, 0.10, glabella),
            (re, 0.10, supraorb), (le, 0.10, supraorb),
        ], n), sk),
        (1.84, body_ring(0.0, 0.01 * skd, 0.20 * skw, 0.25 * skd, &[
            (0.0, 0.50, brow_boss * 0.8),
        ], n), sk),

        // ── FOREHEAD ──
        (1.87 + fh_off, body_ring(0.0, 0.02 * skd, 0.20 * skw, 0.25 * skd, &[], n), sk),

        // ── CRANIAL VAULT — more rings for smooth dome, rear bumps for occipital ──
        (1.89 + fh_off, body_ring(0.0, 0.03 * skd, 0.20 * skw, 0.25 * skd, &[
            (PI, 0.40, 0.025),
        ], n), sk),
        (1.91 + fh_off, body_ring(0.0, 0.04 * skd, 0.19 * skw, 0.24 * skd, &[
            (PI, 0.40, 0.030),
        ], n), sk),
        (1.93 + fh_off, body_ring(0.0, 0.05 * skd, 0.18 * skw, 0.23 * skd, &[
            (PI, 0.40, 0.032),
        ], n), sk),
        (1.95 + fh_off, body_ring(0.0, 0.05 * skd, 0.17 * skw, 0.22 * skd, &[
            (PI, 0.40, 0.028),
        ], n), sk),
        (1.97 + fh_off, body_ring(0.0, 0.04 * skd, 0.15 * skw, 0.20 * skd, &[
            (PI, 0.40, 0.020),
        ], n), sk),

        // ── CROWN — smooth convergence ──
        (1.99 + fh_off, body_ring(0.0, 0.03 * skd, 0.13 * skw, 0.16 * skd, &[], n), sk),
        (2.01 + fh_off, body_ring(0.0, 0.02 * skd, 0.10 * skw, 0.12 * skd, &[], n), sk),
        (2.03 + fh_off, body_ring(0.0, 0.01 * skd, 0.06 * skw, 0.07 * skd, &[], n), sk),
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
        mesh::ellipsoid_tris(tris, side * eye_x, 1.82, -0.310,
            0.030, brow_thick, 0.012, 0, darken(skin, 0.65));
        // Brow tail — follows brow ridge
        mesh::ellipsoid_tris(tris, side * (eye_x + 0.014), 1.823, -0.290,
            0.012, brow_thick * 0.5, 0.008, 0, darken(skin, 0.60));
    }

    // ══════════════════════════════════════════════════════════════
    // NOSE — anatomically defined: bridge with plane changes, defined
    // alar wings, visible columella, enclosed nostrils
    // ══════════════════════════════════════════════════════════════
    let ns = nose_size;
    let nb = nose_br;
    // Nose bridge — prominent ridge
    mesh::ellipsoid_tris(tris, 0.0, 1.75, -0.270, nb * 2.0, 0.050, nb * 1.8, 1, sk);
    // Nose tip — large, prominent
    mesh::ellipsoid_tris(tris, 0.0, 1.715, -0.290, ns * 1.0, 0.030, ns * 0.9, 1, darken(sk, 0.96));
    // Alar wings — wider and more visible
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * ns * 0.80, 1.712, -0.260,
            ns * 0.70, 0.025, ns * 0.55, 1, sk);
    }
    // Nostrils
    for &side in &[-1.0f32, 1.0] {
        mesh::ellipsoid_tris(tris, side * ns * 0.35, 1.706, -0.268,
            ns * 0.32, 0.012, 0.012, 0, darken(sk, 0.30));
    }

    // ══════════════════════════════════════════════════════════════
    // MOUTH — Cupid's bow upper lip, full lower lip, vermilion borders
    // ══════════════════════════════════════════════════════════════
    let lip_col = if fem { 0xFFCC8888 } else { 0xFFBB8877 };
    let lo_lip_col = if fem { 0xFFDD9999 } else { 0xFFCC9988 };
    let lf = lip_full;

    // Muzzle — forward projection for mouth area
    mesh::ellipsoid_tris(tris, 0.0, 1.668, -0.210, 0.048, 0.026, 0.028, 1, sk);

    // Upper lip
    mesh::ellipsoid_tris(tris, 0.0, 1.676, -0.225, 0.038, 0.010 * lf, 0.016 * lf, 1, lip_col);
    // Lower lip
    mesh::ellipsoid_tris(tris, 0.0, 1.656, -0.210, 0.042, 0.013 * lf, 0.018 * lf, 1, lo_lip_col);
    // Mouth line
    push_box(tris, 0.0, 1.666, -0.218, 0.034, 0.001, 0.004, darken(sk, 0.40));

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
    // HAT — headwear generation
    // ══════════════════════════════════════════════════════════════
    if let Some(jc) = is_job_hat {
        gen_job_hat(tris, jc);
    } else {
        gen_hat(tris, app);
    };
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
        // Base — matches torso top ring exactly (same dims + bump) for seamless junction
        (1.48, body_ring(0.0, 0.0, rx, rz, &[
            (PI, 0.5, 0.035 * m),             // trapezius — matches torso top
        ], n), skin),
        // Mid-neck — SCM prominent, laryngeal prominence
        (1.49, body_ring(0.0, 0.0, rx * 0.96, rz * 0.96, &[
            (PI * 0.5, 0.28, 0.018 * m),      // SCM right — diagonal cord
            (PI * 1.5, 0.28, 0.018 * m),      // SCM left
            (0.0, 0.12, if props.has_adams_apple { 0.014 } else { 0.005 }), // larynx
            (PI, 0.4, 0.015 * m),             // nuchal muscles
        ], n), skin),
        // Upper neck — tapers, SCM fading
        (nt, body_ring(0.0, 0.0, rx * 0.88, rz * 0.88, &[
            (PI * 0.5, 0.25, 0.010 * m),      // SCM insertion
            (PI * 1.5, 0.25, 0.010 * m),
            (0.0, 0.10, if props.has_adams_apple { 0.008 } else { 0.003 }),
        ], n), skin),
    ];

    mesh::loft_y_tris(tris, &rings);
}

/// Generate hat/headwear with high detail
fn gen_hat(tris: &mut Vec<WorldTri>, app: &NpcAppearance) -> bool {
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
            // Forehead wig edge detail
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

/// Nude torso — GLTF-extracted anatomical contours lofted into continuous mesh.
/// Crotch taper and shoulder/deltoid/trap zones remain parametric.
fn gen_nude_torso(tris: &mut Vec<WorldTri>, skin: u32, props: &BodyProportions, n: usize) {
    let sk = skin;
    let m = props.muscle_def;
    let sk_deep = darken(sk, 1.0 - 0.07 * m);
    let nipple_col = darken(sk, 0.78);

    use std::f32::consts::{PI, TAU};

    // Scale factors for each torso zone (1.0 for male baseline)
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

    let breast_bump = |y: f32| -> f32 {
        if !props.has_breasts { return 0.0; }
        let by = props.breast_y;
        let dy = (y - by) / 0.10;
        let amp = props.breast_rz * 1.3;
        amp * (-0.5 * dy * dy).exp()
    };

    // Base body half-widths interpolated by Y (from original ring definitions)
    let base_dims = |y: f32| -> (f32, f32) {
        const CP: [(f32, f32, f32); 13] = [
            (0.82, 0.04, 0.04), (0.86, 0.10, 0.08), (0.89, 0.14, 0.11),
            (0.92, 0.18, 0.14), (0.96, 0.17, 0.13), (1.00, 0.15, 0.13),
            (1.04, 0.155, 0.14), (1.08, 0.17, 0.15), (1.12, 0.19, 0.16),
            (1.16, 0.20, 0.17), (1.21, 0.21, 0.20), (1.26, 0.22, 0.21),
            (1.32, 0.22, 0.19),
        ];
        if y <= CP[0].0 { return (CP[0].1, CP[0].2); }
        if y >= CP[12].0 { return (CP[12].1, CP[12].2); }
        for i in 0..12 {
            if y <= CP[i + 1].0 {
                let t = (y - CP[i].0) / (CP[i + 1].0 - CP[i].0);
                return (CP[i].1 + t * (CP[i + 1].1 - CP[i].1),
                        CP[i].2 + t * (CP[i + 1].2 - CP[i].2));
            }
        }
        (0.18, 0.14)
    };

    let mut rings: Vec<(f32, Vec<[f32; 2]>, u32)> = Vec::with_capacity(34);

    // ── CROTCH TAPER — smooth transition from point to hip width ──
    {
        let sh = s(0.86);
        rings.push((0.78, body_ring(0.0, 0.0, 0.02, 0.02, &[], n), sk));
        rings.push((0.80, body_ring(0.0, 0.0, 0.04, 0.04, &[], n), sk));
        rings.push((0.82, body_ring(0.0, 0.0, 0.07 * sh, 0.06 * sh, &[], n), sk));
        rings.push((0.84, body_ring(0.0, 0.0, 0.10 * sh, 0.08 * sh, &[], n), sk));
    }

    // ── ANATOMICAL TORSO ZONE (Y≈0.837 to Y≈1.291) ──
    for i in 0..anatomy::torso_ring_count() {
        let y = anatomy::torso_ring_y(i);
        let (brx, brz) = base_dims(y);
        let sf = s(y);
        let rx = brx * sf;
        let rz = brz * sf;
        let mut pts = anatomy::torso_ring(y, rx, rz, m, n);

        // Add breast bumps as radial displacement at breast angles
        let bb = breast_bump(y);
        if bb > 0.001 {
            for pt in pts.iter_mut() {
                let theta = pt[0].atan2(-pt[1]);
                let st = theta.sin();
                let ct = theta.cos();
                let mut dr = 0.0f32;
                for &(center, width) in &[(0.30f32, 0.55f32), (-0.30, 0.55)] {
                    let mut diff = theta - center;
                    if diff > PI { diff -= TAU; }
                    if diff < -PI { diff += TAU; }
                    dr += bb * (-0.5 * (diff / width).powi(2)).exp();
                }
                pt[0] += dr * st;
                pt[1] -= dr * ct;
            }
        }

        rings.push((y, pts, sk));
    }

    // ── SHOULDER / DELTOID / TRAP ZONE (parametric — above anatomy data) ──
    {
        let hp = PI * 0.5;
        let sf = s(1.32);
        let da = props.shoulder_deltoid_amp;
        rings.push((1.32, body_ring(0.0, 0.0, 0.22 * sf, 0.19 * sf, &[
            (0.35, 0.35, 0.035 * m), (-0.35, 0.35, 0.035 * m),
            (0.0, 0.10, -0.010),
            (hp, 0.30, 0.045 * m), (PI + hp, 0.30, 0.045 * m),
            (PI, 0.15, -0.008 * m),
            (PI - 0.5, 0.25, 0.028 * m), (PI + 0.5, 0.25, 0.028 * m),
        ], n), sk));
        rings.push((1.36, body_ring(0.0, 0.0, 0.20 * sf, 0.19 * sf, &[
            (hp, 0.35, da + 0.03), (PI + hp, 0.35, da + 0.03),
            (0.5, 0.3, 0.020 * m), (-0.5, 0.3, 0.020 * m),
            (PI - 0.5, 0.3, 0.025 * m), (PI + 0.5, 0.3, 0.025 * m),
            (PI, 0.5, 0.05 * sf),
        ], n), sk));
        rings.push((1.39, body_ring(0.0, 0.0, 0.18 * sf, 0.18 * sf, &[
            (hp, 0.35, da + 0.05), (PI + hp, 0.35, da + 0.05),
            (0.5, 0.3, 0.015 * m), (-0.5, 0.3, 0.015 * m),
            (PI - 0.5, 0.3, 0.025 * m), (PI + 0.5, 0.3, 0.025 * m),
            (PI, 0.5, 0.055 * sf),
        ], n), sk));
        rings.push((1.42, body_ring(0.0, 0.0, 0.16 * sf, 0.17 * sf, &[
            (hp, 0.35, da + 0.06), (PI + hp, 0.35, da + 0.06),
            (0.0, 0.4, 0.015 * sf),
            (PI, 0.5, 0.055 * sf),
        ], n), sk));
        rings.push((1.43, body_ring(0.0, 0.0, 0.14 * sf, 0.15 * sf, &[
            (hp, 0.30, da * 0.4 + 0.03), (PI + hp, 0.30, da * 0.4 + 0.03),
            (PI, 0.5, 0.050 * sf),
        ], n), sk));
        rings.push((1.44, body_ring(0.0, 0.0, 0.13 * sf, 0.13 * sf, &[
            (hp, 0.25, 0.015 * sf), (PI + hp, 0.25, 0.015 * sf),
            (PI, 0.5, 0.045 * sf),
        ], n), sk));
    }
    {
        let sf = s(1.45);
        rings.push((1.45, body_ring(0.0, 0.0, 0.13 * sf, 0.12 * sf, &[
            (PI, 0.5, 0.040 * sf),
        ], n), sk));
    }
    {
        let sf = s(1.46);
        rings.push((1.46, body_ring(0.0, 0.0, 0.14 * sf, 0.12 * sf, &[
            (PI, 0.5, 0.04 * sf),
            (PI - 0.4, 0.3, 0.012 * sf), (PI + 0.4, 0.3, 0.012 * sf),
        ], n), sk));
    }
    rings.push((1.48, body_ring(0.0, 0.0, props.neck_rx, props.neck_rz, &[
        (PI, 0.5, 0.035 * m),
    ], n), sk));

    // ── NECK (merged into torso loft for seamless junction) ──
    {
        let rx = props.neck_rx;
        let rz = props.neck_rz;
        let nt = props.neck_top;
        rings.push((1.49, body_ring(0.0, 0.0, rx * 0.96, rz * 0.96, &[
            (PI * 0.5, 0.28, 0.018 * m), (PI * 1.5, 0.28, 0.018 * m),
            (0.0, 0.12, if props.has_adams_apple { 0.014 } else { 0.005 }),
            (PI, 0.4, 0.015 * m),
        ], n), sk));
        rings.push((nt, body_ring(0.0, 0.0, rx * 0.88, rz * 0.88, &[
            (PI * 0.5, 0.25, 0.010 * m), (PI * 1.5, 0.25, 0.010 * m),
            (0.0, 0.10, if props.has_adams_apple { 0.008 } else { 0.003 }),
        ], n), sk));
    }

    let torso_base = tris.len();
    mesh::loft_y_tris(tris, &rings);

    // ── CURVATURE-BASED COLOR VARIATION ──
    // Muscle peaks (convex) get lighter skin, grooves (concave) get darker.
    // Computed as deviation from smooth ellipse at each triangle's position.
    if m > 0.1 {
        for tri in &mut tris[torso_base..] {
            let cy = (tri.v[0][1] + tri.v[1][1] + tri.v[2][1]) / 3.0;
            if cy < 0.86 || cy > 1.29 { continue; } // anatomy zone only
            let cx = (tri.v[0][0] + tri.v[1][0] + tri.v[2][0]) / 3.0;
            let cz = (tri.v[0][2] + tri.v[1][2] + tri.v[2][2]) / 3.0;
            let r_actual = (cx * cx + cz * cz).sqrt();
            // Smooth ellipse radius at this angle and Y
            let (brx, brz) = base_dims(cy);
            let sf = s(cy);
            let theta = cx.atan2(-cz);
            let ex = brx * sf * theta.sin();
            let ez = brz * sf * theta.cos();
            let r_smooth = (ex * ex + ez * ez).sqrt();
            if r_smooth < 0.01 { continue; }
            let dev = (r_actual - r_smooth) / r_smooth;
            // ±3% brightness shift, scaled by muscle_def (subtle to avoid overlap banding)
            let shift = (dev * 4.0).clamp(-0.03, 0.03) * m;
            if shift.abs() > 0.003 {
                tri.color = darken(tri.color, 1.0 + shift);
            }
        }
    }

    // ── MINIMAL SURFACE DETAIL ──
    let nip_y = if props.has_breasts { 1.23 } else { 1.20 };
    for &side in &[-1.0f32, 1.0] {
        let nx = side * 0.12 * sc;
        mesh::sphere_tris(tris, nx, nip_y, -0.22 * sc, 0.004, 0, nipple_col);
        mesh::ellipsoid_tris(tris, nx, nip_y, -0.215 * sc, 0.012, 0.012, 0.003, 0, darken(nipple_col, 0.95));
    }
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

    // ── JOINT POSITIONS — wrist at hip level, not shin ──
    let shoulder = [side * props.shoulder_joint_x, 1.42, 0.0];
    let elbow = [side * (props.shoulder_joint_x + 0.08), 1.10, fwd * 0.30];
    let wrist = [side * (props.shoulder_joint_x + 0.04), 0.78, fwd * 0.12 - bend];

    let shoulder_y = 1.44;
    let elbow_y = 1.10;
    let wrist_y = 0.78;
    let arm_span = shoulder_y - wrist_y; // 0.66

    // GLTF arm x_offset range
    let gltf_max_x = anatomy::arm_ring_x(anatomy::arm_ring_count() - 1);

    // rx/rz profile along the arm — wrist at Y=0.78 (hip level after stretch)
    let arm_dims: &[(f32, f32, f32)] = &[
        (1.44, 0.145, 0.135),  // deltoid cap
        (1.43, 0.140, 0.128),  // deltoid
        (1.42, 0.135, 0.122),  // deltoid
        (1.40, 0.128, 0.114),  // deltoid-to-upper-arm
        (1.38, 0.120, 0.106),  // upper arm
        (1.35, 0.114, 0.100),  // bicep
        (1.30, 0.110, 0.096),  // bicep peak
        (1.24, 0.105, 0.092),  // mid-upper arm
        (1.18, 0.098, 0.086),  // lower bicep
        (1.12, 0.090, 0.080),  // elbow approach
        (1.10, 0.082, 0.074),  // elbow
        (1.08, 0.086, 0.076),  // below elbow
        (1.02, 0.088, 0.078),  // forearm belly
        (0.96, 0.084, 0.074),  // mid forearm
        (0.90, 0.076, 0.066),  // forearm taper
        (0.84, 0.066, 0.056),  // lower forearm
        (0.78, 0.054, 0.046),  // wrist
    ];

    let arm_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = arm_dims.iter().map(|&(y, rx, rz)| {
        // Map arm Y to GLTF x_offset (0 at shoulder, gltf_max_x at wrist)
        let t_arm = ((shoulder_y - y) / arm_span).clamp(0.0, 1.0);
        let x_offset = t_arm * gltf_max_x;

        // Get GLTF-calibrated contour (centroid-origin)
        let mut ring = anatomy::arm_ring(x_offset, rx * a, rz * a, m, n);

        // Mirror for left arm: negate X of each point, reverse for winding
        if side < 0.0 {
            for pt in ring.iter_mut() { pt[0] = -pt[0]; }
            ring.reverse();
        }

        // Interpolate center position along shoulder→elbow→wrist path
        let (cx, cz) = if y >= elbow_y {
            let t_lin = ((shoulder_y - y) / (shoulder_y - elbow_y)).clamp(0.0, 1.0);
            let t = t_lin * t_lin * t_lin;
            (shoulder[0] * (1.0 - t) + elbow[0] * t, shoulder[2] * (1.0 - t) + elbow[2] * t)
        } else {
            let t = (elbow_y - y) / (elbow_y - wrist_y);
            (elbow[0] * (1.0 - t) + wrist[0] * t, elbow[2] * (1.0 - t) + wrist[2] * t)
        };

        // Offset contour to arm center
        for pt in ring.iter_mut() {
            pt[0] += cx;
            pt[1] += cz;
        }

        (y, ring, sk)
    }).collect();

    let arm_base = tris.len();
    mesh::loft_y_tris(tris, &arm_rings);

    // Arm top is open but hidden inside the torso shoulder zone.

    // ── CURVATURE-BASED COLOR VARIATION (arms) ──
    if m > 0.1 {
        for tri in &mut tris[arm_base..] {
            let cy = (tri.v[0][1] + tri.v[1][1] + tri.v[2][1]) / 3.0;
            if cy < 0.80 || cy > 1.10 { continue; } // forearm only, avoid torso overlap
            let cx = (tri.v[0][0] + tri.v[1][0] + tri.v[2][0]) / 3.0;
            let cz = (tri.v[0][2] + tri.v[1][2] + tri.v[2][2]) / 3.0;
            // Find arm center at this Y via same path interpolation
            let (acx, acz) = if cy >= elbow_y {
                let t_lin = ((shoulder_y - cy) / (shoulder_y - elbow_y)).clamp(0.0, 1.0);
                let t = t_lin * t_lin * t_lin;
                (shoulder[0] * (1.0 - t) + elbow[0] * t, shoulder[2] * (1.0 - t) + elbow[2] * t)
            } else {
                let t = ((elbow_y - cy) / (elbow_y - wrist_y)).clamp(0.0, 1.0);
                (elbow[0] * (1.0 - t) + wrist[0] * t, elbow[2] * (1.0 - t) + wrist[2] * t)
            };
            let dx = cx - acx;
            let dz = cz - acz;
            let r_actual = (dx * dx + dz * dz).sqrt();
            // Interpolate rx/rz at this Y
            let mut rx_here = 0.0f32;
            let mut rz_here = 0.0f32;
            for j in 0..arm_dims.len() - 1 {
                if cy >= arm_dims[j + 1].0 && cy <= arm_dims[j].0 {
                    let t = (arm_dims[j].0 - cy) / (arm_dims[j].0 - arm_dims[j + 1].0);
                    rx_here = arm_dims[j].1 + t * (arm_dims[j + 1].1 - arm_dims[j].1);
                    rz_here = arm_dims[j].2 + t * (arm_dims[j + 1].2 - arm_dims[j].2);
                    break;
                }
            }
            if rx_here < 0.01 { continue; }
            let theta = dx.atan2(-dz);
            let ex = rx_here * a * theta.sin();
            let ez = rz_here * a * theta.cos();
            let r_smooth = (ex * ex + ez * ez).sqrt();
            if r_smooth < 0.005 { continue; }
            let dev = (r_actual - r_smooth) / r_smooth;
            let shift = (dev * 4.0).clamp(-0.03, 0.03) * m;
            if shift.abs() > 0.003 {
                tri.color = darken(tri.color, 1.0 + shift);
            }
        }
    }

    // ── CULL ARM TRIS INSIDE TORSO (prevents Z-fighting overlap banding) ──
    {
        let mut keep = arm_base;
        for i in arm_base..tris.len() {
            let cy = (tris[i].v[0][1] + tris[i].v[1][1] + tris[i].v[2][1]) / 3.0;
            let cull = cy > 1.20 && {
                let cx = (tris[i].v[0][0] + tris[i].v[1][0] + tris[i].v[2][0]) / 3.0;
                let cz = (tris[i].v[0][2] + tris[i].v[1][2] + tris[i].v[2][2]) / 3.0;
                // Approximate torso ellipse extent at this Y
                let trx = if cy < 1.32 { 0.22 } else if cy < 1.40 { 0.19 } else { 0.15 };
                let trz = if cy < 1.32 { 0.19 } else if cy < 1.40 { 0.18 } else { 0.14 };
                let nx = cx / trx;
                let nz = cz / trz;
                nx * nx + nz * nz < 0.82 // cull tris inside torso
            };
            if !cull {
                if keep != i { tris.swap(keep, i); }
                keep += 1;
            }
        }
        tris.truncate(keep);
    }

    // Armpit fill removed — was creating floating debris geometry.

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
    // Y coordinates relative to ankle — foot geometry follows ankle position
    let ay = ankle[1]; // default 0.08 in standing
    let y = |offset: f32| -> f32 { ay + offset };

    // Foot scale — Z offsets multiplied to reach ~26cm foot length
    let fz = 1.6f32; // foot Z scale

    // ── HEEL — calcaneus, rounded posterior ──
    mesh::ellipsoid_tris(tris, lx, y(-0.052), az + 0.035 * fz, 0.038, 0.030, 0.040, 0, sk);
    mesh::ellipsoid_tris(tris, lx, y(-0.035), az + 0.04 * fz, 0.022, 0.016, 0.020, 0, sk_dk);

    // ── MIDFOOT — arch structure ──
    mesh::ellipsoid_tris(tris, lx, y(-0.038), az - 0.02 * fz, 0.042, 0.020, 0.070, 0, sk);
    mesh::ellipsoid_tris(tris, lx + side * 0.022, y(-0.065), az - 0.01 * fz, 0.022, 0.016, 0.060, 0, sk);
    mesh::ellipsoid_tris(tris, lx - side * 0.016, y(-0.050), az - 0.005 * fz, 0.020, 0.022, 0.055, 0, sk);

    // ── FOREFOOT — ball of foot ──
    mesh::ellipsoid_tris(tris, lx, y(-0.062), az - 0.065 * fz, 0.046, 0.018, 0.030, 0, sk);
    mesh::ellipsoid_tris(tris, lx - side * 0.022, y(-0.065), az - 0.068 * fz, 0.018, 0.014, 0.018, 0, sk_dk);
    mesh::ellipsoid_tris(tris, lx + side * 0.027, y(-0.067), az - 0.060 * fz, 0.014, 0.012, 0.016, 0, sk_dk);

    // ── EXTENSOR TENDONS ──
    for ti in 0..4 {
        let tx = lx + (ti as f32 - 1.5) * side * 0.011;
        mesh::ellipsoid_tris(tris, tx, y(-0.032), az - 0.030 * fz, 0.003, 0.004, 0.040, 0, darken(sk, 0.97));
    }

    // ── TOES ──
    let btx = lx - side * 0.024;
    mesh::ellipsoid_tris(tris, btx, y(-0.067), az - 0.095 * fz, 0.016, 0.012, 0.022, 0, sk);
    mesh::ellipsoid_tris(tris, btx, y(-0.068), az - 0.118 * fz, 0.014, 0.011, 0.016, 0, sk);
    push_box(tris, btx, y(-0.060), az - 0.128 * fz, 0.009, 0.003, 0.007, nail_col);

    for ti in 0..4 {
        let tx = lx - side * 0.008 + (ti as f32 + 0.5) * side * 0.013;
        let toe_len = 0.016 - ti as f32 * 0.002;
        let toe_r = 0.009 - ti as f32 * 0.001;
        let tz = az - 0.088 * fz + ti as f32 * 0.005;
        mesh::ellipsoid_tris(tris, tx, y(-0.070), tz - toe_len * 0.5, toe_r, 0.007, toe_len, 0, sk);
        push_box(tris, tx, y(-0.065), tz - toe_len + 0.002, 0.006, 0.002, 0.005, nail_col);
    }

    // ── SOLE ──
    mesh::ellipsoid_tris(tris, lx, y(-0.076), az - 0.02 * fz, 0.042, 0.005, 0.085, 0, sk_dk);
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

/// Nude leg — GLTF-extracted anatomical contours lofted from hip to ankle.
fn gen_nude_leg(
    tris: &mut Vec<WorldTri>, side: f32, fwd: f32, knee_bend: f32, skin: u32,
    props: &BodyProportions, n: usize,
) {
    let sk = skin;
    let l = props.leg_rx_scale;
    let m = props.muscle_def;

    let lx = side * props.hip_joint_x;
    let ankle = [lx, 0.08, fwd * 0.25 - knee_bend * 0.4];

    // GLTF → game Y mapping (ankle-to-hip scale)
    const GLTF_LO: f32 = 0.04;
    const GLTF_HI: f32 = 0.62;
    const GAME_LO: f32 = 0.08;
    const GAME_HI: f32 = 0.92;
    const Y_SCALE: f32 = (GAME_HI - GAME_LO) / (GLTF_HI - GLTF_LO);
    let gltf_to_game = |gy: f32| -> f32 { GAME_LO + (gy - GLTF_LO) * Y_SCALE };

    // Base leg dimensions interpolated by game Y — ~10% thicker than original
    let base_dims = |y: f32| -> (f32, f32) {
        const CP: [(f32, f32, f32); 16] = [
            (0.08, 0.052, 0.050), (0.14, 0.063, 0.057), (0.22, 0.076, 0.070),
            (0.30, 0.096, 0.086), (0.36, 0.103, 0.094), (0.42, 0.098, 0.090),
            (0.46, 0.096, 0.090), (0.48, 0.092, 0.086), (0.50, 0.096, 0.090),
            (0.54, 0.120, 0.110), (0.62, 0.148, 0.134), (0.70, 0.165, 0.148),
            (0.78, 0.174, 0.156), (0.84, 0.172, 0.154), (0.88, 0.167, 0.150),
            (0.92, 0.160, 0.145),
        ];
        if y <= CP[0].0 { return (CP[0].1, CP[0].2); }
        if y >= CP[15].0 { return (CP[15].1, CP[15].2); }
        for i in 0..15 {
            if y <= CP[i + 1].0 {
                let t = (y - CP[i].0) / (CP[i + 1].0 - CP[i].0);
                return (CP[i].1 + t * (CP[i + 1].1 - CP[i].1),
                        CP[i].2 + t * (CP[i + 1].2 - CP[i].2));
            }
        }
        (0.10, 0.09)
    };

    // CZ interpolation along hip→knee→ankle path (walking animation)
    let hip_y = 0.92f32;
    let knee_y = 0.48f32;
    let ankle_y = 0.08f32;
    let knee_cz = fwd * 0.5;
    let ankle_cz = fwd * 0.25 - knee_bend * 0.4;

    let compute_cz = |y: f32| -> f32 {
        if y >= knee_y {
            let t = (hip_y - y) / (hip_y - knee_y);
            knee_cz * t
        } else {
            let t = (knee_y - y) / (knee_y - ankle_y);
            knee_cz * (1.0 - t) + ankle_cz * t
        }
    };

    // Build rings from GLTF anatomy data (ascending game Y: ankle → hip)
    let mut leg_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = Vec::with_capacity(35);

    for i in 0..anatomy::leg_ring_count() {
        let gltf_y = anatomy::leg_ring_y(i);
        let game_y = gltf_to_game(gltf_y);
        let (brx, brz) = base_dims(game_y);
        let cz = compute_cz(game_y);

        let mut pts = anatomy::leg_ring(gltf_y, brx * l, brz * l, m, n);
        // Mirror for left leg (negate X, reverse winding)
        if side < 0.0 {
            for pt in pts.iter_mut() { pt[0] = -pt[0]; }
            pts.reverse();
        }
        // Offset to limb center
        for pt in pts.iter_mut() {
            pt[0] += lx;
            pt[1] += cz;
        }

        leg_rings.push((game_y, pts, sk));
    }

    let leg_base = tris.len();
    mesh::loft_y_tris(tris, &leg_rings);

    // Leg top is open but hidden by the torso from normal viewing angles.

    // ── CURVATURE-BASED COLOR VARIATION (legs) ──
    if m > 0.1 {
        for tri in &mut tris[leg_base..] {
            let cy = (tri.v[0][1] + tri.v[1][1] + tri.v[2][1]) / 3.0;
            if cy < 0.10 || cy > 0.90 { continue; }
            let cx = (tri.v[0][0] + tri.v[1][0] + tri.v[2][0]) / 3.0;
            let cz = (tri.v[0][2] + tri.v[1][2] + tri.v[2][2]) / 3.0;
            // Distance from leg center axis
            let dx = cx - lx;
            let dz = cz - compute_cz(cy);
            let r_actual = (dx * dx + dz * dz).sqrt();
            // Smooth ellipse radius at this angle
            let (brx, brz) = base_dims(cy);
            let theta = dx.atan2(-dz);
            let ex = brx * l * theta.sin();
            let ez = brz * l * theta.cos();
            let r_smooth = (ex * ex + ez * ez).sqrt();
            if r_smooth < 0.005 { continue; }
            let dev = (r_actual - r_smooth) / r_smooth;
            let shift = (dev * 4.0).clamp(-0.03, 0.03) * m;
            if shift.abs() > 0.003 {
                tri.color = darken(tri.color, 1.0 + shift);
            }
        }
    }

    // ── BARE FOOT ──
    gen_bare_foot(tris, ankle, side, sk);
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
    let n = 24;

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
    let co = 0.055; // clothing offset over body (increased to prevent Z-fighting)

    // Breast coverage bump — matches nude body breast profile so coat covers chest
    let breast_bump = |y: f32| -> f32 {
        if !props.has_breasts { return 0.0; }
        let dy = (y - props.breast_y) / 0.10;
        (props.breast_rz * 1.3 + 0.015) * (-0.5 * dy * dy).exp() // body breast amp + margin
    };

    // ── COAT/VEST SHELL — lofted tube matching body contour + offset ──
    // Anatomy zone uses GLTF contours for body-matching coat shape.
    // Muscle bumps attenuated to 50% for smooth cloth surface.
    if app.has_coat {
        use std::f32::consts::TAU;
        // Base body dimensions for coat (same profile as gen_nude_torso)
        let coat_base_dims = |y: f32| -> (f32, f32) {
            const CP: [(f32, f32, f32); 13] = [
                (0.82, 0.04, 0.04), (0.86, 0.10, 0.08), (0.89, 0.14, 0.11),
                (0.92, 0.18, 0.14), (0.96, 0.17, 0.13), (1.00, 0.15, 0.13),
                (1.04, 0.155, 0.14), (1.08, 0.17, 0.15), (1.12, 0.19, 0.16),
                (1.16, 0.20, 0.17), (1.21, 0.21, 0.20), (1.26, 0.22, 0.21),
                (1.32, 0.22, 0.19),
            ];
            if y <= CP[0].0 { return (CP[0].1, CP[0].2); }
            if y >= CP[12].0 { return (CP[12].1, CP[12].2); }
            for i in 0..12 {
                if y <= CP[i + 1].0 {
                    let t = (y - CP[i].0) / (CP[i + 1].0 - CP[i].0);
                    return (CP[i].1 + t * (CP[i + 1].1 - CP[i].1),
                            CP[i].2 + t * (CP[i + 1].2 - CP[i].2));
                }
            }
            (0.18, 0.14)
        };
        let coat_ring_parametric = |y: f32, rx: f32, rz: f32, bumps: &[(f32, f32, f32)]| -> (f32, Vec<[f32; 2]>, u32) {
            let sf = s(y);
            let ab: Vec<(f32, f32, f32)> = bumps.iter()
                .map(|&(a, w, amp)| (a, w, amp * 0.5))
                .collect();
            (y, body_ring(0.0, 0.0, rx * sf + co, rz * sf + co, &ab, n), coat)
        };
        // Anatomy coat ring: body contour + offset, half muscle_def, with breast bumps
        let coat_ring_anat = |y: f32| -> (f32, Vec<[f32; 2]>, u32) {
            let (brx, brz) = coat_base_dims(y);
            let sf = s(y);
            let rx = brx * sf + co;
            let rz = brz * sf + co;
            let cm = props.muscle_def * 0.5; // cloth smoothness
            let mut pts = anatomy::torso_ring(y, rx, rz, cm, n);
            // Add breast bumps (with margin for coat coverage)
            let bb = breast_bump(y);
            if bb > 0.001 {
                for pt in pts.iter_mut() {
                    let theta = pt[0].atan2(-pt[1]);
                    let st = theta.sin();
                    let ct = theta.cos();
                    let mut dr = 0.0f32;
                    for &(center, width) in &[(0.30f32, 0.55f32), (-0.30, 0.55)] {
                        let mut diff = theta - center;
                        if diff > PI { diff -= TAU; }
                        if diff < -PI { diff += TAU; }
                        dr += bb * (-0.5 * (diff / width).powi(2)).exp();
                    }
                    pt[0] += dr * st;
                    pt[1] -= dr * ct;
                }
            }
            (y, pts, coat)
        };

        let mut coat_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = Vec::with_capacity(30);
        // Below anatomy zone (parametric)
        coat_rings.push(coat_ring_parametric(0.78, 0.16, 0.14, &[(PI, 0.5, 0.04)]));
        coat_rings.push(coat_ring_parametric(0.82, 0.15, 0.13, &[(PI, 0.5, 0.04)]));
        coat_rings.push(coat_ring_parametric(0.88, 0.14, 0.12, &[(PI, 0.5, 0.04)]));
        // Anatomy zone (Y≈0.92 to Y≈1.26)
        for &y in &[0.92, 0.96, 1.00, 1.04, 1.08, 1.12, 1.16, 1.21, 1.26] {
            coat_rings.push(coat_ring_anat(y));
        }
        // Above anatomy zone (parametric shoulder/deltoid/neck)
        {
            let sf = s(1.32);
            let bb = breast_bump(1.32);
            let pm = props.muscle_def * 0.5;
            let mut b = vec![(hp, 0.45, 0.02), (PI + hp, 0.45, 0.02), (PI, 0.5, 0.02 * sf),
                (0.35, 0.35, 0.018 * pm), (-0.35, 0.35, 0.018 * pm)];
            if bb > 0.001 { b.push((0.30, 0.55, bb)); b.push((-0.30, 0.55, bb)); }
            coat_rings.push((1.32, body_ring(0.0, 0.0, 0.24 * sf + co, 0.20 * sf + co, &b, n), coat));
        }
        {
            let sf = s(1.36); let da = props.shoulder_deltoid_amp * 0.5;
            coat_rings.push((1.36, body_ring(0.0, 0.0, 0.22 * sf + co, 0.20 * sf + co, &[
                (hp, 0.45, da + 0.02), (PI + hp, 0.45, da + 0.02), (PI, 0.5, 0.030 * sf),
            ], n), coat));
        }
        {
            let sf = s(1.39); let da = props.shoulder_deltoid_amp * 0.5;
            coat_rings.push((1.39, body_ring(0.0, 0.0, 0.20 * sf + co, 0.19 * sf + co, &[
                (hp, 0.45, da + 0.03), (PI + hp, 0.45, da + 0.03), (PI, 0.5, 0.035 * sf),
            ], n), coat));
        }
        {
            let sf = s(1.42); let da = props.shoulder_deltoid_amp * 0.5;
            coat_rings.push((1.42, body_ring(0.0, 0.0, 0.18 * sf + co, 0.18 * sf + co, &[
                (hp, 0.45, da + 0.04), (PI + hp, 0.45, da + 0.04), (PI, 0.5, 0.035 * sf),
            ], n), coat));
        }
        {
            let sf = s(1.44);
            coat_rings.push((1.44, body_ring(0.0, 0.0, 0.14 * sf + co, 0.14 * sf + co, &[
                (hp, 0.35, 0.02), (PI + hp, 0.35, 0.02), (PI, 0.5, 0.028 * sf),
            ], n), darken(coat, 0.92)));
        }
        coat_rings.push((1.48, body_ring(0.0, 0.0, props.neck_rx + co, props.neck_rz + co, &[], n),
            darken(coat, 0.88)));
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
                (0.88, body_ring(tx, back_z, 0.06, 0.020, &[], 8), coat),
                (0.74, body_ring(tx, back_z + 0.02 + tail_sway * 0.3, 0.065, 0.018, &[], 8), coat),
                (0.58, body_ring(tx, back_z + 0.04 + tail_sway * 0.6, 0.07, 0.015, &[], 8), coat),
                (0.42, body_ring(tx, back_z + 0.06 + tail_sway, 0.075, 0.012, &[], 8),
                    darken(coat, 0.95)),
            ];
            mesh::loft_y_tris(tris, &tail_rings);
        }
        // Front skirt panels
        let front_z = -(0.14 * sh + co);
        for &tx in &[-0.09f32, 0.09] {
            let skirt_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
                (0.88, body_ring(tx, front_z, 0.05, 0.015, &[], 8), coat),
                (0.76, body_ring(tx, front_z - 0.01 - tail_sway * 0.15, 0.055, 0.012, &[], 8), coat),
                (0.64, body_ring(tx, front_z - 0.02 - tail_sway * 0.3, 0.055, 0.010, &[], 8),
                    darken(coat, 0.96)),
            ];
            mesh::loft_y_tris(tris, &skirt_rings);
        }

        // ── CAPE (short shoulder mantle) ──
        if app.has_cape {
            let cape_col = darken(coat, 0.88);
            let cape_z = 0.21 * sc + co;
            let cape_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
                (1.44, body_ring(0.0, cape_z * 0.5, 0.10, 0.02, &[], 8), cape_col),
                (1.36, body_ring(0.0, cape_z, 0.14, 0.03, &[], 8), cape_col),
                (1.26, body_ring(0.0, cape_z + 0.01, 0.16, 0.03, &[], 8), cape_col),
                (1.16, body_ring(0.0, cape_z + 0.02, 0.16, 0.03, &[], 8),
                    darken(cape_col, 0.95)),
            ];
            mesh::loft_y_tris(tris, &cape_rings);
        }
    } else {
        // No coat — vest/shirt shell (anatomy contours + offset)
        use std::f32::consts::TAU;
        let vco = 0.04;
        // Base body dimensions (same profile as gen_nude_torso)
        let vest_base_dims = |y: f32| -> (f32, f32) {
            const CP: [(f32, f32, f32); 13] = [
                (0.82, 0.04, 0.04), (0.86, 0.10, 0.08), (0.89, 0.14, 0.11),
                (0.92, 0.18, 0.14), (0.96, 0.17, 0.13), (1.00, 0.15, 0.13),
                (1.04, 0.155, 0.14), (1.08, 0.17, 0.15), (1.12, 0.19, 0.16),
                (1.16, 0.20, 0.17), (1.21, 0.21, 0.20), (1.26, 0.22, 0.21),
                (1.32, 0.22, 0.19),
            ];
            if y <= CP[0].0 { return (CP[0].1, CP[0].2); }
            if y >= CP[12].0 { return (CP[12].1, CP[12].2); }
            for i in 0..12 {
                if y <= CP[i + 1].0 {
                    let t = (y - CP[i].0) / (CP[i + 1].0 - CP[i].0);
                    return (CP[i].1 + t * (CP[i + 1].1 - CP[i].1),
                            CP[i].2 + t * (CP[i + 1].2 - CP[i].2));
                }
            }
            (0.18, 0.14)
        };
        let vest_ring_param = |y: f32, rx: f32, rz: f32| -> (f32, Vec<[f32; 2]>, u32) {
            let sf = s(y);
            let bb = breast_bump(y);
            let mut bumps: Vec<(f32, f32, f32)> = Vec::new();
            if bb > 0.001 { bumps.push((0.30, 0.55, bb)); bumps.push((-0.30, 0.55, bb)); }
            (y, body_ring(0.0, 0.0, rx * sf + vco, rz * sf + vco, &bumps, n), vest)
        };
        let vest_ring_anat = |y: f32| -> (f32, Vec<[f32; 2]>, u32) {
            let (brx, brz) = vest_base_dims(y);
            let sf = s(y);
            let mut pts = anatomy::torso_ring(y, brx * sf + vco, brz * sf + vco, props.muscle_def * 0.3, n);
            let bb = breast_bump(y);
            if bb > 0.001 {
                for pt in pts.iter_mut() {
                    let theta = pt[0].atan2(-pt[1]);
                    let st = theta.sin();
                    let ct = theta.cos();
                    let mut dr = 0.0f32;
                    for &(center, width) in &[(0.30f32, 0.55f32), (-0.30, 0.55)] {
                        let mut diff = theta - center;
                        if diff > PI { diff -= TAU; }
                        if diff < -PI { diff += TAU; }
                        dr += bb * (-0.5 * (diff / width).powi(2)).exp();
                    }
                    pt[0] += dr * st;
                    pt[1] -= dr * ct;
                }
            }
            (y, pts, vest)
        };
        let vest_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
            vest_ring_param(0.78, 0.16, 0.14),
            vest_ring_param(0.82, 0.15, 0.13),
            vest_ring_param(0.88, 0.14, 0.12),
            vest_ring_anat(0.92),
            vest_ring_anat(1.00),
            vest_ring_anat(1.08),
            vest_ring_anat(1.16),
            vest_ring_anat(1.26),
            vest_ring_param(1.34, 0.22, 0.19),
            vest_ring_param(1.40, 0.18, 0.18),
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
    // F8: Per-NPC belt color instead of uniform LEATHER_DARK
    let belt_col = app.belt_col;
    mesh::cylinder_tris(tris, 0.0, belt_y, 0.0, belt_r, 0.025, 12, belt_col);
    mesh::cylinder_tris(tris, 0.0, belt_y + 0.013, 0.0, belt_r + 0.003, 0.003, 12, darken(belt_col, 0.7));
    mesh::cylinder_tris(tris, 0.0, belt_y - 0.013, 0.0, belt_r + 0.003, 0.003, 12, darken(belt_col, 0.7));
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
            (0.92, 0.142, 0.128, vec![
                (PI + hp, 0.70, 0.092), (PI, 0.60, 0.065), (0.0, 0.60, 0.060),
            ]),
            (0.88, 0.148, 0.132, vec![
                (PI + hp, 0.70, 0.088), (PI, 0.60, 0.060), (0.0, 0.60, 0.055),
            ]),
            (0.84, 0.154, 0.136, vec![
                (PI + hp, 0.65, 0.070), (PI, 0.55, 0.048), (0.0, 0.55, 0.044),
            ]),
            (0.78, 0.156, 0.138, vec![
                (PI + hp, 0.60, 0.054), (PI, 0.50, 0.038), (0.0, 0.50, 0.033),
            ]),
            (0.70, 0.146, 0.130, vec![
                (PI + hp, 0.55, 0.032), (PI, 0.45, 0.022), (0.0, 0.45, 0.016),
            ]),
            (0.62, 0.128, 0.116, vec![
                (PI + hp, 0.50, 0.016),
            ]),
            (0.54, 0.100, 0.090, vec![]),
            (0.48, 0.076, 0.070, vec![]),
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
            (0.060 + pco) * l, 0.020, 8, darken(pants_col, 0.93));
        for ki in 0..3 {
            let kbx = knee_pt[0] + side * (0.035 + ki as f32 * 0.012) * l;
            mesh::sphere_tris(tris, kbx, knee_pt[1] - 0.02, knee_pt[2] - (0.050 + pco) * l,
                0.005, 0, BUCKLE_BRASS);
        }

        // Boot loft (knee → ankle) — same interpolation as gen_nude_leg lower half
        let bco = 0.018;
        // Boot radii sized to cover legs
        let boot_data: [(f32, f32, f32); 7] = [
            (0.48, 0.076, 0.070),
            (0.42, 0.082, 0.076),
            (0.36, 0.088, 0.080),
            (0.30, 0.080, 0.072),
            (0.22, 0.062, 0.056),
            (0.14, 0.050, 0.046),
            (0.08, 0.042, 0.040),
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
        // F6: Larger boot foot (was 0.10x0.07x0.15)
        mesh::beveled_box_tris(tris, ankle_pt[0], ankle_pt[1] - 0.045, ankle_pt[2] - 0.06,
            0.14 * l, 0.09, 0.22 * l, 0.02, bc);
        push_box(tris, ankle_pt[0], ankle_pt[1] - 0.09, ankle_pt[2] - 0.06,
            0.13 * l, 0.006, 0.20 * l, darken(bc, 0.65));
        push_box(tris, ankle_pt[0], ankle_pt[1] - 0.065, ankle_pt[2] + 0.08 * l,
            0.06 * l, 0.025, 0.04 * l, darken(bc, 0.75));
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
                elbow: [-(sx + 0.10), 0.96, -0.2 * 0.35],
                wrist: [-(sx + 0.06), 0.54, -0.2 * 0.15 - 0.3],
            },
            ArmPose { side: 1.0,
                shoulder: [sx, 1.42, 0.0],
                elbow: [sx + 0.10, 1.00, -0.15 - extend * 0.20],
                wrist: [sx + 0.06, 0.66, -0.35 - extend * 0.35],
            },
        ]
    } else if carrying_item || carrying_bin {
        vec![
            ArmPose { side: -1.0,
                shoulder: [-sx, 1.42, 0.0],
                elbow: [-(sx + 0.10), 0.96, -0.63 * 0.35],
                wrist: [-(sx + 0.06), 0.54, -0.63 * 0.15 - 0.30],
            },
            ArmPose { side: 1.0,
                shoulder: [sx, 1.42, 0.0],
                elbow: [sx + 0.10, 0.96, -0.63 * 0.35],
                wrist: [sx + 0.06, 0.54, -0.63 * 0.15 - 0.30],
            },
        ]
    } else {
        let l_arm_fwd = swing * 0.25;
        let r_arm_fwd = -swing * 0.25;
        let bend = 0.10 + swing.abs() * 0.14;
        vec![
            ArmPose { side: -1.0,
                shoulder: [-sx, 1.42, 0.0],
                elbow: [-(sx + 0.10), 0.96, l_arm_fwd * 0.35],
                wrist: [-(sx + 0.06), 0.54, l_arm_fwd * 0.15 - bend],
            },
            ArmPose { side: 1.0,
                shoulder: [sx, 1.42, 0.0],
                elbow: [sx + 0.10, 0.96, r_arm_fwd * 0.35],
                wrist: [sx + 0.06, 0.54, r_arm_fwd * 0.15 - bend],
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
            (1.10, 0.070, vec![
                (PI + hp, 0.55, 0.045),
                (0.0, 0.35, 0.015), (PI, 0.35, 0.015),
            ]),
            (elbow_y, 0.060, vec![]),
            // F2: Sleeve extends below elbow to cover upper forearm
            (elbow_y - 0.06, 0.058, vec![]),
            (elbow_y - 0.12, 0.054, vec![]),
        ];
        let sleeve_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = sleeve_data.iter().map(|(y, rx, bumps)| {
            let (cx, cz) = if *y >= elbow_y {
                let t_lin = (shoulder_y - *y) / (shoulder_y - elbow_y);
                let t = t_lin * t_lin * t_lin;
                (shoulder[0] * (1.0 - t) + elbow[0] * t,
                 shoulder[2] * (1.0 - t) + elbow[2] * t)
            } else {
                // Below elbow: interpolate toward wrist
                let t = (elbow_y - *y) / (elbow_y - wrist[1]);
                (elbow[0] * (1.0 - t) + wrist[0] * t,
                 elbow[2] * (1.0 - t) + wrist[2] * t)
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
            let brs = 0.36;
            let bre = 0.72;
            mesh::tapered_cylinder_between(tris, lerp(brs), lerp(bre),
                (0.054 + aco) * a, (0.048 + aco) * a, 12, 0xFF4A3828);
            let brs_p = lerp(brs);
            let bre_p = lerp(bre);
            mesh::cylinder_tris(tris, brs_p[0], brs_p[1], brs_p[2], (0.058 + aco) * a, 0.006, 12, darken(0xFF4A3828, 0.82));
            mesh::cylinder_tris(tris, bre_p[0], bre_p[1], bre_p[2], (0.052 + aco) * a, 0.006, 12, darken(0xFF4A3828, 0.82));
            let mb_p = lerp((brs + bre) * 0.5);
            mesh::cylinder_tris(tris, mb_p[0], mb_p[1], mb_p[2], (0.056 + aco) * a, 0.003, 12, 0xFF887755);
            let st2_p = lerp(0.62);
            mesh::cylinder_tris(tris, st2_p[0], st2_p[1], st2_p[2], (0.056 + aco) * a, 0.003, 12, 0xFF887755);
            let strap_r = (0.058 + aco) * a;
            push_box(tris, mb_p[0] + arm.side * strap_r, mb_p[1], mb_p[2], 0.005, 0.014, 0.008, darken(0xFF4A3828, 1.15));
            push_box(tris, mb_p[0] + arm.side * (strap_r + 0.003), mb_p[1], mb_p[2], 0.008, 0.010, 0.006, BUCKLE_BRASS);
            push_box(tris, mb_p[0] + arm.side * (strap_r + 0.005), mb_p[1], mb_p[2], 0.002, 0.006, 0.002, darken(BUCKLE_BRASS, 0.7));
            for li in 0..4 {
                let lt = brs + (bre - brs) * (li as f32 + 0.5) / 4.0;
                let lp = lerp(lt);
                let rivet_x = lp[0] - arm.side * (0.050 + aco) * a;
                mesh::sphere_tris(tris, rivet_x, lp[1], lp[2], 0.004, 0, darken(BUCKLE_BRASS, 0.85));
            }
        }
    }
}

/// Complete player body with animation — male or female via BodyProportions.
/// When clothing is Some, adds ACU-style clothing over the body before the stretch.
fn gen_nude_player_body(
    tris: &mut Vec<WorldTri>,
    swing: f32,
    skin: u32,
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
            skin,
            hat_type: 0, hat_col: 0, coat_col: 0, vest_col: 0,
            has_coat: false, has_cape: false, has_sash: false,
            has_cross_strap: false, has_bracers: false,
            boot_type: 0, boot_col: 0, sash_col: 0, belt_col: 0,
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
            let mid_thigh = [hip_x, 0.47, -0.21];
            let knee_s = [hip_x, 0.46, -0.42];
            let mid_calf = [hip_x, 0.26, -0.43];
            let ankle_s = [hip_x, 0.06, -0.44];
            let l = props.leg_rx_scale;
            // Upper leg — tapers from hip to knee with thigh bulk
            mesh::tapered_cylinder_between(tris, hip_s, mid_thigh, 0.20 * l, 0.18 * l, 10, skin);
            mesh::tapered_cylinder_between(tris, mid_thigh, knee_s, 0.18 * l, 0.12 * l, 10, skin);
            // Knee — smooth bulge
            mesh::ellipsoid_tris(tris, knee_s[0], knee_s[1], knee_s[2],
                0.10 * l, 0.10 * l, 0.08 * l, 0, skin);
            // Lower leg — calf swell then taper
            mesh::tapered_cylinder_between(tris, knee_s, mid_calf, 0.11 * l, 0.10 * l, 8, skin);
            mesh::tapered_cylinder_between(tris, mid_calf, ankle_s, 0.10 * l, 0.06 * l, 8, skin);
            gen_bare_foot(tris, ankle_s, side, skin);
        }
        // Seated arms — lofted anatomy, shift down to match seated torso
        for &side in &[-1.0f32, 1.0] {
            let arm_base = tris.len();
            gen_nude_arm(tris, side, -0.25, 0.30, skin, &props, 16);
            // Shift arm down by torso offset (seated torso is 0.4 lower)
            for tri in &mut tris[arm_base..] {
                for v in &mut tri.v { v[1] -= 0.44; }
            }
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
            let e1 = [tri.v[1][0] - tri.v[0][0], tri.v[1][1] - tri.v[0][1], tri.v[1][2] - tri.v[0][2]];
            let e2 = [tri.v[2][0] - tri.v[0][0], tri.v[2][1] - tri.v[0][1], tri.v[2][2] - tri.v[0][2]];
            let nx = e1[1]*e2[2] - e1[2]*e2[1];
            let ny = e1[2]*e2[0] - e1[0]*e2[2];
            let nz = e1[0]*e2[1] - e1[1]*e2[0];
            let nl = (nx*nx + ny*ny + nz*nz).sqrt();
            if nl > 1e-10 {
                tri.normal = [nx/nl, ny/nl, nz/nl];
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

    // ── BODY (torso+neck as single loft, then limbs) — generated at natural coords, then stretched ──
    let body_base = tris.len();
    gen_nude_torso(tris, skin, &props, 32); // neck merged into torso loft

    // Wider torso hips now cover the leg-torso junction naturally.

    // ── WALKING ANIMATION — hip sway, counter-rotation, speed-dependent arm bend ──
    // Hip lateral sway: weight shifts toward stance leg
    let hip_sway = -swing * 0.015; // subtle lateral shift

    // Legs — stride with hip sway offset + foot lift arc
    // Foot lift: the swinging leg lifts in an arc to clear ground, peaking at mid-swing
    let l_fwd = -swing * 0.40;
    let r_fwd = swing * 0.40;
    let l_knee = if swing > 0.0 { swing * 0.28 } else { 0.0 };
    let r_knee = if swing < 0.0 { (-swing) * 0.28 } else { 0.0 };
    // Foot lift: the backward-moving leg lifts to clear ground
    // Peak lift occurs at max backward extension (sin peak of stride)
    let l_lift = if l_fwd > 0.0 { l_fwd * 0.12 } else { 0.0 }; // left foot lifts when swinging back
    let r_lift = if r_fwd > 0.0 { r_fwd * 0.12 } else { 0.0 }; // right foot lifts when swinging back
    let l_leg_base = tris.len();
    gen_nude_leg(tris, -1.0, l_fwd, l_knee, skin, &props, 24);
    // Lift foot vertices (below ankle Y=0.08) on the swinging leg
    if l_lift > 0.001 {
        for tri in &mut tris[l_leg_base..] {
            for v in &mut tri.v {
                if v[1] < 0.12 { // ankle and below
                    let weight = 1.0 - (v[1] / 0.12).clamp(0.0, 1.0);
                    v[1] += l_lift * weight;
                }
            }
        }
    }
    let r_leg_base = tris.len();
    gen_nude_leg(tris, 1.0, r_fwd, r_knee, skin, &props, 24);
    if r_lift > 0.001 {
        for tri in &mut tris[r_leg_base..] {
            for v in &mut tri.v {
                if v[1] < 0.12 {
                    let weight = 1.0 - (v[1] / 0.12).clamp(0.0, 1.0);
                    v[1] += r_lift * weight;
                }
            }
        }
    }

    // Arms — counter-swing with speed-dependent bend
    if attack_phase > 0.0 {
        let t = (attack_phase / ATTACK_ANIM_DURATION).clamp(0.0, 1.0);
        let extend = 1.0 - (1.0 - t) * (1.0 - t);
        // Punch arm: reuse full anatomy loft (forward + extended bend)
        let punch_fwd = -0.35 - extend * 0.50;
        let punch_bend = 0.20 + extend * 0.30;
        gen_nude_arm(tris, 1.0, punch_fwd, punch_bend, skin, &props, 24);
        // Replace hand with fist on punch arm
        let sx = props.shoulder_joint_x;
        let wrist_x = 1.0 * (sx + 0.06);
        let wrist_z = punch_fwd * 0.15 - punch_bend;
        push_box(tris, wrist_x, 0.49, wrist_z - 0.06,
            0.040 * props.arm_rx_scale, 0.035, 0.025 * props.arm_rx_scale,
            darken(skin, 0.95));
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
        // Shoulder counter-rotation: arms swing opposite to legs, forward arm bends more
        let l_arm_fwd = swing * 0.30;
        let r_arm_fwd = -swing * 0.30;
        let l_bend = 0.10 + l_arm_fwd.abs() * 0.22;
        let r_bend = 0.10 + r_arm_fwd.abs() * 0.22;
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

    // Stretch body vertically (taller), widen, apply hip sway + shoulder counter-twist
    let bs = props.body_stretch;
    let bw = props.body_widen;
    // Shoulder counter-rotation: upper body twists opposite to hips during walk
    let twist_angle = swing * 0.06; // ~3.4° at full swing — subtle but visible
    let twist_sin = twist_angle.sin();
    let twist_cos = twist_angle.cos();
    for tri in &mut tris[body_base..] {
        for v in &mut tri.v {
            // Hip sway: lateral shift proportional to how low on the body (hips sway most)
            let sway_weight = (1.0 - (v[1] / 1.5).clamp(0.0, 1.0)) * 0.5;
            v[0] += hip_sway * sway_weight;
            // Shoulder twist: Y-axis rotation increasing with height (waist=0, shoulders=full)
            let twist_weight = ((v[1] - 0.92) / 0.50).clamp(0.0, 1.0);
            if twist_weight > 0.0 {
                let tw = twist_sin * twist_weight;
                let tc = 1.0 + (twist_cos - 1.0) * twist_weight;
                let ox = v[0];
                let oz = v[2];
                v[0] = ox * tc - oz * tw;
                v[2] = ox * tw + oz * tc;
            }
            v[0] *= bw;
            v[1] *= bs;
            v[2] *= bw;
        }
        // Recalculate normal from stretched vertices for correct lighting
        let e1 = [tri.v[1][0] - tri.v[0][0], tri.v[1][1] - tri.v[0][1], tri.v[1][2] - tri.v[0][2]];
        let e2 = [tri.v[2][0] - tri.v[0][0], tri.v[2][1] - tri.v[0][1], tri.v[2][2] - tri.v[0][2]];
        let nx = e1[1]*e2[2] - e1[2]*e2[1];
        let ny = e1[2]*e2[0] - e1[0]*e2[2];
        let nz = e1[0]*e2[1] - e1[1]*e2[0];
        let nl = (nx*nx + ny*ny + nz*nz).sqrt();
        if nl > 1e-10 {
            tri.normal = [nx/nl, ny/nl, nz/nl];
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
    let app = player_appearance(player.is_female);
    let skin = if player.hit_flash > 0.0 { 0xFFFF4444 } else { app.skin };

    gen_nude_player_body(
        tris,
        player.walk_phase.sin() * 0.4,
        skin,
        player.attack_phase,
        player.carrying_item,
        player.carrying_bin.is_some(),
        player.sitting,
        player.is_female,
        if NUDE_MODE.load(Ordering::Relaxed) { None } else { Some((&app, PLAYER_SHIRT, PLAYER_PANTS)) },
        None,
    );

    place_mesh(tris, base, player.terrain_normal, 25.0, player.rot_y, player.x, player.y, player.z);
}

/// Generate player mesh from a pre-loaded GLTF model.
/// Clones the selected model's triangles, tints them to the player's skin color,
/// then applies terrain rotation and world positioning via place_mesh().
/// If animation_data is provided, applies skeletal animation based on walk/attack phase.
pub fn gen_player_mesh_gltf(
    player: &Player,
    models: &[Vec<WorldTri>],
    tris: &mut Vec<WorldTri>,
    animation_data: Option<&crate::skeleton_anim::AnimationData>,
) {
    let idx = player.model_index.min(models.len().saturating_sub(1));
    if models.is_empty() { return; }
    let model = &models[idx];

    let base = tris.len();

    // Apply skeletal animation — works for all models (bone assignments computed per-model)
    {
        if let Some(anim) = animation_data {
            if let Some((clip_idx, time)) = anim.select_clip(
                player.walk_phase, player.attack_phase, player.sitting,
                player.sprinting, 0,
            ) {
                anim.generate_animated_mesh(model, idx, clip_idx, time, tris);
            } else {
                tris.extend_from_slice(model);
            }
        } else {
            tris.extend_from_slice(model);
        }
    }

    // Tint skin color
    let app = player_appearance(player.is_female);
    let skin = if player.hit_flash > 0.0 { 0xFFFF4444 } else { app.skin };
    let default_skin: u32 = 0xFFBBA088;
    if skin != default_skin {
        for tri in &mut tris[base..] {
            if tri.color == default_skin {
                tri.color = skin;
            }
        }
    }

    place_mesh(tris, base, player.terrain_normal, 25.0, player.rot_y, player.x, player.y, player.z);
}

/// Generate NPC mesh from a pre-loaded GLTF model.
/// Selects male or female model based on NPC appearance, tints skin color,
/// then applies terrain rotation and world positioning via place_mesh().
/// If animation_data is provided, applies skeletal animation based on NPC state.
pub fn gen_npc_mesh_gltf(
    npc: &Npc,
    models: &[Vec<WorldTri>],
    tris: &mut Vec<WorldTri>,
    animation_data: Option<&crate::skeleton_anim::AnimationData>,
) {
    if models.is_empty() { return; }
    let app = npc_appearance(npc.brain_idx as u32);
    // Select model based on brain_idx, cycling through all available models
    let idx = npc.brain_idx % models.len();
    let model = &models[idx];

    let base = tris.len();

    // Apply skeletal animation if available
    if let Some(anim) = animation_data {
        let sitting = false; // NPCs don't have sitting state yet
        if let Some((clip_idx, time)) = anim.select_clip(
            npc.walk_phase, npc.attack_phase, sitting,
            false, // NPCs don't sprint
            npc.attack_intent, // use attack_intent as attack_type for variety
        ) {
            anim.generate_animated_mesh(model, idx, clip_idx, time, tris);
        } else {
            tris.extend_from_slice(model);
        }
    } else {
        tris.extend_from_slice(model);
    }

    // Tint skin color
    let shirt = if npc.hit_flash > 0.0 { 0xFFFF4444 } else { job_shirt_color(npc) };
    let skin = if npc.hit_flash > 0.0 { 0xFFFF4444 } else { app.skin };
    let default_skin: u32 = 0xFFBBA088;
    let _ = shirt; // shirt color not applied to GLTF model (it has no clothing distinction)
    if skin != default_skin {
        for tri in &mut tris[base..] {
            if tri.color == default_skin {
                tri.color = skin;
            }
        }
    }

    // Height variation: scale Y by height_scale, X/Z by height_scale^0.3
    let hs = npc.height_scale;
    if (hs - 1.0).abs() > 0.01 {
        let xz_scale = hs.powf(0.3); // slight width increase for taller NPCs
        for tri in &mut tris[base..] {
            for v in &mut tri.v {
                v[0] *= xz_scale;
                v[1] *= hs;
                v[2] *= xz_scale;
            }
        }
    }

    // Handle ragdoll state — apply the same dual-segment orientation as the procedural body
    if npc.ragdoll_active {
        let p = &npc.ragdoll_points;
        let make_basis = |up_raw: [f32; 3], right_hint: [f32; 3]| -> [[f32; 3]; 3] {
            let ulen = (up_raw[0]*up_raw[0] + up_raw[1]*up_raw[1] + up_raw[2]*up_raw[2]).sqrt().max(0.01);
            let up = [up_raw[0]/ulen, up_raw[1]/ulen, up_raw[2]/ulen];
            let dot = right_hint[0]*up[0] + right_hint[1]*up[1] + right_hint[2]*up[2];
            let rx = right_hint[0] - dot*up[0];
            let ry = right_hint[1] - dot*up[1];
            let rz = right_hint[2] - dot*up[2];
            let rlen = (rx*rx + ry*ry + rz*rz).sqrt().max(0.01);
            let right = [rx/rlen, ry/rlen, rz/rlen];
            let fwd = [
                right[1]*up[2] - right[2]*up[1],
                right[2]*up[0] - right[0]*up[2],
                right[0]*up[1] - right[1]*up[0],
            ];
            [right, up, fwd]
        };

        let foot_mid = [
            (p[5][0]+p[6][0])*0.5, (p[5][1]+p[6][1])*0.5, (p[5][2]+p[6][2])*0.5,
        ];
        let lower_up = [p[0][0]-foot_mid[0], p[0][1]-foot_mid[1], p[0][2]-foot_mid[2]];
        let feet_right = [p[6][0]-p[5][0], p[6][1]-p[5][1], p[6][2]-p[5][2]];
        let lower = make_basis(lower_up, feet_right);

        let upper_up = [p[2][0]-p[1][0], p[2][1]-p[1][1], p[2][2]-p[1][2]];
        let hands_right = [p[4][0]-p[3][0], p[4][1]-p[3][1], p[4][2]-p[3][2]];
        let upper = make_basis(upper_up, hands_right);

        // Flatten [[f32;3];3] basis (rows: right, up, fwd) to column-major [f32;9]
        let flatten = |b: &[[f32; 3]; 3]| -> [f32; 9] {
            [b[0][0],b[0][1],b[0][2], b[1][0],b[1][1],b[1][2], b[2][0],b[2][1],b[2][2]]
        };
        let lower9 = flatten(&lower);
        let upper9 = flatten(&upper);

        // GLTF model is 1.8m tall (Y=0 at feet), blend at mid-body
        let blend_lo = 0.8;
        let blend_hi = 1.1;

        for tri in &mut tris[base..] {
            for v in &mut tri.v {
                let lx = v[0]; let ly = v[1]; let lz = v[2];
                let t = ((ly - blend_lo) / (blend_hi - blend_lo)).clamp(0.0, 1.0);

                if t < 0.01 {
                    let rv = rot3x3_apply(&lower9, [lx, ly, lz]);
                    v[0] = rv[0] + foot_mid[0];
                    v[1] = rv[1] + foot_mid[1];
                    v[2] = rv[2] + foot_mid[2];
                } else if t > 0.99 {
                    let rel_y = ly - blend_hi;
                    let rv = rot3x3_apply(&upper9, [lx, rel_y, lz]);
                    v[0] = rv[0] + p[1][0];
                    v[1] = rv[1] + p[1][1];
                    v[2] = rv[2] + p[1][2];
                } else {
                    let lo = rot3x3_apply(&lower9, [lx, ly, lz]);
                    let rel_y = ly - blend_hi;
                    let hi = rot3x3_apply(&upper9, [lx, rel_y, lz]);
                    v[0] = lo[0]*(1.0-t) + hi[0]*t + foot_mid[0]*(1.0-t) + p[1][0]*t;
                    v[1] = lo[1]*(1.0-t) + hi[1]*t + foot_mid[1]*(1.0-t) + p[1][1]*t;
                    v[2] = lo[2]*(1.0-t) + hi[2]*t + foot_mid[2]*(1.0-t) + p[1][2]*t;
                }
            }
            let nx = tri.normal[0]; let ny = tri.normal[1]; let nz = tri.normal[2];
            let avg_y = (tri.v[0][1] + tri.v[1][1] + tri.v[2][1]) / 3.0;
            let t = ((avg_y - blend_lo) / (blend_hi - blend_lo)).clamp(0.0, 1.0);
            if t < 0.5 {
                tri.normal = rot3x3_apply(&lower9, [nx, ny, nz]);
            } else {
                tri.normal = rot3x3_apply(&upper9, [nx, ny, nz]);
            }
        }
        return;
    }

    // KO pose — rotate face-down
    if npc.state == NpcState::KnockedOut {
        let rot = terrain_rot3x3(clamp_normal_tilt(npc.terrain_normal, 25.0), npc.rot_y);
        for tri in &mut tris[base..] {
            for v in &mut tri.v {
                let local = [v[0], v[2], -v[1]]; // -90° around X
                let rv = rot3x3_apply(&rot, local);
                v[0] = rv[0] + npc.x;
                v[1] = rv[1] + npc.y + 0.15;
                v[2] = rv[2] + npc.z;
            }
            let local_n = [tri.normal[0], tri.normal[2], -tri.normal[1]];
            tri.normal = rot3x3_apply(&rot, local_n);
        }
        return;
    }

    place_mesh(tris, base, npc.terrain_normal, 25.0, npc.rot_y, npc.x, npc.y, npc.z);
}

/// Generate a clothed player body (for model_viewer debug renders).
/// Uses the detailed nude body with ACU-style clothing layered on top.
pub fn gen_clothed_player_body(tris: &mut Vec<WorldTri>, is_female: bool) {
    let app = player_appearance(is_female);
    gen_nude_player_body(
        tris, 0.0, app.skin,
        0.0, false, false, false, is_female,
        Some((&app, PLAYER_SHIRT, PLAYER_PANTS)),
        None,
    );
}

/// Generate a nude player body (no clothing) for debug/anatomy inspection.
pub fn gen_nude_player_body_export(tris: &mut Vec<WorldTri>, is_female: bool) {
    let app = player_appearance(is_female);
    gen_nude_player_body(
        tris, 0.0, app.skin,
        0.0, false, false, false, is_female,
        None, // no clothing
        None,
    );
}

/// Generate a standalone head + neck with face sliders, scaled and positioned.
/// Used by model_viewer for rendering face variations.
pub fn gen_head_standalone(tris: &mut Vec<WorldTri>, face: &FaceSliders, skin: u32, is_female: bool) {
    let props = if is_female { female_proportions() } else { male_proportions() };
    let app = NpcAppearance {
        skin,
        hat_type: 0, hat_col: 0, coat_col: 0, vest_col: 0,
        has_coat: false, has_cape: false, has_sash: false,
        has_cross_strap: false, has_bracers: false,
        boot_type: 0, boot_col: 0, sash_col: 0, belt_col: 0,
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

pub fn gen_vehicle_mesh(v: &Vehicle, tris: &mut Vec<WorldTri>, show_interior: bool, headlights: bool) {
    let base = tris.len();
    gen_rs5_body(tris, v.color, show_interior);
    // Night-time headlight + tail light glow (only added when lights are on)
    if headlights {
        // Headlight glow — directional forward cones
        for &side in &[-1.0f32, 1.0] {
            let hx = side * 0.72;
            mesh::glow_directional(tris, hx, 0.66, -2.36, [0.0, 0.0, -1.0],
                0.06, 0.4, 8, 0x00FFEE88);
        }
        // Tail light glow — directional rear cones (red)
        for &side in &[-1.0f32, 1.0] {
            mesh::glow_directional(tris, side * 0.64, 0.74, 2.35, [0.0, 0.0, 1.0],
                0.04, 0.25, 6, 0x00FF2222);
        }
    }
    // Deformation visual: displace outer shell vertices inward based on crash damage
    let dmg = v.deformation.damage_fraction();
    if dmg > 0.01 {
        for tri in &mut tris[base..] {
            for vert in &mut tri.v {
                // Only displace vertices on the outer body shell (horiz dist > 0.4 from center)
                let dx = vert[0];
                let dz = vert[2];
                let horiz_dist = (dx * dx + dz * dz).sqrt();
                if horiz_dist > 0.4 && vert[1] > 0.1 && vert[1] < 1.5 {
                    let angle = dz.atan2(dx);
                    let deform = v.deformation.sample_at_angle(angle);
                    if deform > 0.001 {
                        // Push vertex inward toward center, scaled by distance from center
                        let edge_frac = ((horiz_dist - 0.4) / 0.6).min(1.0);
                        let disp = deform * edge_frac;
                        let inv_dist = 1.0 / horiz_dist;
                        vert[0] -= dx * inv_dist * disp;
                        vert[2] -= dz * inv_dist * disp;
                        // Slight downward crush for heavy impacts
                        vert[1] -= deform * edge_frac * 0.15;
                    }
                }
            }
        }
    }
    let s = v.scale;
    let ch = ((v.color >> 4) ^ (v.color >> 12) ^ (v.color >> 20)) & 0xFF;
    let sx = s * (1.0 + (ch as f32 - 128.0) * 0.0008);
    let sz = s * (1.0 - (ch as f32 - 128.0) * 0.0005);
    let rot = terrain_rot3x3(clamp_normal_tilt(v.terrain_normal, 30.0), v.rot_y);
    for tri in &mut tris[base..] {
        for vert in &mut tri.v {
            // Apply per-vehicle non-uniform scale, then rotate + translate
            let sv = [vert[0] * sx, vert[1] * s, vert[2] * sz];
            let rv = rot3x3_apply(&rot, sv);
            vert[0] = rv[0] + v.x;
            vert[1] = rv[1] + v.y;
            vert[2] = rv[2] + v.z;
        }
        tri.normal = rot3x3_apply(&rot, tri.normal);
    }
    // H5: Projected ground shadow — approximate vehicle silhouette
    gen_vehicle_shadow(v, s, tris);
}

/// Generate vehicle mesh from a GLTF car model.
/// Falls back to procedural mesh if car_models is empty or index out of range.
pub fn gen_vehicle_mesh_gltf(
    v: &Vehicle, vi: usize, car_models: &[Vec<WorldTri>],
    tris: &mut Vec<WorldTri>, show_interior: bool, headlights: bool,
) {
    if car_models.is_empty() {
        gen_vehicle_mesh(v, tris, show_interior, headlights);
        return;
    }
    let model_idx = vi % car_models.len();
    let model = &car_models[model_idx];

    let base = tris.len();

    // Copy model triangles, tint to vehicle color
    for tri in model {
        let mut t = tri.clone();
        // Tint the model's gray placeholder color to vehicle color
        t.color = v.color;
        tris.push(t);
    }

    // Apply vehicle scale + terrain rotation + world position (same as procedural)
    let s = v.scale;
    let ch = ((v.color >> 4) ^ (v.color >> 12) ^ (v.color >> 20)) & 0xFF;
    let sx = s * (1.0 + (ch as f32 - 128.0) * 0.0008);
    let sz = s * (1.0 - (ch as f32 - 128.0) * 0.0005);
    let rot = terrain_rot3x3(clamp_normal_tilt(v.terrain_normal, 30.0), v.rot_y);
    for tri in &mut tris[base..] {
        for vert in &mut tri.v {
            let sv = [vert[0] * sx, vert[1] * s, vert[2] * sz];
            let rv = rot3x3_apply(&rot, sv);
            vert[0] = rv[0] + v.x;
            vert[1] = rv[1] + v.y;
            vert[2] = rv[2] + v.z;
        }
        tri.normal = rot3x3_apply(&rot, tri.normal);
    }
    gen_vehicle_shadow(v, s, tris);
}

fn gen_vehicle_shadow(v: &Vehicle, s: f32, tris: &mut Vec<WorldTri>) {
    let sy = v.y + 0.05; let sc: u32 = 0xFF1A3A1A;
    let (snr, csr) = v.rot_y.sin_cos();
    let p = |lx: f32, lz: f32| -> [f32; 3] {
        [v.x + lx*s*csr + lz*s*snr, sy, v.z - lx*s*snr + lz*s*csr]
    };
    // Vehicle silhouette outline — tapered nose/tail, wide body, wheel arches
    let outline: [(f32, f32); 16] = [
        ( 0.00, -2.30), ( 0.60, -2.20), ( 0.80, -1.80), ( 0.93, -1.20),
        ( 0.78, -1.41), ( 0.93, -1.60), // front wheel arch notch
        ( 0.93,  1.20), ( 0.78,  1.41), ( 0.93,  1.60), // rear wheel arch notch
        ( 0.90,  1.80), ( 0.84,  2.20), ( 0.00,  2.30),
        (-0.84,  2.20), (-0.90,  1.80), (-0.93,  1.20), (-0.93, -1.20),
    ];
    let center = p(0.0, 0.0);
    let n = outline.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ax, az) = (outline[i].0, outline[i].1);
        let (bx, bz) = (outline[j].0, outline[j].1);
        let a = p(ax, az);
        let b = p(bx, bz);
        mesh::push_tri(tris, center, a, b, [0.0, 1.0, 0.0], sc);
    }
}

/// Audi RS5 Sportback — detailed body model in local space (origin at ground center).
/// Uses volumetric primitives (beveled boxes, ellipsoids) for solid body construction.
/// Real RS5: 4780mm L × 1861mm W × 1387mm H, wheelbase 2826mm, 20" wheels ~710mm OD.
/// Coordinates: -Z = front, +Z = rear, +Y = up, ±X = sides.
fn gen_rs5_body(tris: &mut Vec<WorldTri>, color: u32, show_interior: bool) {
    use std::f32::consts::PI;
    let c = color;
    let c_dk = darken(c, 0.88);
    let c_dkr = darken(c, 0.78);
    let c_shadow = darken(c, 0.70);
    let trim = 0xFF333333_u32;
    let chrome = 0xFFAAAAAA_u32;
    let chrome_dk = 0xFF777777_u32;
    let undercar = 0xFF222222_u32;
    let glass = WINDSHIELD_COLOR;
    let grille_dk = 0xFF1A1A1A_u32;

    // Key dimensions (meters, 1:1 from real RS5 Sportback)
    let fwz: f32 = -1.41;   // front wheel center Z
    let rwz: f32 = 1.41;    // rear wheel center Z
    let wtrk: f32 = 0.80;   // wheel center |X| (half track)
    let wr: f32 = 0.355;    // wheel radius (20" + tire)
    let ww: f32 = 0.255;    // tire width
    let gc: f32 = 0.125;    // ground clearance

    // ══════════════════════════════════════════════════════════════
    //  MAIN BODY SHELL — central box + tapered nose/tail with quad panels
    // ══════════════════════════════════════════════════════════════

    // Mid body (door/cabin area) — main volume between wheel arches
    // y: 0.15 to 0.77
    mesh::beveled_box_tris(tris, 0.0, 0.46, 0.0, 1.86, 0.62, 2.90, 0.06, c);

    // Upper body shoulder ridge
    mesh::beveled_box_tris(tris, 0.0, 0.80, -0.10, 1.84, 0.06, 2.70, 0.03, c);

    // ── FRONT NOSE (tapered with quad side/top panels) ──
    // Key coordinates: body ends at z=-1.45 (half of 2.90)
    //   Body top: y=0.77, width: 1.86 (half=0.93)
    //   Nose tip: z=-2.30, y_top=0.68, y_bot=0.17, width: 1.60 (half=0.80)
    let fb_z = -1.45_f32;  // front body edge Z
    let fn_z = -2.30_f32;  // front nose tip Z
    let fb_yt = 0.77_f32;  // front body top Y
    let fn_yt = 0.68_f32;  // front nose top Y
    let fb_yb = 0.15_f32;  // front body bottom Y
    let fn_yb = 0.17_f32;  // front nose bottom Y
    let fb_xh = 0.93_f32;  // front body half-width
    let fn_xh = 0.80_f32;  // front nose half-width

    // Nose volume fill (inner box for solidity)
    mesh::beveled_box_tris(tris, 0.0, 0.42, -1.88, 1.66, 0.50, 0.86, 0.04, c);

    // Right side panel (visible from right — tapers inward + downward, CCW from +X)
    mesh::push_quad(tris,
        [fb_xh, fb_yb, fb_z], [fn_xh, fn_yb, fn_z],
        [fn_xh, fn_yt, fn_z], [fb_xh, fb_yt, fb_z],
        c);
    // Left side panel (CCW from -X)
    mesh::push_quad(tris,
        [-fb_xh, fb_yt, fb_z], [-fn_xh, fn_yt, fn_z],
        [-fn_xh, fn_yb, fn_z], [-fb_xh, fb_yb, fb_z],
        c);
    // Top panel (hood slope — visible from above and slightly from front)
    mesh::push_quad(tris,
        [fb_xh, fb_yt, fb_z], [fn_xh, fn_yt, fn_z],
        [-fn_xh, fn_yt, fn_z], [-fb_xh, fb_yt, fb_z],
        c);
    // Bottom panel
    mesh::push_quad(tris,
        [fb_xh, fb_yb, fb_z], [-fb_xh, fb_yb, fb_z],
        [-fn_xh, fn_yb, fn_z], [fn_xh, fn_yb, fn_z],
        undercar);
    // Front face (nose face, CCW from -Z)
    mesh::push_quad(tris,
        [-fn_xh, fn_yt, fn_z], [fn_xh, fn_yt, fn_z],
        [fn_xh, fn_yb, fn_z], [-fn_xh, fn_yb, fn_z],
        c_dk);

    // ── REAR TAIL (tapered with quad side/top panels) ──
    let rb_z = 1.45_f32;   // rear body edge Z
    let rn_z = 2.30_f32;   // rear tail tip Z
    let rb_yt = 0.77_f32;  // rear body top Y
    let rn_yt = 0.66_f32;  // rear tail top Y
    let rb_yb = 0.15_f32;  // rear body bottom Y
    let rn_yb = 0.17_f32;  // rear tail bottom Y
    let rb_xh = 0.93_f32;  // rear body half-width
    let rn_xh = 0.84_f32;  // rear tail half-width

    // Rear volume fill
    mesh::beveled_box_tris(tris, 0.0, 0.42, 1.88, 1.72, 0.50, 0.86, 0.04, c);

    // Right side panel (CCW from +X)
    mesh::push_quad(tris,
        [rb_xh, rb_yt, rb_z], [rn_xh, rn_yt, rn_z],
        [rn_xh, rn_yb, rn_z], [rb_xh, rb_yb, rb_z],
        c);
    // Left side panel (CCW from -X)
    mesh::push_quad(tris,
        [-rb_xh, rb_yb, rb_z], [-rn_xh, rn_yb, rn_z],
        [-rn_xh, rn_yt, rn_z], [-rb_xh, rb_yt, rb_z],
        c);
    // Top panel (rear deck)
    mesh::push_quad(tris,
        [rb_xh, rb_yt, rb_z], [-rb_xh, rb_yt, rb_z],
        [-rn_xh, rn_yt, rn_z], [rn_xh, rn_yt, rn_z],
        c);
    // Bottom panel
    mesh::push_quad(tris,
        [rb_xh, rb_yb, rb_z], [rn_xh, rn_yb, rn_z],
        [-rn_xh, rn_yb, rn_z], [-rb_xh, rb_yb, rb_z],
        undercar);
    // Rear face
    mesh::push_quad(tris,
        [-rn_xh, rn_yb, rn_z], [rn_xh, rn_yb, rn_z],
        [rn_xh, rn_yt, rn_z], [-rn_xh, rn_yt, rn_z],
        c_dk);

    // ── FENDER FLARES (RS5 wide-body — prominent muscular bulges) ──
    for &side in &[-1.0f32, 1.0] {
        // Front fender flare (large, wraps over front arch)
        mesh::ellipsoid_tris(tris, side * 0.88, 0.50, fwz, 0.20, 0.40, 0.58, 0, c);
        // Rear fender flare (wider — RS5 muscular rear haunches)
        mesh::ellipsoid_tris(tris, side * 0.90, 0.50, rwz, 0.24, 0.44, 0.62, 0, c);
        // Fender lip ridge above each arch
        push_box(tris, side * 0.96, 0.74, fwz, 0.02, 0.02, 0.42, c_dkr);
        push_box(tris, side * 0.98, 0.74, rwz, 0.02, 0.02, 0.46, c_dkr);
    }

    // ── HOOD SURFACE ──
    // Sloping hood panel (cowl to body front — creates visible slope in side profile)
    mesh::push_quad(tris,
        [-0.90, 0.86, -0.88], [0.90, 0.86, -0.88],
        [0.93, 0.78, -1.45], [-0.93, 0.78, -1.45], c);
    // Power bulge (center hood ridge)
    mesh::ellipsoid_tris(tris, 0.0, 0.82, -1.20, 0.30, 0.03, 0.70, 0, c_dk);
    // Hood crease lines
    push_box(tris, -0.40, 0.81, -1.20, 0.012, 0.004, 0.70, c_dkr);
    push_box(tris, 0.40, 0.81, -1.20, 0.012, 0.004, 0.70, c_dkr);
    // Hood panel gap at cowl
    push_box(tris, 0.0, 0.84, -0.88, 0.85, 0.003, 0.005, c_shadow);

    // ── SINGLEFRAME GRILLE (massive hexagonal opening) ──
    mesh::beveled_box_tris(tris, 0.0, 0.42, -2.34, 1.12, 0.30, 0.06, 0.03, grille_dk);
    // Honeycomb mesh pattern (7 rows × 10 cols)
    for row in 0..7 {
        let y_off = if row % 2 == 1 { 0.05 } else { 0.0 };
        for col in 0..10 {
            let gx = -0.48 + col as f32 * 0.10 + y_off * 0.5;
            let gy = 0.30 + row as f32 * 0.035;
            if gx.abs() < 0.54 {
                push_box(tris, gx, gy, -2.38, 0.032, 0.010, 0.008, 0xFF2A2A2A);
            }
        }
    }
    // Chrome frame around grille
    push_box(tris, 0.0, 0.58, -2.36, 1.14, 0.018, 0.02, chrome);  // top
    push_box(tris, 0.0, 0.26, -2.36, 1.10, 0.018, 0.02, chrome);  // bottom
    push_box(tris, -0.57, 0.42, -2.36, 0.018, 0.32, 0.02, chrome); // left
    push_box(tris, 0.57, 0.42, -2.36, 0.018, 0.32, 0.02, chrome);  // right
    // Signature center horizontal chrome bar
    push_box(tris, 0.0, 0.42, -2.38, 1.10, 0.024, 0.012, chrome);

    // ── LOWER AIR INTAKES (flanking grille) ──
    for &side in &[-1.0f32, 1.0] {
        mesh::beveled_box_tris(tris, side * 0.68, 0.20, -2.30, 0.36, 0.14, 0.06, 0.02, grille_dk);
        for i in 0..3 {
            push_box(tris, side * 0.68, 0.16 + i as f32 * 0.04, -2.32, 0.32, 0.006, 0.015, 0xFF2A2A2A);
        }
        // Fog light area
        mesh::sphere_tris(tris, side * 0.72, 0.20, -2.32, 0.045, 0, chrome_dk);
        // Front canard / aero blade
        push_box(tris, side * 0.86, 0.18, -2.26, 0.05, 0.12, 0.10, trim);
    }

    // ── FRONT BUMPER + SPLITTER ──
    mesh::beveled_box_tris(tris, 0.0, 0.16, -2.28, 1.86, 0.10, 0.12, 0.03, trim);
    // Front splitter lip
    push_box(tris, 0.0, 0.10, -2.36, 1.70, 0.020, 0.06, trim);

    // ── HEADLIGHTS (angular, aggressive RS5 shape) ──
    for &side in &[-1.0f32, 1.0] {
        let hx = side * 0.72;
        // Main housing (dark, angular — extends from grille outward)
        mesh::beveled_box_tris(tris, hx, 0.66, -2.20, 0.38, 0.12, 0.26, 0.02, 0xFF333338);
        // Headlight wraps to side
        push_box(tris, side * 0.92, 0.66, -2.10, 0.02, 0.10, 0.14, 0xFF333338);
        // Chrome inner reflector (behind emissive elements)
        push_box(tris, hx, 0.66, -2.22, 0.30, 0.06, 0.04, chrome_dk);
        // LED DRL strip (upper arc — RS5 signature, proud of housing face)
        push_box(tris, hx, 0.73, -2.34, 0.32, 0.020, 0.04, 0x00FFEE88);
        // Projector lens (main beam, proud of housing)
        mesh::sphere_tris(tris, hx - side * 0.04, 0.66, -2.34, 0.07, 0, 0x00FFEE88);
        // (glow disc removed — too large, creates dark shapes in daylight)
        // Secondary lens (inner, forward of reflector)
        mesh::sphere_tris(tris, hx + side * 0.08, 0.66, -2.34, 0.05, 0, 0x00FFDD66);
        // Turn signal (lower outer corner, emissive amber — proud of bumper)
        push_box(tris, hx + side * 0.14, 0.60, -2.34, 0.07, 0.030, 0.03, 0x00FFAA22);
    }

    // ══════════════════════════════════════════════════════════════
    //  CABIN / GREENHOUSE — lofted cross-sections for flowing profile
    // ══════════════════════════════════════════════════════════════

    // Cross-sections: (z, roof_y, belt_y, roof_half_x, belt_half_x)
    let gh: [(f32,f32,f32,f32,f32); 13] = [
        (-0.88, 0.92, 0.86, 0.72, 0.76),   // cowl (windshield base)
        (-0.52, 1.16, 0.86, 0.71, 0.76),   // windshield mid
        (-0.15, 1.37, 0.86, 0.70, 0.76),   // top of windshield / front roof
        ( 0.08, 1.385, 0.86, 0.70, 0.76),  // roof transition
        ( 0.30, 1.39, 0.86, 0.70, 0.76),   // roof peak
        ( 0.45, 1.388, 0.86, 0.70, 0.76),  // roof plateau mid
        ( 0.60, 1.38, 0.86, 0.70, 0.76),   // roof plateau end
        ( 0.80, 1.31, 0.86, 0.695, 0.755), // sportback start
        ( 1.00, 1.24, 0.86, 0.69, 0.75),   // sportback upper
        ( 1.20, 1.14, 0.855, 0.68, 0.745), // sportback mid-upper
        ( 1.40, 1.04, 0.85, 0.67, 0.74),   // sportback mid
        ( 1.60, 0.96, 0.845, 0.665, 0.73), // sportback mid-lower
        ( 1.80, 0.88, 0.84, 0.66, 0.72),   // sportback end
    ];

    // Loft greenhouse shell between cross-sections
    for i in 0..gh.len()-1 {
        let (z0, ry0, by0, rx0, bx0) = gh[i];
        let (z1, ry1, by1, rx1, bx1) = gh[i+1];

        // Roof surface (normal up)
        mesh::push_quad(tris,
            [-rx1, ry1, z1], [rx1, ry1, z1],
            [rx0, ry0, z0], [-rx0, ry0, z0], c);

        // Right side (tumblehome: roof narrower than belt, normal +X)
        mesh::push_quad(tris,
            [rx0, ry0, z0], [rx1, ry1, z1],
            [bx1, by1, z1], [bx0, by0, z0], c_dk);

        // Left side (normal -X)
        mesh::push_quad(tris,
            [-bx0, by0, z0], [-bx1, by1, z1],
            [-rx1, ry1, z1], [-rx0, ry0, z0], c_dk);
    }

    // Rear face of greenhouse (sportback end cap)
    {
        let (z, ry, by, rx, bx) = gh[gh.len()-1];
        mesh::push_quad(tris,
            [-bx, by, z], [bx, by, z], [rx, ry, z], [-rx, ry, z], c_dk);
    }

    // Volume fill (prevents see-through at extreme angles)
    mesh::beveled_box_tris(tris, 0.0, 1.08, 0.10, 1.28, 0.40, 1.60, 0.04, c_dk);
    mesh::beveled_box_tris(tris, 0.0, 0.96, 1.30, 1.10, 0.22, 0.50, 0.03, c);

    // Shoulder transition (body top to greenhouse beltline)
    for &side in &[-1.0f32, 1.0] {
        let (a, b, d2, d) = (
            [side * 0.92, 0.83, -0.90], [side * 0.92, 0.83, 1.80],
            [side * 0.76, 0.86, 1.80], [side * 0.76, 0.86, -0.90],
        );
        if side > 0.0 { mesh::push_quad(tris, d, d2, b, a, c); }
        else { mesh::push_quad(tris, a, b, d2, d, c); }
    }

    // ── PILLARS ──
    for &side in &[-1.0f32, 1.0] {
        // A-pillar (heavily raked — follows windshield angle)
        let (a, b, p2, d) = (
            [side * 0.75, 0.90, -0.88],
            [side * 0.72, 1.37, -0.15],
            [side * 0.70, 1.33, -0.15],
            [side * 0.73, 0.88, -0.88],
        );
        if side > 0.0 { mesh::push_quad(tris, a, b, p2, d, trim); }
        else { mesh::push_quad(tris, d, p2, b, a, trim); }
        // B-pillar
        push_box(tris, side * 0.76, 1.10, 0.14, 0.04, 0.48, 0.05, trim);
        // C-pillar (sportback sweep)
        let (a, b, p2, d) = (
            [side * 0.72, 1.34, 0.80],
            [side * 0.68, 0.90, 1.76],
            [side * 0.66, 0.86, 1.76],
            [side * 0.70, 1.30, 0.80],
        );
        if side > 0.0 { mesh::push_quad(tris, a, b, p2, d, trim); }
        else { mesh::push_quad(tris, d, p2, b, a, trim); }
    }

    // ── WINDSHIELDS ──
    // Front windshield (raked quad — matches greenhouse slope)
    mesh::push_quad(tris,
        [-0.66, 1.34, -0.17], [0.66, 1.34, -0.17],
        [0.70, 0.92, -0.86], [-0.70, 0.92, -0.86], glass);
    // Rear window (sportback angle)
    mesh::push_quad(tris,
        [-0.62, 0.92, 1.76], [0.62, 0.92, 1.76],
        [0.64, 1.30, 0.84], [-0.64, 1.30, 0.84], glass);

    // ── SIDE WINDOWS ──
    for &side in &[-1.0f32, 1.0] {
        let s = side;
        // Front door window (A-pillar to B-pillar, trapezoidal)
        let (a, b, w2, d) = (
            [s * 0.77, 0.88, -0.76], [s * 0.73, 1.34, -0.22],
            [s * 0.75, 1.32, 0.08], [s * 0.77, 0.88, 0.08],
        );
        if s > 0.0 { mesh::push_quad(tris, a, b, w2, d, glass); }
        else { mesh::push_quad(tris, d, w2, b, a, glass); }
        // Rear door window (B-pillar to C-pillar)
        let (a, b, w2, d) = (
            [s * 0.77, 0.88, 0.24], [s * 0.76, 1.30, 0.24],
            [s * 0.74, 1.28, 0.74], [s * 0.77, 0.88, 0.74],
        );
        if s > 0.0 { mesh::push_quad(tris, a, b, w2, d, glass); }
        else { mesh::push_quad(tris, d, w2, b, a, glass); }
        // Quarter window
        push_box(tris, s * 0.72, 1.06, 1.06, 0.03, 0.16, 0.14, glass);
        // Chrome window surround
        push_box(tris, s * 0.78, 1.26, -0.10, 0.005, 0.005, 1.40, chrome);
    }

    // ══════════════════════════════════════════════════════════════
    //  SIDE DETAILS
    // ══════════════════════════════════════════════════════════════

    // ── CHARACTER LINES ──
    for &side in &[-1.0f32, 1.0] {
        // Upper character line (shoulder crease, headlight → tail)
        push_box(tris, side * 0.93, 0.78, -0.10, 0.006, 0.006, 2.40, c_dkr);
        // Lower character line (door crease)
        push_box(tris, side * 0.93, 0.52, -0.10, 0.006, 0.006, 2.10, c_shadow);
        // Beltline (at window bottom, matches greenhouse belt)
        push_box(tris, side * 0.77, 0.87, -0.10, 0.006, 0.006, 1.70, c_dkr);
    }

    // ── SIDE SKIRTS (gloss black aero) ──
    for &side in &[-1.0f32, 1.0] {
        mesh::beveled_box_tris(tris, side * 0.92, 0.18, 0.0, 0.06, 0.10, 2.00, 0.015, trim);
        push_box(tris, side * 0.94, 0.13, -0.40, 0.015, 0.015, 0.50, trim);
        push_box(tris, side * 0.94, 0.13, 0.40, 0.015, 0.015, 0.50, trim);
    }

    // ── DOOR HANDLES ──
    for &side in &[-1.0f32, 1.0] {
        push_box(tris, side * 0.94, 0.72, -0.22, 0.015, 0.024, 0.12, chrome_dk);
        push_box(tris, side * 0.94, 0.72, 0.52, 0.015, 0.024, 0.12, chrome_dk);
    }

    // ── SIDE MIRRORS ──
    for &side in &[-1.0f32, 1.0] {
        push_box(tris, side * 0.90, 0.95, -0.64, 0.04, 0.025, 0.06, c);
        mesh::beveled_box_tris(tris, side * 1.02, 0.95, -0.62, 0.09, 0.07, 0.15, 0.02, c);
        push_box(tris, side * 1.07, 0.95, -0.62, 0.008, 0.05, 0.11, 0xFF556677);
        push_box(tris, side * 1.04, 1.00, -0.62, 0.008, 0.012, 0.06, 0x00FFAA22);
    }

    // ══════════════════════════════════════════════════════════════
    //  WHEEL WELLS + WHEELS
    // ══════════════════════════════════════════════════════════════

    // Wheel arch inner liners (thin dark panels behind wheels, inboard of tire)
    // Positioned at the inboard edge so they don't obscure the tire/rim from outside
    for &(side, awz) in &[(-1.0f32, fwz), (1.0, fwz), (-1.0, rwz), (1.0, rwz)] {
        let liner_x = side * (wtrk - ww * 0.5 - 0.01); // just inboard of tire inner face
        push_box(tris, liner_x, wr, awz, 0.02, wr + 0.02, wr + 0.08, undercar);
        // Top arch surface (inside fender, above wheel)
        push_box(tris, side * 0.88, wr * 2.0 + 0.04, awz, 0.10, 0.02, wr + 0.08, undercar);
    }

    // 20" split 5-spoke alloy wheels
    let spk = 0xFF444444_u32;
    let hub = 0xFF888888_u32;
    for &(wwx, wwz) in &[(-wtrk, fwz), (wtrk, fwz), (-wtrk, rwz), (wtrk, rwz)] {
        let wy = wr;
        let hw = ww * 0.5; // half tire width
        // Tire (wide, low-profile) — X-axis aligned for vertical wheel
        mesh::cylinder_between(tris, [wwx - hw, wy, wwz], [wwx + hw, wy, wwz], wr, 16, TIRE_COLOR);
        // Sidewall detail
        let sw_hw = (ww + 0.008) * 0.5;
        mesh::cylinder_between(tris, [wwx - sw_hw, wy, wwz], [wwx + sw_hw, wy, wwz], wr - 0.015, 14, darken(TIRE_COLOR, 0.88));
        // Rim face (alloy)
        mesh::cylinder_between(tris, [wwx - 0.009, wy, wwz], [wwx + 0.009, wy, wwz], wr * 0.84, 14, spk);
        // 5 split spokes (rotate in Y-Z plane, perpendicular to axle)
        for s in 0..5 {
            let a = s as f32 * PI * 2.0 / 5.0;
            let (sa, ca) = a.sin_cos();
            let rr = wr * 0.68;
            push_box(tris, wwx, wy + ca * rr * 0.5, wwz + sa * rr * 0.5,
                ww * 0.36, 0.028, 0.032, spk);
            let a2 = a + 0.16;
            let (sa2, ca2) = a2.sin_cos();
            push_box(tris, wwx, wy + ca2 * rr * 0.5, wwz + sa2 * rr * 0.5,
                ww * 0.26, 0.018, 0.024, spk);
        }
        // Center hub
        let hub_hw = (ww + 0.016) * 0.5;
        mesh::cylinder_between(tris, [wwx - hub_hw, wy, wwz], [wwx + hub_hw, wy, wwz], wr * 0.18, 8, hub);
        let lug_hw = (ww + 0.036) * 0.5;
        mesh::cylinder_between(tris, [wwx - lug_hw, wy, wwz], [wwx + lug_hw, wy, wwz], wr * 0.06, 5, 0xFFBBBBBB);
        // Brake disc
        mesh::cylinder_between(tris, [wwx - 0.0125, wy, wwz], [wwx + 0.0125, wy, wwz], wr * 0.66, 12, 0xFF666666);
        // RS red brake caliper
        push_box(tris, wwx, wy - 0.08, wwz - 0.12, ww * 0.28, 0.06, 0.06, 0xFFCC2222);
    }

    // ══════════════════════════════════════════════════════════════
    //  REAR END
    // ══════════════════════════════════════════════════════════════

    // Trunk lid surface
    mesh::beveled_box_tris(tris, 0.0, 0.84, 1.90, 1.68, 0.08, 0.50, 0.03, c);
    // Lip spoiler (subtle, integrated into trunk edge)
    push_box(tris, 0.0, 0.90, 2.18, 1.36, 0.016, 0.04, trim);

    // ── TAIL LIGHTS (wide, connected by LED bar — RS5 signature) ──
    for &side in &[-1.0f32, 1.0] {
        // Outer tail light housing (wraps to side)
        mesh::beveled_box_tris(tris, side * 0.64, 0.74, 2.26, 0.34, 0.10, 0.06, 0.01, 0xFF331111);
        // Inner tail light housing
        push_box(tris, side * 0.30, 0.74, 2.28, 0.18, 0.07, 0.04, 0xFF331111);
        // LED strip (proud of housing, visible from rear)
        push_box(tris, side * 0.64, 0.74, 2.33, 0.30, 0.035, 0.01, 0x00FF2222);
        // (glow disc removed — too large, creates dark shapes in daylight)
        // Inner tail light LED (proud of housing)
        push_box(tris, side * 0.30, 0.74, 2.33, 0.14, 0.025, 0.01, 0x00FF2222);
    }
    // Connected light bar across full width (proud of housing)
    push_box(tris, 0.0, 0.74, 2.33, 0.84, 0.018, 0.01, 0x00FF2222);
    // Reverse lights
    push_box(tris, -0.34, 0.67, 2.28, 0.08, 0.028, 0.02, 0xFFDDDDDD);
    push_box(tris, 0.34, 0.67, 2.28, 0.08, 0.028, 0.02, 0xFFDDDDDD);

    // ── REAR BUMPER ──
    mesh::beveled_box_tris(tris, 0.0, 0.28, 2.24, 1.86, 0.20, 0.14, 0.03, trim);
    // Diffuser with vertical fins
    mesh::beveled_box_tris(tris, 0.0, 0.14, 2.28, 1.46, 0.10, 0.08, 0.02, 0xFF2A2A2A);
    for i in 0..5 {
        push_box(tris, -0.44 + i as f32 * 0.22, 0.16, 2.30, 0.008, 0.06, 0.04, 0xFF333333);
    }

    // ── DUAL OVAL EXHAUST TIPS ──
    for &side in &[-1.0f32, 1.0] {
        let ex = side * 0.56;
        mesh::beveled_box_tris(tris, ex, 0.16, 2.32, 0.22, 0.10, 0.06, 0.015, chrome_dk);
        mesh::ellipsoid_tris(tris, ex, 0.16, 2.34, 0.065, 0.035, 0.025, 0, 0xFF111111);
        mesh::cylinder_tris(tris, ex, 0.16, 2.34, 0.055, 0.04, 6, chrome);
    }

    // ── LICENSE PLATE (rear center) ──
    push_box(tris, 0.0, 0.52, 2.30, 0.32, 0.07, 0.012, 0xFFDDDDDD);
    push_box(tris, 0.0, 0.52, 2.31, 0.35, 0.005, 0.008, 0xFF444444);

    // ── UNDERCARRIAGE PAN ──
    push_box(tris, 0.0, gc - 0.02, 0.0, 1.74, 0.035, 4.60, undercar);

    // ── SHARK FIN ANTENNA ──
    mesh::ellipsoid_tris(tris, 0.0, 1.44, 0.50, 0.030, 0.045, 0.09, 0, trim);

    // ══════════════════════════════════════════════════════════════
    //  INTERIOR (only rendered for player's vehicle)
    // ══════════════════════════════════════════════════════════════
    if show_interior {
        // Dashboard
        mesh::beveled_box_tris(tris, 0.0, 0.92, -0.68, 1.36, 0.12, 0.38, 0.02, DASHBOARD_COLOR);
        // Virtual cockpit screen
        push_box(tris, -0.34, 0.96, -0.80, 0.32, 0.07, 0.015, 0x00334455);
        // MMI touchscreen
        push_box(tris, 0.10, 1.00, -0.68, 0.24, 0.14, 0.015, 0x00334455);
        // Steering wheel (flat-bottom RS style)
        mesh::cylinder_tris(tris, -0.34, 1.02, -0.52, 0.14, 0.018, 10, STEERING_COLOR);
        push_box(tris, -0.34, 0.93, -0.52, 0.11, 0.018, 0.018, STEERING_COLOR);
        mesh::cylinder_tris(tris, -0.34, 0.96, -0.60, 0.026, 0.12, 4, STEERING_COLOR);
        // High center console
        mesh::beveled_box_tris(tris, 0.0, 0.74, -0.02, 0.26, 0.22, 0.84, 0.02,
            darken(DASHBOARD_COLOR, 0.8));
        // Gear selector
        push_box(tris, 0.0, 0.86, -0.14, 0.06, 0.038, 0.10, 0xFF333333);
        // Sport bucket seats
        for &sx in &[-0.38f32, 0.38] {
            mesh::beveled_box_tris(tris, sx, 0.68, -0.04, 0.46, 0.12, 0.50, 0.02, SEAT_COLOR);
            mesh::beveled_box_tris(tris, sx, 1.02, 0.18, 0.46, 0.50, 0.08, 0.02, SEAT_COLOR);
            push_box(tris, sx, 1.30, 0.20, 0.18, 0.11, 0.06, SEAT_COLOR);
            push_box(tris, sx - 0.21, 1.02, 0.18, 0.035, 0.34, 0.06, darken(SEAT_COLOR, 0.85));
            push_box(tris, sx + 0.21, 1.02, 0.18, 0.035, 0.34, 0.06, darken(SEAT_COLOR, 0.85));
        }
        // Rear bench
        mesh::beveled_box_tris(tris, 0.0, 0.66, 0.64, 1.14, 0.10, 0.42, 0.02, SEAT_COLOR);
        mesh::beveled_box_tris(tris, 0.0, 0.94, 0.80, 1.14, 0.34, 0.07, 0.02, SEAT_COLOR);
        // Rearview mirror
        push_box(tris, 0.0, 1.26, -0.54, 0.18, 0.05, 0.018, 0xFF556677);
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

    // Ragdoll rendering: dual-segment orientation with waist blend
    if npc.ragdoll_active {
        let base = tris.len();
        let p = &npc.ragdoll_points;
        // hips=0, chest=1, head=2, l_hand=3, r_hand=4, l_foot=5, r_foot=6

        // Helper: build orthonormal basis from up + hint right
        let make_basis = |up_raw: [f32; 3], right_hint: [f32; 3]| -> [[f32; 3]; 3] {
            let ulen = (up_raw[0]*up_raw[0] + up_raw[1]*up_raw[1] + up_raw[2]*up_raw[2]).sqrt().max(0.01);
            let up = [up_raw[0]/ulen, up_raw[1]/ulen, up_raw[2]/ulen];
            let dot = right_hint[0]*up[0] + right_hint[1]*up[1] + right_hint[2]*up[2];
            let rx = right_hint[0] - dot*up[0];
            let ry = right_hint[1] - dot*up[1];
            let rz = right_hint[2] - dot*up[2];
            let rlen = (rx*rx + ry*ry + rz*rz).sqrt().max(0.01);
            let right = [rx/rlen, ry/rlen, rz/rlen];
            let fwd = [
                right[1]*up[2] - right[2]*up[1],
                right[2]*up[0] - right[0]*up[2],
                right[0]*up[1] - right[1]*up[0],
            ];
            [right, up, fwd]
        };

        // Lower body: hips→feet midpoint for "up", feet spread for "right"
        let foot_mid = [
            (p[5][0]+p[6][0])*0.5, (p[5][1]+p[6][1])*0.5, (p[5][2]+p[6][2])*0.5,
        ];
        let lower_up = [p[0][0]-foot_mid[0], p[0][1]-foot_mid[1], p[0][2]-foot_mid[2]];
        let feet_right = [p[6][0]-p[5][0], p[6][1]-p[5][1], p[6][2]-p[5][2]];
        let lower = make_basis(lower_up, feet_right);

        // Upper body: chest→head for "up", hands spread for "right"
        let upper_up = [p[2][0]-p[1][0], p[2][1]-p[1][1], p[2][2]-p[1][2]];
        let hands_right = [p[4][0]-p[3][0], p[4][1]-p[3][1], p[4][2]-p[3][2]];
        let upper = make_basis(upper_up, hands_right);

        let job_hat = job_hat_color(npc.job);

        gen_nude_player_body(
            tris, 0.0, app.skin, 0.0, false, false, false,
            app.is_female,
            Some((&app, shirt, npc.pants_color)),
            job_hat,
        );

        // Blend zone: waist Y in stretched coords (~1.15 to ~1.55)
        let blend_lo = 1.15; // hip region (post-stretch)
        let blend_hi = 1.55; // chest region (post-stretch)

        for tri in &mut tris[base..] {
            for v in &mut tri.v {
                let lx = v[0]; let ly = v[1]; let lz = v[2];
                // Blend weight: 0=lower body, 1=upper body
                let t = ((ly - blend_lo) / (blend_hi - blend_lo)).clamp(0.0, 1.0);

                if t < 0.01 {
                    // Pure lower body
                    v[0] = lower[0][0]*lx + lower[1][0]*ly + lower[2][0]*lz + foot_mid[0];
                    v[1] = lower[0][1]*lx + lower[1][1]*ly + lower[2][1]*lz + foot_mid[1];
                    v[2] = lower[0][2]*lx + lower[1][2]*ly + lower[2][2]*lz + foot_mid[2];
                } else if t > 0.99 {
                    // Pure upper body — pivot from chest point
                    let rel_y = ly - blend_hi;
                    v[0] = upper[0][0]*lx + upper[1][0]*rel_y + upper[2][0]*lz + p[1][0];
                    v[1] = upper[0][1]*lx + upper[1][1]*rel_y + upper[2][1]*lz + p[1][1];
                    v[2] = upper[0][2]*lx + upper[1][2]*rel_y + upper[2][2]*lz + p[1][2];
                } else {
                    // Blend zone — interpolate between lower and upper transforms
                    let lo_x = lower[0][0]*lx + lower[1][0]*ly + lower[2][0]*lz + foot_mid[0];
                    let lo_y = lower[0][1]*lx + lower[1][1]*ly + lower[2][1]*lz + foot_mid[1];
                    let lo_z = lower[0][2]*lx + lower[1][2]*ly + lower[2][2]*lz + foot_mid[2];
                    let rel_y = ly - blend_hi;
                    let hi_x = upper[0][0]*lx + upper[1][0]*rel_y + upper[2][0]*lz + p[1][0];
                    let hi_y = upper[0][1]*lx + upper[1][1]*rel_y + upper[2][1]*lz + p[1][1];
                    let hi_z = upper[0][2]*lx + upper[1][2]*rel_y + upper[2][2]*lz + p[1][2];
                    v[0] = lo_x * (1.0 - t) + hi_x * t;
                    v[1] = lo_y * (1.0 - t) + hi_y * t;
                    v[2] = lo_z * (1.0 - t) + hi_z * t;
                }
            }
            // Normal: blend rotation only (no translation)
            let nx = tri.normal[0]; let ny = tri.normal[1]; let nz = tri.normal[2];
            let avg_y = (tri.v[0][1] + tri.v[1][1] + tri.v[2][1]) / 3.0;
            let t = ((avg_y - blend_lo) / (blend_hi - blend_lo)).clamp(0.0, 1.0);
            if t < 0.5 {
                tri.normal[0] = lower[0][0]*nx + lower[1][0]*ny + lower[2][0]*nz;
                tri.normal[1] = lower[0][1]*nx + lower[1][1]*ny + lower[2][1]*nz;
                tri.normal[2] = lower[0][2]*nx + lower[1][2]*ny + lower[2][2]*nz;
            } else {
                tri.normal[0] = upper[0][0]*nx + upper[1][0]*ny + upper[2][0]*nz;
                tri.normal[1] = upper[0][1]*nx + upper[1][1]*ny + upper[2][1]*nz;
                tri.normal[2] = upper[0][2]*nx + upper[1][2]*ny + upper[2][2]*nz;
            }
        }
        return;
    }

    let base = tris.len();

    // KO pose — actual character model lying face-down on the ground
    if npc.state == NpcState::KnockedOut {
        let job_hat = job_hat_color(npc.job);

        gen_nude_player_body(
            tris, 0.0, app.skin, 0.0, false, false, false,
            app.is_female,
            Some((&app, shirt, npc.pants_color)),
            job_hat,
        );

        // Rotate -90° around X (face-down), then terrain-aligned heading rotation
        let rot = terrain_rot3x3(clamp_normal_tilt(npc.terrain_normal, 25.0), npc.rot_y);
        for tri in &mut tris[base..] {
            for v in &mut tri.v {
                // First: rotate -90° around X axis (face-down)
                let local = [v[0], v[2], -v[1]];
                // Then: terrain-aligned rotation + translate to world
                let rv = rot3x3_apply(&rot, local);
                v[0] = rv[0] + npc.x;
                v[1] = rv[1] + npc.y + 0.15;
                v[2] = rv[2] + npc.z;
            }
            let local_n = [tri.normal[0], tri.normal[2], -tri.normal[1]];
            tri.normal = rot3x3_apply(&rot, local_n);
        }
        return;
    }

    // Job-specific hat color
    let job_hat = job_hat_color(npc.job);

    gen_nude_player_body(
        tris,
        npc.walk_phase.sin() * 0.4,
        app.skin,
        npc.attack_phase,
        npc.carrying_item,
        npc.carrying_bin.is_some(),
        false,
        app.is_female,
        Some((&app, shirt, npc.pants_color)),
        job_hat,
    );

    // Speech bubble — rounded balloon with pointer tail + "..." dots
    if npc.interacting_with.is_some() {
        // Main balloon (wide oval)
        mesh::ellipsoid_tris(tris, 0.0, 2.85, -0.18, 0.16, 0.08, 0.10, 1, 0xFFF8F8F0);
        // Outline/shadow rim (slightly larger, darker)
        mesh::ellipsoid_tris(tris, 0.0, 2.85, -0.18, 0.165, 0.084, 0.104, 1, 0xFFDDDDCC);
        // Pointer tail — two diminishing spheres toward mouth
        mesh::sphere_tris(tris, 0.0, 2.72, -0.12, 0.035, 0, 0xFFF8F8F0);
        mesh::sphere_tris(tris, 0.0, 2.66, -0.08, 0.018, 0, 0xFFF8F8F0);
        // Ellipsis dots "..." inside balloon
        for i in 0..3 {
            let dx = (i as f32 - 1.0) * 0.055;
            mesh::sphere_tris(tris, dx, 2.85, -0.28, 0.018, 0, 0xFF666666);
        }
    }

    place_mesh(tris, base, npc.terrain_normal, 25.0, npc.rot_y, npc.x, npc.y, npc.z);
}

pub fn gen_item_mesh(item: &Item, tris: &mut Vec<WorldTri>) {
    let color = match item.kind {
        ItemKind::Health => 0xFFFF3333,
        ItemKind::Money => 0xFFFFDD33,
        ItemKind::Stamina => 0xFFFFFF33, // bright yellow — high contrast against green terrain
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
                let nx = tri.normal[0] * cos_s + tri.normal[2] * sin_s;
                let nz = -tri.normal[0] * sin_s + tri.normal[2] * cos_s;
                tri.normal[0] = nx;
                tri.normal[2] = nz;
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
                let nx = tri.normal[0] * cos_s + tri.normal[2] * sin_s;
                let nz = -tri.normal[0] * sin_s + tri.normal[2] * cos_s;
                tri.normal[0] = nx;
                tri.normal[2] = nz;
            }
        }
        ItemKind::Water => {
            // Bottle — lathe profile
            let base = tris.len();
            let profile: [[f32;2]; 5] = [
                [0.0, -0.2], [0.1, -0.18], [0.1, 0.1], [0.05, 0.18], [0.0, 0.2],
            ];
            mesh::lathe_tris(tris, 0.0, 0.0, 0.0, &profile, 6, color);
            let (sin_s, cos_s) = item.spin_phase.sin_cos();
            for tri in &mut tris[base..] {
                for v in &mut tri.v {
                    let rx = v[0] * cos_s + v[2] * sin_s;
                    let rz = -v[0] * sin_s + v[2] * cos_s;
                    v[0] = rx + item.x;
                    v[1] += y;
                    v[2] = rz + item.z;
                }
                let nx = tri.normal[0] * cos_s + tri.normal[2] * sin_s;
                let nz = -tri.normal[0] * sin_s + tri.normal[2] * cos_s;
                tri.normal[0] = nx;
                tri.normal[2] = nz;
            }
        }
        ItemKind::Food => {
            // Apple — sphere body + small brown stem on top
            let base = tris.len();
            mesh::sphere_tris(tris, 0.0, 0.0, 0.0, 0.18, 2, color);
            // Stem — thin brown cylinder poking out the top
            mesh::cylinder_tris(tris, 0.0, 0.18, 0.0, 0.02, 0.08, 4, 0xFF664422);
            // Small green leaf at stem base
            mesh::box_tris(tris, 0.04, 0.19, 0.0, 0.06, 0.02, 0.03, 0xFF44AA44);
            let (sin_s, cos_s) = item.spin_phase.sin_cos();
            for tri in &mut tris[base..] {
                for v in &mut tri.v {
                    let rx = v[0] * cos_s + v[2] * sin_s;
                    let rz = -v[0] * sin_s + v[2] * cos_s;
                    v[0] = rx + item.x;
                    v[1] += y;
                    v[2] = rz + item.z;
                }
                let nx = tri.normal[0] * cos_s + tri.normal[2] * sin_s;
                let nz = -tri.normal[0] * sin_s + tri.normal[2] * cos_s;
                tri.normal[0] = nx;
                tri.normal[2] = nz;
            }
        }
        _ => {
            // Stamina — smooth sphere (subdivision 2 for reduced faceting)
            mesh::sphere_tris(tris, item.x, y, item.z, 0.2, 2, color);
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
    place_mesh(tris, base, bin.terrain_normal, 25.0, 0.0, bin.x, bin.y, bin.z);
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
        mesh::push_tri(tris, c[idx[0]], c[idx[1]], c[idx[2]], normal, color);
        mesh::push_tri(tris, c[idx[0]], c[idx[2]], c[idx[3]], normal, color);
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
    let n = 8; // reduced subdivision for mid-LOD
    let sk = app.skin;

    use std::f32::consts::PI;
    let hp = PI * 0.5;

    let swing = npc.walk_phase.sin() * 0.2;

    // Head — lofted skull shape (no face detail, no ears)
    let head_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
        (1.46, body_ring(0.0, -0.04, 0.06, 0.08, &[], n), sk),
        (1.55, body_ring(0.0, 0.0, 0.14, 0.16, &[], n), sk),
        (1.65, body_ring(0.0, 0.01, 0.19, 0.22, &[], n), sk),
        (1.78, body_ring(0.0, 0.02, 0.20, 0.24, &[], n), sk),
        (1.90, body_ring(0.0, 0.04, 0.18, 0.22, &[], n), sk),
        (2.00, body_ring(0.0, 0.04, 0.13, 0.16, &[], n), sk),
        (2.06, body_ring(0.0, 0.03, 0.06, 0.08, &[], n), sk),
    ];
    mesh::loft_y_tris(tris, &head_rings);

    // Neck
    let neck_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
        (1.42, body_ring(0.0, 0.0, 0.12, 0.11, &[], n), sk),
        (1.48, body_ring(0.0, 0.0, 0.11, 0.10, &[], n), sk),
    ];
    mesh::loft_y_tris(tris, &neck_rings);

    // Torso — lofted with shoulder/hip taper (clothing color)
    let torso_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
        (0.88, body_ring(0.0, 0.0, 0.18, 0.14, &[], n), npc.pants_color),
        (0.96, body_ring(0.0, 0.0, 0.16, 0.13, &[], n), arm_col),
        (1.04, body_ring(0.0, 0.0, 0.17, 0.14, &[], n), arm_col),
        (1.12, body_ring(0.0, 0.0, 0.20, 0.17, &[], n), arm_col),
        (1.22, body_ring(0.0, 0.0, 0.24, 0.19, &[], n), arm_col),
        (1.34, body_ring(0.0, 0.0, 0.28, 0.18, &[
            (hp, 0.3, 0.03), (PI + hp, 0.3, 0.03), // shoulder width
        ], n), arm_col),
        (1.42, body_ring(0.0, 0.0, 0.30, 0.16, &[
            (hp, 0.3, 0.04), (PI + hp, 0.3, 0.04),
        ], n), arm_col),
    ];
    mesh::loft_y_tris(tris, &torso_rings);

    // Arms — lofted tubes with taper
    for &side in &[-1.0f32, 1.0] {
        let arm_fwd = if side < 0.0 { swing * 0.1 } else { -swing * 0.1 };
        let sx = side * 0.24;
        let arm_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
            (1.38, body_ring(sx, arm_fwd * 0.1, 0.06, 0.06, &[], n), arm_col),
            (1.24, body_ring(sx, arm_fwd * 0.3, 0.055, 0.055, &[], n), arm_col),
            (1.10, body_ring(sx, arm_fwd * 0.5, 0.050, 0.050, &[], n), arm_col),
            (0.96, body_ring(sx, arm_fwd * 0.7, 0.045, 0.045, &[], n), arm_col),
            (0.80, body_ring(sx, arm_fwd * 0.9, 0.042, 0.042, &[], n), arm_col),
            (0.64, body_ring(sx, arm_fwd, 0.036, 0.036, &[], n), sk),
            (0.54, body_ring(sx, arm_fwd, 0.030, 0.030, &[], n), sk),
        ];
        mesh::loft_y_tris(tris, &arm_rings);
    }

    // Legs — lofted tubes with thigh/calf taper
    for &side in &[-1.0f32, 1.0] {
        let leg_fwd = if side < 0.0 { -swing * 0.15 } else { swing * 0.15 };
        let lx = side * 0.10;
        let leg_rings: Vec<(f32, Vec<[f32; 2]>, u32)> = vec![
            (0.86, body_ring(lx, leg_fwd * 0.1, 0.10, 0.10, &[], n), npc.pants_color),
            (0.72, body_ring(lx, leg_fwd * 0.3, 0.11, 0.10, &[], n), npc.pants_color),
            (0.58, body_ring(lx, leg_fwd * 0.5, 0.09, 0.08, &[], n), npc.pants_color),
            (0.48, body_ring(lx, leg_fwd * 0.7, 0.08, 0.07, &[], n), npc.pants_color),
            (0.36, body_ring(lx, leg_fwd * 0.85, 0.08, 0.07, &[], n), npc.pants_color),
            (0.22, body_ring(lx, leg_fwd, 0.06, 0.05, &[], n), npc.pants_color),
            (0.10, body_ring(lx, leg_fwd, 0.05, 0.05, &[], n), app.boot_col),
        ];
        mesh::loft_y_tris(tris, &leg_rings);
        // Boot box
        push_box(tris, lx, 0.04, leg_fwd, 0.06, 0.05, 0.10, app.boot_col);
    }

    // Apply body stretch + world transform
    for tri in &mut tris[base..] { for v in &mut tri.v { v[1] *= BODY_STRETCH; } }
    place_mesh(tris, base, npc.terrain_normal, 25.0, npc.rot_y, npc.x, npc.y, npc.z);
}

/// Low-detail NPC: 3 colored boxes (~36 tris vs ~14K full detail)
fn gen_npc_mesh_lod(npc: &Npc, tris: &mut Vec<WorldTri>) {
    let app = npc_appearance(npc.brain_idx as u32);
    let shirt = job_shirt_color(npc);
    let body_col = if app.has_coat { app.coat_col } else { shirt };
    let base = tris.len();
    push_box(tris, 0.0, 0.75, 0.0, 0.20, 0.55, 0.12, body_col);
    push_box(tris, 0.0, 0.25, 0.0, 0.13, 0.25, 0.10, npc.pants_color);
    push_box(tris, 0.0, 1.55, 0.0, 0.10, 0.12, 0.10, app.skin);
    place_mesh(tris, base, npc.terrain_normal, 25.0, npc.rot_y, npc.x, npc.y, npc.z);
}

// LOD distance thresholds (squared)
const LOD_NPC_FULL_SQ: f32 = 625.0;    // < 25m: full detail
const LOD_NPC_MID_SQ: f32 = 6400.0;    // 25-80m: medium detail (~200 tris)
const LOD_NPC_LOW_SQ: f32 = 40000.0;   // 80-200m: low detail boxes
const LOD_VEH_FULL_SQ: f32 = 2500.0;   // < 50m: full detail vehicle
const LOD_VEH_MID_SQ: f32 = 10000.0;   // 50-100m: medium detail vehicle
const LOD_VEH_DIST_SQ: f32 = 40000.0;  // > 200m: skip vehicles

/// Medium-detail vehicle: body + cabin + 4 wheels + windshield, ~100 tris
pub fn gen_vehicle_mesh_mid(v: &Vehicle, tris: &mut Vec<WorldTri>) {
    let base = tris.len();
    let color = v.color;
    let cabin_color = darken(color, VEHICLE_BODY_COLOR_DARKEN);
    // Body
    mesh::beveled_box_tris(tris, 0.0, 0.45, 0.0, 1.8, 0.6, 3.6, 0.08, color);
    // Cabin
    mesh::beveled_box_tris(tris, 0.0, 0.95, 0.2, 1.5, 0.5, 1.8, 0.06, cabin_color);
    // Windshield
    push_box(tris, 0.0, 0.95, -0.70, 1.3, 0.4, 0.05, WINDSHIELD_COLOR);
    // 4 wheels — X-axis aligned for vertical wheel
    for &(wx, wz) in &[(-0.88f32, -1.1f32), (0.88, -1.1), (-0.88, 1.1), (0.88, 1.1)] {
        mesh::cylinder_between(tris, [wx - 0.11, 0.28, wz], [wx + 0.11, 0.28, wz], 0.28, 5, TIRE_COLOR);
    }
    place_mesh(tris, base, v.terrain_normal, 30.0, v.rot_y, v.x, v.y, v.z);
}

/// Low-detail vehicle mesh: 2 colored boxes (body + cabin), ~24 tris
fn gen_vehicle_mesh_lod(v: &Vehicle, tris: &mut Vec<WorldTri>) {
    let base = tris.len();
    push_box(tris, 0.0, 0.35, 0.0, 1.8, 0.5, 3.6, v.color);
    push_box(tris, 0.0, 0.95, 0.2, 1.4, 0.45, 1.8, darken(v.color, 0.85));
    place_mesh(tris, base, v.terrain_normal, 30.0, v.rot_y, v.x, v.y, v.z);
}

/// Generate GPU vertices for dynamic entities only (call each frame)
pub fn generate_dynamic_gpu_vertices(
    world: &WorldData, player: &Player, cam: &Camera,
    scratch: &mut Vec<WorldTri>, out: &mut Vec<GpuVertex>,
    hour: f32,
    character_models: &[Vec<WorldTri>],
    animation_data: Option<&crate::skeleton_anim::AnimationData>,
    car_models: &[Vec<WorldTri>],
) {
    let eye = v3(cam.x, cam.y, cam.z);
    let fog_dist_sq = FOG_DIST_SQ;

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
            let is_night = hour < 6.0 || hour > 20.0;
            if !car_models.is_empty() {
                gen_vehicle_mesh_gltf(v, vi, car_models, scratch, show_interior, is_night);
            } else {
                gen_vehicle_mesh(v, scratch, show_interior, is_night);
            }
        } else if dist_sq < LOD_VEH_MID_SQ {
            gen_vehicle_mesh_mid(v, scratch);
        } else {
            gen_vehicle_mesh_lod(v, scratch);
        }
    }
    // NPCs: frustum + distance-based LOD
    // At night, reduce render distances to avoid floating bright dots (skin-colored head)
    let is_night = hour < 6.0 || hour > 20.0;
    let npc_mid_sq = if is_night { 2500.0 } else { LOD_NPC_MID_SQ }; // 50m at night vs 80m day
    let npc_low_sq = if is_night { npc_mid_sq } else { LOD_NPC_LOW_SQ }; // no low LOD at night
    for npc in &world.npcs {
        if npc.state == NpcState::Sleeping { continue; }
        if npc.in_vehicle { continue; }
        let dist_sq = match view_cull(eye, fwd_x, fwd_z, npc.x, npc.z, fog_dist_sq) {
            Some(d) => d,
            None => continue,
        };
        if dist_sq < LOD_NPC_FULL_SQ {
            if !character_models.is_empty() {
                gen_npc_mesh_gltf(npc, character_models, scratch, animation_data);
            } else {
                gen_npc_mesh(npc, scratch);
            }
        } else if dist_sq < npc_mid_sq {
            gen_npc_mesh_mid(npc, scratch);
        } else if dist_sq < npc_low_sq {
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
        if !character_models.is_empty() {
            gen_player_mesh_gltf(player, character_models, scratch, animation_data);
        } else {
            gen_player_mesh(player, scratch);
        }
    }
    // Night-time street light glow halos (dynamic — only rendered at night)
    if is_night {
        for sl in &world.street_lights {
            let dx = sl.x - eye[0];
            let dz = sl.z - eye[2];
            if dx * dx + dz * dz > 150.0 * 150.0 { continue; }
            let gy = sl.ground_y;
            mesh::glow_halo(scratch, sl.x, gy + 5.2, sl.z, 0.2, 1.5, 8, 0x00FFDD88);
        }
    }
    // Convert to GPU format (raw material colors — GPU shader does lighting)
    out.reserve(scratch.len() * 3 + 4000); // +sky dome, clouds, sun/moon glow, stars
    for tri in scratch.iter() {
        for i in 0..3 {
            out.push(GpuVertex {
                pos: tri.v[i],
                color_packed: tri.color,
                normal: tri.normal,
            });
        }
    }

    // Sky dome: gradient hemisphere centered on camera
    gen_sky_dome_gpu(out, eye, hour);
}

/// Generate sky dome hemisphere as GPU vertices (emissive, no lighting).
/// Smooth gradient from horizon haze through sky color to deep zenith,
/// with atmospheric clouds, sun/moon glow halos, and proper zenith cap.
fn gen_sky_dome_gpu(out: &mut Vec<GpuVertex>, eye: Vec3, hour: f32) {
    const SEGS: usize = 24;   // azimuthal segments (smooth horizon)
    const RINGS: usize = 16;  // altitude rings — smooth gradient
    const RADIUS: f32 = 250.0;

    let tc = time_colors(hour);
    let boost = (2.0 - tc.sun_strength * 2.0).clamp(1.0, 2.5);

    // Horizon color = sky color (matches clear color)
    let hr = ((tc.sky >> 16) & 0xFF) as f32;
    let hg = ((tc.sky >> 8) & 0xFF) as f32;
    let hb = (tc.sky & 0xFF) as f32;

    // Zenith color = deeper/darker version of sky
    let zr = hr * 0.25;
    let zg = hg * 0.35;
    let zb = (hb * 0.7 + 30.0).min(255.0); // keep some blue

    // Horizon haze: warm, slightly brighter band for atmospheric scattering
    let haze_r = (hr * 1.15 + 15.0).min(255.0);
    let haze_g = (hg * 1.05 + 8.0).min(255.0);
    let haze_b = (hb * 0.90 + 5.0).min(255.0);

    // Pack color for emissive vertex (alpha=0), compensated for shader boost
    let pack = |r: f32, g: f32, b: f32| -> u32 {
        let cr = (r / boost).min(255.0).max(0.0) as u32;
        let cg = (g / boost).min(255.0).max(0.0) as u32;
        let cb = (b / boost).min(255.0).max(0.0) as u32;
        (cr << 16) | (cg << 8) | cb  // alpha=0x00 (emissive)
    };

    let color_at = |t: f32| -> u32 {
        if t < 0.15 { let ht = (t / 0.15) * (t / 0.15); pack(haze_r + (hr - haze_r) * ht, haze_g + (hg - haze_g) * ht, haze_b + (hb - haze_b) * ht) }
        else { let st = (t - 0.15) / 0.85; let st2 = st * st; pack(hr + (zr - hr) * st2, hg + (zg - hg) * st2, hb + (zb - hb) * st2) }
    };

    // Vertex positions on hemisphere
    let pos = |ring: usize, seg: usize| -> Vec3 {
        let alt = (ring as f32 / RINGS as f32) * std::f32::consts::FRAC_PI_2;
        let az = (seg as f32 / SEGS as f32) * std::f32::consts::TAU;
        let cos_alt = alt.cos();
        [
            eye[0] + az.cos() * cos_alt * RADIUS,
            eye[1] + alt.sin() * RADIUS,
            eye[2] + az.sin() * cos_alt * RADIUS,
        ]
    };

    // Normal pointing inward (toward center)
    let norm = |ring: usize, seg: usize| -> Vec3 {
        let alt = (ring as f32 / RINGS as f32) * std::f32::consts::FRAC_PI_2;
        let az = (seg as f32 / SEGS as f32) * std::f32::consts::TAU;
        let cos_alt = alt.cos();
        [-az.cos() * cos_alt, -alt.sin(), -az.sin() * cos_alt]
    };

    // Generate quads between rings 0..(RINGS-1); zenith capped separately (E4 fix)
    for ring in 0..(RINGS - 1) {
        let c0 = color_at(ring as f32 / RINGS as f32);
        let c1 = color_at((ring + 1) as f32 / RINGS as f32);
        for seg in 0..SEGS {
            let nseg = (seg + 1) % SEGS;
            let p00 = pos(ring, seg);     let n00 = norm(ring, seg);
            let p10 = pos(ring, nseg);    let n10 = norm(ring, nseg);
            let p01 = pos(ring + 1, seg); let n01 = norm(ring + 1, seg);
            let p11 = pos(ring + 1, nseg); let n11 = norm(ring + 1, nseg);
            // Inward-facing: CW winding so front face points toward viewer inside dome
            out.push(GpuVertex { pos: p00, color_packed: c0, normal: n00 });
            out.push(GpuVertex { pos: p11, color_packed: c1, normal: n11 });
            out.push(GpuVertex { pos: p10, color_packed: c0, normal: n10 });
            out.push(GpuVertex { pos: p00, color_packed: c0, normal: n00 });
            out.push(GpuVertex { pos: p01, color_packed: c1, normal: n01 });
            out.push(GpuVertex { pos: p11, color_packed: c1, normal: n11 });
        }
    }
    // Zenith cap: triangle fan — avoids degenerate near-zero quads at pole (E4)
    {
        let zenith_pos: Vec3 = [eye[0], eye[1] + RADIUS, eye[2]];
        let zenith_norm: Vec3 = [0.0, -1.0, 0.0];
        let zenith_color = color_at(1.0);
        let rim_color = color_at((RINGS - 1) as f32 / RINGS as f32);
        for seg in 0..SEGS {
            let nseg = (seg + 1) % SEGS;
            out.push(GpuVertex { pos: pos(RINGS - 1, seg), color_packed: rim_color, normal: norm(RINGS - 1, seg) });
            out.push(GpuVertex { pos: zenith_pos, color_packed: zenith_color, normal: zenith_norm });
            out.push(GpuVertex { pos: pos(RINGS - 1, nseg), color_packed: rim_color, normal: norm(RINGS - 1, nseg) });
        }
    }

    // Lower hemisphere: horizon → ground-fog color below the terrain.
    // Prevents sky-blue clear color showing through gaps at map edges.
    // Full 90° below horizon to nadir, so even steep downward views are covered.
    const LOW_RINGS: usize = 4;

    // Ground-fog color: dark greenish (blends with terrain at distance)
    let gr = 42.0_f32;
    let gg = 107.0_f32;
    let gb = 42.0_f32;

    let low_ring_color = |ring: usize| -> u32 {
        // ring 0 = horizon, ring LOW_RINGS = nadir
        let t = ring as f32 / LOW_RINGS as f32;
        let r = ((hr + (gr - hr) * t) / boost).min(255.0).max(0.0) as u32;
        let g = ((hg + (gg - hg) * t) / boost).min(255.0).max(0.0) as u32;
        let b = ((hb + (gb - hb) * t) / boost).min(255.0).max(0.0) as u32;
        (r << 16) | (g << 8) | b  // alpha=0x00 (emissive)
    };

    // Vertex positions on lower hemisphere (negative altitude, full 90° to nadir)
    let low_pos = |ring: usize, seg: usize| -> Vec3 {
        let alt = -(ring as f32 / LOW_RINGS as f32) * std::f32::consts::FRAC_PI_2; // full 90° down
        let az = (seg as f32 / SEGS as f32) * std::f32::consts::TAU;
        let cos_alt = alt.cos();
        [
            eye[0] + az.cos() * cos_alt * RADIUS,
            eye[1] + alt.sin() * RADIUS,
            eye[2] + az.sin() * cos_alt * RADIUS,
        ]
    };

    let low_norm = |ring: usize, seg: usize| -> Vec3 {
        let alt = -(ring as f32 / LOW_RINGS as f32) * std::f32::consts::FRAC_PI_2;
        let az = (seg as f32 / SEGS as f32) * std::f32::consts::TAU;
        let cos_alt = alt.cos();
        [-az.cos() * cos_alt, -alt.sin(), -az.sin() * cos_alt]
    };

    for ring in 0..LOW_RINGS {
        let c0 = low_ring_color(ring);
        let c1 = low_ring_color(ring + 1);
        for seg in 0..SEGS {
            let nseg = (seg + 1) % SEGS;
            let p00 = low_pos(ring, seg);     let n00 = low_norm(ring, seg);
            let p10 = low_pos(ring, nseg);    let n10 = low_norm(ring, nseg);
            let p01 = low_pos(ring + 1, seg); let n01 = low_norm(ring + 1, seg);
            let p11 = low_pos(ring + 1, nseg); let n11 = low_norm(ring + 1, nseg);

            // Lower hemisphere faces inward — altitude reversal flips effective winding,
            // so use standard order (opposite of upper hemisphere's reversed order)
            out.push(GpuVertex { pos: p00, color_packed: c0, normal: n00 });
            out.push(GpuVertex { pos: p10, color_packed: c0, normal: n10 });
            out.push(GpuVertex { pos: p11, color_packed: c1, normal: n11 });
            out.push(GpuVertex { pos: p00, color_packed: c0, normal: n00 });
            out.push(GpuVertex { pos: p11, color_packed: c1, normal: n11 });
            out.push(GpuVertex { pos: p01, color_packed: c1, normal: n01 });
        }
    }

    // Clouds — clusters of overlapping puffs on the sky hemisphere
    const CLOUD_RADIUS: f32 = 210.0;
    const NUM_CLOUDS: u32 = 14;

    // Cloud base color by time of day
    let (cloud_r, cloud_g, cloud_b) = if tc.sun_strength > 0.5 {
        (220.0 + tc.sun_strength * 35.0, 220.0 + tc.sun_strength * 35.0, 230.0 + tc.sun_strength * 25.0)
    } else if tc.sun_strength > 0.05 {
        let t = (tc.sun_strength - 0.05) / 0.45;
        (80.0 + t * 175.0, 60.0 + t * 170.0, 70.0 + t * 170.0)
    } else {
        (35.0, 35.0, 45.0)
    };

    // Skip clouds at deep night — they render as dark blobs
    let clouds_visible = tc.sun_strength > 0.02;

    for ci in 0..NUM_CLOUDS {
        if !clouds_visible { break; }
        let ch = ci.wrapping_mul(2654435761);
        let az = (ci as f32 / NUM_CLOUDS as f32) * std::f32::consts::TAU + (ch % 100) as f32 * 0.01;
        let alt_frac = 0.30 + (ch % 40) as f32 * 0.008;
        let alt = alt_frac * std::f32::consts::FRAC_PI_2;
        let cos_alt = alt.cos();

        // Cloud center on sphere
        let base_x = eye[0] + az.cos() * cos_alt * CLOUD_RADIUS;
        let base_y = eye[1] + alt.sin() * CLOUD_RADIUS;
        let base_z = eye[2] + az.sin() * cos_alt * CLOUD_RADIUS;

        let rx = -az.sin();
        let rz = az.cos();
        let n = [-az.cos() * cos_alt, -alt.sin(), -az.sin() * cos_alt];

        // 4-7 puffs per cloud for fuller shapes
        let num_puffs = 4 + (ch % 4) as i32;
        for pi in 0..num_puffs {
            let ph = ch.wrapping_add((pi as u32).wrapping_mul(1013904223));
            let off_r = ((ph % 50) as f32 - 25.0) * 0.6;
            let off_y = ((ph >> 4) % 16) as f32 * 0.3 - 2.0;
            let px = base_x + rx * off_r;
            let py = base_y + off_y;
            let pz = base_z + rz * off_r;

            // Puff size — center puffs larger, edges smaller
            let size_t = 1.0 - (pi as f32 / num_puffs as f32 - 0.5).abs() * 1.5;
            let pw = (14.0 + (ph % 18) as f32 * 1.0) * size_t.max(0.5);
            let phh = (7.0 + (ph % 10) as f32 * 0.8) * size_t.max(0.6);

            let bright = ((ph >> 8) % 25) as f32 - 10.0;
            let cr = ((cloud_r + bright) / boost).clamp(0.0, 255.0) as u32;
            let cg = ((cloud_g + bright) / boost).clamp(0.0, 255.0) as u32;
            let cb = ((cloud_b + bright * 0.5) / boost).clamp(0.0, 255.0) as u32;
            let cc = (cr << 16) | (cg << 8) | cb; // alpha=0 emissive

            let hw = pw * 0.5;
            let hh = phh * 0.5;
            let p0 = [px - rx * hw, py - hh, pz - rz * hw];
            let p1 = [px + rx * hw, py - hh, pz + rz * hw];
            let p2 = [px + rx * hw, py + hh, pz + rz * hw];
            let p3 = [px - rx * hw, py + hh, pz - rz * hw];

            out.push(GpuVertex { pos: p0, color_packed: cc, normal: n });
            out.push(GpuVertex { pos: p1, color_packed: cc, normal: n });
            out.push(GpuVertex { pos: p2, color_packed: cc, normal: n });
            out.push(GpuVertex { pos: p0, color_packed: cc, normal: n });
            out.push(GpuVertex { pos: p2, color_packed: cc, normal: n });
            out.push(GpuVertex { pos: p3, color_packed: cc, normal: n });
        }
    }

    // Sun/moon disc — emissive circles tracking across the sky
    // Sun arcs east→overhead→west; moon is opposite
    let sun_angle = (hour - 6.0) / 12.0 * std::f32::consts::PI; // 0 at 6am, PI at 18pm
    let is_day = tc.sun_strength > 0.05;
    let disc_dist = 220.0;

    // Billboard axes for a point in the sky (reused by disc, glow, stars)
    let billboard_axes = |cx: f32, cy: f32, cz: f32| -> ([f32; 3], [f32; 3], [f32; 3]) {
        let dn = [-(cx - eye[0]), -(cy - eye[1]), -(cz - eye[2])];
        let dl = (dn[0]*dn[0] + dn[1]*dn[1] + dn[2]*dn[2]).sqrt();
        let dn = [dn[0]/dl, dn[1]/dl, dn[2]/dl];
        let up = if dn[1].abs() > 0.99 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let tx = [up[1]*dn[2] - up[2]*dn[1], up[2]*dn[0] - up[0]*dn[2], up[0]*dn[1] - up[1]*dn[0]];
        let tl = (tx[0]*tx[0] + tx[1]*tx[1] + tx[2]*tx[2]).sqrt();
        let tx = [tx[0]/tl, tx[1]/tl, tx[2]/tl];
        let ty = [dn[1]*tx[2] - dn[2]*tx[1], dn[2]*tx[0] - dn[0]*tx[2], dn[0]*tx[1] - dn[1]*tx[0]];
        (dn, tx, ty)
    };

    let emit_disc = |out: &mut Vec<GpuVertex>, cx: f32, cy: f32, cz: f32, radius: f32, color: u32| {
        let (dn, tx, ty) = billboard_axes(cx, cy, cz);
        for i in 0..12u32 {
            let a0 = (i as f32 / 12.0) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / 12.0) * std::f32::consts::TAU;
            let p0 = [cx, cy, cz];
            let p1 = [cx + (a0.cos()*tx[0] + a0.sin()*ty[0])*radius, cy + (a0.cos()*tx[1] + a0.sin()*ty[1])*radius, cz + (a0.cos()*tx[2] + a0.sin()*ty[2])*radius];
            let p2 = [cx + (a1.cos()*tx[0] + a1.sin()*ty[0])*radius, cy + (a1.cos()*tx[1] + a1.sin()*ty[1])*radius, cz + (a1.cos()*tx[2] + a1.sin()*ty[2])*radius];
            out.push(GpuVertex { pos: p0, color_packed: color, normal: dn });
            out.push(GpuVertex { pos: p1, color_packed: color, normal: dn });
            out.push(GpuVertex { pos: p2, color_packed: color, normal: dn });
        }
    };

    // Glow halo: concentric rings with fading brightness (E3 fix)
    let emit_glow = |out: &mut Vec<GpuVertex>, cx: f32, cy: f32, cz: f32, inner_r: f32, cc: (f32, f32, f32), nrings: u32| {
        let (dn, tx, ty) = billboard_axes(cx, cy, cz);
        for ring in 0..nrings {
            let r0 = inner_r + inner_r * (ring as f32 * 0.55);
            let r1 = inner_r + inner_r * ((ring + 1) as f32 * 0.55);
            let f0 = (1.0 - ring as f32 / nrings as f32).powi(2) * 0.50;
            let f1 = (1.0 - (ring + 1) as f32 / nrings as f32).powi(2) * 0.50;
            let ci = ((((cc.0*f0)/boost).clamp(0.0,255.0) as u32) << 16) | ((((cc.1*f0)/boost).clamp(0.0,255.0) as u32) << 8) | (((cc.2*f0)/boost).clamp(0.0,255.0) as u32);
            let co = ((((cc.0*f1)/boost).clamp(0.0,255.0) as u32) << 16) | ((((cc.1*f1)/boost).clamp(0.0,255.0) as u32) << 8) | (((cc.2*f1)/boost).clamp(0.0,255.0) as u32);
            for i in 0..16u32 {
                let a0 = (i as f32 / 16.0) * std::f32::consts::TAU;
                let a1 = ((i+1) as f32 / 16.0) * std::f32::consts::TAU;
                let pi0 = [cx+(a0.cos()*tx[0]+a0.sin()*ty[0])*r0, cy+(a0.cos()*tx[1]+a0.sin()*ty[1])*r0, cz+(a0.cos()*tx[2]+a0.sin()*ty[2])*r0];
                let pi1 = [cx+(a1.cos()*tx[0]+a1.sin()*ty[0])*r0, cy+(a1.cos()*tx[1]+a1.sin()*ty[1])*r0, cz+(a1.cos()*tx[2]+a1.sin()*ty[2])*r0];
                let po0 = [cx+(a0.cos()*tx[0]+a0.sin()*ty[0])*r1, cy+(a0.cos()*tx[1]+a0.sin()*ty[1])*r1, cz+(a0.cos()*tx[2]+a0.sin()*ty[2])*r1];
                let po1 = [cx+(a1.cos()*tx[0]+a1.sin()*ty[0])*r1, cy+(a1.cos()*tx[1]+a1.sin()*ty[1])*r1, cz+(a1.cos()*tx[2]+a1.sin()*ty[2])*r1];
                out.push(GpuVertex { pos: pi0, color_packed: ci, normal: dn });
                out.push(GpuVertex { pos: pi1, color_packed: ci, normal: dn });
                out.push(GpuVertex { pos: po1, color_packed: co, normal: dn });
                out.push(GpuVertex { pos: pi0, color_packed: ci, normal: dn });
                out.push(GpuVertex { pos: po1, color_packed: co, normal: dn });
                out.push(GpuVertex { pos: po0, color_packed: co, normal: dn });
            }
        }
    };

    if is_day {
        let sy = sun_angle.sin().max(0.05);
        let sx = sun_angle.cos();
        let sun_x = eye[0] + sx * disc_dist * 0.8;
        let sun_y = eye[1] + sy * disc_dist;
        let sun_z = eye[2] + disc_dist * 0.3;
        let horizon_t = (1.0 - sy * 2.0).clamp(0.0, 1.0);
        let sr = (255.0 / boost).min(255.0) as u32;
        let sg = (((1.0 - horizon_t * 0.4) * 230.0) / boost).min(255.0) as u32;
        let sb = (((1.0 - horizon_t * 0.7) * 180.0) / boost).min(255.0) as u32;
        let sun_color = (sr << 16) | (sg << 8) | sb;
        // Warm glow halo behind disc
        emit_glow(out, sun_x, sun_y, sun_z, 12.0, (255.0, (1.0-horizon_t*0.3)*220.0, (1.0-horizon_t*0.5)*160.0), 4);
        emit_disc(out, sun_x, sun_y, sun_z, 12.0, sun_color);
    } else {
        let moon_elev = sun_angle.sin().abs().max(0.2);
        let moon_az = sun_angle.cos();
        let moon_x = eye[0] - moon_az * disc_dist * 0.6;
        let moon_y = eye[1] + moon_elev * disc_dist * 0.9;
        let moon_z = eye[2] + disc_dist * 0.4;
        let mr = (210.0 / boost).min(255.0) as u32;
        let mg = (210.0 / boost).min(255.0) as u32;
        let mb = (225.0 / boost).min(255.0) as u32;
        let moon_color = (mr << 16) | (mg << 8) | mb;
        // Cool silver-blue moonlight glow
        emit_glow(out, moon_x, moon_y, moon_z, 8.0, (180.0, 190.0, 210.0), 3);
        emit_disc(out, moon_x, moon_y, moon_z, 8.0, moon_color);
    }

    // Stars — small emissive dots scattered across the sky, visible at night
    if tc.sun_strength < 0.15 {
        let star_alpha = ((0.15 - tc.sun_strength) / 0.15).clamp(0.0, 1.0);
        let star_dist = 230.0;
        for si in 0..80u32 {
            let h = si.wrapping_mul(2654435761).wrapping_add(0xDEAD);
            let h2 = h.wrapping_mul(1664525).wrapping_add(1013904223);
            // Distribute stars across hemisphere using hash-based spherical coords
            let az = (h % 10000) as f32 / 10000.0 * std::f32::consts::TAU;
            let alt = ((h2 % 10000) as f32 / 10000.0 * 0.7 + 0.1) * std::f32::consts::FRAC_PI_2;
            let cos_alt = alt.cos();
            let sx = eye[0] + az.cos() * cos_alt * star_dist;
            let sy = eye[1] + alt.sin() * star_dist;
            let sz = eye[2] + az.sin() * cos_alt * star_dist;

            let brightness = (140.0 + (h % 116) as f32) * star_alpha;
            let bv = (brightness / boost).min(255.0) as u32;
            let (sr, sg, sb) = match h2 % 10 {
                0..=1 => (bv, bv, (bv + 15).min(255)),
                2 => ((bv + 10).min(255), (bv + 5).min(255), bv.saturating_sub(10)),
                _ => (bv, bv, bv),
            };
            let star_color = (sr << 16) | (sg << 8) | sb; // alpha=0 emissive

            let n = [-(az.cos() * cos_alt), -alt.sin(), -(az.sin() * cos_alt)];
            // Proper billboard axes avoid degenerate quads
            let star_r = 0.6 + (h % 5) as f32 * 0.2;
            let (_, stx, sty) = billboard_axes(sx, sy, sz);
            let p0 = [sx-stx[0]*star_r-sty[0]*star_r, sy-stx[1]*star_r-sty[1]*star_r, sz-stx[2]*star_r-sty[2]*star_r];
            let p1 = [sx+stx[0]*star_r-sty[0]*star_r, sy+stx[1]*star_r-sty[1]*star_r, sz+stx[2]*star_r-sty[2]*star_r];
            let p2 = [sx+stx[0]*star_r+sty[0]*star_r, sy+stx[1]*star_r+sty[1]*star_r, sz+stx[2]*star_r+sty[2]*star_r];
            let p3 = [sx-stx[0]*star_r+sty[0]*star_r, sy-stx[1]*star_r+sty[1]*star_r, sz-stx[2]*star_r+sty[2]*star_r];
            out.push(GpuVertex { pos: p0, color_packed: star_color, normal: n });
            out.push(GpuVertex { pos: p1, color_packed: star_color, normal: n });
            out.push(GpuVertex { pos: p2, color_packed: star_color, normal: n });
            out.push(GpuVertex { pos: p0, color_packed: star_color, normal: n });
            out.push(GpuVertex { pos: p2, color_packed: star_color, normal: n });
            out.push(GpuVertex { pos: p3, color_packed: star_color, normal: n });
        }
    }
}

/// Build GPU push constants from current frame state
pub fn gpu_push_constants(hour: f32, eye: Vec3, target: Vec3, vp: &Mat4) -> crate::gpu::GpuPushConstants {
    let tc = time_colors(hour);
    let fog_dist_sq = FOG_DIST_SQ;
    let fdx = target[0] - eye[0];
    let fdz = target[2] - eye[2];
    let flen = (fdx * fdx + fdz * fdz).sqrt().max(0.001);
    // At night, fold moonlight into the GPU sun_strength so the shader's
    // dot(N,L)*sun term provides faint directional moonlight (light_dir already
    // points to the moon). Also ensure ambient doesn't clamp away nighttime variation.
    let gpu_sun = if tc.sun_strength < 0.01 { 0.04 } else { tc.sun_strength };
    let gpu_amb = tc.ambient.max(0.08);
    crate::gpu::GpuPushConstants {
        vp: *vp,
        light_dir_ambient: [tc.light_dir[0], tc.light_dir[1], tc.light_dir[2], gpu_amb],
        sun_fog_params: [gpu_sun, 1.0 / fog_dist_sq, fdx / flen, fdz / flen],
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

/// Software-rasterize a set of triangles from a given camera orbit, returning a pixel buffer.
/// Used by studio and body_view for model preview rendering.
///   tris: world-space triangles to render
///   bearing: camera bearing in degrees (0 = front, 90 = right)
///   pitch: camera pitch in degrees (positive = looking down)
///   dist: camera distance from model center
///   width, height: output image dimensions
///   bg_color: background fill color (ARGB)
pub fn render_model_to_pixels(
    tris: &[WorldTri],
    bearing: f32, pitch: f32, dist: f32,
    width: usize, height: usize,
    bg_color: u32,
) -> Vec<u32> {
    // Find model vertical center for camera target
    let mut cy = 0.0f32;
    let mut count = 0usize;
    for tri in tris {
        for v in &tri.v { cy += v[1]; count += 1; }
    }
    if count > 0 { cy /= count as f32; }

    // Camera orbiting model center
    let br = bearing.to_radians();
    let pr = pitch.to_radians();
    let eye_x = dist * br.sin() * pr.cos();
    let eye_z = -dist * br.cos() * pr.cos();
    let eye_y = cy + dist * pr.sin();

    let view = m4_look_at([eye_x, eye_y, eye_z], [0.0, cy, 0.0], [0.0, 1.0, 0.0]);
    let proj = m4_perspective(45.0f32.to_radians(), width as f32 / height as f32, 0.05, 50.0);
    let vp = m4_mul(&proj, &view);

    // Software rasterize
    let mut fb = Framebuffer::new(width, height);
    fb.clear(bg_color);

    // Ground plane
    let ground_col = 0xFF0F0F1Au32;
    let ground_tris = [
        WorldTri { v: [[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [2.0, 0.0, 2.0]], normal: [0.0, 1.0, 0.0], color: ground_col },
        WorldTri { v: [[-2.0, 0.0, -2.0], [2.0, 0.0, 2.0], [-2.0, 0.0, 2.0]], normal: [0.0, 1.0, 0.0], color: ground_col },
    ];

    // Light direction (sun from above-front-right)
    let light = [0.4f32, 0.8, -0.45];
    let ll = (light[0]*light[0]+light[1]*light[1]+light[2]*light[2]).sqrt();
    let light_n = [light[0]/ll, light[1]/ll, light[2]/ll];

    for tri in ground_tris.iter().chain(tris.iter()) {
        // Transform to clip space
        let mut sv = [[0.0f32; 4]; 3];
        for i in 0..3 {
            sv[i] = m4_transform_no_div(&vp, tri.v[i]);
        }
        // Clip behind near plane
        if sv[0][3] <= 0.01 && sv[1][3] <= 0.01 && sv[2][3] <= 0.01 { continue; }

        // Perspective divide + viewport
        let mut sx = [[0.0f32; 3]; 3];
        let mut any_behind = false;
        for i in 0..3 {
            if sv[i][3] <= 0.01 { any_behind = true; break; }
            let inv_w = 1.0 / sv[i][3];
            sx[i][0] = (0.5 + sv[i][0] * inv_w * 0.5) * width as f32;
            sx[i][1] = (0.5 - sv[i][1] * inv_w * 0.5) * height as f32;
            sx[i][2] = sv[i][2] * inv_w;
        }
        if any_behind { continue; }

        // Backface cull
        let cross = (sx[1][0]-sx[0][0])*(sx[2][1]-sx[0][1]) - (sx[1][1]-sx[0][1])*(sx[2][0]-sx[0][0]);
        if cross <= 0.0 { continue; }

        // Simple diffuse lighting
        let dot = tri.normal[0]*light_n[0] + tri.normal[1]*light_n[1] + tri.normal[2]*light_n[2];
        let shade = 0.45 + 0.55 * dot.max(0.0);

        let r = ((tri.color >> 16) & 0xFF) as f32 * shade;
        let g = ((tri.color >> 8) & 0xFF) as f32 * shade;
        let b = (tri.color & 0xFF) as f32 * shade;
        let col = 0xFF000000 | ((r.min(255.0) as u32) << 16) | ((g.min(255.0) as u32) << 8) | (b.min(255.0) as u32);

        let screen_tri = ScreenTri { v: sx, color: col };
        draw_triangle(&mut fb, &screen_tri);
    }

    fb.pixels
}

/// Build VP matrix, push constants, and clear color for a frame.
pub fn frame_setup(
    width: usize, height: usize,
    eye: Vec3, target: Vec3, hour: f32,
) -> (Mat4, crate::gpu::GpuPushConstants, [f32; 4]) {
    let aspect = width as f32 / height as f32;
    let view = m4_look_at(eye, target, [0.0, 1.0, 0.0]);
    let proj = m4_perspective_vk(60.0_f32.to_radians(), aspect, 0.1, 500.0);
    let vp = m4_mul(&proj, &view);
    let push = gpu_push_constants(hour, eye, target, &vp);
    let clear = sky_color_f32(hour);
    (vp, push, clear)
}
