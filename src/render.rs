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

// ACU-style NPC appearance palettes
const SKIN_TONES: [u32; 5] = [0xFFDEB887, 0xFFD2A87A, 0xFFC89B6E, 0xFFE8C9A0, 0xFFBB9060];
const HAIR_COLORS: [u32; 6] = [0xFF332211, 0xFF443322, 0xFF221100, 0xFF554433, 0xFF664422, 0xFF887755];
const HAT_COLORS: [u32; 5] = [0xFF333333, 0xFF554433, 0xFF222222, 0xFF443344, 0xFF665544];
const COAT_COLORS: [u32; 8] = [
    0xFF443322, 0xFF333355, 0xFF554433, 0xFF444444,
    0xFF553333, 0xFF335544, 0xFF555544, 0xFF443355,
];
const BOOT_COLOR: u32 = 0xFF332211;
const STOCKING_COLOR: u32 = 0xFFCCBBAA;
const SHIRT_WHITE: u32 = 0xFFDDCCBB;
const BUCKLE_COLOR: u32 = 0xFFCCBB88;

/// Derive NPC appearance from a seed. No struct storage needed.
struct NpcAppearance {
    skin: u32,
    hair: u32,
    hat_type: u8,   // 0=none, 1=tricorn, 2=top_hat, 3=cap, 4=bonnet, 5=wide_brim
    hat_col: u32,
    coat_col: u32,
    has_coat: bool,
    is_female: bool,
}

fn npc_appearance(seed: u32) -> NpcAppearance {
    NpcAppearance {
        skin: SKIN_TONES[(seed / 3) as usize % SKIN_TONES.len()],
        hair: HAIR_COLORS[(seed / 5) as usize % HAIR_COLORS.len()],
        hat_type: (seed / 7 % 6) as u8,
        hat_col: HAT_COLORS[(seed / 11) as usize % HAT_COLORS.len()],
        coat_col: COAT_COLORS[(seed / 13) as usize % COAT_COLORS.len()],
        has_coat: seed % 4 != 0,
        is_female: seed % 5 == 0,
    }
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

/// Generate a detailed ACU-style character body in local space (origin at feet).
/// Shared between NPC and player rendering.
///
/// Body proportions (1.78m total):
///   Shoes   0.00-0.06    Calves  0.06-0.40   Knees ~0.40
///   Thighs  0.40-0.80    Hips    0.80-0.88   Waist  0.88
///   Torso   0.88-1.40    Shoulders ~1.42     Neck   1.42-1.52
///   Head    1.52-1.90    Hat crown ~2.0
fn gen_character_body(
    tris: &mut Vec<WorldTri>,
    swing: f32,          // walk animation phase sine (-0.4..0.4)
    app: &NpcAppearance,
    shirt_col: u32,
    pants_col: u32,
    attack_phase: f32,
    carrying_item: bool,
    carrying_bin: bool,
    sitting: bool,
    is_job_hat: Option<u32>, // job-specific hat color override
) {
    let skin = app.skin;
    let hair = app.hair;
    let coat = app.coat_col;
    let arm_outer = if app.has_coat { coat } else { shirt_col };

    if sitting {
        gen_seated_body(tris, app, shirt_col, pants_col, is_job_hat);
        return;
    }

    // ═══════════════════════ HEAD ═══════════════════════
    // Shaped head using lathe (forehead, cheek, jaw, chin)
    let head_profile: [[f32; 2]; 7] = [
        [0.0, -0.20], [0.10, -0.17], [0.15, -0.10], [0.18, 0.0],
        [0.17, 0.10], [0.13, 0.18], [0.0, 0.22],
    ];
    mesh::lathe_tris(tris, 0.0, 1.70, 0.0, &head_profile, 8, skin);

    // Brow ridge — subtle protrusion above eyes
    push_box(tris, 0.0, 1.76, -0.15, 0.18, 0.02, 0.04, darken(skin, 0.92));

    // Eye sockets — dark recesses
    push_box(tris, -0.065, 1.735, -0.155, 0.05, 0.025, 0.02, darken(skin, 0.75));
    push_box(tris, 0.065, 1.735, -0.155, 0.05, 0.025, 0.02, darken(skin, 0.75));
    // Eyeballs — white
    mesh::sphere_tris(tris, -0.065, 1.735, -0.145, 0.022, 0, 0xFFEEDDCC);
    mesh::sphere_tris(tris, 0.065, 1.735, -0.145, 0.022, 0, 0xFFEEDDCC);
    // Pupils — dark
    mesh::sphere_tris(tris, -0.065, 1.735, -0.165, 0.014, 0, 0xFF221100);
    mesh::sphere_tris(tris, 0.065, 1.735, -0.165, 0.014, 0, 0xFF221100);
    // Eyelids — skin-colored arcs over eyes
    push_box(tris, -0.065, 1.75, -0.16, 0.048, 0.008, 0.025, skin);
    push_box(tris, 0.065, 1.75, -0.16, 0.048, 0.008, 0.025, skin);

    // Nose — bridge + tip + nostrils
    push_box(tris, 0.0, 1.72, -0.17, 0.025, 0.05, 0.03, skin); // bridge
    mesh::sphere_tris(tris, 0.0, 1.695, -0.19, 0.025, 0, darken(skin, 0.95)); // tip
    // Nostrils — two tiny dark dots
    mesh::sphere_tris(tris, -0.012, 1.69, -0.19, 0.008, 0, darken(skin, 0.7));
    mesh::sphere_tris(tris, 0.012, 1.69, -0.19, 0.008, 0, darken(skin, 0.7));

    // Mouth — thin dark line
    push_box(tris, 0.0, 1.665, -0.17, 0.06, 0.008, 0.01, darken(skin, 0.7));
    // Lower lip — subtle
    push_box(tris, 0.0, 1.658, -0.165, 0.04, 0.006, 0.015, darken(skin, 0.88));

    // Chin — slight protrusion
    mesh::sphere_tris(tris, 0.0, 1.64, -0.14, 0.03, 0, skin);

    // Ears — shaped with inner cavity suggestion
    for &ex in &[-0.175f32, 0.175] {
        mesh::sphere_tris(tris, ex, 1.73, 0.0, 0.035, 0, skin);
        mesh::sphere_tris(tris, ex, 1.73, 0.0, 0.02, 0, darken(skin, 0.8)); // inner
    }

    // Jaw line — subtle ridge from ears to chin
    push_box(tris, -0.1, 1.655, -0.06, 0.06, 0.015, 0.1, darken(skin, 0.95));
    push_box(tris, 0.1, 1.655, -0.06, 0.06, 0.015, 0.1, darken(skin, 0.95));

    // ═══════════════════════ HAIR / HAT ═══════════════════════
    let hat = if let Some(jc) = is_job_hat {
        // Job-specific hat overrides
        gen_job_hat(tris, jc);
        true
    } else {
        gen_hat(tris, app.hat_type, app.hat_col, hair)
    };

    // Hair visible below/around hat
    if hat {
        // Nape hair
        mesh::sphere_tris(tris, 0.0, 1.65, 0.1, 0.1, 0, hair);
        // Sideburns
        push_box(tris, -0.155, 1.71, -0.02, 0.02, 0.06, 0.04, hair);
        push_box(tris, 0.155, 1.71, -0.02, 0.02, 0.06, 0.04, hair);
    } else {
        // Full hair — top, sides, back
        mesh::sphere_tris(tris, 0.0, 1.85, 0.02, 0.155, 1, hair);
        push_box(tris, -0.13, 1.74, 0.0, 0.04, 0.1, 0.12, hair);
        push_box(tris, 0.13, 1.74, 0.0, 0.04, 0.1, 0.12, hair);
        mesh::sphere_tris(tris, 0.0, 1.7, 0.12, 0.12, 0, hair); // back
    }

    // ═══════════════════════ NECK ═══════════════════════
    mesh::cylinder_tris(tris, 0.0, 1.49, 0.0, 0.055, 0.1, 5, skin);
    // Adam's apple / throat detail
    mesh::sphere_tris(tris, 0.0, 1.48, -0.055, 0.015, 0, skin);

    // ═══════════════════════ TORSO ═══════════════════════
    // Undershirt (white/cream, visible at collar and cuffs)
    mesh::cylinder_tris(tris, 0.0, 1.15, 0.0, 0.17, 0.5, 7, SHIRT_WHITE);
    // Shirt collar — V-neck fold
    push_box(tris, -0.04, 1.42, -0.12, 0.06, 0.04, 0.06, SHIRT_WHITE);
    push_box(tris, 0.04, 1.42, -0.12, 0.06, 0.04, 0.06, SHIRT_WHITE);

    // Waistcoat / vest over shirt
    mesh::beveled_box_tris(tris, 0.0, 1.15, 0.0, 0.38, 0.48, 0.26, 0.02, shirt_col);
    // Waistcoat buttons (4 down center)
    for bi in 0..4 {
        let by = 1.32 - bi as f32 * 0.08;
        mesh::sphere_tris(tris, 0.0, by, -0.14, 0.01, 0, BUCKLE_COLOR);
    }
    // Waistcoat pocket flaps
    push_box(tris, -0.1, 1.08, -0.14, 0.06, 0.015, 0.02, darken(shirt_col, 0.85));
    push_box(tris, 0.1, 1.08, -0.14, 0.06, 0.015, 0.02, darken(shirt_col, 0.85));

    if app.has_coat {
        // ═══════════ LONG COAT ═══════════
        // Shoulders — wider than body
        mesh::beveled_box_tris(tris, 0.0, 1.35, 0.0, 0.52, 0.2, 0.32, 0.03, coat);

        // Coat body — upper half
        mesh::beveled_box_tris(tris, 0.0, 1.1, 0.0, 0.48, 0.52, 0.3, 0.03, coat);

        // Coat collar (raised, folded)
        push_box(tris, 0.0, 1.44, 0.0, 0.4, 0.06, 0.3, darken(coat, 0.8));
        // Lapels (triangular fold at chest)
        push_box(tris, -0.09, 1.32, -0.16, 0.08, 0.16, 0.02, darken(coat, 0.85));
        push_box(tris, 0.09, 1.32, -0.16, 0.08, 0.16, 0.02, darken(coat, 0.85));

        // Coat buttons (brass, 3 down front)
        for bi in 0..3 {
            let by = 1.28 - bi as f32 * 0.1;
            mesh::sphere_tris(tris, 0.0, by, -0.165, 0.012, 0, 0xFFBBAA66);
        }

        // Button holes (tiny dark marks opposite buttons)
        for bi in 0..3 {
            let by = 1.28 - bi as f32 * 0.1;
            push_box(tris, -0.025, by, -0.166, 0.015, 0.005, 0.005, darken(coat, 0.6));
        }

        // Coat pocket flaps
        push_box(tris, -0.13, 1.0, -0.16, 0.08, 0.02, 0.02, darken(coat, 0.85));
        push_box(tris, 0.13, 1.0, -0.16, 0.08, 0.02, 0.02, darken(coat, 0.85));

        // Back seam (center line down back)
        push_box(tris, 0.0, 1.1, 0.16, 0.01, 0.5, 0.005, darken(coat, 0.75));

        // Coat tails — split at back, sway with walk
        let tail_sway = swing * 0.12;
        // Left tail
        push_box(tris, -0.09, 0.58, 0.1 + tail_sway, 0.15, 0.52, 0.1, coat);
        // Right tail
        push_box(tris, 0.09, 0.58, 0.1 - tail_sway, 0.15, 0.52, 0.1, coat);
        // Tail inner lining (slightly different color)
        push_box(tris, -0.09, 0.58, 0.05 + tail_sway, 0.13, 0.48, 0.01, darken(coat, 1.15));
        push_box(tris, 0.09, 0.58, 0.05 - tail_sway, 0.13, 0.48, 0.01, darken(coat, 1.15));

        // Front coat skirts (shorter, to mid-thigh)
        push_box(tris, -0.1, 0.68, -0.1, 0.13, 0.34, 0.08, coat);
        push_box(tris, 0.1, 0.68, -0.1, 0.13, 0.34, 0.08, coat);

        // Shoulder epaulettes (subtle ridge)
        push_box(tris, -0.27, 1.42, 0.0, 0.06, 0.02, 0.08, darken(coat, 0.8));
        push_box(tris, 0.27, 1.42, 0.0, 0.06, 0.02, 0.08, darken(coat, 0.8));
    }

    // ═══════════════════════ BELT / WAIST ═══════════════════════
    mesh::cylinder_tris(tris, 0.0, 0.87, 0.0, 0.19, 0.06, 6, pants_col);
    // Leather belt
    mesh::cylinder_tris(tris, 0.0, 0.87, 0.0, 0.205, 0.025, 8, 0xFF443322);
    // Belt buckle — rectangular brass
    push_box(tris, 0.0, 0.87, -0.21, 0.03, 0.02, 0.01, BUCKLE_COLOR);
    // Belt pouch (small on right hip)
    push_box(tris, 0.18, 0.85, 0.0, 0.05, 0.06, 0.04, 0xFF553322);
    // Pouch flap
    push_box(tris, 0.18, 0.885, 0.0, 0.05, 0.01, 0.045, 0xFF442211);

    // ═══════════════════════ LEGS ═══════════════════════
    // Animated walk: swing controls hip angle, knee bends on trailing leg
    let l_fwd = -swing * 0.35;   // left leg z-offset
    let r_fwd = swing * 0.35;    // right leg z-offset
    // Knee bend: trailing leg bends more (back-swing)
    let l_knee = if swing > 0.0 { swing * 0.12 } else { 0.0 };
    let r_knee = if swing < 0.0 { (-swing) * 0.12 } else { 0.0 };

    // Thighs (breeches — knee-length fitted pants)
    mesh::cylinder_tris(tris, -0.1, 0.6, l_fwd, 0.075, 0.38, 6, pants_col);
    mesh::cylinder_tris(tris, 0.1, 0.6, r_fwd, 0.075, 0.38, 6, pants_col);
    // Breeches knee buttons (period detail)
    mesh::sphere_tris(tris, -0.14, 0.42, l_fwd, 0.01, 0, BUCKLE_COLOR);
    mesh::sphere_tris(tris, 0.14, 0.42, r_fwd, 0.01, 0, BUCKLE_COLOR);

    // Knee joint (slightly wider)
    mesh::sphere_tris(tris, -0.1, 0.4, l_fwd * 0.8, 0.06, 0, pants_col);
    mesh::sphere_tris(tris, 0.1, 0.4, r_fwd * 0.8, 0.06, 0, pants_col);

    // Calves (stockings — lighter color, fitted)
    let l_calf_z = l_fwd * 0.5 - l_knee;
    let r_calf_z = r_fwd * 0.5 - r_knee;
    mesh::cylinder_tris(tris, -0.1, 0.22, l_calf_z, 0.055, 0.32, 5, STOCKING_COLOR);
    mesh::cylinder_tris(tris, 0.1, 0.22, r_calf_z, 0.055, 0.32, 5, STOCKING_COLOR);

    // Stocking tops (garter line — darker ring)
    mesh::cylinder_tris(tris, -0.1, 0.37, l_calf_z + l_knee * 0.3, 0.058, 0.015, 4, darken(STOCKING_COLOR, 0.8));
    mesh::cylinder_tris(tris, 0.1, 0.37, r_calf_z + r_knee * 0.3, 0.058, 0.015, 4, darken(STOCKING_COLOR, 0.8));

    // Shoes — beveled boxes with buckle
    let l_shoe_z = l_calf_z - 0.04;
    let r_shoe_z = r_calf_z - 0.04;
    mesh::beveled_box_tris(tris, -0.1, 0.035, l_shoe_z, 0.08, 0.06, 0.14, 0.01, BOOT_COLOR);
    mesh::beveled_box_tris(tris, 0.1, 0.035, r_shoe_z, 0.08, 0.06, 0.14, 0.01, BOOT_COLOR);
    // Shoe heels (slightly raised back)
    push_box(tris, -0.1, 0.02, l_shoe_z + 0.05, 0.065, 0.03, 0.03, darken(BOOT_COLOR, 0.8));
    push_box(tris, 0.1, 0.02, r_shoe_z + 0.05, 0.065, 0.03, 0.03, darken(BOOT_COLOR, 0.8));
    // Shoe tongue / top flap
    push_box(tris, -0.1, 0.065, l_shoe_z - 0.02, 0.04, 0.02, 0.04, darken(BOOT_COLOR, 1.1));
    push_box(tris, 0.1, 0.065, r_shoe_z - 0.02, 0.04, 0.02, 0.04, darken(BOOT_COLOR, 1.1));
    // Shoe buckles
    mesh::sphere_tris(tris, -0.1, 0.05, l_shoe_z - 0.06, 0.015, 0, BUCKLE_COLOR);
    mesh::sphere_tris(tris, 0.1, 0.05, r_shoe_z - 0.06, 0.015, 0, BUCKLE_COLOR);

    // ═══════════════════════ ARMS ═══════════════════════
    if attack_phase > 0.0 {
        gen_attack_arms(tris, attack_phase, arm_outer, skin, app.has_coat, coat, swing);
    } else if carrying_item {
        gen_carry_arms(tris, arm_outer, skin, app.has_coat, coat);
        mesh::beveled_box_tris(tris, 0.0, 0.88, -0.48, 0.28, 0.28, 0.18, 0.02, BAG_COLOR);
        // Bag strap
        push_box(tris, 0.0, 1.1, -0.35, 0.02, 0.4, 0.02, 0xFF553322);
    } else if carrying_bin {
        gen_carry_arms(tris, arm_outer, skin, app.has_coat, coat);
        mesh::cylinder_tris(tris, 0.0, 0.78, -0.52, 0.2, 0.55, 6, BIN_COLOR);
    } else {
        gen_swing_arms(tris, swing, arm_outer, skin, app.has_coat, coat);
    }
}

fn gen_hat(tris: &mut Vec<WorldTri>, hat_type: u8, hat_col: u32, hair: u32) -> bool {
    match hat_type {
        1 => {
            // Tricorn hat
            // Crown — tapered cylinder
            mesh::cylinder_tris(tris, 0.0, 1.93, 0.0, 0.14, 0.1, 6, hat_col);
            // Brim — wide disc, turned up on 3 sides
            mesh::cylinder_tris(tris, 0.0, 1.88, 0.0, 0.22, 0.018, 8, hat_col);
            // Three upturned brim segments
            push_box(tris, 0.0, 1.91, -0.16, 0.14, 0.04, 0.06, darken(hat_col, 0.88));
            push_box(tris, -0.14, 1.91, 0.08, 0.06, 0.04, 0.12, darken(hat_col, 0.88));
            push_box(tris, 0.14, 1.91, 0.08, 0.06, 0.04, 0.12, darken(hat_col, 0.88));
            // Cockade / ribbon at front
            mesh::sphere_tris(tris, 0.0, 1.93, -0.18, 0.02, 0, 0xFF888866);
            true
        }
        2 => {
            // Top hat — tall cylinder + brim
            mesh::cylinder_tris(tris, 0.0, 2.02, 0.0, 0.11, 0.22, 7, hat_col);
            mesh::cylinder_tris(tris, 0.0, 1.9, 0.0, 0.17, 0.02, 8, hat_col);
            // Hat band
            mesh::cylinder_tris(tris, 0.0, 1.92, 0.0, 0.115, 0.02, 7, darken(hat_col, 0.7));
            true
        }
        3 => {
            // Worker's cap / beret
            mesh::sphere_tris(tris, 0.0, 1.9, -0.02, 0.15, 1, hat_col);
            // Visor
            push_box(tris, 0.0, 1.86, -0.16, 0.12, 0.01, 0.06, darken(hat_col, 0.85));
            true
        }
        4 => {
            // Bonnet — rounded with ribbon tie
            mesh::sphere_tris(tris, 0.0, 1.88, 0.03, 0.16, 1, hat_col);
            // Bonnet brim framing face
            push_box(tris, 0.0, 1.85, -0.14, 0.2, 0.06, 0.02, darken(hat_col, 0.9));
            // Ribbon under chin
            push_box(tris, -0.07, 1.6, -0.07, 0.015, 0.2, 0.01, 0xFFDDCCBB);
            push_box(tris, 0.07, 1.6, -0.07, 0.015, 0.2, 0.01, 0xFFDDCCBB);
            true
        }
        5 => {
            // Wide-brim hat
            mesh::cylinder_tris(tris, 0.0, 1.94, 0.0, 0.12, 0.1, 6, hat_col);
            mesh::cylinder_tris(tris, 0.0, 1.88, 0.0, 0.24, 0.02, 10, hat_col);
            // Hat band
            mesh::cylinder_tris(tris, 0.0, 1.9, 0.0, 0.125, 0.02, 6, darken(hat_col, 0.7));
            true
        }
        _ => false, // no hat — caller renders full hair
    }
}

fn gen_job_hat(tris: &mut Vec<WorldTri>, color: u32) {
    // Simple job-specific headwear (police, fire, construction, etc.)
    mesh::cylinder_tris(tris, 0.0, 1.93, 0.0, 0.17, 0.08, 6, color);
    // Visor/brim
    push_box(tris, 0.0, 1.88, -0.12, 0.14, 0.012, 0.06, darken(color, 0.8));
    // Badge
    mesh::sphere_tris(tris, 0.0, 1.91, -0.17, 0.015, 0, BUCKLE_COLOR);
}

fn gen_swing_arms(
    tris: &mut Vec<WorldTri>, swing: f32,
    arm_outer: u32, skin: u32, has_coat: bool, coat: u32,
) {
    // Upper arms (sleeve)
    mesh::cylinder_tris(tris, -0.27, 1.25, swing * 0.2, 0.055, 0.3, 5, arm_outer);
    mesh::cylinder_tris(tris, 0.27, 1.25, -swing * 0.2, 0.055, 0.3, 5, arm_outer);

    // Elbows (joint spheres)
    let l_elbow_z = swing * 0.2;
    let r_elbow_z = -swing * 0.2;
    mesh::sphere_tris(tris, -0.27, 1.1, l_elbow_z, 0.04, 0, arm_outer);
    mesh::sphere_tris(tris, 0.27, 1.1, r_elbow_z, 0.04, 0, arm_outer);

    // Forearms (skin visible below sleeve)
    let l_bend = 0.04 + swing.abs() * 0.08;
    let r_bend = 0.04 + swing.abs() * 0.08;
    mesh::cylinder_tris(tris, -0.28, 0.95, l_elbow_z - l_bend, 0.04, 0.26, 4, skin);
    mesh::cylinder_tris(tris, 0.28, 0.95, r_elbow_z - r_bend, 0.04, 0.26, 4, skin);

    // Shirt cuffs (white ruffled edge at wrist)
    mesh::cylinder_tris(tris, -0.28, 0.84, l_elbow_z - l_bend, 0.045, 0.025, 4, SHIRT_WHITE);
    mesh::cylinder_tris(tris, 0.28, 0.84, r_elbow_z - r_bend, 0.045, 0.025, 4, SHIRT_WHITE);

    if has_coat {
        // Coat sleeve cuffs (folded back, wider)
        mesh::cylinder_tris(tris, -0.27, 1.0, l_elbow_z * 0.7, 0.06, 0.06, 4, darken(coat, 0.8));
        mesh::cylinder_tris(tris, 0.27, 1.0, r_elbow_z * 0.7, 0.06, 0.06, 4, darken(coat, 0.8));
        // Cuff buttons
        mesh::sphere_tris(tris, -0.31, 1.0, l_elbow_z * 0.7, 0.008, 0, BUCKLE_COLOR);
        mesh::sphere_tris(tris, 0.31, 1.0, r_elbow_z * 0.7, 0.008, 0, BUCKLE_COLOR);
    }

    // Hands — flattened boxes with thumb suggestion
    let lhz = l_elbow_z - l_bend;
    let rhz = r_elbow_z - r_bend;
    push_box(tris, -0.28, 0.82, lhz, 0.04, 0.05, 0.055, skin);
    push_box(tris, 0.28, 0.82, rhz, 0.04, 0.05, 0.055, skin);
    // Thumb (small offset box)
    push_box(tris, -0.25, 0.83, lhz - 0.02, 0.015, 0.025, 0.025, skin);
    push_box(tris, 0.25, 0.83, rhz - 0.02, 0.015, 0.025, 0.025, skin);
    // Finger creases (subtle dark lines)
    push_box(tris, -0.28, 0.81, lhz - 0.02, 0.035, 0.003, 0.003, darken(skin, 0.8));
    push_box(tris, 0.28, 0.81, rhz - 0.02, 0.035, 0.003, 0.003, darken(skin, 0.8));
}

fn gen_attack_arms(
    tris: &mut Vec<WorldTri>, attack_phase: f32,
    arm_outer: u32, skin: u32, has_coat: bool, coat: u32, swing: f32,
) {
    let t = (attack_phase / ATTACK_ANIM_DURATION).clamp(0.0, 1.0);
    let extend = 1.0 - (1.0 - t) * (1.0 - t);

    // Right arm — punching forward
    mesh::cylinder_tris(tris, 0.3, 1.25, -0.1 - extend * 0.2, 0.055, 0.3, 5, arm_outer);
    mesh::cylinder_tris(tris, 0.3, 1.1, -0.3 - extend * 0.35, 0.045, 0.28, 4, skin);
    // Fist (clenched)
    mesh::sphere_tris(tris, 0.3, 1.05, -0.45 - extend * 0.35, 0.055, 0, skin);

    // Left arm — guard position
    mesh::cylinder_tris(tris, -0.27, 1.25, 0.05, 0.055, 0.3, 5, arm_outer);
    mesh::cylinder_tris(tris, -0.28, 1.0, -0.1, 0.04, 0.26, 4, skin);
    push_box(tris, -0.28, 0.88, -0.12, 0.04, 0.05, 0.055, skin);
}

fn gen_carry_arms(
    tris: &mut Vec<WorldTri>,
    arm_outer: u32, skin: u32, has_coat: bool, coat: u32,
) {
    // Both arms forward, holding object
    mesh::cylinder_tris(tris, -0.24, 1.2, -0.2, 0.055, 0.3, 5, arm_outer);
    mesh::cylinder_tris(tris, 0.24, 1.2, -0.2, 0.055, 0.3, 5, arm_outer);
    mesh::cylinder_tris(tris, -0.26, 1.0, -0.35, 0.04, 0.26, 4, skin);
    mesh::cylinder_tris(tris, 0.26, 1.0, -0.35, 0.04, 0.26, 4, skin);
    push_box(tris, -0.26, 0.88, -0.42, 0.04, 0.05, 0.05, skin);
    push_box(tris, 0.26, 0.88, -0.42, 0.04, 0.05, 0.05, skin);
}

fn gen_seated_body(
    tris: &mut Vec<WorldTri>,
    app: &NpcAppearance, shirt_col: u32, pants_col: u32,
    is_job_hat: Option<u32>,
) {
    let skin = app.skin;
    let coat = app.coat_col;
    let arm_outer = if app.has_coat { coat } else { shirt_col };

    // Head (same detail as standing)
    let head_profile: [[f32; 2]; 7] = [
        [0.0, -0.20], [0.10, -0.17], [0.15, -0.10], [0.18, 0.0],
        [0.17, 0.10], [0.13, 0.18], [0.0, 0.22],
    ];
    mesh::lathe_tris(tris, 0.0, 1.3, 0.0, &head_profile, 8, skin);
    // Eyes
    mesh::sphere_tris(tris, -0.065, 1.335, -0.165, 0.014, 0, 0xFF221100);
    mesh::sphere_tris(tris, 0.065, 1.335, -0.165, 0.014, 0, 0xFF221100);
    // Nose
    mesh::sphere_tris(tris, 0.0, 1.295, -0.19, 0.025, 0, skin);
    // Neck
    mesh::cylinder_tris(tris, 0.0, 1.09, 0.0, 0.055, 0.1, 5, skin);
    // Torso
    mesh::cylinder_tris(tris, 0.0, 0.75, 0.0, 0.17, 0.5, 6, SHIRT_WHITE);
    mesh::beveled_box_tris(tris, 0.0, 0.75, 0.0, 0.38, 0.48, 0.26, 0.02, shirt_col);
    if app.has_coat {
        mesh::beveled_box_tris(tris, 0.0, 0.75, 0.0, 0.48, 0.52, 0.3, 0.03, coat);
    }
    // Seated thighs (horizontal)
    mesh::cylinder_tris(tris, -0.12, 0.42, -0.2, 0.07, 0.35, 5, pants_col);
    mesh::cylinder_tris(tris, 0.12, 0.42, -0.2, 0.07, 0.35, 5, pants_col);
    // Shins (hanging down)
    mesh::cylinder_tris(tris, -0.12, 0.15, -0.38, 0.055, 0.3, 4, STOCKING_COLOR);
    mesh::cylinder_tris(tris, 0.12, 0.15, -0.38, 0.055, 0.3, 4, STOCKING_COLOR);
    // Shoes
    mesh::beveled_box_tris(tris, -0.12, 0.02, -0.42, 0.07, 0.05, 0.12, 0.01, BOOT_COLOR);
    mesh::beveled_box_tris(tris, 0.12, 0.02, -0.42, 0.07, 0.05, 0.12, 0.01, BOOT_COLOR);
    // Arms resting
    mesh::cylinder_tris(tris, -0.3, 0.65, -0.12, 0.055, 0.4, 4, arm_outer);
    mesh::cylinder_tris(tris, 0.3, 0.65, -0.12, 0.055, 0.4, 4, arm_outer);
    mesh::cylinder_tris(tris, -0.3, 0.48, -0.25, 0.04, 0.26, 4, skin);
    mesh::cylinder_tris(tris, 0.3, 0.48, -0.25, 0.04, 0.26, 4, skin);
}

pub fn gen_player_mesh(player: &Player, tris: &mut Vec<WorldTri>) {
    let base = tris.len();
    let shirt = if player.hit_flash > 0.0 { 0xFFFF4444 } else { SHIRT_COLOR };
    let app = NpcAppearance {
        skin: SKIN_COLOR, hair: 0xFF332211,
        hat_type: 0, hat_col: 0, coat_col: 0xFF334466,
        has_coat: true, is_female: false,
    };

    gen_character_body(
        tris,
        player.walk_phase.sin() * 0.4,
        &app,
        shirt,
        PANTS_COLOR,
        player.attack_phase,
        player.carrying_item,
        player.carrying_bin.is_some(),
        player.sitting,
        None,
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
        mesh::cylinder_between(tris, p[0], p[1], 0.2, 6, arm_col);
        // Head
        mesh::lathe_tris(tris, p[2][0], p[2][1], p[2][2],
            &[[0.0, -0.15], [0.12, -0.08], [0.16, 0.02], [0.12, 0.12], [0.0, 0.16]], 6, app.skin);
        // Neck
        mesh::cylinder_between(tris, p[1], p[2], 0.05, 4, app.skin);
        // Arms
        mesh::cylinder_between(tris, p[1], p[3], 0.05, 4, arm_col);
        mesh::cylinder_between(tris, p[1], p[4], 0.05, 4, arm_col);
        push_box(tris, p[3][0], p[3][1], p[3][2], 0.05, 0.04, 0.05, app.skin);
        push_box(tris, p[4][0], p[4][1], p[4][2], 0.05, 0.04, 0.05, app.skin);
        // Legs
        mesh::cylinder_between(tris, p[0], p[5], 0.07, 5, npc.pants_color);
        mesh::cylinder_between(tris, p[0], p[6], 0.07, 5, npc.pants_color);
        mesh::beveled_box_tris(tris, p[5][0], p[5][1], p[5][2], 0.07, 0.05, 0.12, 0.01, BOOT_COLOR);
        mesh::beveled_box_tris(tris, p[6][0], p[6][1], p[6][2], 0.07, 0.05, 0.12, 0.01, BOOT_COLOR);
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
        mesh::beveled_box_tris(tris, -0.14, 0.03, 0.72, 0.07, 0.05, 0.12, 0.01, BOOT_COLOR);
        mesh::beveled_box_tris(tris, 0.14, 0.03, 0.72, 0.07, 0.05, 0.12, 0.01, BOOT_COLOR);

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
