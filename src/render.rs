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

pub fn gen_player_mesh(player: &Player, tris: &mut Vec<WorldTri>) {
    let base = tris.len();
    let shirt = if player.hit_flash > 0.0 { 0xFFFF4444 } else { SHIRT_COLOR };

    if player.sitting {
        // Seated pose — cylinder torso, sphere head
        mesh::cylinder_tris(tris, 0.0, 0.65, 0.0, 0.2, 0.7, 6, shirt);
        mesh::sphere_tris(tris, 0.0, 1.35, 0.0, 0.2, 1, SKIN_COLOR);
        // Eyes
        mesh::sphere_tris(tris, -0.08, 1.4, -0.17, 0.03, 0, 0xFF222222);
        mesh::sphere_tris(tris, 0.08, 1.4, -0.17, 0.03, 0, 0xFF222222);
        // Thighs — horizontal cylinders
        mesh::cylinder_tris(tris, -0.12, 0.32, -0.25, 0.07, 0.45, 4, PANTS_COLOR);
        mesh::cylinder_tris(tris, 0.12, 0.32, -0.25, 0.07, 0.45, 4, PANTS_COLOR);
        // Shins
        mesh::cylinder_tris(tris, -0.12, 0.08, -0.45, 0.07, 0.30, 4, PANTS_COLOR);
        mesh::cylinder_tris(tris, 0.12, 0.08, -0.45, 0.07, 0.30, 4, PANTS_COLOR);
        // Arms
        mesh::cylinder_tris(tris, -0.32, 0.55, -0.15, 0.06, 0.45, 4, SKIN_COLOR);
        mesh::cylinder_tris(tris, 0.32, 0.55, -0.15, 0.06, 0.45, 4, SKIN_COLOR);
    } else {
        let phase = player.walk_phase;
        let swing = phase.sin() * 0.4;

        // Torso — cylinder
        mesh::cylinder_tris(tris, 0.0, 1.05, 0.0, 0.2, 0.7, 6, shirt);
        // Hips
        mesh::cylinder_tris(tris, 0.0, 0.65, 0.0, 0.18, 0.15, 5, PANTS_COLOR);
        // Head — sphere with face
        mesh::sphere_tris(tris, 0.0, 1.75, 0.0, 0.2, 1, SKIN_COLOR);
        mesh::sphere_tris(tris, -0.08, 1.8, -0.17, 0.03, 0, 0xFF222222); // eyes
        mesh::sphere_tris(tris, 0.08, 1.8, -0.17, 0.03, 0, 0xFF222222);
        mesh::sphere_tris(tris, 0.0, 1.73, -0.2, 0.025, 0, SKIN_COLOR); // nose

        // Legs — cylinders with swing
        mesh::cylinder_tris(tris, -0.1, 0.35, -swing * 0.35, 0.07, 0.65, 5, PANTS_COLOR);
        mesh::cylinder_tris(tris, 0.1, 0.35, swing * 0.35, 0.07, 0.65, 5, PANTS_COLOR);
        // Shoes
        mesh::sphere_tris(tris, -0.1, 0.05, -swing * 0.35, 0.08, 0, 0xFF333333);
        mesh::sphere_tris(tris, 0.1, 0.05, swing * 0.35, 0.08, 0, 0xFF333333);

        if player.attack_phase > 0.0 {
            let t = (player.attack_phase / ATTACK_ANIM_DURATION).clamp(0.0, 1.0);
            let extend = 1.0 - (1.0 - t) * (1.0 - t);
            let right_z = -0.15 - extend * 0.8;
            let left_z = 0.15 + extend * 0.2;
            mesh::cylinder_tris(tris, -0.35, 1.05, left_z, 0.06, 0.55, 4, SKIN_COLOR);
            mesh::cylinder_tris(tris, 0.35, 1.05, right_z, 0.06, 0.55, 4, SKIN_COLOR);
            mesh::sphere_tris(tris, 0.35, 1.05, right_z - 0.3, 0.07, 0, SKIN_COLOR);
        } else if player.carrying_item {
            mesh::cylinder_tris(tris, -0.25, 1.0, -0.35, 0.06, 0.5, 4, SKIN_COLOR);
            mesh::cylinder_tris(tris, 0.25, 1.0, -0.35, 0.06, 0.5, 4, SKIN_COLOR);
            mesh::beveled_box_tris(tris, 0.0, 0.9, -0.5, 0.3, 0.3, 0.2, 0.03, BAG_COLOR);
        } else if player.carrying_bin.is_some() {
            mesh::cylinder_tris(tris, -0.25, 1.0, -0.4, 0.06, 0.5, 4, SKIN_COLOR);
            mesh::cylinder_tris(tris, 0.25, 1.0, -0.4, 0.06, 0.5, 4, SKIN_COLOR);
            mesh::cylinder_tris(tris, 0.0, 0.8, -0.55, 0.2, 0.6, 6, BIN_COLOR);
        } else {
            mesh::cylinder_tris(tris, -0.3, 1.05, swing * 0.25, 0.06, 0.55, 4, SKIN_COLOR);
            mesh::cylinder_tris(tris, 0.3, 1.05, -swing * 0.25, 0.06, 0.55, 4, SKIN_COLOR);
            mesh::sphere_tris(tris, -0.3, 0.8, swing * 0.25, 0.06, 0, SKIN_COLOR);
            mesh::sphere_tris(tris, 0.3, 0.8, -swing * 0.25, 0.06, 0, SKIN_COLOR);
        }
    }

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

    // Beveled main body
    mesh::beveled_box_tris(tris, 0.0, 0.45, 0.0, 1.8, 0.6, 3.6, 0.08, color);

    // Beveled cabin
    mesh::beveled_box_tris(tris, 0.0, 0.95, 0.2, 1.5, 0.5, 1.8, 0.06, cabin_color);

    // Sloped hood (front end slopes down)
    push_box(tris, 0.0, 0.55, -1.5, 1.6, 0.15, 0.5, darken(color, 0.85));

    // Sloped trunk (rear end)
    push_box(tris, 0.0, 0.6, 1.55, 1.6, 0.1, 0.4, darken(color, 0.85));

    // Windshields — recessed into body (not flush)
    push_box(tris, 0.0, 0.95, -0.7, 1.3, 0.35, 0.08, WINDSHIELD_COLOR);  // front
    push_box(tris, 0.0, 0.95, 1.15, 1.3, 0.35, 0.08, WINDSHIELD_COLOR);  // rear

    // Side windows
    push_box(tris, -0.76, 0.95, 0.2, 0.04, 0.35, 1.2, WINDSHIELD_COLOR);
    push_box(tris, 0.76, 0.95, 0.2, 0.04, 0.35, 1.2, WINDSHIELD_COLOR);

    // Cylinder wheels (6 segments for speed)
    let wheel_r = 0.2;
    let wheel_w = 0.18;
    for (wx, wz) in [(-0.85f32, -1.1f32), (0.85, -1.1), (-0.85, 1.1), (0.85, 1.1)] {
        // Tire — horizontal cylinder (rotated 90° around Z)
        mesh::cylinder_tris(tris, wx, 0.2, wz, wheel_r, wheel_w, 6, TIRE_COLOR);
        // Hub cap
        mesh::cylinder_tris(tris, wx, 0.2, wz, wheel_r * 0.5, wheel_w + 0.02, 4, 0xFF888888);
    }

    // Headlights
    mesh::sphere_tris(tris, -0.6, 0.45, -1.81, 0.12, 0, 0xFFFFEE88);
    mesh::sphere_tris(tris, 0.6, 0.45, -1.81, 0.12, 0, 0xFFFFEE88);

    // Tail lights
    mesh::sphere_tris(tris, -0.6, 0.45, 1.81, 0.1, 0, 0xFFFF2222);
    mesh::sphere_tris(tris, 0.6, 0.45, 1.81, 0.1, 0, 0xFFFF2222);

    // Bumpers
    mesh::box_tris(tris, 0.0, 0.25, -1.85, 1.7, 0.15, 0.08, 0xFF444444);
    mesh::box_tris(tris, 0.0, 0.25, 1.85, 1.7, 0.15, 0.08, 0xFF444444);

    // Interior details (only for player's vehicle)
    if show_interior {
        push_box(tris, 0.0, 0.75, -0.6, 1.3, 0.15, 0.4, DASHBOARD_COLOR);
        push_box(tris, -0.3, 0.85, -0.45, 0.25, 0.25, 0.05, STEERING_COLOR);
        push_box(tris, -0.35, 0.55, 0.0, 0.5, 0.15, 0.5, SEAT_COLOR);
        push_box(tris, -0.35, 0.85, 0.2, 0.5, 0.45, 0.1, SEAT_COLOR);
        push_box(tris, 0.35, 0.55, 0.0, 0.5, 0.15, 0.5, SEAT_COLOR);
        push_box(tris, 0.35, 0.85, 0.2, 0.5, 0.45, 0.1, SEAT_COLOR);
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

    // Ragdoll rendering: sphere joints + cylinder limb segments
    if npc.ragdoll_active {
        let p = &npc.ragdoll_points;
        // hips=0, chest=1, head=2, l_hand=3, r_hand=4, l_foot=5, r_foot=6
        // Torso: cylinder between hips and chest
        mesh::cylinder_between(tris, p[0], p[1], 0.2, 5, shirt);
        // Head: sphere at p[2]
        mesh::sphere_tris(tris, p[2][0], p[2][1], p[2][2], 0.18, 1, SKIN_COLOR);
        // Neck: cylinder from chest to head
        mesh::cylinder_between(tris, p[1], p[2], 0.06, 4, SKIN_COLOR);
        // Arms: cylinder from chest to hands
        mesh::cylinder_between(tris, p[1], p[3], 0.06, 4, SKIN_COLOR);
        mesh::cylinder_between(tris, p[1], p[4], 0.06, 4, SKIN_COLOR);
        // Hands: small spheres
        mesh::sphere_tris(tris, p[3][0], p[3][1], p[3][2], 0.07, 0, SKIN_COLOR);
        mesh::sphere_tris(tris, p[4][0], p[4][1], p[4][2], 0.07, 0, SKIN_COLOR);
        // Legs: cylinder from hips to feet
        mesh::cylinder_between(tris, p[0], p[5], 0.08, 4, npc.pants_color);
        mesh::cylinder_between(tris, p[0], p[6], 0.08, 4, npc.pants_color);
        // Feet: small spheres
        mesh::sphere_tris(tris, p[5][0], p[5][1], p[5][2], 0.08, 0, npc.pants_color);
        mesh::sphere_tris(tris, p[6][0], p[6][1], p[6][2], 0.08, 0, npc.pants_color);
        return;
    }

    let base = tris.len();

    // KO pose: body flat on ground
    if npc.state == NpcState::KnockedOut {
        // Torso — flat cylinder
        mesh::cylinder_tris(tris, 0.0, 0.15, 0.0, 0.2, 0.7, 5, shirt);
        // Head — sphere
        mesh::sphere_tris(tris, 0.0, 0.15, -0.55, 0.18, 1, SKIN_COLOR);
        // Arms — cylinders
        mesh::cylinder_tris(tris, -0.5, 0.1, 0.2, 0.06, 0.6, 4, SKIN_COLOR);
        mesh::cylinder_tris(tris, 0.5, 0.1, -0.1, 0.06, 0.6, 4, SKIN_COLOR);
        // Legs — cylinders
        mesh::cylinder_tris(tris, -0.15, 0.1, 0.5, 0.08, 0.65, 4, npc.pants_color);
        mesh::cylinder_tris(tris, 0.15, 0.1, 0.5, 0.08, 0.65, 4, npc.pants_color);

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

    let swing = npc.walk_phase.sin() * 0.4;

    // Torso — cylinder
    mesh::cylinder_tris(tris, 0.0, 1.05, 0.0, 0.2, 0.7, 6, shirt);
    // Hips — cylinder
    mesh::cylinder_tris(tris, 0.0, 0.65, 0.0, 0.18, 0.15, 5, npc.pants_color);

    // Head — sphere with face features
    mesh::sphere_tris(tris, 0.0, 1.75, 0.0, 0.2, 1, SKIN_COLOR);
    // Eyes — tiny spheres
    mesh::sphere_tris(tris, -0.08, 1.8, -0.17, 0.03, 0, 0xFF222222);
    mesh::sphere_tris(tris, 0.08, 1.8, -0.17, 0.03, 0, 0xFF222222);
    // Nose — tiny sphere
    mesh::sphere_tris(tris, 0.0, 1.73, -0.2, 0.025, 0, SKIN_COLOR);

    // Job accessories (hats/helmets)
    match npc.job {
        NpcJob::PolicePatrol => mesh::cylinder_tris(tris, 0.0, 1.95, 0.0, 0.2, 0.1, 6, 0xFF2222CC),
        NpcJob::Firefighter => mesh::sphere_tris(tris, 0.0, 1.97, 0.0, 0.22, 0, 0xFFCC2222),
        NpcJob::Paramedic => mesh::cylinder_tris(tris, 0.0, 1.95, 0.0, 0.19, 0.08, 6, 0xFFFFFFFF),
        NpcJob::ConstructionWorker => mesh::sphere_tris(tris, 0.0, 1.97, 0.0, 0.22, 0, 0xFFDDAA22),
        NpcJob::MailCarrier => mesh::cylinder_tris(tris, 0.0, 1.95, 0.0, 0.19, 0.08, 6, 0xFF3344CC),
        _ => {}
    }

    // Speech bubble
    if npc.interacting_with.is_some() {
        mesh::sphere_tris(tris, 0.0, 2.2, -0.2, 0.15, 0, 0xFFFFFFFF);
    }

    // Legs — cylinders with swing animation
    mesh::cylinder_tris(tris, -0.1, 0.35, -swing * 0.35, 0.07, 0.65, 5, npc.pants_color);
    mesh::cylinder_tris(tris, 0.1, 0.35, swing * 0.35, 0.07, 0.65, 5, npc.pants_color);
    // Shoes — small spheres at feet
    mesh::sphere_tris(tris, -0.1, 0.05, -swing * 0.35, 0.08, 0, 0xFF333333);
    mesh::sphere_tris(tris, 0.1, 0.05, swing * 0.35, 0.08, 0, 0xFF333333);

    if npc.attack_phase > 0.0 {
        let t = (npc.attack_phase / ATTACK_ANIM_DURATION).clamp(0.0, 1.0);
        let extend = 1.0 - (1.0 - t) * (1.0 - t);
        let right_z = -0.15 - extend * 0.8;
        let left_z = 0.15 + extend * 0.2;
        // Arms — cylinders
        mesh::cylinder_tris(tris, -0.35, 1.05, left_z, 0.06, 0.55, 4, SKIN_COLOR);
        mesh::cylinder_tris(tris, 0.35, 1.05, right_z, 0.06, 0.55, 4, SKIN_COLOR);
        // Fists
        mesh::sphere_tris(tris, 0.35, 1.05, right_z - 0.3, 0.07, 0, SKIN_COLOR);
    } else if npc.carrying_item {
        mesh::cylinder_tris(tris, -0.25, 1.0, -0.35, 0.06, 0.5, 4, SKIN_COLOR);
        mesh::cylinder_tris(tris, 0.25, 1.0, -0.35, 0.06, 0.5, 4, SKIN_COLOR);
        mesh::beveled_box_tris(tris, 0.0, 0.9, -0.5, 0.3, 0.3, 0.2, 0.03, BAG_COLOR);
    } else if npc.carrying_bin.is_some() {
        mesh::cylinder_tris(tris, -0.25, 1.0, -0.4, 0.06, 0.5, 4, SKIN_COLOR);
        mesh::cylinder_tris(tris, 0.25, 1.0, -0.4, 0.06, 0.5, 4, SKIN_COLOR);
        mesh::cylinder_tris(tris, 0.0, 0.8, -0.55, 0.2, 0.6, 6, BIN_COLOR);
    } else {
        // Normal arm swing — cylinders
        mesh::cylinder_tris(tris, -0.3, 1.05, swing * 0.25, 0.06, 0.55, 4, SKIN_COLOR);
        mesh::cylinder_tris(tris, 0.3, 1.05, -swing * 0.25, 0.06, 0.55, 4, SKIN_COLOR);
        // Hands — spheres
        mesh::sphere_tris(tris, -0.3, 0.8, swing * 0.25, 0.06, 0, SKIN_COLOR);
        mesh::sphere_tris(tris, 0.3, 0.8, -swing * 0.25, 0.06, 0, SKIN_COLOR);
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
