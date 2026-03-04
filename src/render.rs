// sys_render: transform world + player geometry to screen, rasterize
// General-purpose triangle renderer with backface culling, distance culling, lighting

use crate::math::*;
use crate::raster::*;
use crate::state::*;

const SKIN_COLOR: u32 = 0xFFDEB887;
const SHIRT_COLOR: u32 = 0xFF3355AA;
const PANTS_COLOR: u32 = 0xFF333355;

const VEHICLE_BODY_COLOR_DARKEN: f32 = 0.7;
const WINDSHIELD_COLOR: u32 = 0xFF88AACC;
const TIRE_COLOR: u32 = 0xFF222222;

// Sky/fog/light colors for time of day
struct TimeColors {
    sky: u32,
    fog_r: f32, fog_g: f32, fog_b: f32,
    light_dir: Vec3,
    ambient: f32,   // minimum light level
    sun_strength: f32,
}

fn time_colors(hour: f32) -> TimeColors {
    // Key times: 6=sunrise, 12=noon, 18=sunset, 0=midnight
    let (sky, amb, sun) = if hour < 5.0 {
        // Night
        (lerp_color(0xFF0A0A20, 0xFF0A0A20, 0.0), 0.15, 0.0)
    } else if hour < 6.5 {
        // Dawn
        let t = (hour - 5.0) / 1.5;
        (lerp_color(0xFF0A0A20, 0xFFDD8844, t), 0.15 + t * 0.3, t * 0.4)
    } else if hour < 8.0 {
        // Sunrise to morning
        let t = (hour - 6.5) / 1.5;
        (lerp_color(0xFFDD8844, 0xFF87CEEB, t), 0.45 + t * 0.2, 0.4 + t * 0.25)
    } else if hour < 16.0 {
        // Day
        (0xFF87CEEB, 0.65, 0.65)
    } else if hour < 18.0 {
        // Afternoon to sunset
        let t = (hour - 16.0) / 2.0;
        (lerp_color(0xFF87CEEB, 0xFFDD6633, t), 0.65 - t * 0.2, 0.65 - t * 0.25)
    } else if hour < 19.5 {
        // Sunset to dusk
        let t = (hour - 18.0) / 1.5;
        (lerp_color(0xFFDD6633, 0xFF1A1A40, t), 0.45 - t * 0.3, 0.4 - t * 0.4)
    } else {
        // Night
        (0xFF0A0A20, 0.15, 0.0)
    };

    let fr = ((sky >> 16) & 0xFF) as f32;
    let fg = ((sky >> 8) & 0xFF) as f32;
    let fb = (sky & 0xFF) as f32;

    // Sun direction rotates with time
    let sun_angle = (hour - 6.0) / 12.0 * std::f32::consts::PI;
    let light_dir = if sun > 0.0 {
        let sy = sun_angle.sin().max(0.1);
        let sx = sun_angle.cos() * 0.5;
        let len = (sx * sx + sy * sy + 0.25).sqrt();
        [sx / len, sy / len, 0.5 / len]
    } else {
        [0.0, 1.0, 0.0] // doesn't matter at night
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

pub fn sys_render(fb: &mut Framebuffer, world: &WorldData, player: &Player, cam: &Camera, hour: f32) {
    let tc = time_colors(hour);
    let aspect = fb.w as f32 / fb.h as f32;
    let eye = v3(cam.x, cam.y, cam.z);
    let target = v3(cam.tx, cam.ty, cam.tz);
    let view = m4_look_at(eye, target, v3(0.0, 1.0, 0.0));
    let proj = m4_perspective(60.0_f32.to_radians(), aspect, 0.1, 200.0);
    let vp = m4_mul(&proj, &view);

    render_tris(fb, &vp, &world.static_tris, eye, &tc);

    for v in &world.vehicles {
        let vehicle_tris = gen_vehicle_mesh(v);
        render_tris(fb, &vp, &vehicle_tris, eye, &tc);
    }

    for npc in &world.npcs {
        let npc_tris = gen_npc_mesh(npc);
        render_tris(fb, &vp, &npc_tris, eye, &tc);
    }

    for item in &world.items {
        if !item.active { continue; }
        let item_tris = gen_item_mesh(item);
        render_tris(fb, &vp, &item_tris, eye, &tc);
    }

    if player.in_vehicle.is_none() {
        let player_tris = gen_player_mesh(player);
        render_tris(fb, &vp, &player_tris, eye, &tc);
    }
}

fn render_tris(fb: &mut Framebuffer, vp: &Mat4, tris: &[WorldTri], cam_pos: Vec3, tc: &TimeColors) {
    let w = fb.w as f32;
    let h = fb.h as f32;

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
        if dist_sq > FOG_DIST * FOG_DIST { continue; }

        let view_dir = [dx, dy, dz];
        let vd_len = dist_sq.sqrt();
        if vd_len < 0.001 { continue; }
        let vd = [view_dir[0]/vd_len, view_dir[1]/vd_len, view_dir[2]/vd_len];
        if v3_dot(tri.normal, vd) < -0.1 { continue; }

        let c0 = m4_transform_no_div(vp, tri.v[0]);
        let c1 = m4_transform_no_div(vp, tri.v[1]);
        let c2 = m4_transform_no_div(vp, tri.v[2]);

        if c0[3] <= 0.01 || c1[3] <= 0.01 || c2[3] <= 0.01 { continue; }

        let s0 = clip_to_screen(c0, w, h);
        let s1 = clip_to_screen(c1, w, h);
        let s2 = clip_to_screen(c2, w, h);

        let sun_lit = v3_dot(tri.normal, tc.light_dir).max(0.0) * tc.sun_strength;
        let intensity = sun_lit + tc.ambient;
        let fog = (dist_sq.sqrt() / FOG_DIST).min(1.0);
        let color = shade_and_fog(tri.color, intensity, fog, tc);

        draw_triangle(fb, &ScreenTri { v: [s0, s1, s2], color });
    }
}

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
    let mix = fog * fog; // quadratic fog falloff
    let ro = ((r * i * (1.0 - mix) + tc.fog_r * mix) as u32).min(255);
    let go = ((g * i * (1.0 - mix) + tc.fog_g * mix) as u32).min(255);
    let bo = ((b * i * (1.0 - mix) + tc.fog_b * mix) as u32).min(255);
    0xFF000000 | (ro << 16) | (go << 8) | bo
}

// Player humanoid mesh: torso, head, arms, legs with walking animation
fn gen_player_mesh(player: &Player) -> Vec<WorldTri> {
    let mut tris = Vec::with_capacity(72);
    let phase = player.walk_phase;
    let swing = phase.sin() * 0.4;

    // Torso
    push_box(&mut tris, 0.0, 1.05, 0.0, 0.6, 0.7, 0.35, SHIRT_COLOR);
    // Head
    push_box(&mut tris, 0.0, 1.75, 0.0, 0.35, 0.35, 0.35, SKIN_COLOR);
    // Left leg
    push_box(&mut tris, -0.15, 0.35, -swing * 0.35, 0.22, 0.65, 0.22, PANTS_COLOR);
    // Right leg
    push_box(&mut tris, 0.15, 0.35, swing * 0.35, 0.22, 0.65, 0.22, PANTS_COLOR);
    // Left arm
    push_box(&mut tris, -0.45, 1.05, swing * 0.25, 0.18, 0.6, 0.18, SKIN_COLOR);
    // Right arm
    push_box(&mut tris, 0.45, 1.05, -swing * 0.25, 0.18, 0.6, 0.18, SKIN_COLOR);

    // Rotate by player.rot_y and translate
    let (sin_r, cos_r) = player.rot_y.sin_cos();
    for tri in &mut tris {
        for v in &mut tri.v {
            let rx = v[0] * cos_r - v[2] * sin_r;
            let rz = v[0] * sin_r + v[2] * cos_r;
            v[0] = rx + player.x;
            v[1] += player.y;
            v[2] = rz + player.z;
        }
        let nx = tri.normal[0] * cos_r - tri.normal[2] * sin_r;
        let nz = tri.normal[0] * sin_r + tri.normal[2] * cos_r;
        tri.normal[0] = nx;
        tri.normal[2] = nz;
    }

    tris
}

// Vehicle mesh: sedan-like shape with body, roof, windshield, tires
fn gen_vehicle_mesh(v: &Vehicle) -> Vec<WorldTri> {
    let mut tris = Vec::with_capacity(48);
    let color = v.color;

    // Lower body (wider, shorter)
    push_box(&mut tris, 0.0, 0.45, 0.0, 1.8, 0.6, 3.6, color);
    // Upper cabin (narrower, on top)
    push_box(&mut tris, 0.0, 0.95, -0.2, 1.5, 0.5, 1.8, darken(color, VEHICLE_BODY_COLOR_DARKEN));
    // Windshield (front of cabin)
    push_box(&mut tris, 0.0, 0.95, 0.75, 1.4, 0.4, 0.05, WINDSHIELD_COLOR);
    // Rear window
    push_box(&mut tris, 0.0, 0.95, -1.15, 1.4, 0.4, 0.05, WINDSHIELD_COLOR);
    // Tires (4 small dark boxes)
    push_box(&mut tris, -0.85, 0.2, 1.1, 0.25, 0.4, 0.5, TIRE_COLOR);
    push_box(&mut tris, 0.85, 0.2, 1.1, 0.25, 0.4, 0.5, TIRE_COLOR);
    push_box(&mut tris, -0.85, 0.2, -1.1, 0.25, 0.4, 0.5, TIRE_COLOR);
    push_box(&mut tris, 0.85, 0.2, -1.1, 0.25, 0.4, 0.5, TIRE_COLOR);

    // Rotate and translate to world position
    let (sin_r, cos_r) = v.rot_y.sin_cos();
    for tri in &mut tris {
        for vert in &mut tri.v {
            let rx = vert[0] * cos_r - vert[2] * sin_r;
            let rz = vert[0] * sin_r + vert[2] * cos_r;
            vert[0] = rx + v.x;
            vert[1] += 0.0; // vehicles sit on ground
            vert[2] = rz + v.z;
        }
        let nx = tri.normal[0] * cos_r - tri.normal[2] * sin_r;
        let nz = tri.normal[0] * sin_r + tri.normal[2] * cos_r;
        tri.normal[0] = nx;
        tri.normal[2] = nz;
    }

    tris
}

// NPC humanoid mesh (same shape as player, different colors)
fn gen_npc_mesh(npc: &Npc) -> Vec<WorldTri> {
    let mut tris = Vec::with_capacity(72);
    let swing = npc.walk_phase.sin() * 0.4;

    push_box(&mut tris, 0.0, 1.05, 0.0, 0.6, 0.7, 0.35, npc.shirt_color);
    push_box(&mut tris, 0.0, 1.75, 0.0, 0.35, 0.35, 0.35, SKIN_COLOR);
    push_box(&mut tris, -0.15, 0.35, -swing * 0.35, 0.22, 0.65, 0.22, npc.pants_color);
    push_box(&mut tris, 0.15, 0.35, swing * 0.35, 0.22, 0.65, 0.22, npc.pants_color);
    push_box(&mut tris, -0.45, 1.05, swing * 0.25, 0.18, 0.6, 0.18, SKIN_COLOR);
    push_box(&mut tris, 0.45, 1.05, -swing * 0.25, 0.18, 0.6, 0.18, SKIN_COLOR);

    let (sin_r, cos_r) = npc.rot_y.sin_cos();
    for tri in &mut tris {
        for v in &mut tri.v {
            let rx = v[0] * cos_r - v[2] * sin_r;
            let rz = v[0] * sin_r + v[2] * cos_r;
            v[0] = rx + npc.x;
            v[2] = rz + npc.z;
        }
        let nx = tri.normal[0] * cos_r - tri.normal[2] * sin_r;
        let nz = tri.normal[0] * sin_r + tri.normal[2] * cos_r;
        tri.normal[0] = nx;
        tri.normal[2] = nz;
    }
    tris
}

// Item mesh: spinning colored shape floating above ground
fn gen_item_mesh(item: &Item) -> Vec<WorldTri> {
    let mut tris = Vec::with_capacity(16);
    let color = match item.kind {
        ItemKind::Health => 0xFFFF3333,  // red
        ItemKind::Money => 0xFFFFDD33,   // gold
        ItemKind::Stamina => 0xFF33FF33, // green
    };
    let y = 0.8 + (item.spin_phase * 2.0).sin() * 0.2; // bob up and down

    // Small spinning octahedron
    let r = 0.35;
    let (sin_s, cos_s) = item.spin_phase.sin_cos();
    let top = [item.x, y + r, item.z];
    let bot = [item.x, y - r, item.z];
    let pts = [
        [item.x + r * cos_s, y, item.z + r * sin_s],
        [item.x - r * sin_s, y, item.z + r * cos_s],
        [item.x - r * cos_s, y, item.z - r * sin_s],
        [item.x + r * sin_s, y, item.z - r * cos_s],
    ];
    for i in 0..4 {
        let a = pts[i];
        let b = pts[(i + 1) % 4];
        let n_top = tri_normal(top, a, b);
        tris.push(WorldTri { v: [top, a, b], normal: n_top, color });
        let n_bot = tri_normal(bot, b, a);
        tris.push(WorldTri { v: [bot, b, a], normal: n_bot, color });
    }
    tris
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
