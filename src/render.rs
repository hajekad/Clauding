// sys_render: transform world + player geometry to screen, rasterize
// Near-plane clipping, backface/distance culling, day/night lighting

use crate::math::*;
use crate::raster::*;
use crate::state::*;

const SKIN_COLOR: u32 = 0xFFDEB887;
const SHIRT_COLOR: u32 = 0xFF3355AA;
const PANTS_COLOR: u32 = 0xFF333355;

const VEHICLE_BODY_COLOR_DARKEN: f32 = 0.7;
const WINDSHIELD_COLOR: u32 = 0xFF88AACC;
const TIRE_COLOR: u32 = 0xFF222222;

const NEAR_W: f32 = 0.1;

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
    for v in &world.vehicles {
        gen_vehicle_mesh(v, scratch);
    }
    for npc in &world.npcs {
        gen_npc_mesh(npc, scratch);
    }
    for item in &world.items {
        if !item.active { continue; }
        gen_item_mesh(item, scratch);
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
        if dist < 0.001 { continue; }
        let inv_dist = 1.0 / dist;
        let vd = [dx * inv_dist, dy * inv_dist, dz * inv_dist];
        if v3_dot(tri.normal, vd) < -0.5 { continue; }

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

fn gen_player_mesh(player: &Player, tris: &mut Vec<WorldTri>) {
    let base = tris.len();
    let phase = player.walk_phase;
    let swing = phase.sin() * 0.4;

    push_box(tris, 0.0, 1.05, 0.0, 0.6, 0.7, 0.35, SHIRT_COLOR);
    push_box(tris, 0.0, 1.75, 0.0, 0.35, 0.35, 0.35, SKIN_COLOR);
    push_box(tris, -0.15, 0.35, -swing * 0.35, 0.22, 0.65, 0.22, PANTS_COLOR);
    push_box(tris, 0.15, 0.35, swing * 0.35, 0.22, 0.65, 0.22, PANTS_COLOR);
    push_box(tris, -0.45, 1.05, swing * 0.25, 0.18, 0.6, 0.18, SKIN_COLOR);
    push_box(tris, 0.45, 1.05, -swing * 0.25, 0.18, 0.6, 0.18, SKIN_COLOR);

    let (sin_r, cos_r) = player.rot_y.sin_cos();
    for tri in &mut tris[base..] {
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
}

fn gen_vehicle_mesh(v: &Vehicle, tris: &mut Vec<WorldTri>) {
    let base = tris.len();
    let color = v.color;

    push_box(tris, 0.0, 0.45, 0.0, 1.8, 0.6, 3.6, color);
    push_box(tris, 0.0, 0.95, -0.2, 1.5, 0.5, 1.8, darken(color, VEHICLE_BODY_COLOR_DARKEN));
    push_box(tris, 0.0, 0.95, 0.75, 1.4, 0.4, 0.05, WINDSHIELD_COLOR);
    push_box(tris, 0.0, 0.95, -1.15, 1.4, 0.4, 0.05, WINDSHIELD_COLOR);
    push_box(tris, -0.85, 0.2, 1.1, 0.25, 0.4, 0.5, TIRE_COLOR);
    push_box(tris, 0.85, 0.2, 1.1, 0.25, 0.4, 0.5, TIRE_COLOR);
    push_box(tris, -0.85, 0.2, -1.1, 0.25, 0.4, 0.5, TIRE_COLOR);
    push_box(tris, 0.85, 0.2, -1.1, 0.25, 0.4, 0.5, TIRE_COLOR);

    let (sin_r, cos_r) = v.rot_y.sin_cos();
    for tri in &mut tris[base..] {
        for vert in &mut tri.v {
            let rx = vert[0] * cos_r - vert[2] * sin_r;
            let rz = vert[0] * sin_r + vert[2] * cos_r;
            vert[0] = rx + v.x;
            vert[2] = rz + v.z;
        }
        let nx = tri.normal[0] * cos_r - tri.normal[2] * sin_r;
        let nz = tri.normal[0] * sin_r + tri.normal[2] * cos_r;
        tri.normal[0] = nx;
        tri.normal[2] = nz;
    }
}

fn gen_npc_mesh(npc: &Npc, tris: &mut Vec<WorldTri>) {
    let base = tris.len();
    let swing = npc.walk_phase.sin() * 0.4;

    push_box(tris, 0.0, 1.05, 0.0, 0.6, 0.7, 0.35, npc.shirt_color);
    push_box(tris, 0.0, 1.75, 0.0, 0.35, 0.35, 0.35, SKIN_COLOR);
    push_box(tris, -0.15, 0.35, -swing * 0.35, 0.22, 0.65, 0.22, npc.pants_color);
    push_box(tris, 0.15, 0.35, swing * 0.35, 0.22, 0.65, 0.22, npc.pants_color);
    push_box(tris, -0.45, 1.05, swing * 0.25, 0.18, 0.6, 0.18, SKIN_COLOR);
    push_box(tris, 0.45, 1.05, -swing * 0.25, 0.18, 0.6, 0.18, SKIN_COLOR);

    let (sin_r, cos_r) = npc.rot_y.sin_cos();
    for tri in &mut tris[base..] {
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
}

fn gen_item_mesh(item: &Item, tris: &mut Vec<WorldTri>) {
    let color = match item.kind {
        ItemKind::Health => 0xFFFF3333,
        ItemKind::Money => 0xFFFFDD33,
        ItemKind::Stamina => 0xFF33FF33,
    };
    let y = 0.8 + (item.spin_phase * 2.0).sin() * 0.2;

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
