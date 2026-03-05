// World generation: heightmap terrain, buildings, roads, trees, rocks, street lights
// All geometry output as WorldTri in world space, generated once at startup

use crate::state::*;
use crate::rng::Rng;

const BUILDING_COLORS: [u32; 10] = [
    0xFF887766, 0xFF776688, 0xFF668877, 0xFF998877, 0xFF778899,
    0xFF886677, 0xFF667788, 0xFF888888, 0xFF997766, 0xFF779988,
];

const CANOPY_COLORS: [u32; 4] = [0xFF338833, 0xFF228822, 0xFF448844, 0xFF2A7A2A];
const TRUNK_COLOR: u32 = 0xFF554422;
const ROCK_COLOR: u32 = 0xFF777777;
const GROUND_LOW: u32 = 0xFF2A6B2A;  // darker green in valleys
const GROUND_HIGH: u32 = 0xFF44AA44; // lighter green on hills
const ROAD_COLOR: u32 = 0xFF444444;
const ROAD_LINE_COLOR: u32 = 0xFFCCCC33;
const SIDEWALK_COLOR: u32 = 0xFF888888;
const FIELD_ROAD_COLOR: u32 = 0xFF665544;
const LAMP_POLE_COLOR: u32 = 0xFF666666;
const LAMP_GLOW_COLOR: u32 = 0xFFFFEE88;

const ROAD_SEG_STEP: f32 = 2.0; // subdivision step for terrain-following road strips

/// Distance from point (px, pz) to line segment (x0,z0)-(x1,z1)
pub fn point_to_segment_dist(px: f32, pz: f32, x0: f32, z0: f32, x1: f32, z1: f32) -> f32 {
    let dx = x1 - x0;
    let dz = z1 - z0;
    let len_sq = dx * dx + dz * dz;
    if len_sq < 1e-8 {
        let ex = px - x0;
        let ez = pz - z0;
        return (ex * ex + ez * ez).sqrt();
    }
    let t = ((px - x0) * dx + (pz - z0) * dz) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = x0 + t * dx;
    let proj_z = z0 + t * dz;
    let ex = px - proj_x;
    let ez = pz - proj_z;
    (ex * ex + ez * ez).sqrt()
}

/// What surface is at world position (x, z)?
pub fn surface_at(x: f32, z: f32, net: &RoadNetwork) -> Surface {
    let mut best_dist = f32::MAX;
    let mut best_tier = RoadTier::CarRoad;
    for seg in &net.segments {
        let d = point_to_segment_dist(x, z, seg.x0, seg.z0, seg.x1, seg.z1);
        if d < best_dist {
            best_dist = d;
            best_tier = seg.tier;
        }
    }
    match best_tier {
        RoadTier::CarRoad => {
            if best_dist < CAR_ROAD_WIDTH * 0.5 { Surface::CarRoad }
            else if best_dist < CAR_ROAD_WIDTH * 0.5 + SIDEWALK_WIDTH { Surface::Sidewalk }
            else { Surface::Terrain }
        }
        RoadTier::FieldRoad => {
            if best_dist < FIELD_ROAD_WIDTH * 0.5 { Surface::FieldRoad }
            else { Surface::Terrain }
        }
    }
}

/// Distance to nearest road segment (for heightmap flattening)
fn road_dist_network(x: f32, z: f32, net: &RoadNetwork) -> (f32, RoadTier) {
    let mut best_dist = f32::MAX;
    let mut best_tier = RoadTier::CarRoad;
    for seg in &net.segments {
        let d = point_to_segment_dist(x, z, seg.x0, seg.z0, seg.x1, seg.z1);
        if d < best_dist {
            best_dist = d;
            best_tier = seg.tier;
        }
    }
    (best_dist, best_tier)
}

/// Check if position is on or near any road (for object placement avoidance)
pub fn on_any_road(x: f32, z: f32, net: &RoadNetwork) -> bool {
    for seg in &net.segments {
        let d = point_to_segment_dist(x, z, seg.x0, seg.z0, seg.x1, seg.z1);
        let clearance = match seg.tier {
            RoadTier::CarRoad => CAR_ROAD_WIDTH * 0.5 + SIDEWALK_WIDTH + 1.0,
            RoadTier::FieldRoad => FIELD_ROAD_WIDTH * 0.5 + 1.0,
        };
        if d < clearance { return true; }
    }
    false
}

/// Generate an organic radial road network from seed
fn generate_road_network(rng: &mut Rng) -> RoadNetwork {
    let mut nodes: Vec<[f32; 2]> = Vec::new();
    let mut segments: Vec<RoadSegment> = Vec::new();

    // Center hub near origin with slight jitter
    let center = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
    nodes.push(center); // index 0

    // Ring 1: 4 nodes at radius ~30
    let ring1_count = 4;
    let ring1_start = nodes.len();
    let base_angle = rng.range(0.0, std::f32::consts::TAU);
    for i in 0..ring1_count {
        let angle = base_angle + (i as f32 / ring1_count as f32) * std::f32::consts::TAU
            + rng.range(-0.3, 0.3);
        let radius = rng.range(25.0, 35.0);
        nodes.push([angle.cos() * radius, angle.sin() * radius]);
    }

    // Ring 2: 6 nodes at radius ~60
    let ring2_count = 6;
    let ring2_start = nodes.len();
    let base_angle2 = rng.range(0.0, std::f32::consts::TAU);
    for i in 0..ring2_count {
        let angle = base_angle2 + (i as f32 / ring2_count as f32) * std::f32::consts::TAU
            + rng.range(-0.25, 0.25);
        let radius = rng.range(52.0, 68.0);
        nodes.push([angle.cos() * radius, angle.sin() * radius]);
    }

    // Edge nodes: 4 nodes at radius ~85
    let edge_count = 4;
    let edge_start = nodes.len();
    let base_angle3 = rng.range(0.0, std::f32::consts::TAU);
    for i in 0..edge_count {
        let angle = base_angle3 + (i as f32 / edge_count as f32) * std::f32::consts::TAU
            + rng.range(-0.3, 0.3);
        let radius = rng.range(78.0, 90.0);
        nodes.push([angle.cos() * radius, angle.sin() * radius]);
    }

    // --- CarRoad connections ---

    // Center to all ring-1 (spokes)
    for i in 0..ring1_count {
        let ni = ring1_start + i;
        segments.push(RoadSegment {
            x0: nodes[0][0], z0: nodes[0][1],
            x1: nodes[ni][0], z1: nodes[ni][1],
            tier: RoadTier::CarRoad,
        });
    }

    // Ring-1 partial ring (connect consecutive, skip one for asymmetry)
    let skip = rng.next() as usize % ring1_count;
    for i in 0..ring1_count {
        if i == skip { continue; }
        let a = ring1_start + i;
        let b = ring1_start + (i + 1) % ring1_count;
        segments.push(RoadSegment {
            x0: nodes[a][0], z0: nodes[a][1],
            x1: nodes[b][0], z1: nodes[b][1],
            tier: RoadTier::CarRoad,
        });
    }

    // Ring-1 to nearest ring-2 nodes
    for i in 0..ring1_count {
        let ni = ring1_start + i;
        // Find 1-2 nearest ring-2 nodes
        let mut dists: Vec<(f32, usize)> = (0..ring2_count).map(|j| {
            let nj = ring2_start + j;
            let dx = nodes[ni][0] - nodes[nj][0];
            let dz = nodes[ni][1] - nodes[nj][1];
            (dx * dx + dz * dz, nj)
        }).collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // Connect to 1 or 2 nearest
        let connect_count = if rng.next() % 3 == 0 { 2 } else { 1 };
        for k in 0..connect_count.min(dists.len()) {
            let nj = dists[k].1;
            // Avoid duplicate segments
            let already = segments.iter().any(|s|
                (s.x0 == nodes[ni][0] && s.z0 == nodes[ni][1] && s.x1 == nodes[nj][0] && s.z1 == nodes[nj][1])
                || (s.x1 == nodes[ni][0] && s.z1 == nodes[ni][1] && s.x0 == nodes[nj][0] && s.z0 == nodes[nj][1])
            );
            if !already {
                segments.push(RoadSegment {
                    x0: nodes[ni][0], z0: nodes[ni][1],
                    x1: nodes[nj][0], z1: nodes[nj][1],
                    tier: RoadTier::CarRoad,
                });
            }
        }
    }

    // Some ring-2 to ring-2 connections (partial ring)
    let r2_connections = 3 + rng.next() as usize % 3;
    for _ in 0..r2_connections {
        let a = ring2_start + rng.next() as usize % ring2_count;
        let b = ring2_start + rng.next() as usize % ring2_count;
        if a == b { continue; }
        let already = segments.iter().any(|s|
            (s.x0 == nodes[a][0] && s.z0 == nodes[a][1] && s.x1 == nodes[b][0] && s.z1 == nodes[b][1])
            || (s.x1 == nodes[a][0] && s.z1 == nodes[a][1] && s.x0 == nodes[b][0] && s.z0 == nodes[b][1])
        );
        if !already {
            segments.push(RoadSegment {
                x0: nodes[a][0], z0: nodes[a][1],
                x1: nodes[b][0], z1: nodes[b][1],
                tier: RoadTier::CarRoad,
            });
        }
    }

    // Some ring-2 to edge connections
    for i in 0..edge_count {
        let ei = edge_start + i;
        // Find nearest ring-2 node
        let mut best_dist = f32::MAX;
        let mut best_j = ring2_start;
        for j in 0..ring2_count {
            let nj = ring2_start + j;
            let dx = nodes[ei][0] - nodes[nj][0];
            let dz = nodes[ei][1] - nodes[nj][1];
            let d = dx * dx + dz * dz;
            if d < best_dist { best_dist = d; best_j = nj; }
        }
        segments.push(RoadSegment {
            x0: nodes[best_j][0], z0: nodes[best_j][1],
            x1: nodes[ei][0], z1: nodes[ei][1],
            tier: RoadTier::CarRoad,
        });
    }

    // --- FieldRoad connections ---
    let field_count = 5 + rng.next() as usize % 4;
    for _ in 0..field_count {
        // Pick two random non-adjacent nodes
        let a = rng.next() as usize % nodes.len();
        let b = rng.next() as usize % nodes.len();
        if a == b { continue; }
        let dx = nodes[a][0] - nodes[b][0];
        let dz = nodes[a][1] - nodes[b][1];
        let dist = (dx * dx + dz * dz).sqrt();
        if dist > 60.0 || dist < 10.0 { continue; } // reasonable length
        let already = segments.iter().any(|s|
            (s.x0 == nodes[a][0] && s.z0 == nodes[a][1] && s.x1 == nodes[b][0] && s.z1 == nodes[b][1])
            || (s.x1 == nodes[a][0] && s.z1 == nodes[a][1] && s.x0 == nodes[b][0] && s.z0 == nodes[b][1])
        );
        if !already {
            segments.push(RoadSegment {
                x0: nodes[a][0], z0: nodes[a][1],
                x1: nodes[b][0], z1: nodes[b][1],
                tier: RoadTier::FieldRoad,
            });
        }
    }

    RoadNetwork { segments, nodes }
}

// Generate axis-aligned box triangles centered at (cx, cy, cz) with full extents (w, h, d)
fn box_tris(tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32, w: f32, h: f32, d: f32, color: u32) {
    let (hw, hh, hd) = (w * 0.5, h * 0.5, d * 0.5);
    let c = [
        [cx-hw, cy-hh, cz+hd], [cx+hw, cy-hh, cz+hd], [cx+hw, cy+hh, cz+hd], [cx-hw, cy+hh, cz+hd],
        [cx-hw, cy-hh, cz-hd], [cx+hw, cy-hh, cz-hd], [cx+hw, cy+hh, cz-hd], [cx-hw, cy+hh, cz-hd],
    ];
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([0,1,2,3], [0.0, 0.0, 1.0]), ([5,4,7,6], [0.0, 0.0,-1.0]),
        ([4,0,3,7], [-1.0,0.0,0.0]),  ([1,5,6,2], [1.0, 0.0, 0.0]),
        ([3,2,6,7], [0.0, 1.0, 0.0]), ([4,5,1,0], [0.0,-1.0, 0.0]),
    ];
    for (idx, normal) in faces {
        tris.push(WorldTri { v: [c[idx[0]], c[idx[1]], c[idx[2]]], normal, color });
        tris.push(WorldTri { v: [c[idx[0]], c[idx[2]], c[idx[3]]], normal, color });
    }
}

// 8-sided approximation of a sphere
fn octahedron_tris(tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32, r: f32, color: u32) {
    let top = [cx, cy + r, cz];
    let bot = [cx, cy - r, cz];
    let pts = [
        [cx + r, cy, cz], [cx, cy, cz + r],
        [cx - r, cy, cz], [cx, cy, cz - r],
    ];
    for i in 0..4 {
        let a = pts[i];
        let b = pts[(i + 1) % 4];
        let n_top = normalize_tri_normal(top, a, b);
        tris.push(WorldTri { v: [top, a, b], normal: n_top, color });
        let n_bot = normalize_tri_normal(bot, b, a);
        tris.push(WorldTri { v: [bot, b, a], normal: n_bot, color });
    }
}

fn normalize_tri_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let e1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let e2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    let n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
    let l = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
    if l < 1e-10 { [0.0, 1.0, 0.0] } else { [n[0]/l, n[1]/l, n[2]/l] }
}

fn lerp_color(a: u32, b: u32, t: f32) -> u32 {
    let t = t.clamp(0.0, 1.0);
    let r = (((a >> 16) & 0xFF) as f32 * (1.0 - t) + ((b >> 16) & 0xFF) as f32 * t) as u32;
    let g = (((a >> 8) & 0xFF) as f32 * (1.0 - t) + ((b >> 8) & 0xFF) as f32 * t) as u32;
    let bl = ((a & 0xFF) as f32 * (1.0 - t) + (b & 0xFF) as f32 * t) as u32;
    0xFF000000 | (r << 16) | (g << 8) | bl
}

/// Generate heightmap from multi-octave sinusoidal waves, flattened near roads/downtown
fn generate_heightmap(terrain: &mut Terrain, seed: u64, net: &RoadNetwork) {
    let grid = terrain.grid;
    let stride = grid + 1;
    let cell = terrain.cell_size;

    // Seed-based phase offsets
    let phase_x = (seed as f32) * 0.1234;
    let phase_z = (seed as f32) * 0.5678;

    for iz in 0..stride {
        for ix in 0..stride {
            let x = -WORLD_HALF + ix as f32 * cell;
            let z = -WORLD_HALF + iz as f32 * cell;

            // Multi-octave sinusoidal terrain
            let mut h = 0.0f32;
            h += ((x * 0.03 + phase_x).sin() * (z * 0.025 + phase_z).sin()) * 4.0;
            h += ((x * 0.07 + phase_z).sin() * (z * 0.06 + phase_x).sin()) * 1.5;
            h += ((x * 0.15 + phase_x * 2.0).sin() * (z * 0.13 + phase_z * 2.0).sin()) * 0.5;

            // Flatten near roads (smooth falloff based on nearest segment)
            let (rd, tier) = road_dist_network(x, z, net);
            let corridor = match tier {
                RoadTier::CarRoad => CAR_ROAD_WIDTH * 0.5 + SIDEWALK_WIDTH,
                RoadTier::FieldRoad => FIELD_ROAD_WIDTH * 0.5 + 1.0,
            };
            let road_flatten = if rd < corridor {
                0.0
            } else if rd < corridor + 4.0 {
                let t = (rd - corridor) / 4.0;
                t * t
            } else {
                1.0
            };

            // Flatten downtown area (near origin)
            let downtown_dist = (x * x + z * z).sqrt();
            let downtown_flatten = if downtown_dist < 15.0 {
                0.2
            } else if downtown_dist < 30.0 {
                0.2 + 0.8 * ((downtown_dist - 15.0) / 15.0)
            } else {
                1.0
            };

            terrain.heights[iz * stride + ix] = h * road_flatten * downtown_flatten;
        }
    }
}

/// Generate terrain mesh triangles from heightmap
fn generate_terrain_mesh(tris: &mut Vec<WorldTri>, terrain: &Terrain) {
    let grid = terrain.grid;
    let stride = grid + 1;
    let cell = terrain.cell_size;

    // Find height range for color interpolation
    let mut h_min = f32::MAX;
    let mut h_max = f32::MIN;
    for &h in &terrain.heights {
        if h < h_min { h_min = h; }
        if h > h_max { h_max = h; }
    }
    let h_range = (h_max - h_min).max(0.1);

    for iz in 0..grid {
        for ix in 0..grid {
            let x0 = -WORLD_HALF + ix as f32 * cell;
            let z0 = -WORLD_HALF + iz as f32 * cell;
            let x1 = x0 + cell;
            let z1 = z0 + cell;

            let h00 = terrain.heights[iz * stride + ix];
            let h10 = terrain.heights[iz * stride + ix + 1];
            let h01 = terrain.heights[(iz + 1) * stride + ix];
            let h11 = terrain.heights[(iz + 1) * stride + ix + 1];

            let v00 = [x0, h00, z0];
            let v10 = [x1, h10, z0];
            let v01 = [x0, h01, z1];
            let v11 = [x1, h11, z1];

            let avg_h = (h00 + h10 + h01 + h11) * 0.25;
            let t = ((avg_h - h_min) / h_range).clamp(0.0, 1.0);
            let color = lerp_color(GROUND_LOW, GROUND_HIGH, t);

            let n1 = normalize_tri_normal(v00, v10, v11);
            tris.push(WorldTri { v: [v00, v10, v11], normal: n1, color });
            let n2 = normalize_tri_normal(v00, v11, v01);
            tris.push(WorldTri { v: [v00, v11, v01], normal: n2, color });
        }
    }
}

/// Generate a road strip along an arbitrary-direction segment, following terrain
fn generate_road_strip(
    tris: &mut Vec<WorldTri>, terrain: &Terrain,
    x0: f32, z0: f32, x1: f32, z1: f32,
    half_width: f32, y_offset: f32, color: u32,
) {
    let dx = x1 - x0;
    let dz = z1 - z0;
    let length = (dx * dx + dz * dz).sqrt();
    if length < 0.01 { return; }

    // Direction along the road
    let dir_x = dx / length;
    let dir_z = dz / length;
    // Perpendicular direction (left side)
    let perp_x = -dir_z;
    let perp_z = dir_x;

    // Subdivide along length
    let num_steps = (length / ROAD_SEG_STEP).ceil() as i32;
    let step = length / num_steps as f32;

    for s in 0..num_steps {
        let t0 = s as f32 * step;
        let t1 = (s + 1) as f32 * step;

        let sx0 = x0 + dir_x * t0;
        let sz0 = z0 + dir_z * t0;
        let sx1 = x0 + dir_x * t1;
        let sz1 = z0 + dir_z * t1;

        // 4 corners of this quad
        let lx0 = sx0 + perp_x * half_width;
        let lz0 = sz0 + perp_z * half_width;
        let rx0 = sx0 - perp_x * half_width;
        let rz0 = sz0 - perp_z * half_width;
        let lx1 = sx1 + perp_x * half_width;
        let lz1 = sz1 + perp_z * half_width;
        let rx1 = sx1 - perp_x * half_width;
        let rz1 = sz1 - perp_z * half_width;

        let hl0 = terrain.height_at(lx0, lz0) + y_offset;
        let hr0 = terrain.height_at(rx0, rz0) + y_offset;
        let hl1 = terrain.height_at(lx1, lz1) + y_offset;
        let hr1 = terrain.height_at(rx1, rz1) + y_offset;

        let v_l0 = [lx0, hl0, lz0];
        let v_r0 = [rx0, hr0, rz0];
        let v_l1 = [lx1, hl1, lz1];
        let v_r1 = [rx1, hr1, rz1];

        tris.push(WorldTri { v: [v_l0, v_r0, v_r1], normal: [0.0, 1.0, 0.0], color });
        tris.push(WorldTri { v: [v_l0, v_r1, v_l1], normal: [0.0, 1.0, 0.0], color });
    }
}

pub fn generate_world(game: &mut GameState) {
    let mut rng = Rng::new(game.world_seed);
    let mut tris = Vec::with_capacity(30000);

    // Generate organic road network
    let net = generate_road_network(&mut rng);
    game.road_network = net;

    // Generate heightmap (needs road network for flattening)
    generate_heightmap(&mut game.terrain, game.world_seed, &game.road_network);

    // Terrain mesh
    generate_terrain_mesh(&mut tris, &game.terrain);

    // Road geometry
    for seg in &game.road_network.segments {
        match seg.tier {
            RoadTier::CarRoad => {
                let hw = CAR_ROAD_WIDTH * 0.5;
                // Road surface
                generate_road_strip(&mut tris, &game.terrain,
                    seg.x0, seg.z0, seg.x1, seg.z1, hw, 0.05, ROAD_COLOR);
                // Center line
                generate_road_strip(&mut tris, &game.terrain,
                    seg.x0, seg.z0, seg.x1, seg.z1, 0.15, 0.08, ROAD_LINE_COLOR);

                // Direction for sidewalk offset
                let dx = seg.x1 - seg.x0;
                let dz = seg.z1 - seg.z0;
                let len = (dx * dx + dz * dz).sqrt();
                if len > 0.01 {
                    let perp_x = -dz / len;
                    let perp_z = dx / len;
                    let sw_hw = SIDEWALK_WIDTH * 0.5;
                    let sw_offset = hw + sw_hw;

                    // Left sidewalk
                    let lx0 = seg.x0 + perp_x * sw_offset;
                    let lz0 = seg.z0 + perp_z * sw_offset;
                    let lx1 = seg.x1 + perp_x * sw_offset;
                    let lz1 = seg.z1 + perp_z * sw_offset;
                    generate_road_strip(&mut tris, &game.terrain,
                        lx0, lz0, lx1, lz1, sw_hw, 0.06, SIDEWALK_COLOR);

                    // Right sidewalk
                    let rx0 = seg.x0 - perp_x * sw_offset;
                    let rz0 = seg.z0 - perp_z * sw_offset;
                    let rx1 = seg.x1 - perp_x * sw_offset;
                    let rz1 = seg.z1 - perp_z * sw_offset;
                    generate_road_strip(&mut tris, &game.terrain,
                        rx0, rz0, rx1, rz1, sw_hw, 0.06, SIDEWALK_COLOR);
                }
            }
            RoadTier::FieldRoad => {
                generate_road_strip(&mut tris, &game.terrain,
                    seg.x0, seg.z0, seg.x1, seg.z1,
                    FIELD_ROAD_WIDTH * 0.5, 0.04, FIELD_ROAD_COLOR);
            }
        }
    }

    // Buildings
    for _ in 0..NUM_BUILDINGS {
        let mut x;
        let mut z;
        loop {
            x = rng.range(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            z = rng.range(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            if !on_any_road(x, z, &game.road_network) { break; }
        }
        let w = rng.range(3.0, 8.0);
        let d = rng.range(3.0, 8.0);
        let h = rng.range(3.0, 20.0);
        let ground_y = game.terrain.height_at(x, z);
        let color = rng.pick(&BUILDING_COLORS);
        box_tris(&mut tris, x, ground_y + h * 0.5, z, w, h, d, color);
        // Window details
        let win_color = 0xFF222244;
        let win_h = 1.2;
        let win_w = 0.8;
        let floors = ((h - 1.0) / 3.0) as i32;
        let cols = ((w - 1.0) / 2.0) as i32;
        for floor in 0..floors {
            let wy = ground_y + 2.0 + floor as f32 * 3.0;
            for col in 0..cols {
                let wx = x - w * 0.5 + 1.2 + col as f32 * 2.0;
                let fz = z + d * 0.5 + 0.01;
                tris.push(WorldTri { v: [[wx, wy, fz], [wx+win_w, wy, fz], [wx+win_w, wy+win_h, fz]], normal: [0.0,0.0,1.0], color: win_color });
                tris.push(WorldTri { v: [[wx, wy, fz], [wx+win_w, wy+win_h, fz], [wx, wy+win_h, fz]], normal: [0.0,0.0,1.0], color: win_color });
                let bz = z - d * 0.5 - 0.01;
                tris.push(WorldTri { v: [[wx+win_w, wy, bz], [wx, wy, bz], [wx, wy+win_h, bz]], normal: [0.0,0.0,-1.0], color: win_color });
                tris.push(WorldTri { v: [[wx+win_w, wy, bz], [wx, wy+win_h, bz], [wx+win_w, wy+win_h, bz]], normal: [0.0,0.0,-1.0], color: win_color });
            }
        }
        game.world.buildings.push(Building { x, z, w, d, h, ground_y });
    }

    // Trees
    for _ in 0..NUM_TREES {
        let mut x;
        let mut z;
        loop {
            x = rng.range(-WORLD_HALF + 2.0, WORLD_HALF - 2.0);
            z = rng.range(-WORLD_HALF + 2.0, WORLD_HALF - 2.0);
            if !on_any_road(x, z, &game.road_network) { break; }
        }
        let ground_y = game.terrain.height_at(x, z);
        let trunk_h = rng.range(1.5, 3.5);
        let canopy_r = rng.range(1.0, 2.5);
        box_tris(&mut tris, x, ground_y + trunk_h * 0.5, z, 0.4, trunk_h, 0.4, TRUNK_COLOR);
        let canopy_color = rng.pick(&CANOPY_COLORS);
        octahedron_tris(&mut tris, x, ground_y + trunk_h + canopy_r * 0.6, z, canopy_r, canopy_color);
        game.world.trees.push(Tree { x, z, trunk_radius: 0.4 });
    }

    // Rocks
    for _ in 0..NUM_ROCKS {
        let x = rng.range(-WORLD_HALF + 3.0, WORLD_HALF - 3.0);
        let z = rng.range(-WORLD_HALF + 3.0, WORLD_HALF - 3.0);
        let ground_y = game.terrain.height_at(x, z);
        let size = rng.range(0.5, 1.5);
        octahedron_tris(&mut tris, x, ground_y + size * 0.4, z, size, ROCK_COLOR);
        game.world.rocks.push(Rock { x, z, size });
    }

    // Street lights alongside CarRoad segments
    let car_segments: Vec<RoadSegment> = game.road_network.segments.iter()
        .filter(|s| s.tier == RoadTier::CarRoad).cloned().collect();
    for _ in 0..NUM_STREET_LIGHTS {
        let seg_idx = rng.next() as usize % car_segments.len().max(1);
        if car_segments.is_empty() { break; }
        let seg = &car_segments[seg_idx];
        let t = rng.range(0.1, 0.9);
        let sx = seg.x0 + (seg.x1 - seg.x0) * t;
        let sz = seg.z0 + (seg.z1 - seg.z0) * t;
        // Perpendicular offset to sidewalk edge
        let dx = seg.x1 - seg.x0;
        let dz = seg.z1 - seg.z0;
        let len = (dx * dx + dz * dz).sqrt();
        if len < 0.01 { continue; }
        let perp_x = -dz / len;
        let perp_z = dx / len;
        let offset = CAR_ROAD_WIDTH * 0.5 + SIDEWALK_WIDTH + 0.5;
        let side = if rng.next() % 2 == 0 { 1.0 } else { -1.0 };
        let x = sx + perp_x * offset * side;
        let z = sz + perp_z * offset * side;
        let ground_y = game.terrain.height_at(x, z);
        box_tris(&mut tris, x, ground_y + 2.5, z, 0.15, 5.0, 0.15, LAMP_POLE_COLOR);
        octahedron_tris(&mut tris, x, ground_y + 5.2, z, 0.3, LAMP_GLOW_COLOR);
        game.world.street_lights.push(StreetLight { x, z });
    }

    // Trash bins at road network nodes (intersections)
    let mut bin_count = 0;
    for node in &game.road_network.nodes {
        if bin_count >= NUM_TRASH_BINS { break; }
        let bx = node[0] + 4.0;
        let bz = node[1] + 4.0;
        let by = game.terrain.height_at(bx, bz);
        game.world.trash_bins.push(TrashBin {
            x: bx, y: by, z: bz, items_held: 0, carried_by: None,
        });
        bin_count += 1;
    }

    // Ambient vehicles (spawned on CarRoad segments)
    for i in 0..NUM_VEHICLES {
        if car_segments.is_empty() { break; }
        let seg_idx = rng.next() as usize % car_segments.len();
        let seg = &car_segments[seg_idx];
        let t = rng.range(0.1, 0.9);
        let x = seg.x0 + (seg.x1 - seg.x0) * t;
        let z = seg.z0 + (seg.z1 - seg.z0) * t;
        let y = game.terrain.height_at(x, z);
        // Rotation aligned with road direction
        let dx = seg.x1 - seg.x0;
        let dz = seg.z1 - seg.z0;
        let rot = (-dx).atan2(-dz);
        let color = rng.pick(&VEHICLE_COLORS);
        let ai_active = rng.next() % 3 == 0;
        let vehicle_rng = rng.fork(i as u64);
        let mut v = Vehicle {
            x, y, z, rot_y: rot, speed: 0.0, color, occupied: false,
            ai_active, ai_target_x: x, ai_target_z: z, rng: vehicle_rng,
            owner_npc: None,
        };
        if ai_active {
            // Pick a random node as target
            let target_node = &game.road_network.nodes[rng.next() as usize % game.road_network.nodes.len()];
            v.ai_target_x = target_node[0];
            v.ai_target_z = target_node[1];
        }
        game.world.vehicles.push(v);
    }

    // NPC-owned vehicles (one per NPC, parked near their home building)
    let ambient_vehicle_count = game.world.vehicles.len();
    for i in 0..NUM_NPCS {
        let home_idx = i % game.world.buildings.len();
        let b = &game.world.buildings[home_idx];
        let park_x = b.x + b.w * 0.5 + 2.0;
        let park_z = b.z;
        let park_y = game.terrain.height_at(park_x, park_z);
        let color = rng.pick(&VEHICLE_COLORS);
        let vehicle_rng = rng.fork(1000 + i as u64);
        game.world.vehicles.push(Vehicle {
            x: park_x, y: park_y, z: park_z,
            rot_y: 0.0, speed: 0.0, color, occupied: false,
            ai_active: false, ai_target_x: park_x, ai_target_z: park_z,
            rng: vehicle_rng, owner_npc: Some(i),
        });
    }

    // NPCs
    for i in 0..NUM_NPCS {
        let home_idx = i % game.world.buildings.len();
        let car_idx = ambient_vehicle_count + i;
        let b = &game.world.buildings[home_idx];

        let side = if rng.next() % 2 == 0 { 1.0 } else { -1.0 };
        let x = b.x + side * (b.w * 0.5 + 1.5 + rng.range(0.0, 2.0));
        let z = b.z + rng.range(-b.d * 0.3, b.d * 0.3);
        let y = game.terrain.height_at(x, z);
        let shirt_color = rng.pick(&NPC_SHIRT_COLORS);
        let pants_color = rng.pick(&NPC_PANTS_COLORS);
        let rot_y = rng.range(0.0, std::f32::consts::TAU);
        let npc_rng = rng.fork(NUM_VEHICLES as u64 + i as u64);
        let wake_hour = 5.0 + (npc_rng.clone().next() as f32 % 400.0) / 100.0;

        game.world.npcs.push(Npc {
            x, y, z, rot_y, walk_phase: rng.range(0.0, 6.0),
            target_x: x, target_z: z,
            shirt_color, pants_color, rng: npc_rng,
            vel_y: 0.0, on_ground: true,
            state: NpcState::Working,
            home_idx, car_idx,
            wake_hour,
            state_timer: 0.0,
            money: 0.0,
            carrying_item: false,
            carrying_bin: None,
            target_item: None,
            target_bin: None,
            items_deposited_today: 0,
            in_vehicle: false,
            parked_x: x, parked_z: z,
            stuck_timer: 0.0,
            detour_x: 0.0, detour_z: 0.0,
            detouring: false,
        });
    }

    // Items
    let item_kinds = [ItemKind::Health, ItemKind::Money, ItemKind::Stamina];
    for _ in 0..NUM_ITEMS {
        let mut x;
        let mut z;
        loop {
            x = rng.range(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            z = rng.range(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            if !on_any_road(x, z, &game.road_network) { break; }
        }
        let y = game.terrain.height_at(x, z);
        let kind = item_kinds[rng.next() as usize % 3];
        game.world.items.push(Item {
            x, y, z, kind, active: true, respawn_timer: 0.0,
            spin_phase: rng.range(0.0, 6.0),
            falling: false, vel_y: 0.0, claimed_by: None,
        });
    }

    // Set player spawn height
    game.player.y = game.terrain.height_at(game.player.x, game.player.z);

    eprintln!("World: {} tris, {} road segments ({} nodes), {} vehicles ({} NPC-owned), {} npcs, {} items, {} bins",
        tris.len(), game.road_network.segments.len(), game.road_network.nodes.len(),
        game.world.vehicles.len(), NUM_NPCS, game.world.npcs.len(),
        game.world.items.len(), game.world.trash_bins.len());
    game.world.static_tris = tris;
}

/// Lightweight collision for NPC walking — static obstacles only (buildings, rocks, trees, lights)
/// Skips vehicles, other NPCs, and the NPC's home building to avoid gridlock
pub fn check_npc_walk_collision(world: &WorldData, x: f32, z: f32, radius: f32, home_idx: usize) -> bool {
    for (bi, b) in world.buildings.iter().enumerate() {
        if bi == home_idx { continue; }
        if x + radius > b.x - b.w * 0.5 && x - radius < b.x + b.w * 0.5
        && z + radius > b.z - b.d * 0.5 && z - radius < b.z + b.d * 0.5 {
            return true;
        }
    }
    for r in &world.rocks {
        let dx = x - r.x;
        let dz = z - r.z;
        if dx * dx + dz * dz < (r.size + radius) * (r.size + radius) {
            return true;
        }
    }
    for t in &world.trees {
        let dx = x - t.x;
        let dz = z - t.z;
        let r2 = t.trunk_radius + radius;
        if dx * dx + dz * dz < r2 * r2 {
            return true;
        }
    }
    for sl in &world.street_lights {
        let dx = x - sl.x;
        let dz = z - sl.z;
        let r2 = 0.15 + radius;
        if dx * dx + dz * dz < r2 * r2 {
            return true;
        }
    }
    false
}

// Collision for vehicles and player — checks buildings, rocks, trees
pub fn check_building_collision(world: &WorldData, x: f32, z: f32, radius: f32) -> bool {
    for b in &world.buildings {
        if x + radius > b.x - b.w * 0.5 && x - radius < b.x + b.w * 0.5
        && z + radius > b.z - b.d * 0.5 && z - radius < b.z + b.d * 0.5 {
            return true;
        }
    }
    for r in &world.rocks {
        let dx = x - r.x;
        let dz = z - r.z;
        if dx * dx + dz * dz < (r.size + radius) * (r.size + radius) {
            return true;
        }
    }
    for t in &world.trees {
        let dx = x - t.x;
        let dz = z - t.z;
        let r2 = t.trunk_radius + radius;
        if dx * dx + dz * dz < r2 * r2 {
            return true;
        }
    }
    false
}
