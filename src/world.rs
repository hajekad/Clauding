// World generation: heightmap terrain, buildings, roads, trees, rocks, street lights
// All geometry output as WorldTri in world space, generated once at startup

use crate::state::*;
use crate::rng::Rng;
use crate::mesh;
use crate::color::{lerp_color, darken};
use crate::noise;
use crate::zone;
use crate::placement;

const BUILDING_COLORS: [u32; 10] = [
    0xFFCCBB99, 0xFFBBAA88, 0xFFDDCCAA, 0xFFBBBBAA, 0xFFCCBBAA,
    0xFFD4C4A8, 0xFFE0D0B0, 0xFFC8B898, 0xFFCCCCBB, 0xFFBBAAAA,
];
// ACU-style building accents
const TIMBER_COLOR: u32 = 0xFF443322;
const SHUTTER_COLORS: [u32; 4] = [0xFF556644, 0xFF445566, 0xFF664433, 0xFF554444];
const AWNING_COLORS: [u32; 5] = [0xFFCC4433, 0xFF336644, 0xFF884422, 0xFF334466, 0xFF997744];
const SHOP_FRONT_COLOR: u32 = 0xFF554433;
const SIGN_COLORS: [u32; 4] = [0xFF884422, 0xFF446633, 0xFF553344, 0xFF666644];
const BALCONY_RAIL_COLOR: u32 = 0xFF444444;
const FLOWER_BOX_COLOR: u32 = 0xFF886644;
const FLOWER_COLORS: [u32; 3] = [0xFFCC3344, 0xFFEECC33, 0xFFCC66AA];

const TRUNK_COLOR: u32 = 0xFF554422;
const ROCK_COLOR: u32 = 0xFF777777;
const GROUND_LOW: u32 = 0xFF2A6B2A;  // darker green in valleys
const GROUND_HIGH: u32 = 0xFF44AA44; // lighter green on hills
const ROAD_COLOR: u32 = 0xFF444444;
const ROAD_LINE_COLOR: u32 = 0xFFBBAA44;
const ROAD_EDGE_COLOR: u32 = 0xFFBBBBBB;
const SIDEWALK_COLOR: u32 = 0xFF888888;
const FIELD_ROAD_COLOR: u32 = 0xFF665544;
const CURB_COLOR: u32 = 0xFF999999; // lighter than road, darker than sidewalk
const LAMP_POLE_COLOR: u32 = 0xFF666666;
const LAMP_BASE_COLOR: u32 = 0xFF555555; // darker base/mounting plate for street lights
const LAMP_GLOW_COLOR: u32 = 0x00FFEE88; // alpha=0 flags emissive (night glow)

const ROAD_SEG_STEP: f32 = 4.0; // subdivision step for terrain-following road strips

// Dockyard colors
const DOCK_GROUND: u32 = 0xFF555544;
const WATER_COLOR: u32 = 0xFF2A5580;
const WAREHOUSE_COLORS: [u32; 4] = [0xFF666655, 0xFF555566, 0xFF665544, 0xFF556655];
const CRANE_COLOR: u32 = 0xFFCC8833;
const CONTAINER_COLORS: [u32; 5] = [0xFFCC3333, 0xFF3333CC, 0xFF33CC33, 0xFFCCCC33, 0xFFCC8833];
const PIER_COLOR: u32 = 0xFF665533;
const SCRAP_COLOR: u32 = 0xFF776655;
const CHIMNEY_COLOR: u32 = 0xFF555555;

// Interactible colors
const VENDING_COLOR: u32 = 0xFFCC2222;
const VENDING_PANEL: u32 = 0xFF888888;
const BENCH_COLOR: u32 = 0xFF886644;
const DUMPSTER_COLOR: u32 = 0xFF334488;
const ATM_COLOR: u32 = 0xFF777788;
const ATM_SCREEN: u32 = 0xFF44AACC;
const PHONE_BOOTH_COLOR: u32 = 0xFF667788;
const HYDRANT_COLOR: u32 = 0xFFCC3333;
const NEWSSTAND_COLOR: u32 = 0xFFCCCC33;
const MAILBOX_COLOR: u32 = 0xFF3344CC;
const PAYPHONE_COLOR: u32 = 0xFF888888;

// River/bridge colors — deep center to lighter edge
const RIVER_DEEP: u32 = 0xFF3070AA;   // deep center — rich blue with riverbed warmth
const RIVER_MID: u32 = 0xFF5599CC;    // mid-channel — lighter cyan-blue
const RIVER_EDGE: u32 = 0xFF88BBCC;   // near banks — very light, clear shallow water
const BANK_MUD: u32 = 0xFF445533;     // muddy brown near waterline
const BANK_GRASS: u32 = 0xFF3A7A3A;   // grassy bank top (blends to terrain)
const BRIDGE_DECK_COLOR: u32 = 0xFF666666;
const BRIDGE_RAIL_COLOR: u32 = 0xFF555555;

// Parking lot colors
const PARKING_ASPHALT: u32 = 0xFF333333;
const PARKING_LINE_COLOR: u32 = 0xFFCCCCCC;
const FENCE_COLOR: u32 = 0xFF775533;

// Market stall colors
const STALL_FRAME_COLOR: u32 = 0xFF886644;
const STALL_COUNTER_COLOR: u32 = 0xFF775533;
const STALL_CANVAS_COLORS: [u32; 4] = [0xFFCC3333, 0xFF3344CC, 0xFF33AA33, 0xFFCCCC33];

// Bus stop colors
const BUS_GLASS_COLOR: u32 = 0xFF88BBDD;
const BUS_ROOF_COLOR: u32 = 0xFF555555;
const BUS_SIGN_COLOR: u32 = 0xFF33AA33;

// Decoration colors
const BOLLARD_COLOR: u32 = 0xFF555555;
const PLANTER_BOX_COLOR: u32 = 0xFF886644;
const PLANTER_GREEN_COLOR: u32 = 0xFF33AA33;
const PICNIC_TABLE_COLOR: u32 = 0xFF886644;
const BILLBOARD_POST_COLOR: u32 = 0xFF666666;
const BILLBOARD_PANEL_COLOR: u32 = 0xFFCCBB88;
const WATER_TOWER_COLOR: u32 = 0xFF888888;
const TRAFFIC_CONE_COLOR: u32 = 0xFFFF6633;
const DECO_BENCH_COLOR: u32 = 0xFF886644;

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

/// Linearly interpolate along a road segment: returns (x, z) at parameter t in [0,1]
fn sample_segment(seg: &RoadSegment, t: f32) -> (f32, f32) {
    (seg.x0 + (seg.x1 - seg.x0) * t, seg.z0 + (seg.z1 - seg.z0) * t)
}

pub fn overlaps_building(x: f32, z: f32, margin: f32, buildings: &[Building]) -> bool {
    buildings.iter().any(|b| {
        (x - b.x).abs() < b.w * 0.5 + margin && (z - b.z).abs() < b.d * 0.5 + margin
    })
}

/// Direction and perpendicular for a road segment. Returns (perp_x, perp_z, dir_x, dir_z, length).
fn segment_perp(seg: &RoadSegment) -> Option<(f32, f32, f32, f32, f32)> {
    let dx = seg.x1 - seg.x0;
    let dz = seg.z1 - seg.z0;
    let len = (dx * dx + dz * dz).sqrt();
    if len < 0.01 { return None; }
    Some((-dz / len, dx / len, dx / len, dz / len, len))
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
pub fn road_dist_network(x: f32, z: f32, net: &RoadNetwork) -> (f32, RoadTier) {
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

/// Check if position is on the driveable road surface (for objects that can be on sidewalks)
pub fn on_road_surface(x: f32, z: f32, net: &RoadNetwork) -> bool {
    for seg in &net.segments {
        let d = point_to_segment_dist(x, z, seg.x0, seg.z0, seg.x1, seg.z1);
        let half_width = match seg.tier {
            RoadTier::CarRoad => CAR_ROAD_WIDTH * 0.5 + 0.5,
            RoadTier::FieldRoad => FIELD_ROAD_WIDTH * 0.5 + 0.3,
        };
        if d < half_width { return true; }
    }
    false
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

/// Generate an organic radial road network centered on a settlement position
fn generate_road_network(rng: &mut Rng, settlement_center: [f32; 2]) -> RoadNetwork {
    let mut nodes: Vec<[f32; 2]> = Vec::new();
    let mut segments: Vec<RoadSegment> = Vec::new();

    // Center hub at settlement position with slight jitter
    let center = [
        settlement_center[0] + rng.range(-3.0, 3.0),
        settlement_center[1] + rng.range(-3.0, 3.0),
    ];
    nodes.push(center); // index 0

    // Ring 1: 4 nodes at radius ~150m
    let ring1_count = 4;
    let ring1_start = nodes.len();
    let base_angle = rng.range(0.0, std::f32::consts::TAU);
    for i in 0..ring1_count {
        let angle = base_angle + (i as f32 / ring1_count as f32) * std::f32::consts::TAU
            + rng.range(-0.3, 0.3);
        let radius = rng.range(120.0, 180.0);
        nodes.push([angle.cos() * radius, angle.sin() * radius]);
    }

    // Ring 2: 6 nodes at radius ~350m
    let ring2_count = 6;
    let ring2_start = nodes.len();
    let base_angle2 = rng.range(0.0, std::f32::consts::TAU);
    for i in 0..ring2_count {
        let angle = base_angle2 + (i as f32 / ring2_count as f32) * std::f32::consts::TAU
            + rng.range(-0.25, 0.25);
        let radius = rng.range(280.0, 400.0);
        nodes.push([angle.cos() * radius, angle.sin() * radius]);
    }

    // Edge nodes: 4 nodes at radius ~600m
    let edge_count = 4;
    let edge_start = nodes.len();
    let base_angle3 = rng.range(0.0, std::f32::consts::TAU);
    for i in 0..edge_count {
        let angle = base_angle3 + (i as f32 / edge_count as f32) * std::f32::consts::TAU
            + rng.range(-0.3, 0.3);
        let radius = rng.range(500.0, 700.0);
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
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
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

    // Build road graph: adjacency list from CarRoad segments
    let graph = build_road_graph(&nodes, &segments);

    RoadNetwork {
        segments, nodes, graph,
        parking_spots: Vec::new(), // filled in generate_world
    }
}

fn build_road_graph(nodes: &[[f32; 2]], segments: &[RoadSegment]) -> RoadGraph {
    let mut adjacency: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); nodes.len()];
    let mut segment_nodes: Vec<(usize, usize)> = Vec::new();

    for (seg_idx, seg) in segments.iter().enumerate() {
        if seg.tier != RoadTier::CarRoad { segment_nodes.push((0, 0)); continue; }

        // Find nearest node to each endpoint
        let mut best_a = 0usize;
        let mut best_a_dist = f32::MAX;
        let mut best_b = 0usize;
        let mut best_b_dist = f32::MAX;
        for (ni, node) in nodes.iter().enumerate() {
            let da = (seg.x0 - node[0]) * (seg.x0 - node[0]) + (seg.z0 - node[1]) * (seg.z0 - node[1]);
            let db = (seg.x1 - node[0]) * (seg.x1 - node[0]) + (seg.z1 - node[1]) * (seg.z1 - node[1]);
            if da < best_a_dist { best_a_dist = da; best_a = ni; }
            if db < best_b_dist { best_b_dist = db; best_b = ni; }
        }

        let dx = seg.x1 - seg.x0;
        let dz = seg.z1 - seg.z0;
        let dist = (dx * dx + dz * dz).sqrt();

        segment_nodes.push((best_a, best_b));

        // Bidirectional
        if !adjacency[best_a].iter().any(|&(n, s, _)| n == best_b && s == seg_idx) {
            adjacency[best_a].push((best_b, seg_idx, dist));
        }
        if !adjacency[best_b].iter().any(|&(n, s, _)| n == best_a && s == seg_idx) {
            adjacency[best_b].push((best_a, seg_idx, dist));
        }
    }

    RoadGraph { adjacency, segment_nodes }
}

fn generate_parking_spots(net: &RoadNetwork, buildings: &[Building], terrain: &Terrain, river_segments: &[RiverSegment]) -> Vec<ParkingSpot> {
    let mut spots = Vec::new();

    for seg in &net.segments {
        if seg.tier != RoadTier::CarRoad { continue; }
        let Some((perp_x, perp_z, dir_x, dir_z, len)) = segment_perp(seg) else { continue };
        if len < 20.0 { continue; }

        let rot_y = (-dir_x).atan2(-dir_z);

        // Place parking spots along segment, both sides
        let spot_spacing = PARKING_SPOT_LENGTH + 1.0; // 6m between spot starts
        let num_spots = ((len - 10.0) / spot_spacing) as i32;
        let num_spots = num_spots.min(6); // cap at 6 per side per segment

        for side in [-1.0f32, 1.0] {
            let curb_offset = CAR_ROAD_WIDTH * 0.5 - 0.5; // just inside road edge
            for k in 0..num_spots {
                let t = 0.2 + (k as f32 + 0.5) * spot_spacing / len;
                if t > 0.8 { break; }
                let (pt_x, pt_z) = sample_segment(seg, t);
                let sx = pt_x + perp_x * curb_offset * side;
                let sz = pt_z + perp_z * curb_offset * side;

                // Skip if overlapping buildings
                if overlaps_building(sx, sz, 1.0, buildings) { continue; }

                // Skip if on river
                if on_river(sx, sz, river_segments) { continue; }

                let _ = terrain; // spot is on road surface, no height check needed
                spots.push(ParkingSpot {
                    x: sx, z: sz, rot_y: rot_y + if side > 0.0 { 0.0 } else { std::f32::consts::PI },
                    occupied_by: None,
                });
            }
        }
    }

    spots
}


fn jitter_color(c: u32, delta: i32) -> u32 {
    let r = (((c >> 16) & 0xFF) as i32 + delta).clamp(0, 255) as u32;
    let g = (((c >> 8) & 0xFF) as i32 + delta).clamp(0, 255) as u32;
    let b = ((c & 0xFF) as i32 + delta).clamp(0, 255) as u32;
    (c & 0xFF000000) | (r << 16) | (g << 8) | b
}

/// Compute a building's ground_y by sampling terrain across the footprint and returning
/// the minimum height. This prevents buildings from floating above terrain at corners/edges.
/// Also returns the maximum height for foundation depth calculation.
fn building_ground_height(terrain: &Terrain, cx: f32, cz: f32, w: f32, d: f32) -> (f32, f32) {
    let hw = w * 0.5;
    let hd = d * 0.5;
    // Sample corners, edge midpoints, and center (9 points)
    let samples = [
        terrain.height_at(cx, cz),           // center
        terrain.height_at(cx - hw, cz - hd), // corners
        terrain.height_at(cx + hw, cz - hd),
        terrain.height_at(cx - hw, cz + hd),
        terrain.height_at(cx + hw, cz + hd),
        terrain.height_at(cx - hw, cz),       // edge midpoints
        terrain.height_at(cx + hw, cz),
        terrain.height_at(cx, cz - hd),
        terrain.height_at(cx, cz + hd),
    ];
    let mut min_h = samples[0];
    let mut max_h = samples[0];
    for &h in &samples[1..] {
        if h < min_h { min_h = h; }
        if h > max_h { max_h = h; }
    }
    (min_h, max_h)
}

/// Add foundation geometry around a building to seal gaps between building base and terrain.
/// Creates a skirt (4 walls) extending from ground_y down past the lowest terrain point,
/// plus a floor quad at ground_y.
fn add_building_foundation(
    tris: &mut Vec<WorldTri>, terrain: &Terrain,
    cx: f32, ground_y: f32, cz: f32, w: f32, d: f32, color: u32,
) {
    let hw = w * 0.5 + 0.05; // slightly wider than building to prevent seams
    let hd = d * 0.5 + 0.05;
    let foundation_color = darken(color, 0.4);

    // Sample terrain at corners and edges to find how deep foundation must go
    let (min_h, _max_h) = building_ground_height(terrain, cx, cz, w + 1.0, d + 1.0);
    // Extend foundation 1m below lowest terrain point to ensure full coverage
    let foundation_bottom = min_h - 1.0;
    let skirt_h = ground_y - foundation_bottom;
    if skirt_h < 0.05 { return; } // terrain is flat under building, no skirt needed

    let top = ground_y;
    let bot = foundation_bottom;

    // Front skirt wall (z+) — CCW outward
    mesh::push_quad(tris,
        [cx - hw, bot, cz + hd],
        [cx + hw, bot, cz + hd],
        [cx + hw, top, cz + hd],
        [cx - hw, top, cz + hd],
        foundation_color,
    );
    // Back skirt wall (z-) — CCW outward
    mesh::push_quad(tris,
        [cx + hw, bot, cz - hd],
        [cx - hw, bot, cz - hd],
        [cx - hw, top, cz - hd],
        [cx + hw, top, cz - hd],
        foundation_color,
    );
    // Left skirt wall (x-) — CCW outward
    mesh::push_quad(tris,
        [cx - hw, bot, cz - hd],
        [cx - hw, bot, cz + hd],
        [cx - hw, top, cz + hd],
        [cx - hw, top, cz - hd],
        foundation_color,
    );
    // Right skirt wall (x+) — CCW outward
    mesh::push_quad(tris,
        [cx + hw, bot, cz + hd],
        [cx + hw, bot, cz - hd],
        [cx + hw, top, cz - hd],
        [cx + hw, top, cz + hd],
        foundation_color,
    );
    // Bottom cap (y-) — CCW facing down
    mesh::push_quad(tris,
        [cx - hw, bot, cz - hd],
        [cx + hw, bot, cz - hd],
        [cx + hw, bot, cz + hd],
        [cx - hw, bot, cz + hd],
        foundation_color,
    );
}

/// Generate heightmap from noise-based terrain — models real geological processes:
/// 1. Continental-scale tectonic structure (large features: mountain ranges, basins)
/// 2. Regional fBm terrain (hills, ridges, valleys)
/// 3. Ridged noise for mountain peaks and sharp features
/// 4. Hydraulic erosion simulation (particle-based, carves realistic river channels)
///
/// Seed-dependent roughness produces variety from Bohemian rolling hills to Alpine peaks.
fn generate_heightmap_noise(terrain: &mut Terrain, seed: u64, roughness: f32) {
    let grid = terrain.grid;
    let stride = grid + 1;
    let cell = terrain.cell_size;

    // Base amplitude scales with roughness:
    // roughness 0.3 = gentle rolling hills (~15m variation)
    // roughness 1.0 = hilly terrain (~50m)
    // roughness 2.0 = mountainous (~100m)
    let base_amp = 50.0;

    for iz in 0..stride {
        for ix in 0..stride {
            let x = -WORLD_HALF + ix as f32 * cell;
            let z = -WORLD_HALF + iz as f32 * cell;

            // 1. Large-scale terrain structure (period ~1.5km — 2 major features across 3km)
            let tectonic = noise::fbm(x, z, 3, 0.0007, 2.0, 0.5, seed + 3333);
            let tectonic_h = tectonic * base_amp * roughness;

            // 2. Domain-warped fBm for regional terrain (period ~300m)
            let (wx, wz) = noise::warp_2d(x, z, 80.0, 0.001, seed + 1111);
            let regional = noise::fbm(wx, wz, 5, 0.0015, 2.2, 0.45, seed);
            let regional_h = regional * base_amp * 0.5 * roughness;

            // 3. Ridged noise for ridgelines and sharp features
            let ridge = noise::ridged(x, z, 4, 0.001, 2.0, 0.5, seed + 2222);
            let ridge_h = ridge * base_amp * 0.3 * roughness * roughness;

            // 4. Fine detail noise (period ~30m) — small hillocks
            let detail = noise::fbm(x, z, 3, 0.008, 2.0, 0.5, seed + 4444);
            let detail_h = detail * 5.0 * roughness;

            // Combine
            let mut h = tectonic_h + regional_h + ridge_h + detail_h;

            // Border uplift: hills ring the edges
            let edge_x = (x.abs() / WORLD_HALF).max(0.0);
            let edge_z = (z.abs() / WORLD_HALF).max(0.0);
            let edge = (edge_x.max(edge_z) - 0.6).max(0.0) * 2.5;
            let border_uplift = edge * edge * base_amp * roughness;
            h += border_uplift;

            // Ensure terrain stays above a minimum (no deep holes)
            h = h.max(-5.0);

            terrain.heights[iz * stride + ix] = h;
        }
    }

    // Hydraulic erosion: simulate water particles carving channels
    erode_terrain(terrain, seed, roughness);
}

/// Particle-based hydraulic erosion — drops thousands of water particles that
/// flow downhill, eroding terrain where they move fast and depositing where slow.
/// This creates natural-looking river channels, valleys, and alluvial fans.
fn erode_terrain(terrain: &mut Terrain, seed: u64, roughness: f32) {
    let grid = terrain.grid;
    let stride = grid + 1;
    let cell = terrain.cell_size;

    let num_drops = 50_000; // scaled for 500-cell grid
    let max_lifetime = 128;
    let erosion_rate = 0.3 * roughness;
    let deposition_rate = 0.2;
    let evaporation = 0.01;
    let min_slope = 0.001;

    let mut rng = crate::rng::Rng::new(seed.wrapping_add(99999));

    for _ in 0..num_drops {
        // Random start position
        let mut px = rng.range(1.0, (grid - 2) as f32);
        let mut pz = rng.range(1.0, (grid - 2) as f32);
        let mut vx = 0.0f32;
        let mut vz = 0.0f32;
        let mut water = 1.0f32;
        let mut sediment = 0.0f32;

        for _ in 0..max_lifetime {
            let ix = px as usize;
            let iz = pz as usize;
            if ix < 1 || ix >= grid - 1 || iz < 1 || iz >= grid - 1 { break; }

            // Gradient from neighboring heights
            let h = terrain.heights[iz * stride + ix];
            let h_r = terrain.heights[iz * stride + ix + 1];
            let h_l = terrain.heights[iz * stride + ix.wrapping_sub(1).max(0)];
            let h_d = terrain.heights[(iz + 1) * stride + ix];
            let h_u = terrain.heights[iz.wrapping_sub(1).max(0) * stride + ix];

            let gx = (h_l - h_r) * 0.5;
            let gz = (h_u - h_d) * 0.5;

            // Update velocity (inertia + gradient)
            vx = vx * 0.6 + gx * 2.0;
            vz = vz * 0.6 + gz * 2.0;
            let speed = (vx * vx + vz * vz).sqrt();
            if speed < 0.0001 { break; }

            // Normalize and move
            let nx = px + vx / speed;
            let nz = pz + vz / speed;
            let nix = nx as usize;
            let niz = nz as usize;
            if nix < 1 || nix >= grid - 1 || niz < 1 || niz >= grid - 1 { break; }

            let h_new = terrain.heights[niz * stride + nix];
            let dh = h - h_new;

            // Carry capacity proportional to speed and slope
            let capacity = speed.max(min_slope) * water * 4.0;

            if dh > 0.0 {
                // Moving downhill — erode
                let erode = (capacity - sediment).max(0.0).min(dh) * erosion_rate;
                terrain.heights[iz * stride + ix] -= erode;
                sediment += erode;
            } else {
                // Moving uphill or flat — deposit
                let deposit = sediment * deposition_rate;
                terrain.heights[iz * stride + ix] += deposit;
                sediment -= deposit;
            }

            water *= 1.0 - evaporation;
            px = nx;
            pz = nz;
        }

        // Deposit remaining sediment at final position
        let ix = px as usize;
        let iz = pz as usize;
        if ix < stride && iz < stride {
            let idx = iz * stride + ix;
            if idx < terrain.heights.len() {
                terrain.heights[idx] += sediment * 0.5;
            }
        }
    }

    // Light smoothing pass to blend erosion artifacts
    let _ = cell; // suppress unused
    let mut smoothed = terrain.heights.clone();
    for iz in 1..grid {
        for ix in 1..grid {
            let idx = iz * stride + ix;
            let avg = (terrain.heights[idx]
                + terrain.heights[idx - 1]
                + terrain.heights[idx + 1]
                + terrain.heights[(iz - 1) * stride + ix]
                + terrain.heights[(iz + 1) * stride + ix]) * 0.2;
            smoothed[idx] = terrain.heights[idx] * 0.7 + avg * 0.3;
        }
    }
    terrain.heights = smoothed;
}

/// Flatten terrain near roads and settlement centers (runs after road network is built)
fn flatten_terrain_for_roads(
    terrain: &mut Terrain, net: &RoadNetwork,
    _settlement_center: [f32; 2], zone_map: &zone::ZoneMap,
) {
    let grid = terrain.grid;
    let stride = grid + 1;
    let cell = terrain.cell_size;

    for iz in 0..stride {
        for ix in 0..stride {
            let x = -WORLD_HALF + ix as f32 * cell;
            let z = -WORLD_HALF + iz as f32 * cell;
            let h = terrain.heights[iz * stride + ix];

            // Flatten near roads (smooth falloff)
            let (rd, tier) = road_dist_network(x, z, net);
            let corridor = match tier {
                RoadTier::CarRoad => CAR_ROAD_WIDTH * 0.5 + SIDEWALK_WIDTH,
                RoadTier::FieldRoad => FIELD_ROAD_WIDTH * 0.5 + 1.0,
            };
            let road_flatten = if rd < corridor {
                0.0
            } else if rd < corridor + 8.0 {
                let t = (rd - corridor) / 8.0;
                t * t
            } else {
                1.0
            };

            // Flatten all settlement center areas
            let mut settle_flatten = 1.0f32;
            for center in &zone_map.settlement_centers {
                let dx = x - center[0];
                let dz = z - center[1];
                let settle_dist = (dx * dx + dz * dz).sqrt();
                let f = if settle_dist < 60.0 {
                    0.15
                } else if settle_dist < 150.0 {
                    0.15 + 0.85 * ((settle_dist - 60.0) / 90.0)
                } else {
                    1.0
                };
                settle_flatten = settle_flatten.min(f);
            }

            // Flatten industrial/waterfront zones
            let zone_cell = zone_map.zone_at(x, z);
            let zone_flatten = match zone_cell.kind {
                zone::ZoneKind::Industrial | zone::ZoneKind::Waterfront => {
                    let t = zone_cell.density;
                    1.0 - t * 0.7 // flatten proportional to density
                }
                _ => 1.0,
            };

            terrain.heights[iz * stride + ix] = h * road_flatten * settle_flatten * zone_flatten;
        }
    }
}

/// Generate terrain mesh triangles from heightmap with smoothed vertex normals
/// and zone-aware coloring.
fn generate_terrain_mesh(tris: &mut Vec<WorldTri>, terrain: &Terrain, zone_map: &zone::ZoneMap) {
    let grid = terrain.grid;
    let stride = grid + 1;
    let cell = terrain.cell_size;
    let num_verts = stride * stride;

    // Find height range for color interpolation
    let mut h_min = f32::MAX;
    let mut h_max = f32::MIN;
    for &h in &terrain.heights {
        if h < h_min { h_min = h; }
        if h > h_max { h_max = h; }
    }
    let h_range = (h_max - h_min).max(0.1);

    // Pre-compute smoothed vertex normals by averaging face normals of adjacent triangles.
    // Each vertex is shared by up to 6 triangles. Accumulate then normalize.
    let mut vert_normals = vec![[0.0f32; 3]; num_verts];
    for iz in 0..grid {
        for ix in 0..grid {
            let vh00 = terrain.heights[iz * stride + ix];
            let vh10 = terrain.heights[iz * stride + ix + 1];
            let vh01 = terrain.heights[(iz + 1) * stride + ix];
            let vh11 = terrain.heights[(iz + 1) * stride + ix + 1];

            let vx0 = -WORLD_HALF + ix as f32 * cell;
            let vz0 = -WORLD_HALF + iz as f32 * cell;
            let vx1 = vx0 + cell;
            let vz1 = vz0 + cell;

            // Tri 1: v00, v11, v10 — face normal (CCW viewed from above → +Y normal)
            let fn1 = mesh::tri_normal([vx0,vh00,vz0], [vx1,vh11,vz1], [vx1,vh10,vz0]);
            // Tri 2: v00, v01, v11 — face normal (CCW viewed from above → +Y normal)
            let fn2 = mesh::tri_normal([vx0,vh00,vz0], [vx0,vh01,vz1], [vx1,vh11,vz1]);

            let i00 = iz * stride + ix;
            let i10 = iz * stride + ix + 1;
            let i01 = (iz + 1) * stride + ix;
            let i11 = (iz + 1) * stride + ix + 1;

            // Accumulate face normals to each vertex of tri 1
            for &vi in &[i00, i10, i11] {
                vert_normals[vi][0] += fn1[0];
                vert_normals[vi][1] += fn1[1];
                vert_normals[vi][2] += fn1[2];
            }
            // Accumulate face normals to each vertex of tri 2
            for &vi in &[i00, i11, i01] {
                vert_normals[vi][0] += fn2[0];
                vert_normals[vi][1] += fn2[1];
                vert_normals[vi][2] += fn2[2];
            }
        }
    }
    // Normalize accumulated vertex normals
    for vn in &mut vert_normals {
        let l = (vn[0]*vn[0] + vn[1]*vn[1] + vn[2]*vn[2]).sqrt();
        if l > 1e-10 { vn[0] /= l; vn[1] /= l; vn[2] /= l; }
        else { *vn = [0.0, 1.0, 0.0]; }
    }

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

            let mid_z = z0 + cell * 0.5;

            // Slope: measure max height difference across the cell
            let dh_x = ((h10 - h00).abs() + (h11 - h01).abs()) * 0.5 / cell;
            let dh_z = ((h01 - h00).abs() + (h11 - h10).abs()) * 0.5 / cell;
            let slope = (dh_x * dh_x + dh_z * dh_z).sqrt();
            let slope_t = (slope * 3.0).clamp(0.0, 1.0); // steeper → more rock

            // Per-cell color from average height — both triangles in the cell share the
            // same base color to eliminate visible triangle edges. Only a minimal per-cell
            // noise (+-1) provides subtle variation without causing faceting.
            let h_avg = (h00 + h10 + h01 + h11) * 0.25;
            let t = ((h_avg - h_min) / h_range).clamp(0.0, 1.0);
            let cell_hash = (ix as u32).wrapping_mul(73856093)
                ^ (iz as u32).wrapping_mul(19349663);
            let noise = (cell_hash % 3) as i32 - 1; // +-1 max
            let base_green = lerp_color(GROUND_LOW, GROUND_HIGH, t);
            let rocky = lerp_color(base_green, 0xFF6B6455, slope_t);
            // Zone-aware terrain coloring
            let mid_x = x0 + cell * 0.5;
            let cell_color = if zone_map.grid_size > 0 {
                let zone_cell = zone_map.zone_at(mid_x, mid_z);
                match zone_cell.kind {
                    zone::ZoneKind::Industrial | zone::ZoneKind::Waterfront => {
                        let dock_t = zone_cell.density.min(1.0);
                        jitter_color(lerp_color(rocky, DOCK_GROUND, dock_t), noise)
                    }
                    _ => jitter_color(rocky, noise),
                }
            } else {
                jitter_color(rocky, noise)
            };

            let c1 = cell_color;
            let c2 = cell_color;

            // Use smoothed vertex normals averaged across the triangle
            let i00 = iz * stride + ix;
            let i10 = iz * stride + ix + 1;
            let i01 = (iz + 1) * stride + ix;
            let i11 = (iz + 1) * stride + ix + 1;
            let n1 = avg_normal3(vert_normals[i00], vert_normals[i10], vert_normals[i11]);
            let n2 = avg_normal3(vert_normals[i00], vert_normals[i11], vert_normals[i01]);

            // CCW winding for Vulkan front-face (VK_FRONT_FACE_COUNTER_CLOCKWISE)
            mesh::push_tri(tris, v00, v11, v10, n1, c1);
            mesh::push_tri(tris, v00, v01, v11, n2, c2);
        }
    }

    // Ground skirt: vertical walls dropping from terrain edges + extended floor plane.
    // Prevents sky bleed-through when camera looks toward or beyond map edges.
    // CCW winding for VK_FRONT_FACE_COUNTER_CLOCKWISE.
    let skirt_y = -50.0; // floor level well below any terrain height
    let skirt_color = GROUND_LOW; // dark green matches terrain base
    let skirt_n_up: [f32; 3] = [0.0, 1.0, 0.0];

    // South edge (iz = 0, z = -WORLD_HALF): visible from inside (+Z)
    let south_n: [f32; 3] = [0.0, 0.0, 1.0];
    for ix in 0..grid {
        let x0 = -WORLD_HALF + ix as f32 * cell;
        let x1 = x0 + cell;
        let h0 = terrain.heights[0 * stride + ix];
        let h1 = terrain.heights[0 * stride + ix + 1];
        let z = -WORLD_HALF;
        mesh::push_quad_n(tris, [x0,h0,z], [x0,skirt_y,z], [x1,skirt_y,z], [x1,h1,z], south_n, skirt_color);
    }

    // North edge (iz = grid, z = +WORLD_HALF): visible from inside (-Z)
    let north_n: [f32; 3] = [0.0, 0.0, -1.0];
    for ix in 0..grid {
        let x0 = -WORLD_HALF + ix as f32 * cell;
        let x1 = x0 + cell;
        let h0 = terrain.heights[grid * stride + ix];
        let h1 = terrain.heights[grid * stride + ix + 1];
        let z = WORLD_HALF;
        mesh::push_quad_n(tris, [x1,h1,z], [x1,skirt_y,z], [x0,skirt_y,z], [x0,h0,z], north_n, skirt_color);
    }

    // West edge (ix = 0, x = -WORLD_HALF): visible from inside (+X)
    let west_n: [f32; 3] = [1.0, 0.0, 0.0];
    for iz in 0..grid {
        let z0 = -WORLD_HALF + iz as f32 * cell;
        let z1 = z0 + cell;
        let h0 = terrain.heights[iz * stride + 0];
        let h1 = terrain.heights[(iz + 1) * stride + 0];
        let x = -WORLD_HALF;
        mesh::push_quad_n(tris, [x,h1,z1], [x,skirt_y,z1], [x,skirt_y,z0], [x,h0,z0], west_n, skirt_color);
    }

    // East edge (ix = grid, x = +WORLD_HALF): visible from inside (-X)
    let east_n: [f32; 3] = [-1.0, 0.0, 0.0];
    for iz in 0..grid {
        let z0 = -WORLD_HALF + iz as f32 * cell;
        let z1 = z0 + cell;
        let h0 = terrain.heights[iz * stride + grid];
        let h1 = terrain.heights[(iz + 1) * stride + grid];
        let x = WORLD_HALF;
        mesh::push_quad_n(tris, [x,h0,z0], [x,skirt_y,z0], [x,skirt_y,z1], [x,h1,z1], east_n, skirt_color);
    }

    // Extended ground floor plane
    if true {
        let ext = WORLD_HALF * 2.0;
        let floor_color = darken(GROUND_LOW, 0.7);
        let floor_tiles = 4;
        let tile_size = (ext * 2.0) / floor_tiles as f32;
        for tz in 0..floor_tiles {
            for tx in 0..floor_tiles {
                let fx0 = -ext + tx as f32 * tile_size;
                let fz0 = -ext + tz as f32 * tile_size;
                let fx1 = fx0 + tile_size;
                let fz1 = fz0 + tile_size;
                let a = [fx0, skirt_y, fz0];
                let b = [fx1, skirt_y, fz0];
                let c = [fx1, skirt_y, fz1];
                let d = [fx0, skirt_y, fz1];
                mesh::push_quad_n(tris, a, d, c, b, skirt_n_up, floor_color);
            }
        }
    }
}

/// Average 3 colors (per-channel mean)
#[allow(dead_code)]
fn avg_color3(a: u32, b: u32, c: u32) -> u32 {
    let r = (((a >> 16) & 0xFF) + ((b >> 16) & 0xFF) + ((c >> 16) & 0xFF)) / 3;
    let g = (((a >> 8) & 0xFF) + ((b >> 8) & 0xFF) + ((c >> 8) & 0xFF)) / 3;
    let bl = ((a & 0xFF) + (b & 0xFF) + (c & 0xFF)) / 3;
    0xFF000000 | (r << 16) | (g << 8) | bl
}

/// Average and normalize 3 normal vectors
fn avg_normal3(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let nx = a[0] + b[0] + c[0];
    let ny = a[1] + b[1] + c[1];
    let nz = a[2] + b[2] + c[2];
    let l = (nx*nx + ny*ny + nz*nz).sqrt();
    if l < 1e-10 { [0.0, 1.0, 0.0] } else { [nx/l, ny/l, nz/l] }
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

        // Per-quad color jitter for asphalt variation
        let h = (sx0 as i32 as u32).wrapping_mul(73856093)
            ^ (sz0 as i32 as u32).wrapping_mul(19349663)
            ^ (s as u32).wrapping_mul(2654435761);
        let noise = (h % 12) as i32 - 6;
        let c = jitter_color(color, noise);

        mesh::push_quad_n(tris, v_l0, v_l1, v_r1, v_r0, [0.0, 1.0, 0.0], c);
    }
}

/// Generate the industrial dockyard biome at the given zone center
fn generate_dockyard_at(
    tris: &mut Vec<WorldTri>, terrain: &Terrain, rng: &mut Rng,
    buildings: &mut Vec<Building>, interactibles: &mut Vec<Interactible>,
    center: [f32; 2], density_scale: f32,
) {
    let cx = center[0];
    let cz = center[1];

    // Wave water surface near the dockyard
    let x_min = (cx - 60.0).max(-WORLD_HALF + 5.0);
    let x_max = (cx + 60.0).min(WORLD_HALF - 5.0);
    let z_min = (cz + 5.0).max(-WORLD_HALF + 5.0);
    let z_max = (cz + 50.0).min(WORLD_HALF - 5.0);
    if z_max > z_min && x_max > x_min {
        mesh::wave_surface_tris(tris, x_min, x_max, z_min, z_max,
            WATER_Y, 0.5, 0.5,
            ((x_max - x_min) / 5.0) as usize, ((z_max - z_min) / 5.0).max(1.0) as usize,
            WATER_COLOR);
    }

    // 6 Warehouses
    for i in 0..(6.0 * density_scale) as usize {
        let wx = cx - 50.0 + i as f32 * 18.0 + rng.range(-2.0, 2.0);
        let wz = cz + rng.range(0.0, 8.0);
        let ww = rng.range(8.0, 14.0);
        let wd = rng.range(6.0, 10.0);
        let wh = rng.range(4.0, 7.0);
        let (min_h, _) = building_ground_height(terrain, wx, wz, ww, wd);
        let gy = min_h;
        let color = WAREHOUSE_COLORS[i % WAREHOUSE_COLORS.len()];
        add_building_foundation(tris, terrain, wx, gy, wz, ww, wd, color);
        mesh::beveled_box_tris(tris, wx, gy + wh * 0.5, wz, ww, wh, wd, 0.1, color);
        mesh::box_tris(tris, wx, gy + 2.0, wz - wd * 0.5 + 0.08, ww * 0.4, 4.0, 0.16, 0xFF333322);
        mesh::pitched_roof_tris(tris, wx, gy + wh, wz, ww + 0.2, wd + 0.2, 1.5, darken(color, 0.7));
        buildings.push(Building { x: wx, z: wz, w: ww, d: wd, h: wh, ground_y: gy });
    }

    // 3 Cranes
    for i in 0..(3.0 * density_scale) as usize {
        let kx = cx - 30.0 + i as f32 * 30.0;
        let kz = cz + 22.0;
        let gy = terrain.height_at(kx, kz);
        let crane_h = 25.0;
        mesh::cylinder_tris(tris, kx, gy + crane_h * 0.5, kz, 0.35, crane_h, 8, CRANE_COLOR);
        mesh::cylinder_between(tris,
            [kx, gy + crane_h, kz],
            [kx + 11.0, gy + crane_h - 0.5, kz],
            0.15, 6, CRANE_COLOR);
        mesh::beveled_box_tris(tris, kx - 3.0, gy + crane_h - 1.0, kz, 2.0, 2.0, 1.5, 0.1, CHIMNEY_COLOR);
        mesh::beveled_box_tris(tris, kx, gy + crane_h - 2.0, kz, 1.5, 2.0, 1.5, 0.08, 0xFF888833);
    }

    // 15 Cargo containers
    for _ in 0..(15.0 * density_scale) as usize {
        let bx = cx + rng.range(-40.0, 40.0);
        let bz = cz + rng.range(5.0, 25.0);
        let gy = terrain.height_at(bx, bz);
        let color = CONTAINER_COLORS[rng.next() as usize % CONTAINER_COLORS.len()];
        let stack = 1 + rng.next() as usize % 3;
        for s in 0..stack {
            mesh::beveled_box_tris(tris, bx, gy + 1.3 + s as f32 * 2.5, bz, 6.0, 2.5, 2.5, 0.05, color);
        }
    }

    // 3 Fishing piers
    for i in 0..(3.0 * density_scale) as usize {
        let px = cx - 30.0 + i as f32 * 30.0;
        let pz_start = cz + 25.0;
        let pier_len = 12.0;
        let gy = terrain.height_at(px, pz_start);
        mesh::box_tris(tris, px, gy + 0.5, pz_start + pier_len * 0.5, 2.0, 0.25, pier_len, PIER_COLOR);
        for s in 0..3 {
            let sz = pz_start + s as f32 * 4.0 + 2.0;
            mesh::cylinder_tris(tris, px - 0.8, gy * 0.5, sz, 0.08, gy.abs() + 1.0, 5, PIER_COLOR);
            mesh::cylinder_tris(tris, px + 0.8, gy * 0.5, sz, 0.08, gy.abs() + 1.0, 5, PIER_COLOR);
        }
    }

    // Scrap yard
    for si in 0..(20.0 * density_scale) as usize {
        let sx = cx + rng.range(25.0, 55.0);
        let sz = cz + rng.range(0.0, 12.0);
        let gy = terrain.height_at(sx, sz);
        let size = rng.range(0.3, 1.5);
        mesh::perturbed_sphere_tris(tris, sx, gy + size * 0.5, sz, size, 0, 0.3, si as u64 * 7777, SCRAP_COLOR);
    }

    // 2 Smokestacks
    for i in 0..(2.0 * density_scale).max(1.0) as usize {
        let sx = cx - 20.0 + i as f32 * 40.0;
        let sz = cz + 5.0;
        let gy = terrain.height_at(sx, sz);
        mesh::cylinder_tris(tris, sx, gy + 10.0, sz, 0.6, 20.0, 8, CHIMNEY_COLOR);
        mesh::cylinder_tris(tris, sx, gy + 20.3, sz, 0.8, 0.6, 8, 0xFF444444);
    }

    // Dockyard dumpsters
    for i in 0..(4.0 * density_scale) as usize {
        let dx = cx - 40.0 + i as f32 * 25.0;
        let dz = cz + 2.0;
        let dy = terrain.height_at(dx, dz);
        interactibles.push(Interactible {
            x: dx, y: dy, z: dz,
            kind: InteractibleKind::Dumpster,
            cooldown: 0.0, state_val: 0.0,
        });
    }
}

/// Place interactible objects near roads and buildings
fn generate_interactibles(
    tris: &mut Vec<WorldTri>, terrain: &Terrain, rng: &mut Rng,
    net: &RoadNetwork, buildings: &[Building],
    interactibles: &mut Vec<Interactible>, density_scale: f32,
) {
    let car_segs: Vec<&RoadSegment> = net.segments.iter()
        .filter(|s| s.tier == RoadTier::CarRoad).collect();

    // Helper: pick a sidewalk position along a car road segment
    let sidewalk_pos = |rng: &mut Rng, seg: &RoadSegment, side: f32| -> (f32, f32) {
        let t = rng.range(0.2, 0.8);
        let (sx, sz) = sample_segment(seg, t);
        let dx = seg.x1 - seg.x0;
        let dz = seg.z1 - seg.z0;
        let len = (dx * dx + dz * dz).sqrt().max(0.01);
        let px = -dz / len;
        let pz = dx / len;
        let offset = CAR_ROAD_WIDTH * 0.5 + SIDEWALK_WIDTH * 0.5;
        (sx + px * offset * side, sz + pz * offset * side)
    };

    // Phone Booths (4) — beveled body with cylinder roof
    for i in 0..(4.0 * density_scale) as usize {
        if i >= net.nodes.len() { break; }
        let node = &net.nodes[i.min(net.nodes.len() - 1)];
        let mut x = node[0] + rng.range(3.0, 5.0);
        let mut z = node[1] + rng.range(3.0, 5.0);
        for _ in 0..20 {
            if !on_road_surface(x, z, net) { break; }
            let angle = rng.range(0.0, std::f32::consts::TAU);
            let dist = rng.range(6.0, 15.0);
            x = node[0] + angle.cos() * dist;
            z = node[1] + angle.sin() * dist;
        }
        if on_road_surface(x, z, net) { continue; }
        let y = terrain.height_at(x, z);
        mesh::beveled_box_tris(tris, x, y + 1.1, z, 0.8, 2.2, 0.8, 0.06, PHONE_BOOTH_COLOR);
        // Domed roof
        mesh::sphere_tris(tris, x, y + 2.25, z, 0.45, 1, PHONE_BOOTH_COLOR);
        interactibles.push(Interactible {
            x, y, z, kind: InteractibleKind::PhoneBooth,
            cooldown: 0.0, state_val: 0.0,
        });
    }

    // Vending Machines (20) — beveled body with recessed screen
    for i in 0..(20.0 * density_scale) as usize {
        let bi = (i * 6 + 1) % buildings.len();
        let b = &buildings[bi];
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let x = b.x + side * (b.w * 0.5 + 1.2);
        let z = b.z;
        if on_road_surface(x, z, net) { continue; }
        let y = terrain.height_at(x, z);
        mesh::beveled_box_tris(tris, x, y + 0.75, z, 0.7, 1.5, 0.6, 0.04, VENDING_COLOR);
        // Recessed panel (into body, not flush)
        mesh::box_tris(tris, x, y + 0.9, z - 0.25, 0.55, 0.7, 0.06, VENDING_PANEL);
        interactibles.push(Interactible {
            x, y, z, kind: InteractibleKind::VendingMachine,
            cooldown: 0.0, state_val: 0.0,
        });
    }

    // Park Benches (8) — slatted seat + back using thin boxes
    for i in 0..(8.0 * density_scale) as usize {
        if car_segs.is_empty() { break; }
        let seg = car_segs[rng.next() as usize % car_segs.len()];
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let (x, z) = sidewalk_pos(rng, seg, side);
        if on_road_surface(x, z, net) { continue; }
        let y = terrain.height_at(x, z);
        // Seat slats (3)
        for si in 0..3 {
            let sz = z - 0.15 + si as f32 * 0.15;
            mesh::box_tris(tris, x, y + 0.25, sz, 1.5, 0.05, 0.1, BENCH_COLOR);
        }
        // Backrest slats (2)
        for si in 0..2 {
            let bh = 0.38 + si as f32 * 0.18;
            mesh::box_tris(tris, x, y + bh, z + 0.22, 1.5, 0.05, 0.04, BENCH_COLOR);
        }
        // Legs (cylinder)
        for lx in [-0.6f32, 0.6] {
            mesh::cylinder_tris(tris, x + lx, y + 0.125, z, 0.025, 0.25, 4, 0xFF554422);
        }
        interactibles.push(Interactible {
            x, y, z, kind: InteractibleKind::ParkBench,
            cooldown: 0.0, state_val: 0.0,
        });
    }

    // Dumpsters (6) — beveled body with lid
    for i in 0..(6.0 * density_scale) as usize {
        let bi = (i * 5 + 3) % buildings.len();
        let b = &buildings[bi];
        let x = b.x;
        let z = b.z - b.d * 0.5 - 1.5;
        if on_road_surface(x, z, net) { continue; }
        let y = terrain.height_at(x, z);
        mesh::beveled_box_tris(tris, x, y + 0.5, z, 1.2, 1.0, 0.8, 0.05, DUMPSTER_COLOR);
        mesh::box_tris(tris, x, y + 1.05, z, 1.25, 0.08, 0.82, 0xFF445599); // lid
        interactibles.push(Interactible {
            x, y, z, kind: InteractibleKind::Dumpster,
            cooldown: 0.0, state_val: 0.0,
        });
    }

    // ATMs (3) — beveled body with recessed screen
    for i in 0..(3.0 * density_scale) as usize {
        let bi = i % buildings.len().min(10);
        let b = &buildings[bi];
        let x = b.x + b.w * 0.5 + 0.4;
        let z = b.z;
        if on_road_surface(x, z, net) { continue; }
        let y = terrain.height_at(x, z);
        mesh::beveled_box_tris(tris, x, y + 0.7, z, 0.6, 1.4, 0.3, 0.03, ATM_COLOR);
        // Recessed screen (into body)
        mesh::box_tris(tris, x - 0.15, y + 1.0, z - 0.1, 0.22, 0.25, 0.05, ATM_SCREEN);
        interactibles.push(Interactible {
            x, y, z, kind: InteractibleKind::Atm,
            cooldown: 0.0, state_val: 0.0,
        });
    }

    // Newspaper Stands (4) — beveled box
    for i in 0..(4.0 * density_scale) as usize {
        let ni = (i + 1) % net.nodes.len().max(1);
        let node = &net.nodes[ni];
        let mut x = node[0] - rng.range(3.0, 5.0);
        let mut z = node[1] - rng.range(3.0, 5.0);
        for _ in 0..20 {
            if !on_road_surface(x, z, net) { break; }
            let angle = rng.range(0.0, std::f32::consts::TAU);
            let dist = rng.range(6.0, 15.0);
            x = node[0] + angle.cos() * dist;
            z = node[1] + angle.sin() * dist;
        }
        if on_road_surface(x, z, net) { continue; }
        let y = terrain.height_at(x, z);
        mesh::beveled_box_tris(tris, x, y + 0.5, z, 0.6, 1.0, 0.4, 0.03, NEWSSTAND_COLOR);
        interactibles.push(Interactible {
            x, y, z, kind: InteractibleKind::NewspaperStand,
            cooldown: 0.0, state_val: 0.0,
        });
    }

    // Mailboxes (8) — beveled box with rounded top
    for i in 0..(8.0 * density_scale) as usize {
        if car_segs.is_empty() { break; }
        let seg = car_segs[rng.next() as usize % car_segs.len()];
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let (x, z) = sidewalk_pos(rng, seg, side);
        if on_road_surface(x, z, net) { continue; }
        let y = terrain.height_at(x, z);
        mesh::beveled_box_tris(tris, x, y + 0.5, z, 0.4, 1.0, 0.3, 0.03, MAILBOX_COLOR);
        // Rounded top
        mesh::sphere_tris(tris, x, y + 1.0, z, 0.2, 0, 0xFF4455DD);
        interactibles.push(Interactible {
            x, y, z, kind: InteractibleKind::Mailbox,
            cooldown: 0.0, state_val: 0.0,
        });
    }

    // Fire Hydrants (6) — lathe profile
    for i in 0..(6.0 * density_scale) as usize {
        if car_segs.is_empty() { break; }
        let seg = car_segs[rng.next() as usize % car_segs.len()];
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let (x, z) = sidewalk_pos(rng, seg, side);
        if on_road_surface(x, z, net) { continue; }
        let y = terrain.height_at(x, z);
        // Lathe profile for hydrant shape
        let profile: [[f32;2]; 6] = [
            [0.0, 0.0],   // bottom center
            [0.12, 0.0],  // base
            [0.1, 0.25],  // body
            [0.15, 0.35], // cap bulge
            [0.08, 0.5],  // top neck
            [0.0, 0.55],  // top center
        ];
        mesh::lathe_tris(tris, x, y, z, &profile, 6, HYDRANT_COLOR);
        interactibles.push(Interactible {
            x, y, z, kind: InteractibleKind::FireHydrant,
            cooldown: 0.0, state_val: 0.0,
        });
    }

    // Payphones (2) — beveled body with recessed screen
    for i in 0..(2.0 * density_scale) as usize {
        let ni = (i * 2) % net.nodes.len().max(1);
        let node = &net.nodes[ni];
        let mut x = node[0] + rng.range(-2.0, 2.0);
        let mut z = node[1] + rng.range(5.0, 7.0);
        for _ in 0..20 {
            if !on_road_surface(x, z, net) { break; }
            let angle = rng.range(0.0, std::f32::consts::TAU);
            let dist = rng.range(6.0, 15.0);
            x = node[0] + angle.cos() * dist;
            z = node[1] + angle.sin() * dist;
        }
        if on_road_surface(x, z, net) { continue; }
        let y = terrain.height_at(x, z);
        mesh::beveled_box_tris(tris, x, y + 0.9, z, 0.4, 1.8, 0.3, 0.03, PAYPHONE_COLOR);
        // Recessed screen (into body)
        mesh::box_tris(tris, x, y + 1.3, z - 0.1, 0.25, 0.25, 0.05, 0xFF222222);
        interactibles.push(Interactible {
            x, y, z, kind: InteractibleKind::Payphone,
            cooldown: 0.0, state_val: 0.0,
        });
    }
}

/// River centerline Z at given X
/// River path: noise-warped curve across the map. Entry/exit positions vary by seed.
fn river_z(x: f32, seed: u64) -> f32 {
    // Base offset and exit offset determined by seed
    let entry_z = noise::hash_2d(0, 0, seed.wrapping_add(77777)) * WORLD_HALF * 0.3;
    let exit_z = noise::hash_2d(1, 0, seed.wrapping_add(77777)) * WORLD_HALF * 0.3;
    let t = (x + WORLD_HALF) / WORLD_SIZE;
    let base_z = entry_z + (exit_z - entry_z) * t;
    // Meandering scaled to world size
    let meander = noise::fbm(x * 0.002, 0.0, 3, 1.0, 2.0, 0.5, seed + 77700) * WORLD_HALF * 0.15;
    base_z + meander
}

/// Check if a world position is on the river
pub fn on_bridge(x: f32, z: f32, bridges: &[Bridge]) -> bool {
    for b in bridges {
        let dx = x - b.cx;
        let dz = z - b.cz;
        let along = (dx * b.dir_x + dz * b.dir_z).abs();
        let across = (dx * (-b.dir_z) + dz * b.dir_x).abs();
        if along < b.hl && across < b.hw {
            return true;
        }
    }
    false
}

pub fn on_river(x: f32, z: f32, river_segments: &[RiverSegment]) -> bool {
    for seg in river_segments {
        let d = point_to_segment_dist(x, z, seg.x1, seg.z1, seg.x2, seg.z2);
        if d < seg.width * 0.5 {
            return true;
        }
    }
    false
}

/// River check that respects bridge exemptions
pub fn on_river_not_bridge(x: f32, z: f32, river_segments: &[RiverSegment], bridges: &[Bridge]) -> bool {
    on_river(x, z, river_segments) && !on_bridge(x, z, bridges)
}

/// Generate river: carve heightmap, store segments, add water surface tris.
/// Water surface follows terrain height (pre-carve bank level - 0.5m).
fn generate_river(
    terrain: &mut Terrain, tris: &mut Vec<WorldTri>,
    river_segments: &mut Vec<RiverSegment>, seed: u64,
) {
    let step = 20.0; // 20m segments for 3km world
    let half = WORLD_HALF;

    // Build segments
    let mut x = -half;
    while x < half {
        let x0 = x;
        let x1 = (x + step).min(half);
        let z0 = river_z(x0, seed);
        let z1 = river_z(x1, seed);
        river_segments.push(RiverSegment {
            x1: x0, z1: z0, x2: x1, z2: z1, width: RIVER_WIDTH,
        });
        x += step;
    }

    // Sample bank heights BEFORE carving (terrain at river edge = bank height)
    let bank_heights: Vec<(f32, f32)> = river_segments.iter().map(|seg| {
        let h0 = terrain.height_at(seg.x1, seg.z1);
        let h1 = terrain.height_at(seg.x2, seg.z2);
        (h0, h1)
    }).collect();

    // Carve heightmap — lower terrain within river channel
    // Optimized: only process grid cells near each segment (not full grid scan)
    let grid = terrain.grid;
    let stride = grid + 1;
    let cell = terrain.cell_size;
    let hw = RIVER_WIDTH * 0.5 + cell; // half-width + cell margin
    for seg in river_segments.iter() {
        let min_x = seg.x1.min(seg.x2) - hw;
        let max_x = seg.x1.max(seg.x2) + hw;
        let min_z = seg.z1.min(seg.z2) - hw;
        let max_z = seg.z1.max(seg.z2) + hw;
        let ix0 = ((min_x + half) / cell).max(0.0) as usize;
        let ix1 = ((max_x + half) / cell).min(grid as f32) as usize + 1;
        let iz0 = ((min_z + half) / cell).max(0.0) as usize;
        let iz1 = ((max_z + half) / cell).min(grid as f32) as usize + 1;
        for iz in iz0..iz1.min(stride) {
            for ix in ix0..ix1.min(stride) {
                let wx = -half + ix as f32 * cell;
                let wz = -half + iz as f32 * cell;
                let d = point_to_segment_dist(wx, wz, seg.x1, seg.z1, seg.x2, seg.z2);
                if d < RIVER_WIDTH * 0.5 {
                    terrain.heights[iz * stride + ix] -= RIVER_DEPTH;
                }
            }
        }
    }

    // Wave water surface — never coplanar with banks (sinusoidal undulation)
    // Compute overall river bounding box for wave surface
    if !river_segments.is_empty() {
        let x_min = river_segments.iter().map(|s| s.x1.min(s.x2)).fold(f32::MAX, f32::min) - RIVER_WIDTH * 0.5;
        let x_max = river_segments.iter().map(|s| s.x1.max(s.x2)).fold(f32::MIN, f32::max) + RIVER_WIDTH * 0.5;
        // Average bank height
        let avg_bank_h: f32 = bank_heights.iter().map(|(h0, h1)| (h0 + h1) * 0.5).sum::<f32>() / bank_heights.len().max(1) as f32;
        let water_y = avg_bank_h - 0.5;

        // For each segment, generate wave surface within the segment's river corridor
        for (_si, seg) in river_segments.iter().enumerate() {
            let dx = seg.x2 - seg.x1;
            let dz = seg.z2 - seg.z1;
            let len = (dx * dx + dz * dz).sqrt();
            if len < 0.01 { continue; }
            let perp_x = -dz / len;
            let perp_z = dx / len;
            let hw = RIVER_WIDTH * 0.5;

            // Subdivide along segment length
            let sub_count = (len / 2.0).ceil() as usize;
            let cross_count = 6_usize;
            for si in 0..sub_count {
                for ci in 0..cross_count {
                    let t0 = si as f32 / sub_count as f32;
                    let t1 = (si + 1) as f32 / sub_count as f32;
                    let c0 = ci as f32 / cross_count as f32 * 2.0 - 1.0;
                    let c1 = (ci + 1) as f32 / cross_count as f32 * 2.0 - 1.0;

                    let x00 = seg.x1 + dx * t0 + perp_x * hw * c0;
                    let z00 = seg.z1 + dz * t0 + perp_z * hw * c0;
                    let x10 = seg.x1 + dx * t1 + perp_x * hw * c0;
                    let z10 = seg.z1 + dz * t1 + perp_z * hw * c0;
                    let x01 = seg.x1 + dx * t0 + perp_x * hw * c1;
                    let z01 = seg.z1 + dz * t0 + perp_z * hw * c1;
                    let x11 = seg.x1 + dx * t1 + perp_x * hw * c1;
                    let z11 = seg.z1 + dz * t1 + perp_z * hw * c1;

                    // Multi-octave wave offset using global water level
                    let wave = |x: f32, z: f32| -> f32 {
                        water_y + (x * 0.8).sin() * (z * 0.5).cos() * 0.15
                            + (x * 1.5 + 1.0).sin() * 0.08
                            + (x * 2.3 + z * 1.7).sin() * 0.04
                    };

                    let v00 = [x00, wave(x00, z00), z00];
                    let v10 = [x10, wave(x10, z10), z10];
                    let v01 = [x01, wave(x01, z01), z01];
                    let v11 = [x11, wave(x11, z11), z11];

                    // Color varies by cross-stream position: edge→lighter, center→deeper
                    let cross_mid = (c0.abs() + c1.abs()) * 0.5; // 0=center, 1=edge
                    let depth_t = 1.0 - cross_mid; // 1=center (deep), 0=edge
                    let base = if depth_t > 0.5 {
                        lerp_color(RIVER_MID, RIVER_DEEP, (depth_t - 0.5) * 2.0)
                    } else {
                        lerp_color(RIVER_EDGE, RIVER_MID, depth_t * 2.0)
                    };
                    // Hash-based per-quad noise for ripple variation
                    let h = (si as u32).wrapping_mul(73856093) ^ (ci as u32).wrapping_mul(19349663);
                    let noise = (h % 16) as i32 - 8;
                    let color = jitter_color(base, noise);

                    // CCW winding for VK_FRONT_FACE_COUNTER_CLOCKWISE
                    let n1 = mesh::tri_normal(v00, v11, v10);
                    mesh::push_tri(tris, v00, v11, v10, n1, color);
                    let n2 = mesh::tri_normal(v00, v01, v11);
                    mesh::push_tri(tris, v00, v01, v11, n2, color);
                }
            }
        }
        let _ = (x_min, x_max, water_y); // used for bounding box reference

        // ── River bank / shoreline geometry ──────────────────────────────
        // Wider sloped bank strips on both sides of each segment. Transitions
        // from wet sand/gravel at the waterline through muddy brown to grassy
        // green at the outer edge. Bank is raised above water to avoid z-fighting.
        const BANK_WET_SAND: u32 = 0xFF5A5A44;  // wet sand/gravel at waterline
        const BANK_PEBBLE: u32 = 0xFF6B6B55;    // pebble/gravel strip
        let bank_width = 6.0_f32;  // wider banks for more visible transition
        let bank_steps = 6_usize;  // more steps for smoother gradient
        for (bsi, seg) in river_segments.iter().enumerate() {
            let sdx = seg.x2 - seg.x1;
            let sdz = seg.z2 - seg.z1;
            let slen = (sdx * sdx + sdz * sdz).sqrt();
            if slen < 0.01 { continue; }
            let perp_x = -sdz / slen;
            let perp_z = sdx / slen;
            let hw = RIVER_WIDTH * 0.5;
            let wy = (bank_heights[bsi].0 + bank_heights[bsi].1) * 0.5 - 0.5;
            let along_count = (slen / 2.0).ceil() as usize;

            for &side in &[-1.0_f32, 1.0_f32] {
                for ai in 0..along_count {
                    let t0 = ai as f32 / along_count as f32;
                    let t1 = (ai + 1) as f32 / along_count as f32;

                    for bi in 0..bank_steps {
                        let b0 = bi as f32 / bank_steps as f32;
                        let b1 = (bi + 1) as f32 / bank_steps as f32;
                        let outer_off0 = hw + bank_width * b0;
                        let outer_off1 = hw + bank_width * b1;

                        let cx00 = seg.x1 + sdx * t0 + perp_x * side * outer_off0;
                        let cz00 = seg.z1 + sdz * t0 + perp_z * side * outer_off0;
                        let cx10 = seg.x1 + sdx * t1 + perp_x * side * outer_off0;
                        let cz10 = seg.z1 + sdz * t1 + perp_z * side * outer_off0;
                        let cx01 = seg.x1 + sdx * t0 + perp_x * side * outer_off1;
                        let cz01 = seg.z1 + sdz * t0 + perp_z * side * outer_off1;
                        let cx11 = seg.x1 + sdx * t1 + perp_x * side * outer_off1;
                        let cz11 = seg.z1 + sdz * t1 + perp_z * side * outer_off1;

                        // Smoothstep height: water level at inner edge, terrain at outer
                        let slope0 = b0 * b0 * (3.0 - 2.0 * b0);
                        let slope1 = b1 * b1 * (3.0 - 2.0 * b1);
                        let ty00 = terrain.height_at(cx00, cz00);
                        let ty10 = terrain.height_at(cx10, cz10);
                        let ty01 = terrain.height_at(cx01, cz01);
                        let ty11 = terrain.height_at(cx11, cz11);

                        // Higher offset near water (0.35) to stay above wave peaks,
                        // decreasing toward terrain level at outer edge
                        let inner_lift = 0.35 * (1.0 - slope0);
                        let outer_lift = 0.35 * (1.0 - slope1);
                        let v00 = [cx00, wy + (ty00 - wy) * slope0 + inner_lift, cz00];
                        let v10 = [cx10, wy + (ty10 - wy) * slope0 + inner_lift, cz10];
                        let v01 = [cx01, wy + (ty01 - wy) * slope1 + outer_lift, cz01];
                        let v11 = [cx11, wy + (ty11 - wy) * slope1 + outer_lift, cz11];

                        // Color gradient: wet sand -> pebble -> mud -> grass
                        let color_t = (b0 + b1) * 0.5;
                        let bank_color = if color_t < 0.2 {
                            lerp_color(BANK_WET_SAND, BANK_PEBBLE, color_t * 5.0)
                        } else if color_t < 0.5 {
                            lerp_color(BANK_PEBBLE, BANK_MUD, (color_t - 0.2) * 3.333)
                        } else {
                            lerp_color(BANK_MUD, BANK_GRASS, (color_t - 0.5) * 2.0)
                        };
                        let bh = (ai as u32).wrapping_mul(73856093)
                            ^ (bi as u32).wrapping_mul(19349663)
                            ^ (bsi as u32).wrapping_mul(2654435761);
                        let bnoise = (bh % 12) as i32 - 6;
                        let color = jitter_color(bank_color, bnoise);

                        // CCW winding depends on side: mirrored grid flips winding
                        if side > 0.0 {
                            let n1 = mesh::tri_normal(v00, v11, v10);
                            mesh::push_tri(tris, v00, v11, v10, n1, color);
                            let n2 = mesh::tri_normal(v00, v01, v11);
                            mesh::push_tri(tris, v00, v01, v11, n2, color);
                        } else {
                            let n1 = mesh::tri_normal(v00, v10, v11);
                            mesh::push_tri(tris, v00, v10, v11, n1, color);
                            let n2 = mesh::tri_normal(v00, v11, v01);
                            mesh::push_tri(tris, v00, v11, v01, n2, color);
                        }
                    }
                }

                // Scatter pebbles/small rocks along the waterline for each side
                let pebble_seed_base = (bsi as u64).wrapping_mul(48611)
                    ^ if side > 0.0 { 99991 } else { 0 };
                let mut ph = pebble_seed_base;
                let peb_next = |ph: &mut u64| -> f32 {
                    *ph = ph.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((*ph >> 16) & 0xFFFF) as f32 / 65535.0
                };
                let pebble_count = (slen * 0.8) as usize;
                for _pi in 0..pebble_count {
                    let pt = peb_next(&mut ph);
                    let px = seg.x1 + sdx * pt + perp_x * side * (hw + peb_next(&mut ph) * 1.5);
                    let pz = seg.z1 + sdz * pt + perp_z * side * (hw + peb_next(&mut ph) * 1.5);
                    let py = wy + 0.38;
                    let pr = 0.05 + peb_next(&mut ph) * 0.12;
                    let peb_shade = (ph % 30) as i32 - 15;
                    let peb_color = jitter_color(BANK_PEBBLE, peb_shade);
                    // Small flat pebble — 4 tris (flattened diamond, CCW winding)
                    let pa = peb_next(&mut ph) * std::f32::consts::TAU;
                    let (ps, pc) = (pa.sin(), pa.cos());
                    let pn = [0.0_f32, 1.0, 0.0];
                    mesh::push_tri(tris, [px, py, pz], [px - ps * pr * 0.6, py, pz + pc * pr * 0.6], [px + pc * pr, py, pz + ps * pr], pn, peb_color);
                    mesh::push_tri(tris, [px, py, pz], [px - pc * pr, py, pz - ps * pr], [px - ps * pr * 0.6, py, pz + pc * pr * 0.6], pn, peb_color);
                    mesh::push_tri(tris, [px, py, pz], [px + ps * pr * 0.6, py, pz - pc * pr * 0.6], [px - pc * pr, py, pz - ps * pr], pn, peb_color);
                    mesh::push_tri(tris, [px, py, pz], [px + pc * pr, py, pz + ps * pr], [px + ps * pr * 0.6, py, pz - pc * pr * 0.6], pn, peb_color);
                }
            }
        }
    }
}

/// Generate bridges where car roads cross the river.
/// Must run AFTER river carving but BEFORE terrain mesh generation.
/// Restores heightmap under bridge so road/terrain mesh is flat across river.
fn generate_bridges(
    tris: &mut Vec<WorldTri>, terrain: &mut Terrain,
    net: &RoadNetwork, river_segments: &[RiverSegment],
    walls: &mut Vec<Wall>, bridges: &mut Vec<Bridge>,
) {
    for seg in &net.segments {
        if seg.tier != RoadTier::CarRoad { continue; }

        // Find the crossing point of this road with the river
        let Some((perp_x, perp_z, dir_x, dir_z, len)) = segment_perp(seg) else { continue };
        if len < 5.0 { continue; }

        let mut cross_x = 0.0f32;
        let mut cross_z = 0.0f32;
        let mut crosses = false;
        let sample_count = (len / 2.0).ceil() as i32;
        for s in 0..=sample_count {
            let t = s as f32 / sample_count as f32;
            let (px, pz) = sample_segment(seg, t);
            for rseg in river_segments {
                let d = point_to_segment_dist(px, pz, rseg.x1, rseg.z1, rseg.x2, rseg.z2);
                if d < RIVER_WIDTH * 0.5 {
                    cross_x = px;
                    cross_z = pz;
                    crosses = true;
                    break;
                }
            }
            if crosses { break; }
        }
        if !crosses { continue; }

        // Bridge center at crossing point
        let cx = cross_x;
        let cz = cross_z;

        // Find deck height: sample bank heights at bridge edges (just outside river)
        let edge_dist = RIVER_WIDTH * 0.5 + 3.0;
        let bank0_x = cx - dir_x * edge_dist;
        let bank0_z = cz - dir_z * edge_dist;
        let bank1_x = cx + dir_x * edge_dist;
        let bank1_z = cz + dir_z * edge_dist;
        let bank_h0 = terrain.height_at(bank0_x, bank0_z);
        let bank_h1 = terrain.height_at(bank1_x, bank1_z);
        let road_h = bank_h0.max(bank_h1);
        let deck_y = road_h + 0.1; // bridge slightly above road surface

        let bridge_hw = CAR_ROAD_WIDTH * 0.5 + 0.5;
        // Bridge spans river width + 10m (5m clearance each side)
        let bridge_len = RIVER_WIDTH + 10.0;

        // Restore heightmap under bridge FIRST (before terrain mesh is generated)
        let grid = terrain.grid;
        let stride = grid + 1;
        let cell = terrain.cell_size;
        for iz in 0..stride {
            for ix in 0..stride {
                let wx = -WORLD_HALF + ix as f32 * cell;
                let wz = -WORLD_HALF + iz as f32 * cell;
                let to_x = wx - cx;
                let to_z = wz - cz;
                let along = to_x * dir_x + to_z * dir_z;
                let across = (to_x * perp_x + to_z * perp_z).abs();
                if along.abs() < bridge_len * 0.5 && across < bridge_hw {
                    let h = &mut terrain.heights[iz * stride + ix];
                    if *h < road_h {
                        *h = road_h;
                    }
                }
            }
        }

        // Beveled deck (thick enough to avoid terrain z-fighting)
        mesh::beveled_box_tris(tris, cx, deck_y - 0.2, cz, bridge_hw * 2.0, 0.4, bridge_len, 0.05, BRIDGE_DECK_COLOR);

        // Girder under deck
        mesh::box_tris(tris, cx, deck_y - 0.5, cz, bridge_hw * 1.5, 0.2, bridge_len, darken(BRIDGE_DECK_COLOR, 0.7));

        // Cylinder pillar supports
        let pillar_count = (bridge_len / 8.0).ceil() as i32;
        for pi in 0..pillar_count {
            let t = (pi as f32 + 0.5) / pillar_count as f32;
            let px = cx + dir_x * (t - 0.5) * bridge_len;
            let pz = cz + dir_z * (t - 0.5) * bridge_len;
            let base_y = terrain.height_at(px, pz) - RIVER_DEPTH;
            let pillar_h = deck_y - base_y;
            if pillar_h > 0.5 {
                mesh::cylinder_tris(tris, px, base_y + pillar_h * 0.5, pz, 0.25, pillar_h, 6, BRIDGE_RAIL_COLOR);
            }
        }

        // Railing posts + rail bars
        let rail_x_l = cx + perp_x * bridge_hw;
        let rail_z_l = cz + perp_z * bridge_hw;
        let rail_x_r = cx - perp_x * bridge_hw;
        let rail_z_r = cz - perp_z * bridge_hw;
        // Rail bars (cylinder between endpoints, 6 segments for visibility from all angles)
        let rail_start_l = [rail_x_l - dir_x * bridge_len * 0.5, deck_y + 0.8, rail_z_l - dir_z * bridge_len * 0.5];
        let rail_end_l = [rail_x_l + dir_x * bridge_len * 0.5, deck_y + 0.8, rail_z_l + dir_z * bridge_len * 0.5];
        mesh::cylinder_between(tris, rail_start_l, rail_end_l, 0.06, 6, BRIDGE_RAIL_COLOR);
        let rail_start_r = [rail_x_r - dir_x * bridge_len * 0.5, deck_y + 0.8, rail_z_r - dir_z * bridge_len * 0.5];
        let rail_end_r = [rail_x_r + dir_x * bridge_len * 0.5, deck_y + 0.8, rail_z_r + dir_z * bridge_len * 0.5];
        mesh::cylinder_between(tris, rail_start_r, rail_end_r, 0.06, 6, BRIDGE_RAIL_COLOR);
        // Lower rail bar for additional visibility
        let lower_start_l = [rail_x_l - dir_x * bridge_len * 0.5, deck_y + 0.4, rail_z_l - dir_z * bridge_len * 0.5];
        let lower_end_l = [rail_x_l + dir_x * bridge_len * 0.5, deck_y + 0.4, rail_z_l + dir_z * bridge_len * 0.5];
        mesh::cylinder_between(tris, lower_start_l, lower_end_l, 0.04, 6, BRIDGE_RAIL_COLOR);
        let lower_start_r = [rail_x_r - dir_x * bridge_len * 0.5, deck_y + 0.4, rail_z_r - dir_z * bridge_len * 0.5];
        let lower_end_r = [rail_x_r + dir_x * bridge_len * 0.5, deck_y + 0.4, rail_z_r + dir_z * bridge_len * 0.5];
        mesh::cylinder_between(tris, lower_start_r, lower_end_r, 0.04, 6, BRIDGE_RAIL_COLOR);
        // Railing posts (vertical cylinders, 6 segments for all-angle visibility)
        let post_count = (bridge_len / 3.0).ceil() as i32;
        for pi in 0..post_count {
            let t = (pi as f32 + 0.5) / post_count as f32 - 0.5;
            let pxl = rail_x_l + dir_x * t * bridge_len;
            let pzl = rail_z_l + dir_z * t * bridge_len;
            mesh::cylinder_tris(tris, pxl, deck_y + 0.4, pzl, 0.04, 0.8, 6, BRIDGE_RAIL_COLOR);
            let pxr = rail_x_r + dir_x * t * bridge_len;
            let pzr = rail_z_r + dir_z * t * bridge_len;
            mesh::cylinder_tris(tris, pxr, deck_y + 0.4, pzr, 0.04, 0.8, 6, BRIDGE_RAIL_COLOR);
        }

        // Railing walls for collision
        walls.push(Wall { x: rail_x_l, z: rail_z_l, hw: 0.15, hd: bridge_len * 0.5, height: 1.0 });
        walls.push(Wall { x: rail_x_r, z: rail_z_r, hw: 0.15, hd: bridge_len * 0.5, height: 1.0 });

        // Store bridge zone for on_river exemption
        bridges.push(Bridge { cx, cz, dir_x, dir_z, hw: bridge_hw, hl: bridge_len * 0.5 });
    }
}

/// Generate 4 parking lots near ring-1 intersections
fn generate_parking_lots(
    tris: &mut Vec<WorldTri>, terrain: &Terrain, rng: &mut Rng,
    nodes: &[[f32; 2]], buildings: &[Building],
    parking_spots: &mut Vec<ParkingSpot>,
    walls: &mut Vec<Wall>,
    trees: &mut Vec<Tree>,
    street_lights: &mut Vec<StreetLight>,
    net: &RoadNetwork, river_segments: &[RiverSegment],
) {
    // Use ring-1 nodes (indices 1..5) for lot placement
    let ring1_start = 1;
    let ring1_end = (ring1_start + PARKING_LOT_COUNT).min(nodes.len());

    for ni in ring1_start..ring1_end {
        let node = &nodes[ni];

        // Try several angles to find a placement that avoids roads/river/buildings
        let mut lot_cx = 0.0f32;
        let mut lot_cz = 0.0f32;
        let mut found = false;
        for attempt in 0..8 {
            let angle = (ni as f32 + attempt as f32 * 0.7) * 1.5;
            let radius = 18.0 + rng.range(0.0, 8.0);
            let cx = node[0] + angle.cos() * radius;
            let cz = node[1] + angle.sin() * radius;

            // Check road overlap (sample corners + center of lot area)
            let on_road = on_any_road(cx, cz, net)
                || on_any_road(cx - 7.0, cz - 14.0, net)
                || on_any_road(cx + 7.0, cz + 14.0, net);
            if on_road { continue; }

            // Check river (center + all 4 corners of lot area)
            if on_river(cx, cz, river_segments)
                || on_river(cx - 7.0, cz - 14.0, river_segments)
                || on_river(cx + 7.0, cz - 14.0, river_segments)
                || on_river(cx - 7.0, cz + 14.0, river_segments)
                || on_river(cx + 7.0, cz + 14.0, river_segments) { continue; }

            // Check building overlap
            let overlaps = buildings.iter().any(|b| {
                (cx - b.x).abs() < 12.0 + b.w * 0.5 && (cz - b.z).abs() < 18.0 + b.d * 0.5
            });
            if overlaps { continue; }

            lot_cx = cx;
            lot_cz = cz;
            found = true;
            break;
        }
        if !found { continue; }

        let lot_w = 15.0;
        let lot_d = 30.0;
        let lot_hw = lot_w * 0.5;
        let lot_hd = lot_d * 0.5;
        let gy = terrain.height_at(lot_cx, lot_cz);

        // Asphalt surface
        generate_road_strip(tris, terrain,
            lot_cx - lot_hw, lot_cz - lot_hd,
            lot_cx - lot_hw, lot_cz + lot_hd,
            lot_hw, 0.03, PARKING_ASPHALT);
        generate_road_strip(tris, terrain,
            lot_cx + lot_hw, lot_cz - lot_hd,
            lot_cx + lot_hw, lot_cz + lot_hd,
            lot_hw, 0.03, PARKING_ASPHALT);

        // Parking spots in 2 rows
        let spot_count = 6; // per row
        let spot_spacing = (lot_d - 4.0) / spot_count as f32;
        for row in 0..2 {
            let row_x = lot_cx + (if row == 0 { -lot_hw * 0.5 } else { lot_hw * 0.5 });
            let rot = if row == 0 { 0.0 } else { std::f32::consts::PI };
            for k in 0..spot_count {
                let spot_z = lot_cz - lot_hd + 2.0 + (k as f32 + 0.5) * spot_spacing;
                parking_spots.push(ParkingSpot {
                    x: row_x, z: spot_z, rot_y: rot, occupied_by: None,
                });
                // Raised line markings (no z-fighting with asphalt)
                let line_y = gy + 0.04;
                let lz0 = spot_z - spot_spacing * 0.45;
                let lz1 = spot_z + spot_spacing * 0.45;
                let lx = row_x - PARKING_SPOT_WIDTH * 0.5;
                mesh::raised_strip_tris(tris,
                    &[[lx, line_y, lz0], [lx, line_y, lz1]],
                    0.1, 0.02, PARKING_LINE_COLOR);
            }
        }

        // Perimeter fence (3 sides, road-adjacent side open) — cylinder posts + rail
        let fence_h = 1.5;
        // Back fence — posts
        let back_post_count = (lot_w / 2.0) as usize;
        for fp in 0..=back_post_count {
            let t = fp as f32 / back_post_count as f32;
            let fx = lot_cx - lot_hw + t * lot_w;
            let fz = lot_cz - lot_hd;
            mesh::cylinder_tris(tris, fx, gy + fence_h * 0.5, fz, 0.04, fence_h, 4, FENCE_COLOR);
        }
        // Back fence rail
        mesh::cylinder_between(tris,
            [lot_cx - lot_hw, gy + fence_h * 0.8, lot_cz - lot_hd],
            [lot_cx + lot_hw, gy + fence_h * 0.8, lot_cz - lot_hd],
            0.03, 4, FENCE_COLOR);
        walls.push(Wall { x: lot_cx, z: lot_cz - lot_hd, hw: lot_hw, hd: 0.15, height: fence_h });

        // Side fences — posts + rail
        for side_x in [-1.0f32, 1.0] {
            let fx = lot_cx + side_x * lot_hw;
            let side_post_count = (lot_d / 2.0) as usize;
            for fp in 0..=side_post_count {
                let t = fp as f32 / side_post_count as f32;
                let fz = lot_cz - lot_hd + t * lot_d;
                mesh::cylinder_tris(tris, fx, gy + fence_h * 0.5, fz, 0.04, fence_h, 4, FENCE_COLOR);
            }
            mesh::cylinder_between(tris,
                [fx, gy + fence_h * 0.8, lot_cz - lot_hd],
                [fx, gy + fence_h * 0.8, lot_cz + lot_hd],
                0.03, 4, FENCE_COLOR);
            walls.push(Wall { x: fx, z: lot_cz, hw: 0.15, hd: lot_hd, height: fence_h });
        }

        // Corner trees — bark trunk + leaf canopy
        const LOT_LEAF_COLORS: [u32; 6] = [
            0xFF2D7A2D, 0xFF338833, 0xFF228822, 0xFF448844, 0xFF2A6B2A, 0xFF3A8A3A,
        ];
        for (ci, corner) in [(-1.0f32, -1.0f32), (1.0, -1.0)].iter().enumerate() {
            let tx = lot_cx + corner.0 * (lot_hw - 1.0);
            let tz = lot_cz + corner.1 * (lot_hd - 1.0);
            let tgy = terrain.height_at(tx, tz);
            mesh::bark_cylinder_tris(tris, tx, tgy + 1.5, tz, 0.12, 3.0, 8,
                0.015, (ni * 100 + ci) as u64, TRUNK_COLOR, 0xFF332211);
            // 2 leaf clusters instead of single sphere
            for lci in 0..2 {
                let la = (lci as f32 / 2.0) * std::f32::consts::TAU + ci as f32;
                let lcx = tx + la.cos() * 0.4;
                let lcz = tz + la.sin() * 0.4;
                mesh::leaf_canopy_tris(tris, lcx, tgy + 3.5 + lci as f32 * 0.3, lcz,
                    0.9, 40, 0.07, (ni * 200 + ci * 10 + lci) as u64, &LOT_LEAF_COLORS);
            }
            trees.push(Tree { x: tx, z: tz, trunk_radius: 0.3 });
        }

        // Corner lights — base plate + tapered base + cylinder pole + sphere globe
        let lx = lot_cx + lot_hw - 0.5;
        let lz = lot_cz + lot_hd - 0.5;
        let lgy = terrain.height_at(lx, lz);
        // Base mounting plate
        let lbase_y = lgy + 0.04;
        for pi in 0..8u32 {
            let a0 = (pi as f32 / 8.0) * std::f32::consts::TAU;
            let a1 = ((pi + 1) as f32 / 8.0) * std::f32::consts::TAU;
            mesh::push_tri(tris, [lx, lbase_y, lz], [lx + a1.cos() * 0.25, lbase_y, lz + a1.sin() * 0.25], [lx + a0.cos() * 0.25, lbase_y, lz + a0.sin() * 0.25], [0.0, 1.0, 0.0], LAMP_BASE_COLOR);
        }
        // Wider base section
        mesh::cylinder_tris(tris, lx, lgy + 0.2, lz, 0.12, 0.4, 6, LAMP_BASE_COLOR);
        // Decorative collar ring where base meets pole
        mesh::ring_tris(tris, lx, lgy + 0.42, lz, 0.10, 0.03, 8, 4, LAMP_BASE_COLOR);
        // Main pole (8 segments for rounder appearance)
        mesh::cylinder_tris(tris, lx, lgy + 2.7, lz, 0.06, 4.6, 8, LAMP_POLE_COLOR);
        mesh::sphere_tris(tris, lx, lgy + 5.2, lz, 0.2, 1, LAMP_GLOW_COLOR);
        // (glow halo removed — too large, creates dark umbrella shapes in daylight)
        // Ground light pool
        let pool_y = lgy + 0.03;
        let pool_color: u32 = 0x00553810;
        for pi in 0..8u32 {
            let a0 = (pi as f32 / 8.0) * std::f32::consts::TAU;
            let a1 = ((pi + 1) as f32 / 8.0) * std::f32::consts::TAU;
            mesh::push_tri(tris, [lx, pool_y, lz], [lx + a1.cos() * 3.0, pool_y, lz + a1.sin() * 3.0], [lx + a0.cos() * 3.0, pool_y, lz + a0.sin() * 3.0], [0.0, 1.0, 0.0], pool_color);
        }
        street_lights.push(StreetLight { x: lx, z: lz, ground_y: lgy });
    }
}

/// Generate 6-8 market stalls near town center
fn generate_market_stalls(
    tris: &mut Vec<WorldTri>, terrain: &Terrain, rng: &mut Rng,
    net: &RoadNetwork, buildings: &[Building], walls: &mut Vec<Wall>,
    settlement_center: [f32; 2], density_scale: f32,
) {
    let stall_count = ((6 + rng.next() as usize % 3) as f32 * density_scale) as usize;
    let angle_step = std::f32::consts::TAU / stall_count as f32;

    for i in 0..stall_count {
        let angle = i as f32 * angle_step + rng.range(-0.3, 0.3);
        let radius = rng.range(30.0, 50.0);
        let sx = settlement_center[0] + angle.cos() * radius;
        let sz = settlement_center[1] + angle.sin() * radius;

        // Skip if overlapping buildings
        if overlaps_building(sx, sz, 2.0, buildings) { continue; }
        if on_road_surface(sx, sz, net) { continue; }

        let gy = terrain.height_at(sx, sz);
        let canvas_color = STALL_CANVAS_COLORS[i % 4];

        // Wooden frame (4 cylinder posts)
        let sw = 3.0;
        let sd = 2.0;
        let sh = 2.5;
        for dx in [-1.0f32, 1.0] {
            for dz in [-1.0f32, 1.0] {
                mesh::cylinder_tris(tris, sx + dx * sw * 0.45, gy + sh * 0.5, sz + dz * sd * 0.45,
                    0.04, sh, 4, STALL_FRAME_COLOR);
            }
        }

        // Canvas roof (angled slightly — quad, not coplanar with anything)
        let roof_y = gy + sh;
        let v0 = [sx - sw * 0.5, roof_y + 0.3, sz - sd * 0.5];
        let v1 = [sx + sw * 0.5, roof_y + 0.3, sz - sd * 0.5];
        let v2 = [sx + sw * 0.5, roof_y - 0.1, sz + sd * 0.5];
        let v3 = [sx - sw * 0.5, roof_y - 0.1, sz + sd * 0.5];
        mesh::push_quad(tris, v0, v3, v2, v1, canvas_color);

        // Counter front — beveled
        mesh::beveled_box_tris(tris, sx, gy + 0.5, sz - sd * 0.5 + 0.1, sw * 0.9, 1.0, 0.2, 0.03, STALL_COUNTER_COLOR);
        walls.push(Wall { x: sx, z: sz - sd * 0.5 + 0.1, hw: sw * 0.45, hd: 0.15, height: 1.0 });
    }
}

/// Generate 4-6 bus stops along major road segments
fn generate_bus_stops(
    tris: &mut Vec<WorldTri>, terrain: &Terrain, rng: &mut Rng,
    net: &RoadNetwork, buildings: &[Building], walls: &mut Vec<Wall>,
    density_scale: f32,
) {
    let car_segs: Vec<&RoadSegment> = net.segments.iter()
        .filter(|s| s.tier == RoadTier::CarRoad).collect();
    if car_segs.is_empty() { return; }

    let stop_count = ((4 + rng.next() as usize % 3) as f32 * density_scale) as usize;
    for i in 0..stop_count {
        let seg = car_segs[i % car_segs.len()];
        let t = rng.range(0.3, 0.7);
        let (sx, sz) = sample_segment(seg, t);

        let Some((perp_x, perp_z, _dir_x, _dir_z, _len)) = segment_perp(seg) else { continue };
        let offset = CAR_ROAD_WIDTH * 0.5 + SIDEWALK_WIDTH + 1.5;
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let bx = sx + perp_x * offset * side;
        let bz = sz + perp_z * offset * side;

        // Skip if overlapping buildings
        if overlaps_building(bx, bz, 2.0, buildings) { continue; }

        let gy = terrain.height_at(bx, bz);

        // Shelter: 3 glass walls (beveled) + roof
        let shelter_w = 2.5;
        let shelter_d = 1.5;
        let shelter_h = 2.5;

        // Back wall — beveled
        mesh::beveled_box_tris(tris, bx, gy + shelter_h * 0.5, bz - shelter_d * 0.5,
            shelter_w, shelter_h, 0.1, 0.02, BUS_GLASS_COLOR);
        // Left wall
        mesh::beveled_box_tris(tris, bx - shelter_w * 0.5, gy + shelter_h * 0.5, bz,
            0.1, shelter_h, shelter_d, 0.02, BUS_GLASS_COLOR);
        // Right wall
        mesh::beveled_box_tris(tris, bx + shelter_w * 0.5, gy + shelter_h * 0.5, bz,
            0.1, shelter_h, shelter_d, 0.02, BUS_GLASS_COLOR);
        // Roof — beveled
        mesh::beveled_box_tris(tris, bx, gy + shelter_h + 0.05, bz, shelter_w + 0.3, 0.1, shelter_d + 0.3, 0.02, BUS_ROOF_COLOR);

        // Walls for collision
        walls.push(Wall { x: bx, z: bz - shelter_d * 0.5, hw: shelter_w * 0.5, hd: 0.1, height: shelter_h });
        walls.push(Wall { x: bx - shelter_w * 0.5, z: bz, hw: 0.1, hd: shelter_d * 0.5, height: shelter_h });
        walls.push(Wall { x: bx + shelter_w * 0.5, z: bz, hw: 0.1, hd: shelter_d * 0.5, height: shelter_h });

        // Bench — slatted
        for si in 0..3 {
            let bsz = bz - 0.15 + si as f32 * 0.15;
            mesh::box_tris(tris, bx, gy + 0.25, bsz, 1.5, 0.04, 0.1, DECO_BENCH_COLOR);
        }

        // Sign post — cylinder + sign plate
        let sign_x = bx + shelter_w * 0.5 + 0.5;
        mesh::cylinder_tris(tris, sign_x, gy + 1.5, bz, 0.04, 3.0, 4, LAMP_POLE_COLOR);
        mesh::beveled_box_tris(tris, sign_x, gy + 3.1, bz, 0.4, 0.4, 0.08, 0.02, BUS_SIGN_COLOR);
    }
}

/// Generate decorative objects scattered throughout town area
fn generate_decorations(
    tris: &mut Vec<WorldTri>, terrain: &Terrain, rng: &mut Rng,
    net: &RoadNetwork, buildings: &[Building],
    walls: &mut Vec<Wall>, rocks: &mut Vec<Rock>,
    _street_lights: &mut Vec<StreetLight>,
    clutter: &mut Vec<[f32; 3]>,
    settlement_center: [f32; 2], density_scale: f32,
) {
    let car_segs: Vec<&RoadSegment> = net.segments.iter()
        .filter(|s| s.tier == RoadTier::CarRoad).collect();

    // Helper: random position near town center
    let town_pos = |rng: &mut Rng| -> (f32, f32) {
        let angle = rng.range(0.0, std::f32::consts::TAU);
        let radius = rng.range(10.0, 150.0);
        (settlement_center[0] + angle.cos() * radius, settlement_center[1] + angle.sin() * radius)
    };

    // Bollards (25) — cylinder posts
    for _ in 0..(25.0 * density_scale) as usize {
        let (bx, bz) = town_pos(rng);
        if on_any_road(bx, bz, net) { continue; }
        if overlaps_building(bx, bz, 0.5, buildings) { continue; }
        let gy = terrain.height_at(bx, bz);
        mesh::cylinder_tris(tris, bx, gy + 0.25, bz, 0.08, 0.5, 6, BOLLARD_COLOR);
        // Rounded top
        mesh::sphere_tris(tris, bx, gy + 0.5, bz, 0.08, 0, BOLLARD_COLOR);
        rocks.push(Rock { x: bx, z: bz, size: 0.15 });
    }

    // Planters (12) — beveled box + sphere shrub
    for _ in 0..(12.0 * density_scale) as usize {
        let (px, pz) = town_pos(rng);
        if on_any_road(px, pz, net) { continue; }
        if overlaps_building(px, pz, 0.5, buildings) { continue; }
        let gy = terrain.height_at(px, pz);
        mesh::beveled_box_tris(tris, px, gy + 0.2, pz, 0.5, 0.4, 0.5, 0.04, PLANTER_BOX_COLOR);
        mesh::sphere_tris(tris, px, gy + 0.55, pz, 0.3, 1, PLANTER_GREEN_COLOR);
        walls.push(Wall { x: px, z: pz, hw: 0.25, hd: 0.25, height: 0.4 });
    }

    // Picnic tables (5) — box top + benches + cylinder legs
    for _ in 0..(5.0 * density_scale) as usize {
        let (px, pz) = town_pos(rng);
        if on_any_road(px, pz, net) { continue; }
        if overlaps_building(px, pz, 1.5, buildings) { continue; }
        let gy = terrain.height_at(px, pz);
        // Table top
        mesh::box_tris(tris, px, gy + 0.75, pz, 1.8, 0.08, 0.9, PICNIC_TABLE_COLOR);
        // Two bench slabs
        mesh::box_tris(tris, px, gy + 0.3, pz - 0.7, 1.8, 0.06, 0.25, PICNIC_TABLE_COLOR);
        mesh::box_tris(tris, px, gy + 0.3, pz + 0.7, 1.8, 0.06, 0.25, PICNIC_TABLE_COLOR);
        // Legs — cylinders
        for lx in [-0.7f32, 0.7] {
            mesh::cylinder_tris(tris, px + lx, gy + 0.375, pz, 0.03, 0.75, 4, PICNIC_TABLE_COLOR);
        }
        walls.push(Wall { x: px, z: pz, hw: 0.9, hd: 0.8, height: 0.8 });
    }

    // Billboards (3) — cylinder post + beveled panel
    for _ in 0..(3.0 * density_scale) as usize {
        let (bx, bz) = town_pos(rng);
        if on_any_road(bx, bz, net) { continue; }
        if overlaps_building(bx, bz, 1.0, buildings) { continue; }
        let gy = terrain.height_at(bx, bz);
        // Cylinder post
        mesh::cylinder_tris(tris, bx, gy + 2.5, bz, 0.12, 5.0, 6, BILLBOARD_POST_COLOR);
        // Beveled panel
        mesh::beveled_box_tris(tris, bx, gy + 5.5, bz, 3.0, 2.0, 0.5, 0.03, BILLBOARD_PANEL_COLOR);
        walls.push(Wall { x: bx, z: bz, hw: 0.2, hd: 0.2, height: 5.0 });
    }

    // Water towers (2) — cylinder legs + sphere tank
    for i in 0..(2.0 * density_scale) as usize {
        let wx = settlement_center[0] - 30.0 + i as f32 * 60.0;
        let wz = settlement_center[1] + 60.0;
        let gy = terrain.height_at(wx, wz);
        // Cylinder legs (4)
        for (lx, lz) in [(-0.5f32, -0.5f32), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
            mesh::cylinder_tris(tris, wx + lx, gy + 1.5, wz + lz, 0.08, 3.0, 5, WATER_TOWER_COLOR);
        }
        // Tank — sphere
        mesh::sphere_tris(tris, wx, gy + 4.0, wz, 1.5, 1, WATER_TOWER_COLOR);
        walls.push(Wall { x: wx, z: wz, hw: 0.8, hd: 0.8, height: 4.0 });
    }

    // Traffic cones (18) — proper cones
    for _ in 0..(18.0 * density_scale) as usize {
        let (cx, cz) = town_pos(rng);
        let gy = terrain.height_at(cx, cz);
        mesh::cone_tris(tris, cx, gy + 0.2, cz, 0.1, 0.35, 6, TRAFFIC_CONE_COLOR);
    }

    // Benches (10) — slatted seat + back + cylinder legs
    for _ in 0..(10.0 * density_scale) as usize {
        if car_segs.is_empty() { break; }
        let seg = car_segs[rng.next() as usize % car_segs.len()];
        let t = rng.range(0.2, 0.8);
        let (sx, sz) = sample_segment(seg, t);
        let Some((perp_x, perp_z, _dir_x, _dir_z, _len)) = segment_perp(seg) else { continue };
        let offset = CAR_ROAD_WIDTH * 0.5 + SIDEWALK_WIDTH + 0.8;
        let side = if rng.next() % 2 == 0 { 1.0 } else { -1.0 };
        let bx = sx + perp_x * offset * side;
        let bz = sz + perp_z * offset * side;

        if overlaps_building(bx, bz, 0.5, buildings) { continue; }

        let gy = terrain.height_at(bx, bz);
        // Seat slats (3)
        for si in 0..3 {
            let bsz = bz - 0.12 + si as f32 * 0.12;
            mesh::box_tris(tris, bx, gy + 0.25, bsz, 1.2, 0.04, 0.08, DECO_BENCH_COLOR);
        }
        // Back slats (2)
        for si in 0..2 {
            let bh = 0.38 + si as f32 * 0.17;
            mesh::box_tris(tris, bx, gy + bh, bz + 0.2, 1.2, 0.04, 0.04, DECO_BENCH_COLOR);
        }
        // Legs (cylinder)
        for lx in [-0.5f32, 0.5] {
            mesh::cylinder_tris(tris, bx + lx, gy + 0.125, bz, 0.02, 0.25, 4, 0xFF554422);
        }
        walls.push(Wall { x: bx, z: bz, hw: 0.6, hd: 0.25, height: 0.6 });
    }

    // --- ACU-style street clutter ---

    // Wooden barrels (30) — cylinder with iron rim bands
    for _ in 0..(30.0 * density_scale) as usize {
        let (bx, bz) = town_pos(rng);
        if on_any_road(bx, bz, net) { continue; }
        if overlaps_building(bx, bz, 0.5, buildings) { continue; }
        let gy = terrain.height_at(bx, bz);
        let barrel_h = rng.range(0.6, 0.9);
        let barrel_r = rng.range(0.2, 0.3);
        mesh::cylinder_tris(tris, bx, gy + barrel_h * 0.5, bz, barrel_r, barrel_h, 16, 0xFF664422);
        mesh::cylinder_tris(tris, bx, gy + barrel_h * 0.15, bz, barrel_r + 0.01, 0.04, 16, 0xFF444444);
        mesh::cylinder_tris(tris, bx, gy + barrel_h * 0.85, bz, barrel_r + 0.01, 0.04, 16, 0xFF444444);
        mesh::cylinder_tris(tris, bx, gy + barrel_h + 0.01, bz, barrel_r - 0.02, 0.02, 16, 0xFF775533);
        clutter.push([bx, bz, barrel_r]);
    }

    // Wooden crates (25) — with cross-braces
    for _ in 0..(25.0 * density_scale) as usize {
        let (cx, cz) = town_pos(rng);
        if on_any_road(cx, cz, net) { continue; }
        if overlaps_building(cx, cz, 0.5, buildings) { continue; }
        let gy = terrain.height_at(cx, cz);
        let crate_s = rng.range(0.3, 0.5);
        mesh::box_tris(tris, cx, gy + crate_s * 0.5, cz, crate_s, crate_s, crate_s, 0xFF886644);
        mesh::box_tris(tris, cx, gy + crate_s * 0.5, cz + crate_s * 0.5 + 0.01,
            crate_s * 0.8, 0.03, 0.02, 0xFF553311);
        if rng.next() % 3 == 0 {
            let s2 = crate_s * 0.85;
            mesh::box_tris(tris, cx + 0.05, gy + crate_s + s2 * 0.5, cz - 0.03,
                s2, s2, s2, 0xFF775533);
        }
        clutter.push([cx, cz, crate_s * 0.5]);
    }

    // Sacks/grain bags (20)
    for _ in 0..(20.0 * density_scale) as usize {
        let (sx, sz) = town_pos(rng);
        if on_any_road(sx, sz, net) { continue; }
        if overlaps_building(sx, sz, 0.5, buildings) { continue; }
        let gy = terrain.height_at(sx, sz);
        let sack_r = rng.range(0.15, 0.25);
        mesh::perturbed_sphere_tris(tris, sx, gy + sack_r * 0.7, sz,
            sack_r, 0, 0.15, rng.next() as u64, 0xFF998866);
        clutter.push([sx, sz, sack_r]);
    }

    // Wooden planks/debris on streets (15 clusters)
    for _ in 0..(15.0 * density_scale) as usize {
        if car_segs.is_empty() { break; }
        let seg = car_segs[rng.next() as usize % car_segs.len()];
        let t = rng.range(0.2, 0.8);
        let (px, pz) = sample_segment(seg, t);
        let gy = terrain.height_at(px, pz);
        let num_planks = 2 + (rng.next() % 3) as i32;
        for _ in 0..num_planks {
            let ox = rng.range(-1.5, 1.5);
            let oz = rng.range(-1.5, 1.5);
            mesh::box_tris(tris, px + ox, gy + 0.02, pz + oz,
                rng.range(0.6, 1.5), 0.03, 0.12, 0xFF775533);
        }
    }

    // Hanging bunting/pennant flags between buildings (8)
    for _ in 0..(8.0 * density_scale) as usize {
        if buildings.len() < 2 { break; }
        let b1_idx = rng.next() as usize % buildings.len();
        let b1 = &buildings[b1_idx];
        let mut best_dist = f32::MAX;
        let mut b2_idx = 0;
        for (i, b2) in buildings.iter().enumerate() {
            if i == b1_idx { continue; }
            let ddx = b1.x - b2.x;
            let ddz = b1.z - b2.z;
            let dist = (ddx * ddx + ddz * ddz).sqrt();
            if dist < best_dist && dist < 15.0 && dist > 4.0 {
                best_dist = dist;
                b2_idx = i;
            }
        }
        if best_dist > 15.0 { continue; }
        let b2 = &buildings[b2_idx];
        let flag_y = b1.ground_y.max(b2.ground_y) + 4.0;
        let bunting_color = AWNING_COLORS[rng.next() as usize % AWNING_COLORS.len()];
        let num_flags = ((best_dist / 1.2) as i32).max(3).min(10);
        for fi in 0..num_flags {
            let ft = (fi as f32 + 0.5) / num_flags as f32;
            let fx = b1.x + (b2.x - b1.x) * ft;
            let fz = b1.z + (b2.z - b1.z) * ft;
            let sag = (ft - 0.5).abs() * 2.0 - 1.0;
            mesh::box_tris(tris, fx, flag_y + sag * 0.5, fz, 0.3, 0.25, 0.02, bunting_color);
        }
        mesh::cylinder_between(tris,
            [b1.x, flag_y, b1.z], [b2.x, flag_y, b2.z],
            0.01, 4, 0xFF443322);
    }

    // Hanging laundry lines (5)
    for _ in 0..(5.0 * density_scale) as usize {
        if buildings.len() < 2 { break; }
        let b1_idx = rng.next() as usize % buildings.len();
        let b1 = &buildings[b1_idx];
        let mut best_dist = f32::MAX;
        let mut b2_idx = 0;
        for (i, b2) in buildings.iter().enumerate() {
            if i == b1_idx { continue; }
            let ddx = b1.x - b2.x;
            let ddz = b1.z - b2.z;
            let dist = (ddx * ddx + ddz * ddz).sqrt();
            if dist < best_dist && dist < 10.0 && dist > 3.0 {
                best_dist = dist;
                b2_idx = i;
            }
        }
        if best_dist > 10.0 { continue; }
        let b2 = &buildings[b2_idx];
        let line_y = b1.ground_y.max(b2.ground_y) + 5.5;
        mesh::cylinder_between(tris,
            [b1.x, line_y, b1.z], [b2.x, line_y, b2.z],
            0.008, 4, 0xFF886655);
        let num_clothes = ((best_dist / 1.5) as i32).max(2).min(5);
        for ci in 0..num_clothes {
            let ct = (ci as f32 + 0.5) / num_clothes as f32;
            let clx = b1.x + (b2.x - b1.x) * ct;
            let clz = b1.z + (b2.z - b1.z) * ct;
            let cloth_color = match rng.next() % 4 {
                0 => 0xFFCCCCCC, 1 => 0xFF8888AA, 2 => 0xFFAA8866, _ => 0xFFBBBB99,
            };
            mesh::box_tris(tris, clx, line_y - 0.3, clz, 0.3, 0.4, 0.02, cloth_color);
        }
    }

    // Wooden handcarts (3)
    for _ in 0..(3.0 * density_scale) as usize {
        let cx = settlement_center[0] + rng.range(-30.0, 30.0);
        let cz = settlement_center[1] + rng.range(-30.0, 30.0);
        if on_any_road(cx, cz, net) { continue; }
        if overlaps_building(cx, cz, 1.5, buildings) { continue; }
        let gy = terrain.height_at(cx, cz);
        mesh::box_tris(tris, cx, gy + 0.5, cz, 1.8, 0.08, 1.0, 0xFF775533);
        mesh::box_tris(tris, cx, gy + 0.65, cz - 0.48, 1.8, 0.22, 0.04, 0xFF664422);
        mesh::box_tris(tris, cx, gy + 0.65, cz + 0.48, 1.8, 0.22, 0.04, 0xFF664422);
        for wside in [-0.55f32, 0.55] {
            mesh::cylinder_tris(tris, cx - 0.6, gy + 0.3, cz + wside, 0.3, 0.06, 8, 0xFF443322);
        }
        mesh::cylinder_between(tris,
            [cx + 0.9, gy + 0.55, cz - 0.3], [cx + 1.6, gy + 0.8, cz - 0.3],
            0.025, 4, 0xFF664422);
        mesh::cylinder_between(tris,
            [cx + 0.9, gy + 0.55, cz + 0.3], [cx + 1.6, gy + 0.8, cz + 0.3],
            0.025, 4, 0xFF664422);
        clutter.push([cx, cz, 0.9]); // half-length of cart as radius
    }
}

// Suburban colors
const SUBURB_HOUSE_COLORS: [u32; 6] = [
    0xFF99887A, 0xFF8888AA, 0xFFAA9988, 0xFF889988, 0xFFBBAA88, 0xFF7788AA,
];
const SUBURB_ROOF_COLORS: [u32; 5] = [
    0xFF554433, // dark brown shingle
    0xFF443838, // slate grey-brown
    0xFF5A4A3A, // warm brown
    0xFF3A3A44, // dark slate
    0xFF664433, // reddish brown
];
const SUBURB_FENCE_COLOR: u32 = 0xFF998866;
const SUBURB_DOOR_COLOR: u32 = 0xFF553322;

/// Generate suburban houses along road segments in outer areas (50-170m from center).
/// Each road segment gets 1-3 houses per side, placed with driveways and garden fences.
fn generate_suburbs(
    tris: &mut Vec<WorldTri>, terrain: &Terrain, rng: &mut Rng,
    net: &RoadNetwork, buildings: &mut Vec<Building>,
    _walls: &mut Vec<Wall>, parking_spots: &mut Vec<ParkingSpot>,
    river_segments: &[RiverSegment], settlement_center: [f32; 2],
) {
    for seg in &net.segments {
        if seg.tier != RoadTier::CarRoad { continue; }

        let Some((perp_x, perp_z, dir_x, dir_z, len)) = segment_perp(seg) else { continue };
        if len < 25.0 { continue; }

        let mid_x = (seg.x0 + seg.x1) * 0.5;
        let mid_z = (seg.z0 + seg.z1) * 0.5;
        let dist_from_center = ((mid_x - settlement_center[0]).powi(2) + (mid_z - settlement_center[1]).powi(2)).sqrt();
        // Only place suburbs in outer ring, away from dockyard
        if dist_from_center < 50.0 || dist_from_center > 170.0 { continue; }
        if mid_z > WORLD_HALF - 30.0 { continue; }

        let houses_per_side = ((len - 10.0) / 25.0).ceil() as i32;
        let houses_per_side = houses_per_side.min(3);

        for side in [-1.0f32, 1.0] {
            let house_offset = CAR_ROAD_WIDTH * 0.5 + SIDEWALK_WIDTH + 8.0;

            for k in 0..houses_per_side {
                let t = 0.15 + (k as f32 + 0.5) / houses_per_side as f32 * 0.7;
                let (sx, sz) = sample_segment(seg, t);
                let hx = sx + perp_x * house_offset * side;
                let hz = sz + perp_z * house_offset * side;

                if on_river(hx, hz, river_segments) { continue; }

                // Check building overlap
                if overlaps_building(hx, hz, 6.0, buildings) { continue; }

                let hw = rng.range(4.0, 7.0);
                let hd = rng.range(4.0, 7.0);
                let hh = rng.range(3.0, 5.0);
                let color = rng.pick(&SUBURB_HOUSE_COLORS);
                let roof_color = rng.pick(&SUBURB_ROOF_COLORS);
                let shutter_c = SHUTTER_COLORS[k as usize % SHUTTER_COLORS.len()];

                // Use minimum terrain height across footprint so house never floats
                let (min_h, _) = building_ground_height(terrain, hx, hz, hw, hd);
                let gy = min_h;

                // Foundation skirt to seal gap between house base and terrain
                add_building_foundation(tris, terrain, hx, gy, hz, hw, hd, color);

                // House body — beveled
                mesh::beveled_box_tris(tris, hx, gy + hh * 0.5, hz, hw, hh, hd, 0.08, color);

                // Pitched roof with overhang
                let roof_peak = hh * 0.35 + 0.6;
                mesh::pitched_roof_tris(tris, hx, gy + hh, hz, hw + 0.6, hd + 0.6, roof_peak, roof_color);

                // Chimney on roof — offset along X (ridge direction) to stay on ridge
                let chim_ox = hw * 0.2;
                let chim_oz = 0.0;
                mesh::box_tris(tris, hx + chim_ox, gy + hh + roof_peak * 0.6, hz + chim_oz,
                    0.35, roof_peak * 0.8 + 0.5, 0.35, darken(color, 0.45));
                mesh::box_tris(tris, hx + chim_ox, gy + hh + roof_peak * 0.6 + roof_peak * 0.4 + 0.3, hz + chim_oz,
                    0.45, 0.08, 0.45, darken(color, 0.35));

                // Front face direction (toward road)
                let face_nx = -perp_x * side;
                let face_nz = -perp_z * side;

                // Door with frame and step
                let door_x = hx + face_nx * hd * 0.5;
                let door_z = hz + face_nz * hd * 0.5;
                let door_depth = 0.12;
                // Door frame (slightly larger than door)
                mesh::box_tris(tris, door_x, gy + 1.05, door_z - face_nx * door_depth * 0.3,
                    1.1, 2.3, door_depth * 0.6, darken(color, 0.7));
                // Door itself
                mesh::box_tris(tris, door_x, gy + 0.95, door_z - face_nx * door_depth * 0.5,
                    0.85, 1.9, door_depth, SUBURB_DOOR_COLOR);
                // Doorstep
                mesh::box_tris(tris, door_x + face_nz * 0.0, gy + 0.06, door_z + face_nx * 0.3,
                    1.2, 0.12, 0.5, darken(color, 0.6));

                // Front windows with shutters and sills
                const SUBURB_WIN_LIT: [u32; 3] = [0x00AA7722, 0x00BB8833, 0x00997722];
                const SUBURB_WIN_DARK: u32 = 0xFF1A1A2A;
                for (wi_idx, wi) in [-1.0f32, 1.0].iter().enumerate() {
                    let wx = hx + dir_x * wi * (hw * 0.3);
                    let wz = hz + dir_z * wi * (hw * 0.3);
                    let fwx = wx + face_nx * hd * 0.5;
                    let fwz = wz + face_nz * hd * 0.5;
                    // Window color: ~70% lit, ~30% dark
                    let wh = (k as u32).wrapping_mul(2654435761).wrapping_add(wi_idx as u32 * 1013904223);
                    let win_color = if wh % 100 < 70 {
                        SUBURB_WIN_LIT[(wh / 100) as usize % SUBURB_WIN_LIT.len()]
                    } else { SUBURB_WIN_DARK };
                    // Oriented window pane (faces outward along house face normal)
                    let pane_off = 0.08;
                    let px = fwx + face_nx * pane_off;
                    let pz = fwz + face_nz * pane_off;
                    let cy = gy + hh * 0.55;
                    let qhw = 0.28; let qhh = 0.35;
                    mesh::push_quad(tris,
                        [px - dir_x*qhw, cy - qhh, pz - dir_z*qhw],
                        [px + dir_x*qhw, cy - qhh, pz + dir_z*qhw],
                        [px + dir_x*qhw, cy + qhh, pz + dir_z*qhw],
                        [px - dir_x*qhw, cy + qhh, pz - dir_z*qhw],
                        win_color,
                    );
                    // Window sill
                    let sill_ox = face_nx * 0.06;
                    let sill_oz = face_nz * 0.06;
                    mesh::box_tris(tris, fwx + sill_ox, gy + hh * 0.55 - 0.45, fwz + sill_oz,
                        0.85, 0.06, 0.08, darken(color, 0.7));
                    // Left shutter
                    let ls_ox = -dir_x * 0.45 + face_nx * 0.02;
                    let ls_oz = -dir_z * 0.45 + face_nz * 0.02;
                    mesh::box_tris(tris, fwx + ls_ox, gy + hh * 0.55, fwz + ls_oz,
                        0.14, 0.8, 0.04, shutter_c);
                    // Right shutter
                    let rs_ox = dir_x * 0.45 + face_nx * 0.02;
                    let rs_oz = dir_z * 0.45 + face_nz * 0.02;
                    mesh::box_tris(tris, fwx + rs_ox, gy + hh * 0.55, fwz + rs_oz,
                        0.14, 0.8, 0.04, shutter_c);
                }

                // Upper floor window (gable — oriented quad facing outward)
                let upper_wh = (k as u32).wrapping_mul(2654435761).wrapping_add(7);
                let upper_win_color = if upper_wh % 100 < 70 {
                    SUBURB_WIN_LIT[(upper_wh / 100) as usize % SUBURB_WIN_LIT.len()]
                } else { SUBURB_WIN_DARK };
                {
                    let gx = hx + face_nx * (hd * 0.5 + 0.08);
                    let gz = hz + face_nz * (hd * 0.5 + 0.08);
                    let gy2 = gy + hh + roof_peak * 0.2;
                    let ghw = 0.22; let ghh = 0.22;
                    mesh::push_quad(tris,
                        [gx - dir_x*ghw, gy2 - ghh, gz - dir_z*ghw],
                        [gx + dir_x*ghw, gy2 - ghh, gz + dir_z*ghw],
                        [gx + dir_x*ghw, gy2 + ghh, gz + dir_z*ghw],
                        [gx - dir_x*ghw, gy2 + ghh, gz - dir_z*ghw],
                        upper_win_color,
                    );
                }

                // Side windows (one per side — oriented quads facing outward)
                for (si_idx, si) in [-1.0f32, 1.0].iter().enumerate() {
                    let swx = hx + dir_x * si * hw * 0.5;
                    let swz = hz + dir_z * si * hw * 0.5;
                    let snx = dir_x * si;
                    let snz = dir_z * si;
                    let sx = swx + snx * 0.08;
                    let sz = swz + snz * 0.08;
                    let sy = gy + hh * 0.55;
                    let sh = (k as u32).wrapping_mul(741103597).wrapping_add((si_idx as u32).wrapping_mul(2246822519));
                    let sc = if sh % 100 < 60 { 0x00997722 } else { SUBURB_WIN_DARK };
                    let shw = 0.2; let shh = 0.25;
                    mesh::push_quad(tris,
                        [sx - face_nx*shw, sy - shh, sz - face_nz*shw],
                        [sx + face_nx*shw, sy - shh, sz + face_nz*shw],
                        [sx + face_nx*shw, sy + shh, sz + face_nz*shw],
                        [sx - face_nx*shw, sy + shh, sz - face_nz*shw],
                        sc,
                    );
                }

                // Porch overhang (small roof over door)
                let porch_x = door_x + face_nx * 0.5;
                let porch_z = door_z + face_nz * 0.5;
                mesh::box_tris(tris, porch_x, gy + 2.15, porch_z,
                    1.4, 0.06, 0.8, roof_color);
                // Porch support posts
                for ps in [-0.55f32, 0.55] {
                    let ppx = porch_x + dir_x * ps;
                    let ppz = porch_z + dir_z * ps;
                    mesh::cylinder_tris(tris, ppx, gy + 1.05, ppz, 0.04, 2.1, 4, darken(color, 0.6));
                }

                // Foundation / base course
                mesh::box_tris(tris, hx, gy + 0.1, hz,
                    hw + 0.1, 0.2, hd + 0.1, darken(color, 0.5));

                // Flower box under one front window
                if k % 2 == 0 {
                    let fbx = hx + dir_x * (hw * 0.3) + face_nx * (hd * 0.5 + 0.1);
                    let fbz = hz + dir_z * (hw * 0.3) + face_nz * (hd * 0.5 + 0.1);
                    mesh::box_tris(tris, fbx, gy + hh * 0.55 - 0.5, fbz,
                        0.7, 0.12, 0.15, FLOWER_BOX_COLOR);
                    let fc = FLOWER_COLORS[k as usize % FLOWER_COLORS.len()];
                    mesh::sphere_tris(tris, fbx, gy + hh * 0.55 - 0.35, fbz, 0.08, 0, fc);
                }

                buildings.push(Building { x: hx, z: hz, w: hw, d: hd, h: hh, ground_y: gy });

                // Picket fence with proper posts + rails + gate
                let fence_extent = 5.5;
                let fence_h = 0.9;
                let num_posts = 10;
                for fp in 0..num_posts {
                    let t_fence = (fp as f32 + 0.5) / num_posts as f32 * 2.0 - 1.0;
                    // Skip posts near driveway
                    if t_fence.abs() < 0.15 { continue; }
                    let fx = hx + dir_x * t_fence * fence_extent;
                    let fz = hz + dir_z * t_fence * fence_extent;
                    let fgy = terrain.height_at(fx, fz);
                    // Post (thicker, taller)
                    mesh::cylinder_tris(tris, fx, fgy + fence_h * 0.5, fz, 0.035, fence_h, 4, SUBURB_FENCE_COLOR);
                    // Pointed cap
                    mesh::cone_tris(tris, fx, fgy + fence_h + 0.04, fz, 0.04, 0.08, 4, SUBURB_FENCE_COLOR);
                }
                // Horizontal rails (top and mid)
                for ri in [0.7f32, 0.35] {
                    let r1x = hx + dir_x * (-fence_extent);
                    let r1z = hz + dir_z * (-fence_extent);
                    let r2x = hx + dir_x * fence_extent;
                    let r2z = hz + dir_z * fence_extent;
                    let rgy = terrain.height_at(hx, hz);
                    mesh::cylinder_between(tris,
                        [r1x, rgy + ri, r1z], [r2x, rgy + ri, r2z],
                        0.02, 4, SUBURB_FENCE_COLOR);
                }

                // Driveway parking spot (between house and road)
                let drv_offset = CAR_ROAD_WIDTH * 0.5 + SIDEWALK_WIDTH + 3.0;
                let (ds_x, ds_z) = sample_segment(seg, t);
                let drv_x = ds_x + perp_x * drv_offset * side;
                let drv_z = ds_z + perp_z * drv_offset * side;
                if !on_river(drv_x, drv_z, river_segments) {
                    let rot = (-dir_x).atan2(-dir_z) + if side > 0.0 { 0.0 } else { std::f32::consts::PI };
                    parking_spots.push(ParkingSpot { x: drv_x, z: drv_z, rot_y: rot, occupied_by: None });
                }
            }
        }
    }
}

/// Check which buildings are accessible from the road network.
/// For each building, sample 8 directions; from each clear sample, walk toward nearest
/// road node checking collision every 2m.
fn validate_building_accessibility(world: &WorldData, net: &RoadNetwork) -> Vec<bool> {
    let mut reachable = vec![false; world.buildings.len()];
    let angles: [f32; 8] = [0.0, std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2,
        3.0 * std::f32::consts::FRAC_PI_4, std::f32::consts::PI,
        5.0 * std::f32::consts::FRAC_PI_4, 3.0 * std::f32::consts::FRAC_PI_2,
        7.0 * std::f32::consts::FRAC_PI_4];

    for (bi, b) in world.buildings.iter().enumerate() {
        let half_extent = b.w.max(b.d) * 0.5 + 2.0;
        for &angle in &angles {
            let sx = b.x + angle.cos() * half_extent;
            let sz = b.z + angle.sin() * half_extent;
            // Sample must be clear
            if check_walk_collision(world, sx, sz, 0.4, Some(bi))
                || on_river_not_bridge(sx, sz, &world.river_segments, &world.bridges) {
                continue;
            }
            // Walk from sample toward nearest road node (up to 30 steps at 2m)
            let mut best_node_dist = f32::MAX;
            let mut target_x = 0.0f32;
            let mut target_z = 0.0f32;
            for node in &net.nodes {
                let d = (node[0] - sx).powi(2) + (node[1] - sz).powi(2);
                if d < best_node_dist {
                    best_node_dist = d;
                    target_x = node[0];
                    target_z = node[1];
                }
            }
            if best_node_dist > 60.0 * 60.0 { continue; } // too far from any road
            let dx = target_x - sx;
            let dz = target_z - sz;
            let dist = best_node_dist.sqrt();
            let steps = ((dist / 2.0) as usize).min(30);
            let mut blocked = false;
            for s in 1..=steps {
                let t = s as f32 / (steps + 1) as f32;
                let px = sx + dx * t;
                let pz = sz + dz * t;
                if check_walk_collision(world, px, pz, 0.4, Some(bi))
                    || on_river_not_bridge(px, pz, &world.river_segments, &world.bridges) {
                    blocked = true;
                    break;
                }
            }
            if !blocked {
                reachable[bi] = true;
                break;
            }
        }
    }
    reachable
}

/// Darken ground-facing triangles near building footprints to simulate
/// ambient occlusion / contact shadows at building bases.
fn apply_building_base_ao(tris: &mut Vec<WorldTri>, buildings: &[Building]) {
    const AO_RADIUS: f32 = 3.5;  // how far the darkening extends from building edge
    const AO_STRENGTH: f32 = 0.38; // darken factor at the building edge (0=black, 1=no change)

    for tri in tris.iter_mut() {
        // Only affect roughly horizontal surfaces (ground, sidewalk, road)
        if tri.normal[1] < 0.5 { continue; }

        let cx = (tri.v[0][0] + tri.v[1][0] + tri.v[2][0]) * 0.333;
        let cz = (tri.v[0][2] + tri.v[1][2] + tri.v[2][2]) * 0.333;

        let mut min_dist = f32::MAX;
        for b in buildings {
            // Signed distance from centroid to building footprint AABB
            let dx = (cx - b.x).abs() - b.w * 0.5;
            let dz = (cz - b.z).abs() - b.d * 0.5;
            // Euclidean distance to AABB
            let ex = dx.max(0.0);
            let ez = dz.max(0.0);
            let d = (ex * ex + ez * ez).sqrt();
            // If inside footprint, distance is 0
            let d = if dx < 0.0 && dz < 0.0 { 0.0 } else { d };
            if d < min_dist { min_dist = d; }
        }

        if min_dist < AO_RADIUS {
            // Smooth falloff: strongest at edge, fading to nothing at AO_RADIUS
            let t = min_dist / AO_RADIUS; // 0 at edge, 1 at max radius
            let factor = AO_STRENGTH + (1.0 - AO_STRENGTH) * t * t; // quadratic ease-out
            tri.color = darken(tri.color, factor);
        }
    }
}

pub fn generate_world(game: &mut GameState) {
    let seed = game.world_seed;
    let mut rng = Rng::new(seed);
    let mut tris = Vec::with_capacity(500_000);

    // === Layer 1: Terrain from noise (before any flattening) ===
    // Roughness is seed-derived: each world has unique terrain character
    let roughness = {
        let mut r = Rng::new(seed.wrapping_add(77777));
        0.3 + r.range(0.0, 1.7)  // 0.3 (gentle Czech) to 2.0 (Alpine)
    };
    generate_heightmap_noise(&mut game.terrain, seed, roughness);

    // === Layer 2: River (carved into terrain, path varies by seed) ===
    generate_river(&mut game.terrain, &mut tris, &mut game.world.river_segments, seed);

    // === Layer 3: Zone map from terrain analysis ===
    let zone_map = zone::ZoneMap::generate(
        &game.terrain, &game.world.river_segments, seed,
    );
    let settlement_center = zone_map.main_settlement();
    eprintln!("{} settlements, main at ({:.0}, {:.0}), {} zones classified",
        zone_map.settlement_centers.len(),
        settlement_center[0], settlement_center[1],
        zone_map.cells.iter().filter(|c| c.kind != zone::ZoneKind::Wilderness).count());

    // === Layer 4: Road network centered on settlement ===
    game.road_network = generate_road_network(&mut rng, settlement_center);

    // === Flatten terrain under roads + settlement + zones ===
    flatten_terrain_for_roads(
        &mut game.terrain, &game.road_network, settlement_center, &zone_map,
    );

    // === Bridges over river ===
    generate_bridges(&mut tris, &mut game.terrain,
        &game.road_network, &game.world.river_segments, &mut game.world.walls,
        &mut game.world.bridges);

    // === Terrain mesh (after all terrain modifications) ===
    generate_terrain_mesh(&mut tris, &game.terrain, &zone_map);

    // === Road geometry ===
    emit_road_geometry(&mut tris, &game.terrain, &game.road_network);

    // === Asset placement via zone-driven rules ===

    // Buildings: placed by zone density, near roads
    let building_spots = placement::place_assets(
        &placement::BUILDING_RULE, &zone_map, &game.terrain,
        &game.road_network, &game.world.river_segments,
        &game.world.buildings, &mut rng,
    );
    for (bi, spot) in building_spots.iter().enumerate() {
        emit_building(&mut tris, &mut game.world.buildings, &game.terrain,
            &mut rng, spot.x, spot.z, bi);
    }

    // Trees: dense in wilderness, sparse in town
    let tree_spots = placement::place_assets(
        &placement::TREE_RULE, &zone_map, &game.terrain,
        &game.road_network, &game.world.river_segments,
        &game.world.buildings, &mut rng,
    );
    for (ti, spot) in tree_spots.iter().enumerate() {
        emit_tree(&mut tris, &mut game.world.trees, &game.terrain,
            &mut rng, spot.x, spot.z, ti, seed);
    }

    // Bushes
    let bush_spots = placement::place_assets(
        &placement::BUSH_RULE, &zone_map, &game.terrain,
        &game.road_network, &game.world.river_segments,
        &game.world.buildings, &mut rng,
    );
    for (bi, spot) in bush_spots.iter().enumerate() {
        emit_bush(&mut tris, &game.terrain, spot.x, spot.z, bi, seed);
    }

    // Grass patches
    let grass_spots = placement::place_assets(
        &placement::GRASS_RULE, &zone_map, &game.terrain,
        &game.road_network, &game.world.river_segments,
        &game.world.buildings, &mut rng,
    );
    for (gi, spot) in grass_spots.iter().enumerate() {
        emit_grass(&mut tris, &game.terrain, &mut rng, spot.x, spot.z, gi, seed);
    }

    // Rocks
    let rock_spots = placement::place_assets(
        &placement::ROCK_RULE, &zone_map, &game.terrain,
        &game.road_network, &game.world.river_segments,
        &game.world.buildings, &mut rng,
    );
    for (ri, spot) in rock_spots.iter().enumerate() {
        emit_rock(&mut tris, &mut game.world.rocks, &game.terrain,
            spot.x, spot.z, spot.scale, ri, seed);
    }

    // Street lights: along roads in developed zones
    let light_spots = placement::place_assets(
        &placement::STREET_LIGHT_RULE, &zone_map, &game.terrain,
        &game.road_network, &game.world.river_segments,
        &game.world.buildings, &mut rng,
    );
    for spot in &light_spots {
        emit_street_light(&mut tris, &mut game.world.street_lights, &game.terrain,
            spot.x, spot.z, seed);
    }

    // Trash bins: along roads in town/industrial zones
    let bin_spots = placement::place_assets(
        &placement::TRASH_BIN_RULE, &zone_map, &game.terrain,
        &game.road_network, &game.world.river_segments,
        &game.world.buildings, &mut rng,
    );
    for spot in &bin_spots {
        game.world.trash_bins.push(TrashBin {
            x: spot.x,
            y: game.terrain.height_at(spot.x, spot.z),
            z: spot.z,
            items_held: 0,
            carried_by: None,
            terrain_normal: [0.0, 1.0, 0.0],
        });
    }

    // === Zone-specific features ===
    // Density scale derived from zone map: average Town zone density near settlement
    let d = zone_map.zone_at(settlement_center[0], settlement_center[1]).density.max(0.5);

    // Suburban houses along outer roads
    {
        let mut suburb_spots = Vec::new();
        generate_suburbs(&mut tris, &game.terrain, &mut rng,
            &game.road_network, &mut game.world.buildings,
            &mut game.world.walls, &mut suburb_spots,
            &game.world.river_segments, settlement_center);
        game.road_network.parking_spots.extend(suburb_spots);
    }

    // Industrial dockyard — position from zone map
    if let Some(dock_center) = zone_map.find_zone_center(zone::ZoneKind::Industrial) {
        generate_dockyard_at(&mut tris, &game.terrain, &mut rng,
            &mut game.world.buildings, &mut game.world.interactibles,
            dock_center, d);
    } else if let Some(water_center) = zone_map.find_zone_center(zone::ZoneKind::Waterfront) {
        generate_dockyard_at(&mut tris, &game.terrain, &mut rng,
            &mut game.world.buildings, &mut game.world.interactibles,
            water_center, d);
    }

    // Interactibles (phone booths, vending machines, benches, etc.)
    generate_interactibles(&mut tris, &game.terrain, &mut rng,
        &game.road_network, &game.world.buildings, &mut game.world.interactibles, d);

    // Parking lots near intersections
    {
        let mut lot_spots = Vec::new();
        generate_parking_lots(&mut tris, &game.terrain, &mut rng,
            &game.road_network.nodes, &game.world.buildings,
            &mut lot_spots,
            &mut game.world.walls, &mut game.world.trees, &mut game.world.street_lights,
            &game.road_network, &game.world.river_segments);
        game.road_network.parking_spots.extend(lot_spots);
    }

    // Roadside parking spots
    let roadside_spots = generate_parking_spots(
        &game.road_network, &game.world.buildings,
        &game.terrain, &game.world.river_segments,
    );
    game.road_network.parking_spots.extend(roadside_spots);

    // Market stalls near town center (using settlement center, not origin)
    generate_market_stalls(&mut tris, &game.terrain, &mut rng,
        &game.road_network, &game.world.buildings, &mut game.world.walls,
        settlement_center, d);

    // Bus stops along major roads
    generate_bus_stops(&mut tris, &game.terrain, &mut rng,
        &game.road_network, &game.world.buildings, &mut game.world.walls, d);

    // Decorations throughout developed areas (using settlement center, not origin)
    generate_decorations(&mut tris, &game.terrain, &mut rng,
        &game.road_network, &game.world.buildings,
        &mut game.world.walls, &mut game.world.rocks, &mut game.world.street_lights,
        &mut game.world.clutter, settlement_center, d);

    // Building ground shadows
    for b in &game.world.buildings {
        let shadow_y = b.ground_y + 0.05;
        let shadow_hw = b.w * 0.5 + 0.3;
        let shadow_hd = b.d * 0.5 + 0.3;
        let shadow_color: u32 = 0xFF1E3E1E;
        let v0 = [b.x - shadow_hw, shadow_y, b.z - shadow_hd];
        let v1 = [b.x + shadow_hw, shadow_y, b.z - shadow_hd];
        let v2 = [b.x + shadow_hw, shadow_y, b.z + shadow_hd];
        let v3 = [b.x - shadow_hw, shadow_y, b.z + shadow_hd];
        mesh::push_quad_n(&mut tris, v0, v3, v2, v1, [0.0, 1.0, 0.0], shadow_color);
    }

    // === NPC-owned vehicles (100 — fixed count for NEAT compatibility) ===
    let total_spots = game.road_network.parking_spots.len();
    for i in 0..NUM_NPCS {
        let spot_offset = i;
        let (park_x, park_y, park_z, park_rot, spot_idx) = if spot_offset < total_spots {
            let spot = &game.road_network.parking_spots[spot_offset];
            (spot.x, game.terrain.height_at(spot.x, spot.z) + VEHICLE_GROUND_OFFSET, spot.z, spot.rot_y, Some(spot_offset))
        } else {
            let home_idx = i % game.world.buildings.len().max(1);
            if !game.world.buildings.is_empty() {
                let b = &game.world.buildings[home_idx];
                let px = b.x + b.w * 0.5 + 2.0;
                let pz = b.z;
                (px, game.terrain.height_at(px, pz) + VEHICLE_GROUND_OFFSET, pz, 0.0, None)
            } else {
                let px = settlement_center[0] + rng.range(-20.0, 20.0);
                let pz = settlement_center[1] + rng.range(-20.0, 20.0);
                (px, game.terrain.height_at(px, pz) + VEHICLE_GROUND_OFFSET, pz, 0.0, None)
            }
        };

        let vi = game.world.vehicles.len();
        if let Some(si) = spot_idx {
            game.road_network.parking_spots[si].occupied_by = Some(vi);
        }

        let color = rng.pick(&VEHICLE_COLORS);
        let mut vehicle_rng = rng.fork(1000 + i as u64);
        let cruise_speed = vehicle_rng.range(7.0, 12.0);
        let scale = vehicle_rng.range(0.82, 1.18);
        let (phys_body, phys_wheels, phys_susp, phys_drive) =
            Vehicle::default_physics(park_x, park_y, park_z, park_rot, scale);
        game.world.vehicles.push(Vehicle {
            x: park_x, y: park_y, z: park_z,
            rot_y: park_rot, speed: 0.0, color, occupied: false,
            ai_active: false, ai_target_x: park_x, ai_target_z: park_z,
            rng: vehicle_rng, owner_npc: Some(i),
            path: Vec::new(), path_idx: 0, current_segment: None,
            lane_dir: LaneDirection::Forward,
            intersection_state: IntersectionState::Cruising,
            intersection_wait_timer: 0.0,
            cruise_speed, target_speed: 0.0,
            parking_target: spot_idx, parked: true,
            idle_timer: 0.0,
            terrain_normal: crate::math::clamp_normal_tilt(game.terrain.normal_at(park_x, park_z), 30.0),
            scale,
            body: phys_body, wheels: phys_wheels, suspension: phys_susp, drivetrain: phys_drive,
            deformation: crate::deform::VehicleDeformation::new(),
            surface_override: None,
        });
    }

    // === NPCs (100 — fixed count for NEAT brain compatibility) ===
    let npc_jobs = [
        NpcJob::Collector, NpcJob::GarbageCollector, NpcJob::TaxiDriver,
        NpcJob::DeliveryCourier, NpcJob::MailCarrier, NpcJob::Paramedic,
        NpcJob::Firefighter, NpcJob::PolicePatrol, NpcJob::StreetVendor,
        NpcJob::Mechanic, NpcJob::ConstructionWorker, NpcJob::Fisherman,
        NpcJob::Farmer, NpcJob::Lumberjack, NpcJob::Scavenger,
    ];
    let reachable = validate_building_accessibility(&game.world, &game.road_network);
    let mut building_by_dist: Vec<usize> = (0..game.world.buildings.len())
        .filter(|&bi| reachable[bi])
        .collect();
    // Sort by distance from settlement center (not origin)
    building_by_dist.sort_by(|&a, &b| {
        let ba = &game.world.buildings[a];
        let bb = &game.world.buildings[b];
        let da = (ba.x - settlement_center[0]).powi(2) + (ba.z - settlement_center[1]).powi(2);
        let db = (bb.x - settlement_center[0]).powi(2) + (bb.z - settlement_center[1]).powi(2);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    if building_by_dist.is_empty() {
        building_by_dist = (0..game.world.buildings.len()).collect();
    }
    for i in 0..NUM_NPCS {
        let home_idx = if !building_by_dist.is_empty() {
            building_by_dist[i % building_by_dist.len()]
        } else { 0 };
        let car_idx = i;
        let (bx, bz, bw, bd) = if !game.world.buildings.is_empty() {
            let b = &game.world.buildings[home_idx];
            (b.x, b.z, b.w, b.d)
        } else {
            (settlement_center[0], settlement_center[1], 4.0, 4.0)
        };

        let mut x = bx + (bw * 0.5 + 2.0);
        let mut z = bz;
        let spawn_angles: [f32; 8] = [0.0, std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2,
            3.0 * std::f32::consts::FRAC_PI_4, std::f32::consts::PI,
            5.0 * std::f32::consts::FRAC_PI_4, 3.0 * std::f32::consts::FRAC_PI_2,
            7.0 * std::f32::consts::FRAC_PI_4];
        let mut spawn_ok = false;
        for &angle in &spawn_angles {
            let extent = bw.max(bd) * 0.5 + 1.5 + rng.range(0.0, 2.0);
            let sx = bx + angle.cos() * extent;
            let sz = bz + angle.sin() * extent;
            if !check_walk_collision(&game.world, sx, sz, 0.4, Some(home_idx))
                && !on_river_not_bridge(sx, sz, &game.world.river_segments, &game.world.bridges)
            {
                x = sx; z = sz; spawn_ok = true; break;
            }
        }
        if !spawn_ok {
            let mut best_d = f32::MAX;
            for node in &game.road_network.nodes {
                let d = (node[0] - bx).powi(2) + (node[1] - bz).powi(2);
                if d < best_d {
                    best_d = d;
                    x = node[0]; z = node[1];
                }
            }
        }
        let y = game.terrain.height_at(x, z);
        let shirt_color = rng.pick(&NPC_SHIRT_COLORS);
        let pants_color = rng.pick(&NPC_PANTS_COLORS);
        let rot_y = rng.range(0.0, std::f32::consts::TAU);
        let npc_rng = rng.fork(500 + i as u64);
        let wake_hour = 5.0 + (npc_rng.clone().next() as f32 % 400.0) / 100.0;
        let job = npc_jobs[i % NPC_JOB_COUNT];

        game.world.npcs.push(Npc {
            x, y, z, rot_y, walk_phase: rng.range(0.0, 6.0),
            target_x: x, target_z: z,
            shirt_color, pants_color, rng: npc_rng,
            vel_y: 0.0, on_ground: true,
            state: NpcState::Working,
            home_idx, car_idx,
            wake_hour,
            state_timer: 0.0,
            money: NPC_STARTING_MONEY,
            carrying_item: false,
            carrying_bin: None,
            target_item: None,
            target_bin: None,
            items_deposited_today: 0,
            in_vehicle: false,
            parked_x: x, parked_z: z,
            nav_path: Vec::new(), nav_path_idx: 0,
            nav_target_x: 0.0, nav_target_z: 0.0,
            stuck_timer: 0.0,
            job,
            job_timer: 0.0,
            job_target_x: x, job_target_z: z,
            interaction_target: None,
            interacting_with: None,
            interaction_timer: 0.0,
            brain_idx: i,
            fitness_money_earned: 0.0,
            fitness_items_picked: 0,
            fitness_interactions: 0,
            fitness_distance: 0.0,
            fitness_stuck_time: 0.0,
            prev_x: x, prev_z: z,
            health: NPC_HEALTH_MAX,
            attack_cooldown: 0.0,
            attack_phase: 0.0,
            hit_flash: 0.0,
            knockout_timer: 0.0,
            knockback_vx: 0.0,
            knockback_vz: 0.0,
            attack_intent: 0,
            fitness_knockouts: 0,
            fitness_hits_landed: 0,
            hunger: 100.0,
            thirst: 100.0,
            starving_dead: false,
            fitness_starve_time: 0.0,
            find_item_failures: 0,
            find_bin_failures: 0,
            stuck_recoveries: 0,
            sound: [0.0; 3],
            fitness_sounds_made: 0,
            fitness_npcs_heard: 0,
            fitness_proximity: 0.0,
            ragdoll_active: false,
            ragdoll_points: [[0.0; 3]; 7],
            ragdoll_timer: 0.0,
            skeleton: crate::skeleton::Skeleton::new_humanoid(),
            body: {
                let shape = crate::physics::CollisionShape::Capsule { radius: 0.3, half_height: 0.625 };
                let inertia = shape.inertia_diag(75.0);
                let mut b = crate::physics::RigidBody::new_dynamic([x, y, z], 75.0, inertia);
                b.quat = crate::math::quat_from_rot_y(rot_y);
                b
            },
            wanted: false,
            bounty: 0.0,
            violation_timer: 0.0,
            police_target: None,
            terrain_normal: [0.0, 1.0, 0.0],
        });
    }

    // === Items (250 — fixed count for gameplay balance) ===
    // Items cluster near settlements with falloff into wilderness
    let item_kinds = [ItemKind::Health, ItemKind::Money, ItemKind::Stamina, ItemKind::Food, ItemKind::Water];
    for _ in 0..NUM_ITEMS {
        let mut x;
        let mut z;
        let mut attempts = 0;
        loop {
            if attempts > 50 && !game.road_network.nodes.is_empty() {
                let ni = rng.next() as usize % game.road_network.nodes.len();
                let angle = rng.range(0.0, std::f32::consts::TAU);
                let dist = rng.range(10.0, 20.0);
                x = game.road_network.nodes[ni][0] + angle.cos() * dist;
                z = game.road_network.nodes[ni][1] + angle.sin() * dist;
                if !on_any_road(x, z, &game.road_network) { break; }
                x = game.road_network.nodes[ni][0] + angle.cos() * 25.0;
                z = game.road_network.nodes[ni][1] + angle.sin() * 25.0;
                break;
            }
            // Gaussian-ish distribution centered on settlement (σ ≈ 80m)
            let angle = rng.range(0.0, std::f32::consts::TAU);
            let r = (rng.range(0.0_f32, 1.0).sqrt()) * 120.0; // sqrt for uniform disk, radius 120m
            x = (settlement_center[0] + angle.cos() * r).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            z = (settlement_center[1] + angle.sin() * r).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            attempts += 1;
            if on_any_road(x, z, &game.road_network) { continue; }
            if check_walk_collision(&game.world, x, z, 0.5, Some(usize::MAX)) { continue; }
            if on_river_not_bridge(x, z, &game.world.river_segments, &game.world.bridges) { continue; }
            break;
        }
        let y = game.terrain.height_at(x, z);
        let kind = item_kinds[rng.next() as usize % item_kinds.len()];
        game.world.items.push(Item {
            x, y, z, kind, active: true,
            spin_phase: rng.range(0.0, 6.0),
            falling: false, vel_y: 0.0, claimed_by: None, skip_until: 0.0,
        });
    }

    // Set player spawn near settlement center with camera clearance from buildings
    {
        let mut sx = settlement_center[0];
        let mut sz = settlement_center[1] + 15.0;
        let mut spawn_rng = Rng::new(seed.wrapping_add(88888));
        let cam_clearance = 5.0; // camera orbit ~8m but doesn't need full clearance
        let mut found = false;
        for _ in 0..500 {
            let clear = !on_any_road(sx, sz, &game.road_network)
                && !overlaps_building(sx, sz, cam_clearance, &game.world.buildings)
                && !on_river_not_bridge(sx, sz, &game.world.river_segments, &game.world.bridges);
            if clear { found = true; break; }
            let angle = spawn_rng.range(0.0, std::f32::consts::TAU);
            let r = spawn_rng.range(10.0, 80.0);
            sx = settlement_center[0] + angle.cos() * r;
            sz = settlement_center[1] + angle.sin() * r;
        }
        if !found {
            // Fallback: avoid building overlap and roads
            sx = settlement_center[0];
            sz = settlement_center[1] + 20.0;
            for _ in 0..200 {
                let ok = !overlaps_building(sx, sz, 2.0, &game.world.buildings)
                    && !on_any_road(sx, sz, &game.road_network);
                if ok { break; }
                let angle = spawn_rng.range(0.0, std::f32::consts::TAU);
                let r = spawn_rng.range(15.0, 100.0);
                sx = settlement_center[0] + angle.cos() * r;
                sz = settlement_center[1] + angle.sin() * r;
            }
        }
        game.player.x = sx;
        game.player.z = sz;
        game.player.y = game.terrain.height_at(sx, sz);
        game.player.body.pos = [game.player.x, game.player.y, game.player.z];
    }

    // Ambient occlusion
    apply_building_base_ao(&mut tris, &game.world.buildings);

    // Build navigation grid
    game.walk_grid = crate::navmesh::build_walk_grid(
        &game.world.buildings, &game.world.rocks, &game.world.trees,
        &game.world.walls, &game.world.clutter,
        &game.world.river_segments, &game.world.bridges, &game.terrain,
    );
    eprintln!("Walk grid: {}x{} ({}m cells)", game.walk_grid.grid_w, game.walk_grid.grid_h, game.walk_grid.cell_size);

    // Store zone map for runtime use (jobs, particle effects, etc.)
    game.zone_map = zone_map;

    eprintln!("World: {} tris, {} road segs ({} nodes), {} buildings, {} trees, {} rocks, {} lights, {} vehicles, {} npcs, {} items, {} bins, {} parking",
        tris.len(), game.road_network.segments.len(), game.road_network.nodes.len(),
        game.world.buildings.len(), game.world.trees.len(), game.world.rocks.len(),
        game.world.street_lights.len(),
        game.world.vehicles.len(), game.world.npcs.len(),
        game.world.items.len(), game.world.trash_bins.len(), game.road_network.parking_spots.len());
    game.world.static_tris = tris;
}

// ==================== Road Geometry ====================

fn emit_road_geometry(tris: &mut Vec<WorldTri>, terrain: &Terrain, net: &RoadNetwork) {
    for seg in &net.segments {
        match seg.tier {
            RoadTier::CarRoad => {
                let hw = CAR_ROAD_WIDTH * 0.5;
                generate_road_strip(tris, terrain,
                    seg.x0, seg.z0, seg.x1, seg.z1, hw, 0.05, ROAD_COLOR);
                generate_road_strip(tris, terrain,
                    seg.x0, seg.z0, seg.x1, seg.z1, 0.15, 0.08, ROAD_LINE_COLOR);
                if let Some((perp_x, perp_z, _dir_x, _dir_z, _len)) = segment_perp(seg) {
                    let edge_offset = hw - 0.2;
                    let el_x0 = seg.x0 + perp_x * edge_offset;
                    let el_z0 = seg.z0 + perp_z * edge_offset;
                    let el_x1 = seg.x1 + perp_x * edge_offset;
                    let el_z1 = seg.z1 + perp_z * edge_offset;
                    generate_road_strip(tris, terrain, el_x0, el_z0, el_x1, el_z1, 0.08, 0.11, ROAD_EDGE_COLOR);
                    let er_x0 = seg.x0 - perp_x * edge_offset;
                    let er_z0 = seg.z0 - perp_z * edge_offset;
                    let er_x1 = seg.x1 - perp_x * edge_offset;
                    let er_z1 = seg.z1 - perp_z * edge_offset;
                    generate_road_strip(tris, terrain, er_x0, er_z0, er_x1, er_z1, 0.08, 0.11, ROAD_EDGE_COLOR);
                    let sw_hw = SIDEWALK_WIDTH * 0.5;
                    let sw_offset = hw + sw_hw;
                    let curb_hw = 0.12;
                    let curb_offset = hw + curb_hw;
                    let cl_x0 = seg.x0 + perp_x * curb_offset;
                    let cl_z0 = seg.z0 + perp_z * curb_offset;
                    let cl_x1 = seg.x1 + perp_x * curb_offset;
                    let cl_z1 = seg.z1 + perp_z * curb_offset;
                    generate_road_strip(tris, terrain, cl_x0, cl_z0, cl_x1, cl_z1, curb_hw, 0.18, CURB_COLOR);
                    let cr_x0 = seg.x0 - perp_x * curb_offset;
                    let cr_z0 = seg.z0 - perp_z * curb_offset;
                    let cr_x1 = seg.x1 - perp_x * curb_offset;
                    let cr_z1 = seg.z1 - perp_z * curb_offset;
                    generate_road_strip(tris, terrain, cr_x0, cr_z0, cr_x1, cr_z1, curb_hw, 0.18, CURB_COLOR);
                    let lx0 = seg.x0 + perp_x * sw_offset;
                    let lz0 = seg.z0 + perp_z * sw_offset;
                    let lx1 = seg.x1 + perp_x * sw_offset;
                    let lz1 = seg.z1 + perp_z * sw_offset;
                    generate_road_strip(tris, terrain, lx0, lz0, lx1, lz1, sw_hw, 0.15, SIDEWALK_COLOR);
                    let rx0 = seg.x0 - perp_x * sw_offset;
                    let rz0 = seg.z0 - perp_z * sw_offset;
                    let rx1 = seg.x1 - perp_x * sw_offset;
                    let rz1 = seg.z1 - perp_z * sw_offset;
                    generate_road_strip(tris, terrain, rx0, rz0, rx1, rz1, sw_hw, 0.15, SIDEWALK_COLOR);
                }
            }
            RoadTier::FieldRoad => {
                generate_road_strip(tris, terrain,
                    seg.x0, seg.z0, seg.x1, seg.z1,
                    FIELD_ROAD_WIDTH * 0.5, 0.04, FIELD_ROAD_COLOR);
            }
        }
    }
}

// ==================== Emit Functions ====================
// Each emit_* function generates mesh geometry for a single placed asset.

fn emit_building(
    tris: &mut Vec<WorldTri>, buildings: &mut Vec<Building>,
    terrain: &Terrain, rng: &mut Rng, x: f32, z: f32, bi: usize,
) {
    let w = rng.range(3.0, 8.0);
    let d = rng.range(3.0, 8.0);
    let h = rng.range(3.0, 20.0);
    let (min_h, _max_h) = building_ground_height(terrain, x, z, w, d);
    let ground_y = min_h;
    let color = rng.pick(&BUILDING_COLORS);

    add_building_foundation(tris, terrain, x, ground_y, z, w, d, color);

    let bevel = 0.15_f32.min(w * 0.1).min(d * 0.1);
    let recess_depth = 0.15;
    let wall_inset = recess_depth + 0.02;
    mesh::beveled_box_tris(tris, x, ground_y + h * 0.5, z,
        w - wall_inset * 2.0, h, d - wall_inset * 2.0, bevel, color);

    let win_h = 1.2;
    let win_w = 0.8;
    let floor_height = 3.0;
    let floors = ((h - 1.0) / floor_height) as i32;
    let cols = ((w - 1.0) / 2.0) as i32;
    let is_timber = bi % 4 == 0;
    let has_shop = bi % 3 != 2 && floors > 1;
    let has_balcony = bi % 5 == 0 && floors > 1;
    let shutter_color = SHUTTER_COLORS[bi % SHUTTER_COLORS.len()];

    const WIN_LIT_COLORS: [u32; 4] = [
        0x00AA7722, 0x00BB8833, 0x00997722, 0x00CCAA44,
    ];
        const WIN_DARK: u32 = 0xFF1A1A2A; // dark glass (non-emissive)

        // Window holes for front/back
        let mut win_holes: Vec<mesh::WallHole> = Vec::new();
        let first_window_floor = if has_shop { 1 } else { 0 };
        for floor in first_window_floor..floors {
            let wy = 2.0 + floor as f32 * floor_height;
            for col in 0..cols {
                let wx = 1.2 + col as f32 * 2.0;
                win_holes.push(mesh::WallHole { x: wx, y: wy, w: win_w, h: win_h });
            }
        }

        // Generate per-window colors for front/back (hash-based randomization)
        let front_colors: Vec<u32> = win_holes.iter().enumerate().map(|(i, _)| {
            let h = (bi as u32).wrapping_mul(2654435761).wrapping_add(i as u32 * 1013904223);
            if h % 100 < 65 { WIN_LIT_COLORS[(h / 100) as usize % WIN_LIT_COLORS.len()] }
            else { WIN_DARK }
        }).collect();
        // Back face uses shifted hash for different pattern
        let back_colors: Vec<u32> = win_holes.iter().enumerate().map(|(i, _)| {
            let h = (bi as u32).wrapping_mul(1597334677).wrapping_add(i as u32 * 741103597);
            if h % 100 < 65 { WIN_LIT_COLORS[(h / 100) as usize % WIN_LIT_COLORS.len()] }
            else { WIN_DARK }
        }).collect();

        let zf = 0.01_f32;
        // Front face (z+) with recessed windows
        mesh::wall_with_holes_tris(
            tris,
            x - w * 0.5, ground_y, z + d * 0.5 + zf,
            w, h, &win_holes, recess_depth,
            color, &front_colors, 1.0, 1.0, false,
        );
        // Back face (z-)
        mesh::wall_with_holes_tris(
            tris,
            x + w * 0.5, ground_y, z - d * 0.5 - zf,
            w, h, &win_holes, recess_depth,
            color, &back_colors, -1.0, -1.0, false,
        );

        // Side windows
        let side_cols = ((d - 1.0) / 2.5) as i32;
        let mut side_holes: Vec<mesh::WallHole> = Vec::new();
        for floor in first_window_floor..floors {
            let wy = 2.0 + floor as f32 * floor_height;
            for col in 0..side_cols {
                let wz = 1.5 + col as f32 * 2.5;
                side_holes.push(mesh::WallHole { x: wz, y: wy, w: win_w, h: win_h });
            }
        }
        let side_l_colors: Vec<u32> = side_holes.iter().enumerate().map(|(i, _)| {
            let h = (bi as u32).wrapping_mul(3266489917).wrapping_add(i as u32 * 668265263);
            if h % 100 < 65 { WIN_LIT_COLORS[(h / 100) as usize % WIN_LIT_COLORS.len()] }
            else { WIN_DARK }
        }).collect();
        let side_r_colors: Vec<u32> = side_holes.iter().enumerate().map(|(i, _)| {
            let h = (bi as u32).wrapping_mul(2246822519).wrapping_add(i as u32 * 387276957);
            if h % 100 < 65 { WIN_LIT_COLORS[(h / 100) as usize % WIN_LIT_COLORS.len()] }
            else { WIN_DARK }
        }).collect();
        mesh::wall_with_holes_tris(
            tris,
            z - d * 0.5, ground_y, x + w * 0.5 + zf,
            d, h, &side_holes, recess_depth,
            color, &side_l_colors, -1.0, 1.0, true,
        );
        mesh::wall_with_holes_tris(
            tris,
            z + d * 0.5, ground_y, x - w * 0.5 - zf,
            d, h, &side_holes, recess_depth,
            color, &side_r_colors, 1.0, -1.0, true,
        );

        // --- Window shutters (thin boxes flanking each window on front face) ---
        let front_z = z + d * 0.5 + 0.02;
        for floor in first_window_floor..floors {
            let wy = ground_y + 2.0 + floor as f32 * floor_height;
            for col in 0..cols {
                let wx = x - w * 0.5 + 1.2 + col as f32 * 2.0;
                // Left shutter
                mesh::box_tris(tris, wx - win_w * 0.5 - 0.08, wy + win_h * 0.5, front_z,
                    0.12, win_h, 0.04, shutter_color);
                // Right shutter
                mesh::box_tris(tris, wx + win_w * 0.5 + win_w + 0.08, wy + win_h * 0.5, front_z,
                    0.12, win_h, 0.04, shutter_color);
                // Window sill (small ledge under window)
                mesh::box_tris(tris, wx + win_w * 0.5, wy - 0.02, front_z + 0.04,
                    win_w + 0.2, 0.06, 0.08, darken(color, 0.8));
            }
        }

        // --- Timber framing on ~25% of buildings (half-timbered ACU facade) ---
        if is_timber {
            let tw = 0.08; // timber beam thickness
            let front_tz = z + d * 0.5 + 0.03;
            let timber_c = TIMBER_COLOR;
            // Horizontal beams at each floor line
            for floor in 0..=floors {
                let by = ground_y + 2.0 + floor as f32 * floor_height - 0.5;
                mesh::box_tris(tris, x, by, front_tz, w - 0.2, tw, tw, timber_c);
            }
            // Vertical beams between windows
            for col in 0..=cols {
                let bx = x - w * 0.5 + 0.6 + col as f32 * 2.0;
                mesh::box_tris(tris, bx, ground_y + h * 0.5, front_tz,
                    tw, h - 1.0, tw, timber_c);
            }
            // Diagonal braces (X-pattern between some windows)
            if floors > 1 {
                for col in 0..cols.min(3) {
                    let bx = x - w * 0.5 + 1.6 + col as f32 * 2.0;
                    let by = ground_y + 2.0 + floor_height;
                    // Diagonal beam as a thin rotated box
                    let diag_len = (1.6_f32 * 1.6 + floor_height * floor_height).sqrt();
                    let _diag_angle = (floor_height).atan2(1.6);
                    // Approximate with a thin box (slight tilt baked in position)
                    mesh::box_tris(tris, bx, by + floor_height * 0.5, front_tz,
                        tw, diag_len * 0.8, tw, timber_c);
                }
            }
        }

        // --- Ground-floor shop front with awning ---
        if has_shop {
            let shop_h = 2.5;
            let shop_front_z = z + d * 0.5 + 0.03;
            // Shop front panel (darker wood, full width)
            mesh::box_tris(tris, x, ground_y + shop_h * 0.5, shop_front_z,
                w - 0.4, shop_h, 0.06, SHOP_FRONT_COLOR);
            // Shop window (large glass, recessed)
            let shop_win_w = (w - 2.0).max(1.0);
            mesh::box_tris(tris, x + 0.5, ground_y + 1.4, shop_front_z - 0.04,
                shop_win_w * 0.5, 1.2, 0.04, 0x00BBAA44); // emissive shop (cooler white)
            // Door opening
            mesh::box_tris(tris, x - w * 0.25, ground_y + 1.1, shop_front_z - 0.04,
                0.9, 2.0, 0.06, 0xFF332211);
            // Awning (angled box protruding outward)
            let awning_color = AWNING_COLORS[bi % AWNING_COLORS.len()];
            mesh::box_tris(tris, x, ground_y + shop_h + 0.1, shop_front_z + 0.6,
                w - 0.2, 0.06, 1.2, awning_color);
            // Awning underside brace
            mesh::box_tris(tris, x, ground_y + shop_h + 0.05, shop_front_z + 0.6,
                w - 0.3, 0.02, 1.1, darken(awning_color, 0.6));
            // Hanging shop sign (on bracket from facade)
            let sign_color = SIGN_COLORS[bi % SIGN_COLORS.len()];
            let sign_x = x + w * 0.25;
            // Iron bracket (horizontal cylinder)
            mesh::cylinder_between(tris,
                [sign_x, ground_y + shop_h + 1.5, shop_front_z],
                [sign_x, ground_y + shop_h + 1.5, shop_front_z + 0.5],
                0.02, 4, BALCONY_RAIL_COLOR);
            // Sign board
            mesh::box_tris(tris, sign_x, ground_y + shop_h + 1.0, shop_front_z + 0.5,
                0.6, 0.5, 0.04, sign_color);
        } else {
            // Simple door for non-shop buildings
            mesh::box_tris(tris, x, ground_y + 1.1, z + d * 0.5 - recess_depth * 0.5,
                1.0, 2.2, recess_depth, 0xFF443322);
        }

        // --- Balconies (on ~20% of buildings, 2nd floor front face) ---
        if has_balcony {
            let bal_y = ground_y + 2.0 + floor_height; // 2nd floor
            let bal_z = z + d * 0.5;
            let bal_depth = 0.8;
            let bal_w = w * 0.6;
            // Platform
            mesh::box_tris(tris, x, bal_y - 0.04, bal_z + bal_depth * 0.5,
                bal_w, 0.08, bal_depth, darken(color, 0.75));
            // Railing posts
            let num_posts = ((bal_w / 0.4) as i32).max(2);
            for pi in 0..=num_posts {
                let t = pi as f32 / num_posts as f32;
                let px = x - bal_w * 0.5 + t * bal_w;
                mesh::cylinder_tris(tris, px, bal_y + 0.3, bal_z + bal_depth,
                    0.02, 0.6, 4, BALCONY_RAIL_COLOR);
            }
            // Top rail
            mesh::cylinder_between(tris,
                [x - bal_w * 0.5, bal_y + 0.6, bal_z + bal_depth],
                [x + bal_w * 0.5, bal_y + 0.6, bal_z + bal_depth],
                0.02, 4, BALCONY_RAIL_COLOR);
            // Front railing bottom
            mesh::cylinder_between(tris,
                [x - bal_w * 0.5, bal_y + 0.02, bal_z + bal_depth],
                [x + bal_w * 0.5, bal_y + 0.02, bal_z + bal_depth],
                0.015, 4, BALCONY_RAIL_COLOR);
        }

        // --- Flower boxes under some windows ---
        if bi % 3 == 0 && floors > 1 {
            let flower_floor = 1 + (bi % floors.max(1) as usize) as i32;
            if flower_floor < floors {
                let fy = ground_y + 2.0 + flower_floor as f32 * floor_height - 0.15;
                let fz_pos = z + d * 0.5 + 0.12;
                for col in 0..cols.min(2) {
                    let fx = x - w * 0.5 + 1.2 + col as f32 * 2.0 + win_w * 0.5;
                    mesh::box_tris(tris, fx, fy, fz_pos, win_w + 0.1, 0.12, 0.15, FLOWER_BOX_COLOR);
                    // Flowers
                    let fc = FLOWER_COLORS[col as usize % FLOWER_COLORS.len()];
                    mesh::sphere_tris(tris, fx - 0.12, fy + 0.12, fz_pos + 0.02, 0.08, 0, fc);
                    mesh::sphere_tris(tris, fx + 0.12, fy + 0.12, fz_pos + 0.02, 0.08, 0, fc);
                }
            }
        }

        // --- Roof (3 varieties) ---
        let roof_type = (bi + (color & 0xFF) as usize) % 3;
        let roof_color = darken(color, 0.55);
        match roof_type {
            0 => {
                // Flat roof with parapet
                mesh::box_tris(tris, x, ground_y + h + 0.15, z,
                    w + 0.2, 0.3, d + 0.2, roof_color);
            }
            1 => {
                let peak = h * 0.15 + 1.0;
                mesh::pitched_roof_tris(tris, x, ground_y + h, z, w + 0.3, d + 0.3, peak, roof_color);
                // Dormer window on pitched roof
                if w > 4.0 {
                    let dorm_y = ground_y + h + peak * 0.35;
                    let dorm_z = z + d * 0.5 + 0.1;
                    mesh::box_tris(tris, x, dorm_y, dorm_z, 1.0, 0.8, 0.5, color);
                    mesh::pitched_roof_tris(tris, x, dorm_y + 0.4, dorm_z, 1.2, 0.7, 0.4, roof_color);
                    mesh::box_tris(tris, x, dorm_y, dorm_z + 0.26, 0.5, 0.5, 0.04, 0x00AA7722);
                }
            }
            _ => {
                let peak = h * 0.12 + 0.8;
                mesh::hip_roof_tris(tris, x, ground_y + h, z, w + 0.3, d + 0.3, peak, roof_color);
            }
        }

        // Cornice with more detail (double ledge)
        let cornice_color = darken(color, 0.8);
        mesh::cornice_tris(tris, x, ground_y + h - 0.1, z,
            w, d, 0.12, 0.175, cornice_color);
        mesh::cornice_tris(tris, x, ground_y + h - 0.25, z,
            w, d, 0.08, 0.125, cornice_color);

        // Belt course on taller buildings
        if h > 8.0 {
            mesh::cornice_tris(tris, x, ground_y + h * 0.5, z,
                w, d, 0.15, 0.075, cornice_color);
        }

        // Chimney (40% of buildings)
        if bi % 5 < 2 && h > 4.0 {
            let chim_x = x + w * 0.3;
            let chim_z = z - d * 0.3;
            mesh::box_tris(tris, chim_x, ground_y + h + 1.2, chim_z,
                0.4, 2.4, 0.4, darken(color, 0.5));
            // Chimney cap
            mesh::box_tris(tris, chim_x, ground_y + h + 2.5, chim_z,
                0.5, 0.1, 0.5, darken(color, 0.4));
        }

        buildings.push(Building { x, z, w, d, h, ground_y });
}

fn emit_tree(
    tris: &mut Vec<WorldTri>, trees: &mut Vec<Tree>, terrain: &Terrain,
    rng: &mut Rng, x: f32, z: f32, ti: usize, seed: u64,
) {
    const LEAF_COLORS: [u32; 8] = [
        0xFF38883A, 0xFF44AA44, 0xFF2D8830, 0xFF55BB55,
        0xFF3A9A3A, 0xFF48B048, 0xFF2A7A2C, 0xFF40A840,
    ];
    const BARK_COLOR: u32 = 0xFF443322;
    const BARK_RIDGE_COLOR: u32 = 0xFF332211;

    let ground_y = terrain.height_at(x, z);
    let trunk_h = rng.range(3.5, 6.0);
    let trunk_r = rng.range(0.18, 0.35);
    let canopy_r = rng.range(1.3, 2.5);
    let tree_seed = ti as u64 * 7919 + seed;

    mesh::bark_cylinder_tris(tris, x, ground_y + trunk_h * 0.5, z,
        trunk_r, trunk_h, 10, trunk_r * 0.15, tree_seed, BARK_COLOR, BARK_RIDGE_COLOR);

    let root_count = 2 + (ti % 2);
    for ri in 0..root_count {
        let ra = (ri as f32 / root_count as f32) * std::f32::consts::TAU + (ti as f32 * 2.1);
        let rx = x + ra.cos() * trunk_r * 1.5;
        let rz = z + ra.sin() * trunk_r * 1.5;
        mesh::sphere_tris(tris, rx, ground_y + 0.05, rz, trunk_r * 0.5, 0, BARK_COLOR);
    }

    let num_branches = 2 + (ti % 3);
    let branch_base_y = ground_y + trunk_h * 0.7;
    for bi in 0..num_branches {
        let angle = (bi as f32 / num_branches as f32) * std::f32::consts::TAU + (ti as f32 * 1.23);
        let blen = canopy_r * 0.7;
        let bx = x + angle.cos() * blen * 0.6;
        let bz = z + angle.sin() * blen * 0.6;
        let by = branch_base_y + blen * 0.5;
        mesh::cylinder_between(tris,
            [x, branch_base_y, z], [bx, by, bz],
            trunk_r * 0.45, 5, BARK_COLOR);
        if bi < 2 {
            let sub_angle = angle + 0.6;
            let sbx = bx + sub_angle.cos() * blen * 0.3;
            let sbz = bz + sub_angle.sin() * blen * 0.3;
            let sby = by + blen * 0.25;
            mesh::cylinder_between(tris,
                [bx, by, bz], [sbx, sby, sbz],
                trunk_r * 0.2, 3, BARK_COLOR);
        }
    }

    let num_canopies = 3 + (ti % 3);
    let canopy_base_y = ground_y + trunk_h + canopy_r * 0.2;
    let central_cr = canopy_r * 0.5;
    mesh::leaf_canopy_tris(tris, x, canopy_base_y + canopy_r * 0.1, z,
        central_cr, 60 + (ti * 5) % 20, central_cr * 0.2,
        tree_seed.wrapping_add(9999), &LEAF_COLORS);

    for ci in 0..num_canopies {
        let angle = (ci as f32 / num_canopies as f32) * std::f32::consts::TAU + (ti as f32 * 0.77);
        let spread = canopy_r * 0.5;
        let clx = x + angle.cos() * spread;
        let clz = z + angle.sin() * spread;
        let cly = canopy_base_y + (ci as f32 * 0.15);
        let cr = canopy_r * rng.range(0.35, 0.55);
        let leaves_per_cluster = 50 + (ti * 7 + ci) % 25;
        let leaf_sz = cr * 0.2;
        mesh::leaf_canopy_tris(tris, clx, cly, clz, cr,
            leaves_per_cluster, leaf_sz,
            tree_seed.wrapping_add(ci as u64 * 3571), &LEAF_COLORS);
    }

    trees.push(Tree { x, z, trunk_radius: trunk_r + 0.1 });
}

fn emit_bush(
    tris: &mut Vec<WorldTri>, terrain: &Terrain,
    x: f32, z: f32, bi: usize, seed: u64,
) {
    const BUSH_LEAF_COLORS: [u32; 6] = [
        0xFF2D6B2D, 0xFF336633, 0xFF225522, 0xFF3A7A3A, 0xFF1F5F1F, 0xFF2A6A2A,
    ];
    const BARK_COLOR: u32 = 0xFF443322;
    let gy = terrain.height_at(x, z);
    let hash = (bi as u32).wrapping_mul(2654435761);
    let br = 0.4 + (hash % 500) as f32 / 1000.0;
    let bh = 0.5 + (hash.wrapping_mul(3) % 500) as f32 / 1000.0;
    mesh::bush_tris(tris, x, gy, z, br, bh,
        bi as u64 * 6131 + seed, &BUSH_LEAF_COLORS, BARK_COLOR);
}

fn emit_grass(
    tris: &mut Vec<WorldTri>, terrain: &Terrain, rng: &mut Rng,
    x: f32, z: f32, gi: usize, seed: u64,
) {
    const GRASS_COLORS: [u32; 8] = [
        0xFF55BB55, 0xFF66CC55, 0xFF44AA44, 0xFF5DBB4D,
        0xFF77CC66, 0xFF55AA55, 0xFF6BC060, 0xFF88DD77,
    ];
    let gy = terrain.height_at(x, z);
    let patch_r = rng.range(1.5, 3.5);
    let blade_count = 20 + (gi % 20);
    let blade_h = rng.range(0.2, 0.45);
    let blade_w = rng.range(0.06, 0.12);
    mesh::grass_patch_tris(tris, x, gy, z, patch_r, blade_count,
        blade_h, blade_w, gi as u64 * 4877 + seed,
        &GRASS_COLORS, Some(&|gx, gz| terrain.height_at(gx, gz)));
}

fn emit_rock(
    tris: &mut Vec<WorldTri>, rocks: &mut Vec<Rock>, terrain: &Terrain,
    x: f32, z: f32, scale: f32, ri: usize, seed: u64,
) {
    let ground_y = terrain.height_at(x, z);
    let size = scale;
    mesh::perturbed_sphere_tris(tris, x, ground_y + size * 0.4, z,
        size, 1, 0.25, ri as u64 * 12345 + seed, ROCK_COLOR);
    rocks.push(Rock { x, z, size });
}

fn emit_street_light(
    tris: &mut Vec<WorldTri>, street_lights: &mut Vec<StreetLight>,
    terrain: &Terrain, x: f32, z: f32, _seed: u64,
) {
    let ground_y = terrain.height_at(x, z);

    // Base mounting plate
    let base_y = ground_y + 0.04;
    for pi in 0..8u32 {
        let a0 = (pi as f32 / 8.0) * std::f32::consts::TAU;
        let a1 = ((pi + 1) as f32 / 8.0) * std::f32::consts::TAU;
        mesh::push_tri(tris, [x, base_y, z],
            [x + a1.cos() * 0.25, base_y, z + a1.sin() * 0.25],
            [x + a0.cos() * 0.25, base_y, z + a0.sin() * 0.25],
            [0.0, 1.0, 0.0], LAMP_BASE_COLOR);
    }
    mesh::cylinder_tris(tris, x, ground_y + 0.2, z, 0.12, 0.4, 6, LAMP_BASE_COLOR);
    mesh::cylinder_tris(tris, x, ground_y + 2.7, z, 0.06, 4.6, 8, LAMP_POLE_COLOR);

    // Arm toward road (use arbitrary direction since placement handles positioning)
    let arm_dx = 0.8_f32;
    let arm_dz = 0.0_f32;
    mesh::cylinder_between(tris,
        [x, ground_y + 5.0, z],
        [x + arm_dx, ground_y + 5.1, z + arm_dz],
        0.03, 4, LAMP_POLE_COLOR);

    mesh::sphere_tris(tris, x + arm_dx, ground_y + 5.1, z + arm_dz,
        0.2, 1, LAMP_GLOW_COLOR);

    // Ground light pool
    let pool_x = x + arm_dx;
    let pool_z = z + arm_dz;
    let pool_y = ground_y + 0.03;
    let pool_r = 3.0;
    let pool_color: u32 = 0x00553810;
    for pi in 0..8u32 {
        let a0 = (pi as f32 / 8.0) * std::f32::consts::TAU;
        let a1 = ((pi + 1) as f32 / 8.0) * std::f32::consts::TAU;
        mesh::push_tri(tris, [pool_x, pool_y, pool_z],
            [pool_x + a1.cos() * pool_r, pool_y, pool_z + a1.sin() * pool_r],
            [pool_x + a0.cos() * pool_r, pool_y, pool_z + a0.sin() * pool_r],
            [0.0, 1.0, 0.0], pool_color);
    }

    street_lights.push(StreetLight { x, z, ground_y });
}

/// Unified walk collision — static obstacles (buildings, rocks, trees, lights, walls, bins, interactibles).
/// When `home_idx` is `Some(i)`, skip building `i` and use smaller street light radius (NPC walking).
/// When `home_idx` is `None`, check all buildings and use larger street light radius (player/vehicle).
pub fn check_walk_collision(world: &WorldData, x: f32, z: f32, radius: f32, home_idx: Option<usize>) -> bool {
    for (bi, b) in world.buildings.iter().enumerate() {
        if home_idx == Some(bi) { continue; }
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
    let light_r = if home_idx.is_none() { 0.3 } else { 0.15 };
    for sl in &world.street_lights {
        let dx = x - sl.x;
        let dz = z - sl.z;
        let r2 = light_r + radius;
        if dx * dx + dz * dz < r2 * r2 {
            return true;
        }
    }
    for w in &world.walls {
        if (x - w.x).abs() < w.hw + radius && (z - w.z).abs() < w.hd + radius {
            return true;
        }
    }
    for tb in &world.trash_bins {
        if tb.carried_by.is_some() { continue; }
        let dx = x - tb.x;
        let dz = z - tb.z;
        let r2 = 0.4 + radius;
        if dx * dx + dz * dz < r2 * r2 {
            return true;
        }
    }
    for inter in &world.interactibles {
        let dx = x - inter.x;
        let dz = z - inter.z;
        let r2 = 0.5 + radius;
        if dx * dx + dz * dz < r2 * r2 {
            return true;
        }
    }
    for c in &world.clutter {
        let dx = x - c[0];
        let dz = z - c[1];
        let r2 = c[2] + radius;
        if dx * dx + dz * dz < r2 * r2 {
            return true;
        }
    }
    false
}
