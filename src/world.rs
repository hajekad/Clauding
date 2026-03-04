// World generation: buildings, roads, trees, rocks, street lights
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
const GROUND_COLOR: u32 = 0xFF337733;
const ROAD_COLOR: u32 = 0xFF444444;
const ROAD_LINE_COLOR: u32 = 0xFFCCCC33;
const LAMP_POLE_COLOR: u32 = 0xFF666666;
const LAMP_GLOW_COLOR: u32 = 0xFFFFEE88;

const NUM_ROADS: usize = 5;

fn on_road(x: f32, z: f32, road_positions: &[f32]) -> bool {
    for &r in road_positions {
        if (x - r).abs() < ROAD_WIDTH * 0.5 + 1.0 { return true; }
        if (z - r).abs() < ROAD_WIDTH * 0.5 + 1.0 { return true; }
    }
    false
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

pub fn generate_world(game: &mut GameState) {
    let mut rng = Rng::new(game.world_seed);
    let mut tris = Vec::with_capacity(8000);

    // Generate road positions from seed (5 roads, evenly spaced + jitter)
    let spacing = WORLD_SIZE / (NUM_ROADS as f32 + 1.0);
    let mut road_positions = Vec::with_capacity(NUM_ROADS);
    for i in 0..NUM_ROADS {
        let base = -WORLD_HALF + spacing * (i as f32 + 1.0);
        let jitter = rng.range(-spacing * 0.15, spacing * 0.15);
        road_positions.push(base + jitter);
    }
    game.road_positions = road_positions;
    let roads = &game.road_positions;

    // Ground plane (split into tiles for better z-buffer precision)
    let tile = 50.0;
    let tiles = (WORLD_SIZE / tile) as i32;
    for tx in -tiles/2..tiles/2 {
        for tz in -tiles/2..tiles/2 {
            let x0 = tx as f32 * tile;
            let z0 = tz as f32 * tile;
            let x1 = x0 + tile;
            let z1 = z0 + tile;
            tris.push(WorldTri { v: [[x0,0.0,z0],[x1,0.0,z0],[x1,0.0,z1]], normal: [0.0,1.0,0.0], color: GROUND_COLOR });
            tris.push(WorldTri { v: [[x0,0.0,z0],[x1,0.0,z1],[x0,0.0,z1]], normal: [0.0,1.0,0.0], color: GROUND_COLOR });
        }
    }

    // Roads
    let rh = ROAD_WIDTH * 0.5;
    let line_w = 0.15;
    for &r in roads {
        // Horizontal road (along X)
        tris.push(WorldTri { v: [[-WORLD_HALF,0.05,r-rh],[WORLD_HALF,0.05,r-rh],[WORLD_HALF,0.05,r+rh]], normal: [0.0,1.0,0.0], color: ROAD_COLOR });
        tris.push(WorldTri { v: [[-WORLD_HALF,0.05,r-rh],[WORLD_HALF,0.05,r+rh],[-WORLD_HALF,0.05,r+rh]], normal: [0.0,1.0,0.0], color: ROAD_COLOR });
        // Center line
        tris.push(WorldTri { v: [[-WORLD_HALF,0.08,r-line_w],[WORLD_HALF,0.08,r-line_w],[WORLD_HALF,0.08,r+line_w]], normal: [0.0,1.0,0.0], color: ROAD_LINE_COLOR });
        tris.push(WorldTri { v: [[-WORLD_HALF,0.08,r-line_w],[WORLD_HALF,0.08,r+line_w],[-WORLD_HALF,0.08,r+line_w]], normal: [0.0,1.0,0.0], color: ROAD_LINE_COLOR });
        // Vertical road (along Z)
        tris.push(WorldTri { v: [[r-rh,0.05,-WORLD_HALF],[r+rh,0.05,-WORLD_HALF],[r+rh,0.05,WORLD_HALF]], normal: [0.0,1.0,0.0], color: ROAD_COLOR });
        tris.push(WorldTri { v: [[r-rh,0.05,-WORLD_HALF],[r+rh,0.05,WORLD_HALF],[r-rh,0.05,WORLD_HALF]], normal: [0.0,1.0,0.0], color: ROAD_COLOR });
        // Center line
        tris.push(WorldTri { v: [[r-line_w,0.08,-WORLD_HALF],[r+line_w,0.08,-WORLD_HALF],[r+line_w,0.08,WORLD_HALF]], normal: [0.0,1.0,0.0], color: ROAD_LINE_COLOR });
        tris.push(WorldTri { v: [[r-line_w,0.08,-WORLD_HALF],[r+line_w,0.08,WORLD_HALF],[r-line_w,0.08,WORLD_HALF]], normal: [0.0,1.0,0.0], color: ROAD_LINE_COLOR });
    }

    // Buildings
    for _ in 0..NUM_BUILDINGS {
        let mut x;
        let mut z;
        loop {
            x = rng.range(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            z = rng.range(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            if !on_road(x, z, roads) { break; }
        }
        let w = rng.range(3.0, 8.0);
        let d = rng.range(3.0, 8.0);
        let h = rng.range(3.0, 20.0);
        let color = rng.pick(&BUILDING_COLORS);
        box_tris(&mut tris, x, h * 0.5, z, w, h, d, color);
        // Window details: darker rectangles on front and back faces
        let win_color = 0xFF222244;
        let win_h = 1.2;
        let win_w = 0.8;
        let floors = ((h - 1.0) / 3.0) as i32;
        let cols = ((w - 1.0) / 2.0) as i32;
        for floor in 0..floors {
            let wy = 2.0 + floor as f32 * 3.0;
            for col in 0..cols {
                let wx = x - w * 0.5 + 1.2 + col as f32 * 2.0;
                // Front face windows (z + d/2)
                let fz = z + d * 0.5 + 0.01;
                tris.push(WorldTri { v: [[wx, wy, fz], [wx+win_w, wy, fz], [wx+win_w, wy+win_h, fz]], normal: [0.0,0.0,1.0], color: win_color });
                tris.push(WorldTri { v: [[wx, wy, fz], [wx+win_w, wy+win_h, fz], [wx, wy+win_h, fz]], normal: [0.0,0.0,1.0], color: win_color });
                // Back face windows
                let bz = z - d * 0.5 - 0.01;
                tris.push(WorldTri { v: [[wx+win_w, wy, bz], [wx, wy, bz], [wx, wy+win_h, bz]], normal: [0.0,0.0,-1.0], color: win_color });
                tris.push(WorldTri { v: [[wx+win_w, wy, bz], [wx, wy+win_h, bz], [wx+win_w, wy+win_h, bz]], normal: [0.0,0.0,-1.0], color: win_color });
            }
        }
        game.world.buildings.push(Building { x, z, w, d });
    }

    // Trees
    for _ in 0..NUM_TREES {
        let mut x;
        let mut z;
        loop {
            x = rng.range(-WORLD_HALF + 2.0, WORLD_HALF - 2.0);
            z = rng.range(-WORLD_HALF + 2.0, WORLD_HALF - 2.0);
            if !on_road(x, z, roads) { break; }
        }
        let trunk_h = rng.range(1.5, 3.5);
        let canopy_r = rng.range(1.0, 2.5);
        // Trunk
        box_tris(&mut tris, x, trunk_h * 0.5, z, 0.4, trunk_h, 0.4, TRUNK_COLOR);
        // Canopy
        let canopy_color = rng.pick(&CANOPY_COLORS);
        octahedron_tris(&mut tris, x, trunk_h + canopy_r * 0.6, z, canopy_r, canopy_color);
    }

    // Rocks
    for _ in 0..NUM_ROCKS {
        let x = rng.range(-WORLD_HALF + 3.0, WORLD_HALF - 3.0);
        let z = rng.range(-WORLD_HALF + 3.0, WORLD_HALF - 3.0);
        let size = rng.range(0.5, 1.5);
        // Slightly irregular octahedron
        octahedron_tris(&mut tris, x, size * 0.4, z, size, ROCK_COLOR);
        game.world.rocks.push(Rock { x, z, size });
    }

    // Street lights
    let num_roads = roads.len();
    for _ in 0..NUM_STREET_LIGHTS {
        let road_idx = rng.next() as usize % num_roads;
        let is_horiz = rng.next() % 2 == 0;
        let along = rng.range(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
        let offset = ROAD_WIDTH * 0.5 + 0.5;
        let side = if rng.next() % 2 == 0 { offset } else { -offset };
        let (x, z) = if is_horiz {
            (along, roads[road_idx] + side)
        } else {
            (roads[road_idx] + side, along)
        };
        // Pole
        box_tris(&mut tris, x, 2.5, z, 0.15, 5.0, 0.15, LAMP_POLE_COLOR);
        // Lamp
        octahedron_tris(&mut tris, x, 5.2, z, 0.3, LAMP_GLOW_COLOR);
    }

    // Vehicles (spawned on roads)
    for i in 0..NUM_VEHICLES {
        let road_idx = rng.next() as usize % num_roads;
        let is_horiz = rng.next() % 2 == 0;
        let along = rng.range(-WORLD_HALF + 10.0, WORLD_HALF - 10.0);
        let offset = rng.range(-ROAD_WIDTH * 0.3, ROAD_WIDTH * 0.3);
        let (x, z, rot) = if is_horiz {
            (along, roads[road_idx] + offset, std::f32::consts::FRAC_PI_2)
        } else {
            (roads[road_idx] + offset, along, 0.0)
        };
        let color = rng.pick(&VEHICLE_COLORS);
        let ai_active = rng.next() % 3 == 0; // 1/3 of cars drive around
        let vehicle_rng = rng.fork(i as u64);
        let mut v = Vehicle {
            x, z, rot_y: rot, speed: 0.0, color, occupied: false,
            ai_active, ai_target_x: x, ai_target_z: z, rng: vehicle_rng,
        };
        if ai_active {
            // Give them an initial target
            let t = rng.range(-80.0, 80.0);
            if is_horiz {
                v.ai_target_x = t;
                v.ai_target_z = roads[road_idx];
            } else {
                v.ai_target_x = roads[road_idx];
                v.ai_target_z = t;
            }
        }
        game.world.vehicles.push(v);
    }

    // NPCs (pedestrians near roads)
    for i in 0..NUM_NPCS {
        let road_idx = rng.next() as usize % num_roads;
        let is_horiz = rng.next() % 2 == 0;
        let along = rng.range(-WORLD_HALF + 10.0, WORLD_HALF - 10.0);
        let sidewalk = ROAD_WIDTH * 0.5 + 1.5;
        let side = if rng.next() % 2 == 0 { sidewalk } else { -sidewalk };
        let (x, z) = if is_horiz {
            (along, roads[road_idx] + side)
        } else {
            (roads[road_idx] + side, along)
        };
        let shirt_color = rng.pick(&NPC_SHIRT_COLORS);
        let pants_color = rng.pick(&NPC_PANTS_COLORS);
        let rot_y = rng.range(0.0, std::f32::consts::TAU);
        let npc_rng = rng.fork(NUM_VEHICLES as u64 + i as u64);
        game.world.npcs.push(Npc {
            x, z, rot_y, walk_phase: rng.range(0.0, 6.0),
            target_x: x + rng.range(-10.0, 10.0),
            target_z: z + rng.range(-10.0, 10.0),
            shirt_color, pants_color, rng: npc_rng,
        });
    }

    // Items (scattered around the world, not on roads or inside buildings)
    let item_kinds = [ItemKind::Health, ItemKind::Money, ItemKind::Stamina];
    for _ in 0..NUM_ITEMS {
        let mut x;
        let mut z;
        loop {
            x = rng.range(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            z = rng.range(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            if !on_road(x, z, roads) { break; }
        }
        let kind = item_kinds[rng.next() as usize % 3];
        game.world.items.push(Item {
            x, z, kind, active: true, respawn_timer: 0.0,
            spin_phase: rng.range(0.0, 6.0),
        });
    }

    eprintln!("World: {} tris, {} vehicles, {} npcs, {} items",
        tris.len(), game.world.vehicles.len(), game.world.npcs.len(), game.world.items.len());
    game.world.static_tris = tris;
}

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
    false
}
