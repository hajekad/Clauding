// Top-down world map renderer — generates PPM image for visual inspection
// Usage: cargo run --bin render_map -- [seed] [time_hours]
// Outputs: debug/map.ppm (terrain + structures + entities)
//          debug/map_collision.ppm (collision geometry overlay)

use clauding::{state, world};

const IMG_SIZE: usize = 2048;
const WORLD: f32 = state::WORLD_SIZE;
const HALF: f32 = state::WORLD_HALF;

fn world_to_px(wx: f32, wz: f32) -> (usize, usize) {
    let px = ((wx + HALF) / WORLD * IMG_SIZE as f32) as usize;
    let pz = ((wz + HALF) / WORLD * IMG_SIZE as f32) as usize;
    (px.min(IMG_SIZE - 1), pz.min(IMG_SIZE - 1))
}

fn set_pixel(buf: &mut [u8], x: usize, z: usize, r: u8, g: u8, b: u8) {
    if x < IMG_SIZE && z < IMG_SIZE {
        let idx = (z * IMG_SIZE + x) * 3;
        buf[idx] = r;
        buf[idx + 1] = g;
        buf[idx + 2] = b;
    }
}

fn blend_pixel(buf: &mut [u8], x: usize, z: usize, r: u8, g: u8, b: u8, alpha: f32) {
    if x < IMG_SIZE && z < IMG_SIZE {
        let idx = (z * IMG_SIZE + x) * 3;
        let a = alpha.clamp(0.0, 1.0);
        let inv = 1.0 - a;
        buf[idx] = (buf[idx] as f32 * inv + r as f32 * a) as u8;
        buf[idx + 1] = (buf[idx + 1] as f32 * inv + g as f32 * a) as u8;
        buf[idx + 2] = (buf[idx + 2] as f32 * inv + b as f32 * a) as u8;
    }
}

fn fill_rect(buf: &mut [u8], cx: f32, cz: f32, hw: f32, hd: f32, r: u8, g: u8, b: u8) {
    let (x0, z0) = world_to_px(cx - hw, cz - hd);
    let (x1, z1) = world_to_px(cx + hw, cz + hd);
    for z in z0..=z1 {
        for x in x0..=x1 {
            set_pixel(buf, x, z, r, g, b);
        }
    }
}

fn fill_circle(buf: &mut [u8], wx: f32, wz: f32, radius: f32, r: u8, g: u8, b: u8) {
    let px_radius = (radius / WORLD * IMG_SIZE as f32).max(1.0) as i32;
    let (cx, cz) = world_to_px(wx, wz);
    let cx = cx as i32;
    let cz = cz as i32;
    for dz in -px_radius..=px_radius {
        for dx in -px_radius..=px_radius {
            if dx * dx + dz * dz <= px_radius * px_radius {
                let px = (cx + dx) as usize;
                let pz = (cz + dz) as usize;
                set_pixel(buf, px, pz, r, g, b);
            }
        }
    }
}

fn draw_line(buf: &mut [u8], x0: f32, z0: f32, x1: f32, z1: f32, width: f32, r: u8, g: u8, b: u8) {
    let dx = x1 - x0;
    let dz = z1 - z0;
    let len = (dx * dx + dz * dz).sqrt();
    if len < 0.01 { return; }
    let steps = (len / WORLD * IMG_SIZE as f32 * 2.0) as usize + 1;
    let hw = width * 0.5;
    // Perpendicular
    let px = -dz / len;
    let pz = dx / len;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let cx = x0 + dx * t;
        let cz = z0 + dz * t;
        // Draw width by stepping perpendicular
        let w_steps = (hw / WORLD * IMG_SIZE as f32).max(1.0) as i32;
        for w in -w_steps..=w_steps {
            let wf = w as f32 / w_steps as f32 * hw;
            let wx = cx + px * wf;
            let wz = cz + pz * wf;
            let (ppx, ppz) = world_to_px(wx, wz);
            set_pixel(buf, ppx, ppz, r, g, b);
        }
    }
}

fn draw_cross(buf: &mut [u8], wx: f32, wz: f32, size: i32, r: u8, g: u8, b: u8) {
    let (cx, cz) = world_to_px(wx, wz);
    for d in -size..=size {
        set_pixel(buf, (cx as i32 + d) as usize, cz, r, g, b);
        set_pixel(buf, cx, (cz as i32 + d) as usize, r, g, b);
    }
}

fn main() {
    let _ = std::fs::create_dir_all("debug");
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let sim_hours: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(14.0);

    eprintln!("Generating world (seed={})...", seed);
    let mut game = state::GameState::init(1, 1, seed);

    // Simulate to get NPCs into working positions
    let target_time = sim_hours;
    let dt: f32 = 1.0 / 30.0;
    let mut ticks = 0u64;
    while game.time_of_day < target_time || ticks < 100 {
        game.tick_headless(dt);
        ticks += 1;
        if game.time_of_day >= target_time && ticks > 100 { break; }
    }
    eprintln!("Simulated to {:.1}h ({} ticks), rendering...", game.time_of_day, ticks);

    // === RENDER MAIN MAP ===
    let mut buf = vec![0u8; IMG_SIZE * IMG_SIZE * 3];

    // 1. Terrain heightmap (green-brown gradient)
    for pz in 0..IMG_SIZE {
        for px in 0..IMG_SIZE {
            let wx = px as f32 / IMG_SIZE as f32 * WORLD - HALF;
            let wz = pz as f32 / IMG_SIZE as f32 * WORLD - HALF;
            let h = game.terrain.height_at(wx, wz);
            let t = ((h + 2.0) / 8.0).clamp(0.0, 1.0);
            let r = (60.0 + t * 80.0) as u8;
            let g = (80.0 + t * 60.0) as u8;
            let b = (40.0 + t * 30.0) as u8;
            set_pixel(&mut buf, px, pz, r, g, b);
        }
    }

    // 2. Dockyard area (blue-gray)
    for pz in 0..IMG_SIZE {
        let wz = pz as f32 / IMG_SIZE as f32 * WORLD - HALF;
        if wz > state::DOCK_Z_START {
            for px in 0..IMG_SIZE {
                blend_pixel(&mut buf, px, pz, 60, 80, 120, 0.5);
            }
        }
    }

    // 3. River (blue)
    for seg in &game.world.river_segments {
        draw_line(&mut buf, seg.x1, seg.z1, seg.x2, seg.z2, seg.width, 40, 100, 180);
    }

    // 4. Roads (dark gray for car, light brown for field)
    for seg in &game.road_network.segments {
        let (r, g, b, w) = match seg.tier {
            state::RoadTier::CarRoad => (80, 80, 80, state::CAR_ROAD_WIDTH + state::SIDEWALK_WIDTH * 2.0),
            state::RoadTier::FieldRoad => (110, 95, 70, state::FIELD_ROAD_WIDTH),
        };
        draw_line(&mut buf, seg.x0, seg.z0, seg.x1, seg.z1, w, r, g, b);
    }
    // Car road surface (darker center)
    for seg in &game.road_network.segments {
        if seg.tier == state::RoadTier::CarRoad {
            draw_line(&mut buf, seg.x0, seg.z0, seg.x1, seg.z1, state::CAR_ROAD_WIDTH, 60, 60, 65);
        }
    }

    // 5. Road nodes (white dots)
    for node in &game.road_network.nodes {
        fill_circle(&mut buf, node[0], node[1], 2.0, 255, 255, 255);
    }

    // 6. Buildings (gray rectangles with dark outline)
    for (bi, b) in game.world.buildings.iter().enumerate() {
        fill_rect(&mut buf, b.x, b.z, b.w, b.d, 140, 130, 120);
        // Highlight NPC homes
        let is_home = game.world.npcs.iter().any(|n| n.home_idx == bi);
        if is_home {
            // Thin border
            let (x0, z0) = world_to_px(b.x - b.w, b.z - b.d);
            let (x1, z1) = world_to_px(b.x + b.w, b.z + b.d);
            for x in x0..=x1 { set_pixel(&mut buf, x, z0, 200, 180, 80); set_pixel(&mut buf, x, z1, 200, 180, 80); }
            for z in z0..=z1 { set_pixel(&mut buf, x0, z, 200, 180, 80); set_pixel(&mut buf, x1, z, 200, 180, 80); }
        }
    }

    // 7. Walls (brown)
    for w in &game.world.walls {
        fill_rect(&mut buf, w.x, w.z, w.hw, w.hd, 120, 80, 40);
    }

    // 8. Trees (green circles)
    for t in &game.world.trees {
        fill_circle(&mut buf, t.x, t.z, 1.5, 30, 130, 30);
    }

    // 9. Rocks (gray)
    for r in &game.world.rocks {
        fill_circle(&mut buf, r.x, r.z, r.size, 110, 105, 100);
    }

    // 10. Street lights (yellow dots)
    for l in &game.world.street_lights {
        fill_circle(&mut buf, l.x, l.z, 0.8, 255, 230, 100);
    }

    // 11. Trash bins (brown squares)
    for bin in &game.world.trash_bins {
        fill_rect(&mut buf, bin.x, bin.z, 0.5, 0.5, 100, 70, 30);
    }

    // 12. Interactibles (colored by kind)
    for inter in &game.world.interactibles {
        let (r, g, b) = match inter.kind {
            state::InteractibleKind::VendingMachine => (0, 200, 200),
            state::InteractibleKind::Dumpster => (80, 100, 60),
            state::InteractibleKind::Mailbox => (50, 50, 200),
            state::InteractibleKind::FireHydrant => (200, 50, 50),
            state::InteractibleKind::NewspaperStand => (200, 200, 150),
            state::InteractibleKind::ParkBench => (139, 90, 43),
            state::InteractibleKind::Atm => (200, 200, 0),
            state::InteractibleKind::PhoneBooth => (150, 150, 200),
            state::InteractibleKind::Payphone => (180, 180, 180),
        };
        fill_circle(&mut buf, inter.x, inter.z, 1.0, r, g, b);
    }

    // 13. Parking spots (light gray outlines)
    for spot in &game.road_network.parking_spots {
        let hw = state::PARKING_SPOT_WIDTH * 0.5;
        let hd = state::PARKING_SPOT_LENGTH * 0.5;
        let (x0, z0) = world_to_px(spot.x - hw, spot.z - hd);
        let (x1, z1) = world_to_px(spot.x + hw, spot.z + hd);
        for x in x0..=x1 { blend_pixel(&mut buf, x, z0, 180, 180, 180, 0.3); blend_pixel(&mut buf, x, z1, 180, 180, 180, 0.3); }
        for z in z0..=z1 { blend_pixel(&mut buf, x0, z, 180, 180, 180, 0.3); blend_pixel(&mut buf, x1, z, 180, 180, 180, 0.3); }
    }

    // 14. Vehicles (colored rectangles)
    for v in &game.world.vehicles {
        let (r, g, b) = ((v.color >> 16 & 0xFF) as u8, (v.color >> 8 & 0xFF) as u8, (v.color & 0xFF) as u8);
        if v.parked {
            fill_circle(&mut buf, v.x, v.z, 1.2, r / 2, g / 2, b / 2); // dim parked
        } else {
            fill_circle(&mut buf, v.x, v.z, 1.5, r, g, b);
        }
    }

    // 15. Items (colored diamonds)
    for item in &game.world.items {
        if !item.active && !item.falling { continue; }
        let (r, g, b) = match item.kind {
            state::ItemKind::Health => (255, 50, 50),
            state::ItemKind::Money => (255, 215, 0),
            state::ItemKind::Stamina => (255, 255, 50),
            state::ItemKind::Food => (255, 160, 50),
            state::ItemKind::Water => (50, 200, 255),
        };
        draw_cross(&mut buf, item.x, item.z, 1, r, g, b);
    }

    // 16. NPCs (colored by state, with direction indicator)
    for (i, npc) in game.world.npcs.iter().enumerate() {
        let (r, g, b) = match npc.state {
            state::NpcState::Sleeping => (100, 100, 150),      // blue-gray
            state::NpcState::HomeTask => (150, 150, 100),      // tan
            state::NpcState::GoingToWork => (200, 200, 50),    // yellow
            state::NpcState::Working => (50, 255, 50),         // bright green
            state::NpcState::GoingHome => (200, 150, 50),      // orange
            state::NpcState::Driving => (255, 255, 255),       // white
            state::NpcState::Interacting => (200, 50, 200),    // purple
            state::NpcState::KnockedOut => (255, 0, 0),        // red
        };
        fill_circle(&mut buf, npc.x, npc.z, 1.5, r, g, b);
        // Label with index
        let (px, pz) = world_to_px(npc.x, npc.z);
        // Direction arrow
        let fwd_x = -npc.rot_y.sin() * 3.0;
        let fwd_z = -npc.rot_y.cos() * 3.0;
        let (ax, az) = world_to_px(npc.x + fwd_x, npc.z + fwd_z);
        set_pixel(&mut buf, ax, az, 255, 255, 255);

        // Mark stuck NPCs with red X
        if npc.stuck_timer > 2.0 {
            draw_cross(&mut buf, npc.x, npc.z, 3, 255, 0, 0);
        }
        let _ = (px, pz, i);
    }

    // Write PPM
    let path = "debug/map.ppm";
    let mut ppm = Vec::with_capacity(IMG_SIZE * IMG_SIZE * 3 + 50);
    ppm.extend_from_slice(format!("P6\n{} {}\n255\n", IMG_SIZE, IMG_SIZE).as_bytes());
    ppm.extend_from_slice(&buf);
    std::fs::write(path, &ppm).unwrap();
    eprintln!("Wrote main map to {} ({:.1}MB)", path, ppm.len() as f32 / 1e6);

    // === RENDER COLLISION MAP ===
    // Shows what NPCs actually collide with — buildings, rocks, trees, walls, lights, bins, interactibles, river
    let mut cbuf = vec![0u8; IMG_SIZE * IMG_SIZE * 3];

    // Terrain base (dark)
    for i in 0..IMG_SIZE * IMG_SIZE {
        cbuf[i * 3] = 20;
        cbuf[i * 3 + 1] = 25;
        cbuf[i * 3 + 2] = 20;
    }

    // River collision zone (blue, with margin)
    for pz in 0..IMG_SIZE {
        for px in 0..IMG_SIZE {
            let wx = px as f32 / IMG_SIZE as f32 * WORLD - HALF;
            let wz = pz as f32 / IMG_SIZE as f32 * WORLD - HALF;
            // Check if on river (with 2m NPC avoidance margin)
            for seg in &game.world.river_segments {
                let d = world::point_to_segment_dist(wx, wz, seg.x1, seg.z1, seg.x2, seg.z2);
                if d < seg.width * 0.5 {
                    set_pixel(&mut cbuf, px, pz, 30, 60, 150);
                    break;
                } else if d < seg.width * 0.5 + 2.0 {
                    blend_pixel(&mut cbuf, px, pz, 30, 60, 120, 0.3);
                    break;
                }
            }
        }
    }

    // Building collision (red — NPC collision exclusion zones, with 0.4m radius)
    for b in &game.world.buildings {
        let margin = 0.4;
        fill_rect(&mut cbuf, b.x, b.z, b.w + margin, b.d + margin, 180, 40, 40);
    }

    // Wall collision (orange)
    for w in &game.world.walls {
        fill_rect(&mut cbuf, w.x, w.z, w.hw + 0.4, w.hd + 0.4, 200, 120, 40);
    }

    // Tree collision (green circles, trunk_radius + 0.4)
    for t in &game.world.trees {
        fill_circle(&mut cbuf, t.x, t.z, t.trunk_radius + 0.4, 40, 150, 40);
    }

    // Rock collision (gray, size + 0.4)
    for r in &game.world.rocks {
        fill_circle(&mut cbuf, r.x, r.z, r.size + 0.4, 120, 120, 100);
    }

    // Street light collision (yellow, 0.15 + 0.4 = 0.55m)
    for l in &game.world.street_lights {
        fill_circle(&mut cbuf, l.x, l.z, 0.55, 200, 200, 50);
    }

    // Trash bin collision (brown, 0.4 + 0.4 = 0.8m)
    for bin in &game.world.trash_bins {
        fill_circle(&mut cbuf, bin.x, bin.z, 0.8, 150, 100, 30);
    }

    // Interactible collision (cyan, 0.5 + 0.4 = 0.9m)
    for inter in &game.world.interactibles {
        fill_circle(&mut cbuf, inter.x, inter.z, 0.9, 50, 180, 180);
    }

    // NPC positions (bright dots, color by stuck state)
    for npc in &game.world.npcs {
        let (r, g, b) = if npc.stuck_timer > 2.0 {
            (255, 0, 0) // stuck = red
        } else {
            (255, 255, 255) // normal = white
        };
        fill_circle(&mut cbuf, npc.x, npc.z, 1.0, r, g, b);
    }

    // NPC home positions (yellow outline circles)
    for npc in &game.world.npcs {
        let b = &game.world.buildings[npc.home_idx];
        let (px, pz) = world_to_px(b.x, b.z);
        // Draw small ring
        for a in 0..32 {
            let angle = a as f32 / 32.0 * std::f32::consts::TAU;
            let dx = (angle.cos() * 4.0) as i32;
            let dz = (angle.sin() * 4.0) as i32;
            set_pixel(&mut cbuf, (px as i32 + dx) as usize, (pz as i32 + dz) as usize, 255, 220, 50);
        }
    }

    let path2 = "debug/map_collision.ppm";
    let mut ppm2 = Vec::with_capacity(IMG_SIZE * IMG_SIZE * 3 + 50);
    ppm2.extend_from_slice(format!("P6\n{} {}\n255\n", IMG_SIZE, IMG_SIZE).as_bytes());
    ppm2.extend_from_slice(&cbuf);
    std::fs::write(path2, &ppm2).unwrap();
    eprintln!("Wrote collision map to {} ({:.1}MB)", path2, ppm2.len() as f32 / 1e6);

    // === STATS DUMP ===
    eprintln!("\n=== WORLD STATS ===");
    eprintln!("Buildings: {}", game.world.buildings.len());
    eprintln!("Trees: {}", game.world.trees.len());
    eprintln!("Rocks: {}", game.world.rocks.len());
    eprintln!("Street lights: {}", game.world.street_lights.len());
    eprintln!("Trash bins: {}", game.world.trash_bins.len());
    eprintln!("Interactibles: {}", game.world.interactibles.len());
    eprintln!("Walls: {}", game.world.walls.len());
    eprintln!("River segments: {}", game.world.river_segments.len());
    eprintln!("Parking spots: {}", game.road_network.parking_spots.len());
    eprintln!("Road segments: {}", game.road_network.segments.len());
    eprintln!("Road nodes: {}", game.road_network.nodes.len());
    eprintln!("Vehicles: {} (parked: {}, moving: {})",
        game.world.vehicles.len(),
        game.world.vehicles.iter().filter(|v| v.parked).count(),
        game.world.vehicles.iter().filter(|v| v.speed.abs() > 0.5).count());
    eprintln!("NPCs: {} (working: {}, stuck: {})",
        game.world.npcs.len(),
        game.world.npcs.iter().filter(|n| n.state == state::NpcState::Working).count(),
        game.world.npcs.iter().filter(|n| n.stuck_timer > 2.0).count());
    eprintln!("Items: {} active / {} total",
        game.world.items.iter().filter(|i| i.active).count(),
        game.world.items.len());

    // List stuck NPCs with positions
    eprintln!("\n=== STUCK NPCs (stuck_timer > 2s) ===");
    for (i, npc) in game.world.npcs.iter().enumerate() {
        if npc.stuck_timer > 2.0 {
            let home = &game.world.buildings[npc.home_idx];
            eprintln!("  NPC[{:2}] pos=({:6.1},{:6.1}) stuck={:.1}s home=({:.1},{:.1}) job={:?} state={:?}",
                i, npc.x, npc.z, npc.stuck_timer, home.x, home.z,
                npc.job, npc.state);
        }
    }

    // Collision density analysis — find hotspots
    eprintln!("\n=== COLLISION DENSITY HOTSPOTS (10m grid) ===");
    let grid = 50; // 10m cells
    let cell_size = WORLD / grid as f32;
    let mut density = vec![0u32; grid * grid];
    // Count collision objects per cell
    for b in &game.world.buildings {
        let gx = ((b.x + HALF) / cell_size) as usize;
        let gz = ((b.z + HALF) / cell_size) as usize;
        if gx < grid && gz < grid { density[gz * grid + gx] += 3; } // buildings count more
    }
    for w in &game.world.walls {
        let gx = ((w.x + HALF) / cell_size) as usize;
        let gz = ((w.z + HALF) / cell_size) as usize;
        if gx < grid && gz < grid { density[gz * grid + gx] += 2; }
    }
    for t in &game.world.trees {
        let gx = ((t.x + HALF) / cell_size) as usize;
        let gz = ((t.z + HALF) / cell_size) as usize;
        if gx < grid && gz < grid { density[gz * grid + gx] += 1; }
    }
    for l in &game.world.street_lights {
        let gx = ((l.x + HALF) / cell_size) as usize;
        let gz = ((l.z + HALF) / cell_size) as usize;
        if gx < grid && gz < grid { density[gz * grid + gx] += 1; }
    }
    for inter in &game.world.interactibles {
        let gx = ((inter.x + HALF) / cell_size) as usize;
        let gz = ((inter.z + HALF) / cell_size) as usize;
        if gx < grid && gz < grid { density[gz * grid + gx] += 1; }
    }
    // Find top 10 densest cells
    let mut cells: Vec<(usize, u32)> = density.iter().enumerate().map(|(i, &d)| (i, d)).collect();
    cells.sort_by(|a, b| b.1.cmp(&a.1));
    for &(idx, count) in cells.iter().take(10) {
        let gx = idx % grid;
        let gz = idx / grid;
        let wx = gx as f32 * cell_size - HALF + cell_size * 0.5;
        let wz = gz as f32 * cell_size - HALF + cell_size * 0.5;
        eprintln!("  Cell ({:3.0},{:3.0}) density={} objects", wx, wz, count);
    }

    // NPC homes near collision hotspots
    eprintln!("\n=== NPC HOMES IN DENSE COLLISION AREAS ===");
    for (i, npc) in game.world.npcs.iter().enumerate() {
        let home = &game.world.buildings[npc.home_idx];
        let gx = ((home.x + HALF) / cell_size) as usize;
        let gz = ((home.z + HALF) / cell_size) as usize;
        if gx < grid && gz < grid && density[gz * grid + gx] > 5 {
            eprintln!("  NPC[{:2}] home=({:6.1},{:6.1}) density={}", i, home.x, home.z, density[gz * grid + gx]);
        }
    }
}
