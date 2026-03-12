// 3D Space Probe — queries world positions and generates cross-section views
// Usage: cargo run --bin probe3d -- [seed] [command] [args...]
// Commands:
//   point <x> <z>           — full info at world position
//   line <x0> <z0> <x1> <z1> [steps] — sample line between two points
//   grid <cx> <cz> <size> <step> — grid scan around center
//   cross_section <x|z> <pos> [from] [to] [step] — height profile
//   npc_homes               — analyze all NPC home positions
//   collision_scan [step]   — full collision map analysis
//   density [radius]        — object density analysis
//   buildings               — list all buildings
//   roads                   — road network analysis
//   river                   — river analysis
//   parking                 — parking spot analysis
//   vehicles                — vehicle analysis
//   reachability            — walkability grid, flood-fill components, building/NPC reachability
//   slopes [step]           — terrain slope analysis + entities on slopes with tilt angles
//   pathfinding <x0> <z0> <x1> <z1> — simulate NPC walking between two points
//   bins                    — trash bin accessibility analysis

use clauding::{npc, rng, state, world};

fn surf_str(s: state::Surface) -> &'static str {
    match s {
        state::Surface::CarRoad => "CarRoad",
        state::Surface::Sidewalk => "Sidewalk",
        state::Surface::FieldRoad => "FieldRd",
        state::Surface::Terrain => "Terrain",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);

    // Generate world using GameState (same as render_map)
    let mut game = state::GameState::new(1, 1, seed);
    world::generate_world(&mut game);

    let cmd = args.get(2).map(|s| s.as_str()).unwrap_or("summary");

    match cmd {
        "point" => {
            let x: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let z: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            probe_point(&game.world, &game.road_network, &game.terrain, x, z);
        }
        "line" => {
            let x0: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(-100.0);
            let z0: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let x1: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(100.0);
            let z1: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let steps: usize = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(50);
            probe_line(&game.world, &game.road_network, &game.terrain, x0, z0, x1, z1, steps);
        }
        "grid" => {
            let cx: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let cz: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let size: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(50.0);
            let step: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(2.0);
            probe_grid(&game.world, &game.road_network, cx, cz, size, step);
        }
        "cross_section" => {
            let axis = args.get(3).map(|s| s.as_str()).unwrap_or("x");
            let pos: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let from: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(-250.0);
            let to: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(250.0);
            let step: f32 = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(2.0);
            probe_cross_section(&game.world, &game.road_network, &game.terrain, axis, pos, from, to, step);
        }
        "npc_homes" => {
            analyze_npc_homes(&game.world, &game.road_network, &game.terrain);
        }
        "collision_scan" => {
            let step: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2.0);
            collision_full_scan(&game.world, step);
        }
        "density" => {
            let _radius: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20.0);
            density_analysis(&game.world);
        }
        "buildings" => {
            list_buildings(&game.world, &game.road_network, &game.terrain);
        }
        "roads" => {
            analyze_roads(&game.road_network);
        }
        "river" => {
            analyze_river(&game.world, &game.road_network, &game.terrain);
        }
        "parking" => {
            analyze_parking(&game.world, &game.road_network);
        }
        "vehicles" => {
            analyze_vehicles(&game.world);
        }
        "reachability" => {
            analyze_reachability(&game.world, &game.road_network);
        }
        "slopes" => {
            let step: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5.0);
            analyze_slopes(&game.world, &game.terrain, step);
        }
        "pathfinding" => {
            analyze_pathfinding(&mut game, &args);
        }
        "bins" => {
            analyze_bins(&game);
        }
        _ => {
            print_summary(&game.world, &game.road_network, &game.terrain, seed);
        }
    }
}

fn probe_point(world: &state::WorldData, net: &state::RoadNetwork, terrain: &state::Terrain, x: f32, z: f32) {
    println!("=== PROBE POINT ({:.1}, {:.1}) ===", x, z);
    println!();

    let h = terrain.height_at(x, z);
    let normal = terrain.normal_at(x, z);
    println!("Terrain height: {:.2}m", h);
    println!("Normal: ({:.3}, {:.3}, {:.3})", normal[0], normal[1], normal[2]);

    let surface = world::surface_at(x, z, net);
    println!("Surface: {}", surf_str(surface));

    let on_riv = world::on_river(x, z, &world.river_segments);
    println!("On river: {}", on_riv);
    if !world.river_segments.is_empty() {
        let min_dist = world.river_segments.iter()
            .map(|s| world::point_to_segment_dist(x, z, s.x1, s.z1, s.x2, s.z2))
            .fold(f32::MAX, f32::min);
        println!("Distance to river: {:.1}m", min_dist);
    }

    let building_col = world::check_walk_collision(world, x, z, 0.4, None);
    let npc_col = world::check_walk_collision(world, x, z, 0.4, Some(usize::MAX));
    println!("Building collision (r=0.4): {}", building_col);
    println!("NPC walk collision (r=0.4): {}", npc_col);

    let on_road = world::on_any_road(x, z, net);
    println!("On/near road: {}", on_road);

    if !net.nodes.is_empty() {
        let (ni, nd) = nearest_node(x, z, net);
        println!("Nearest road node: #{} at ({:.1}, {:.1}) dist={:.1}m",
            ni, net.nodes[ni][0], net.nodes[ni][1], nd.sqrt());
    }

    println!("\n--- Nearby Objects (within 15m) ---");
    let r2 = 15.0 * 15.0;

    for (i, b) in world.buildings.iter().enumerate() {
        let dx = x - b.x; let dz = z - b.z;
        if dx*dx + dz*dz < r2 {
            println!("  Building[{}] at ({:.1},{:.1}) size={:.1}x{:.1}x{:.1} dist={:.1}m",
                i, b.x, b.z, b.w * 2.0, b.d * 2.0, b.h,
                (dx*dx + dz*dz).sqrt());
        }
    }
    for (i, t) in world.trees.iter().enumerate() {
        let dx = x - t.x; let dz = z - t.z;
        if dx*dx + dz*dz < r2 {
            println!("  Tree[{}] at ({:.1},{:.1}) trunk_r={:.2} dist={:.1}m",
                i, t.x, t.z, t.trunk_radius, (dx*dx + dz*dz).sqrt());
        }
    }
    for (i, r) in world.rocks.iter().enumerate() {
        let dx = x - r.x; let dz = z - r.z;
        if dx*dx + dz*dz < r2 {
            println!("  Rock[{}] at ({:.1},{:.1}) size={:.2} dist={:.1}m",
                i, r.x, r.z, r.size, (dx*dx + dz*dz).sqrt());
        }
    }
    for (i, l) in world.street_lights.iter().enumerate() {
        let dx = x - l.x; let dz = z - l.z;
        if dx*dx + dz*dz < r2 {
            println!("  StreetLight[{}] at ({:.1},{:.1}) dist={:.1}m",
                i, l.x, l.z, (dx*dx + dz*dz).sqrt());
        }
    }
    for (i, b) in world.trash_bins.iter().enumerate() {
        let dx = x - b.x; let dz = z - b.z;
        if dx*dx + dz*dz < r2 {
            println!("  TrashBin[{}] at ({:.1},{:.1}) dist={:.1}m",
                i, b.x, b.z, (dx*dx + dz*dz).sqrt());
        }
    }
    for (i, inter) in world.interactibles.iter().enumerate() {
        let dx = x - inter.x; let dz = z - inter.z;
        if dx*dx + dz*dz < r2 {
            println!("  Interactible[{}] at ({:.1},{:.1}) dist={:.1}m",
                i, inter.x, inter.z, (dx*dx + dz*dz).sqrt());
        }
    }
    for (i, w) in world.walls.iter().enumerate() {
        let dx = (x - w.x).abs(); let dz = (z - w.z).abs();
        if dx < w.hw + 15.0 && dz < w.hd + 15.0 {
            println!("  Wall[{}] at ({:.1},{:.1}) hw={:.1} hd={:.1} h={:.1}",
                i, w.x, w.z, w.hw, w.hd, w.height);
        }
    }

    for (i, npc) in world.npcs.iter().enumerate() {
        let dx = x - npc.x; let dz = z - npc.z;
        if dx*dx + dz*dz < r2 {
            println!("  NPC[{}] at ({:.1},{:.1}) job={:?} state={:?} stuck={:.1}s",
                i, npc.x, npc.z, npc.job, npc.state, npc.stuck_timer);
        }
    }
    for (i, v) in world.vehicles.iter().enumerate() {
        let dx = x - v.x; let dz = z - v.z;
        if dx*dx + dz*dz < r2 {
            println!("  Vehicle[{}] at ({:.1},{:.1}) parked={} ai={}",
                i, v.x, v.z, v.parked, v.ai_active);
        }
    }
    for (i, ps) in net.parking_spots.iter().enumerate() {
        let dx = x - ps.x; let dz = z - ps.z;
        if dx*dx + dz*dz < r2 {
            println!("  ParkingSpot[{}] at ({:.1},{:.1}) occupied={:?}",
                i, ps.x, ps.z, ps.occupied_by);
        }
    }
}

fn probe_line(world: &state::WorldData, net: &state::RoadNetwork, terrain: &state::Terrain,
              x0: f32, z0: f32, x1: f32, z1: f32, steps: usize) {
    println!("=== LINE PROBE ({:.1},{:.1}) -> ({:.1},{:.1}), {} samples ===", x0, z0, x1, z1, steps);
    println!("{:>8} {:>8} {:>7} {:>10} {:>6} {:>6} {:>6}",
        "X", "Z", "Height", "Surface", "River", "BldCl", "NpcCl");
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = x0 + (x1 - x0) * t;
        let z = z0 + (z1 - z0) * t;
        let h = terrain.height_at(x, z);
        let surf = world::surface_at(x, z, net);
        let riv = world::on_river(x, z, &world.river_segments);
        let bc = world::check_walk_collision(world, x, z, 0.4, None);
        let nc = world::check_walk_collision(world, x, z, 0.4, Some(usize::MAX));
        println!("{:>8.1} {:>8.1} {:>7.2} {:>10} {:>6} {:>6} {:>6}",
            x, z, h, surf_str(surf), riv, bc, nc);
    }
}

fn probe_grid(world: &state::WorldData, net: &state::RoadNetwork,
              cx: f32, cz: f32, size: f32, step: f32) {
    let half = size / 2.0;
    println!("=== GRID PROBE center=({:.1},{:.1}) size={:.0}m step={:.1}m ===", cx, cz, size, step);
    println!("Legend: .=terrain R=road S=sidewalk F=fieldroad #=collision ~=river X=npcblock");

    let mut z = cz - half;
    while z <= cz + half {
        print!("{:>7.1} |", z);
        let mut x = cx - half;
        while x <= cx + half {
            let riv = world::on_river(x, z, &world.river_segments);
            let bc = world::check_walk_collision(world, x, z, 0.3, None);
            let nc = world::check_walk_collision(world, x, z, 0.3, Some(usize::MAX));
            let surf = world::surface_at(x, z, net);

            let ch = if riv { '~' }
            else if bc { '#' }
            else if nc { 'X' }
            else { match surf {
                state::Surface::CarRoad => 'R',
                state::Surface::Sidewalk => 'S',
                state::Surface::FieldRoad => 'F',
                state::Surface::Terrain => '.',
            }};
            print!("{}", ch);
            x += step;
        }
        println!("|");
        z += step;
    }
}

fn probe_cross_section(world: &state::WorldData, net: &state::RoadNetwork, terrain: &state::Terrain,
                       axis: &str, pos: f32, from: f32, to: f32, step: f32) {
    println!("=== CROSS SECTION along {} at {}={:.1}, from {:.1} to {:.1} ===",
        if axis == "x" { "X axis" } else { "Z axis" },
        if axis == "x" { "Z" } else { "X" }, pos, from, to);

    let mut points = Vec::new();
    let mut p = from;
    while p <= to {
        let (x, z) = if axis == "x" { (p, pos) } else { (pos, p) };
        let h = terrain.height_at(x, z);
        let riv = world::on_river(x, z, &world.river_segments);
        let col = world::check_walk_collision(world, x, z, 0.3, None);
        let surf = world::surface_at(x, z, net);
        points.push((p, h, riv, col, surf));
        p += step;
    }

    let min_h = points.iter().map(|p| p.1).fold(f32::MAX, f32::min);
    let max_h = points.iter().map(|p| p.1).fold(f32::MIN, f32::max);
    let h_range = (max_h - min_h).max(1.0);

    let rows = 20;
    println!("Height range: {:.1}m to {:.1}m", min_h, max_h);
    for row in (0..rows).rev() {
        let h_level = min_h + h_range * row as f32 / (rows - 1) as f32;
        print!("{:>6.1}m |", h_level);
        for (_, h, riv, col, _) in &points {
            if *riv && h_level < min_h + h_range * 0.3 { print!("~"); }
            else if *col && (h_level - h).abs() < h_range / rows as f32 * 2.0 { print!("#"); }
            else if (h_level - h).abs() < h_range / rows as f32 { print!("*"); }
            else if h_level < *h { print!(":"); }
            else { print!(" "); }
        }
        println!("|");
    }

    print!("         ");
    for (_, _, riv, _, surf) in &points {
        if *riv { print!("~"); }
        else { match surf {
            state::Surface::CarRoad => print!("R"),
            state::Surface::Sidewalk => print!("S"),
            state::Surface::FieldRoad => print!("F"),
            state::Surface::Terrain => print!("."),
        }}
    }
    println!();
}

fn analyze_npc_homes(world: &state::WorldData, net: &state::RoadNetwork, _terrain: &state::Terrain) {
    println!("=== NPC HOME ANALYSIS ===");
    println!("{:>4} {:>8} {:>8} {:>8} {:>8} {:>10} {:>6} {:>6} {:>8} {:>10}",
        "NPC", "HomeX", "HomeZ", "NpcX", "NpcZ", "Job", "River", "BCol", "DistHome", "Surface");

    let mut homes_on_river = 0;
    let mut homes_in_collision = 0;
    let mut homes_off_road = 0;

    for (i, npc) in world.npcs.iter().enumerate() {
        let bi = npc.home_idx;
        let (hx, hz) = if bi < world.buildings.len() {
            (world.buildings[bi].x, world.buildings[bi].z)
        } else { (0.0, 0.0) };

        let riv = world::on_river(hx, hz, &world.river_segments);
        let col = world::check_walk_collision(world, hx, hz, 0.5, None);
        let dist = ((npc.x - hx).powi(2) + (npc.z - hz).powi(2)).sqrt();
        let surf = world::surface_at(hx, hz, net);

        if riv { homes_on_river += 1; }
        if col { homes_in_collision += 1; }
        if surf == state::Surface::Terrain { homes_off_road += 1; }

        println!("{:>4} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>10?} {:>6} {:>6} {:>8.1} {:>10}",
            i, hx, hz, npc.x, npc.z, npc.job, riv, col, dist, surf_str(surf));
    }

    println!("\n--- Summary ---");
    println!("Homes on river: {}", homes_on_river);
    println!("Homes with building collision at entrance: {}", homes_in_collision);
    println!("Homes on raw terrain (no road access): {}", homes_off_road);

    println!("\n--- Home Building Distribution ---");
    let mut building_use: Vec<usize> = vec![0; world.buildings.len()];
    for npc in &world.npcs {
        if npc.home_idx < building_use.len() {
            building_use[npc.home_idx] += 1;
        }
    }
    let used_buildings = building_use.iter().filter(|c| **c > 0).count();
    let multi_use = building_use.iter().filter(|c| **c > 1).count();
    let max_use = building_use.iter().copied().max().unwrap_or(0);
    println!("Buildings used as homes: {}/{}", used_buildings, world.buildings.len());
    println!("Buildings with 2+ NPCs: {}", multi_use);
    println!("Max NPCs per building: {}", max_use);

    for (bi, count) in building_use.iter().enumerate() {
        if *count > 2 {
            let b = &world.buildings[bi];
            println!("  Building[{}] at ({:.1},{:.1}): {} NPCs, size {:.1}x{:.1}",
                bi, b.x, b.z, count, b.w * 2.0, b.d * 2.0);
        }
    }
}

fn collision_full_scan(world: &state::WorldData, step: f32) {
    println!("=== FULL COLLISION SCAN (step={:.1}m) ===", step);

    let half = state::WORLD_HALF;
    let mut total = 0u64;
    let mut building_hits = 0u64;
    let mut npc_walk_hits = 0u64;
    let mut river_hits = 0u64;

    let cell_size = 50.0;
    let grid_n = (state::WORLD_SIZE / cell_size) as usize;
    let mut building_density = vec![0u32; grid_n * grid_n];

    let mut z = -half;
    while z < half {
        let mut x = -half;
        while x < half {
            total += 1;
            let bc = world::check_walk_collision(world, x, z, 0.4, None);
            let nc = world::check_walk_collision(world, x, z, 0.4, Some(usize::MAX));
            let riv = world::on_river(x, z, &world.river_segments);
            if bc { building_hits += 1; }
            if nc { npc_walk_hits += 1; }
            if riv { river_hits += 1; }

            let gi = ((x + half) / cell_size) as usize;
            let gj = ((z + half) / cell_size) as usize;
            if gi < grid_n && gj < grid_n {
                if bc { building_density[gj * grid_n + gi] += 1; }
            }

            x += step;
        }
        z += step;
    }

    println!("Total samples: {}", total);
    println!("Building collision: {} ({:.1}%)", building_hits, building_hits as f64 / total as f64 * 100.0);
    println!("NPC walk collision: {} ({:.1}%)", npc_walk_hits, npc_walk_hits as f64 / total as f64 * 100.0);
    println!("River: {} ({:.1}%)", river_hits, river_hits as f64 / total as f64 * 100.0);
    println!("Total blocked for NPC: {} ({:.1}%)",
        npc_walk_hits + river_hits, (npc_walk_hits + river_hits) as f64 / total as f64 * 100.0);

    println!("\n--- Building Collision Density (50m grid) ---");
    let samples_per_cell = (cell_size / step) * (cell_size / step);
    for j in 0..grid_n {
        for i in 0..grid_n {
            let v = building_density[j * grid_n + i] as f32 / samples_per_cell * 100.0;
            if v > 50.0 { print!("##"); }
            else if v > 25.0 { print!("%%"); }
            else if v > 10.0 { print!("++"); }
            else if v > 2.0 { print!(".."); }
            else { print!("  "); }
        }
        println!();
    }
}

fn density_analysis(world: &state::WorldData) {
    println!("=== OBJECT DENSITY ANALYSIS (radius=20m) ===");

    let radius = 20.0;
    let r2 = radius * radius;
    let half = state::WORLD_HALF;
    let step = 25.0;

    println!("{:>8} {:>8} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
        "X", "Z", "Bldgs", "Trees", "Rocks", "Light", "Bins", "Walls", "Items", "NPCs");

    let mut z = -half + step;
    while z < half {
        let mut x = -half + step;
        while x < half {
            let mut nb = 0u16; let mut nt = 0u16; let mut nr = 0u16;
            let mut nl = 0u16; let mut nbi = 0u16; let mut nw = 0u16;
            let mut ni = 0u16; let mut nn = 0u16;

            for b in &world.buildings { if dist2(x, z, b.x, b.z) < r2 { nb += 1; } }
            for t in &world.trees { if dist2(x, z, t.x, t.z) < r2 { nt += 1; } }
            for r in &world.rocks { if dist2(x, z, r.x, r.z) < r2 { nr += 1; } }
            for l in &world.street_lights { if dist2(x, z, l.x, l.z) < r2 { nl += 1; } }
            for b in &world.trash_bins { if dist2(x, z, b.x, b.z) < r2 { nbi += 1; } }
            for w in &world.walls { if dist2(x, z, w.x, w.z) < r2 { nw += 1; } }
            for item in &world.items { if dist2(x, z, item.x, item.z) < r2 { ni += 1; } }
            for npc in &world.npcs { if dist2(x, z, npc.x, npc.z) < r2 { nn += 1; } }

            if nb + nt + nr + nl + nbi + nw > 3 {
                println!("{:>8.1} {:>8.1} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}",
                    x, z, nb, nt, nr, nl, nbi, nw, ni, nn);
            }

            x += step;
        }
        z += step;
    }
}

fn list_buildings(world: &state::WorldData, net: &state::RoadNetwork, terrain: &state::Terrain) {
    println!("=== BUILDING LIST ({} buildings) ===", world.buildings.len());
    println!("{:>4} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>6} {:>10}",
        "ID", "X", "Z", "Width", "Depth", "Height", "TerH", "River", "Surface");

    for (i, b) in world.buildings.iter().enumerate() {
        let h = terrain.height_at(b.x, b.z);
        let riv = world::on_river(b.x, b.z, &world.river_segments);
        let surf = world::surface_at(b.x, b.z, net);
        println!("{:>4} {:>8.1} {:>8.1} {:>7.1} {:>7.1} {:>7.1} {:>7.2} {:>6} {:>10}",
            i, b.x, b.z, b.w * 2.0, b.d * 2.0, b.h, h, riv, surf_str(surf));
    }

    let on_river: Vec<_> = world.buildings.iter().enumerate()
        .filter(|(_, b)| world::on_river(b.x, b.z, &world.river_segments))
        .collect();
    println!("\nBuildings on river: {}", on_river.len());
    for (i, b) in &on_river {
        println!("  Building[{}] at ({:.1},{:.1})", i, b.x, b.z);
    }

    println!("\n--- Building Overlaps ---");
    let mut overlaps = 0;
    for i in 0..world.buildings.len() {
        for j in (i+1)..world.buildings.len() {
            let a = &world.buildings[i];
            let b = &world.buildings[j];
            let dx = (a.x - b.x).abs();
            let dz = (a.z - b.z).abs();
            if dx < (a.w + b.w) * 0.5 && dz < (a.d + b.d) * 0.5 {
                println!("  Overlap: Building[{}] ({:.1},{:.1}) <-> Building[{}] ({:.1},{:.1})",
                    i, a.x, a.z, j, b.x, b.z);
                overlaps += 1;
            }
        }
    }
    println!("Total overlaps: {}", overlaps);
}

fn analyze_roads(net: &state::RoadNetwork) {
    println!("=== ROAD NETWORK ANALYSIS ===");
    println!("Segments: {}", net.segments.len());
    println!("Nodes: {}", net.nodes.len());
    println!("Parking spots: {}", net.parking_spots.len());

    let mut car_roads = 0;
    let mut field_roads = 0;
    let mut total_car_len = 0.0f32;
    let mut total_field_len = 0.0f32;

    for seg in &net.segments {
        let len = ((seg.x1 - seg.x0).powi(2) + (seg.z1 - seg.z0).powi(2)).sqrt();
        match seg.tier {
            state::RoadTier::CarRoad => { car_roads += 1; total_car_len += len; }
            state::RoadTier::FieldRoad => { field_roads += 1; total_field_len += len; }
        }
    }
    println!("Car roads: {} segments, total {:.0}m", car_roads, total_car_len);
    println!("Field roads: {} segments, total {:.0}m", field_roads, total_field_len);

    println!("\n--- Road Segments ---");
    for (i, seg) in net.segments.iter().enumerate() {
        let len = ((seg.x1 - seg.x0).powi(2) + (seg.z1 - seg.z0).powi(2)).sqrt();
        let tier = match seg.tier {
            state::RoadTier::CarRoad => "Car",
            state::RoadTier::FieldRoad => "Field",
        };
        println!("  [{:>2}] ({:>7.1},{:>7.1})->({:>7.1},{:>7.1}) {:<5} len={:.0}m",
            i, seg.x0, seg.z0, seg.x1, seg.z1, tier, len);
    }

    println!("\n--- Road Nodes ---");
    for (i, node) in net.nodes.iter().enumerate() {
        println!("  [{:>2}] ({:>7.1},{:>7.1})", i, node[0], node[1]);
    }

    let occupied = net.parking_spots.iter().filter(|p| p.occupied_by.is_some()).count();
    println!("\nParking spots: {}/{} occupied", occupied, net.parking_spots.len());
}

fn analyze_river(world: &state::WorldData, net: &state::RoadNetwork, terrain: &state::Terrain) {
    println!("=== RIVER ANALYSIS ===");
    println!("Segments: {}", world.river_segments.len());

    for (i, seg) in world.river_segments.iter().enumerate() {
        let len = ((seg.x2 - seg.x1).powi(2) + (seg.z2 - seg.z1).powi(2)).sqrt();
        let h1 = terrain.height_at(seg.x1, seg.z1);
        let h2 = terrain.height_at(seg.x2, seg.z2);
        if i < 5 || i == world.river_segments.len() - 1 {
            println!("  Seg[{}] ({:.1},{:.1})->({:.1},{:.1}) len={:.1}m w={:.1} h={:.2}-{:.2}",
                i, seg.x1, seg.z1, seg.x2, seg.z2, len, seg.width, h1, h2);
        }
    }

    println!("\n--- Objects ON River ---");
    for (i, b) in world.buildings.iter().enumerate() {
        if world::on_river(b.x, b.z, &world.river_segments) {
            println!("  Building[{}] at ({:.1},{:.1}) ON RIVER!", i, b.x, b.z);
        }
    }
    for (i, t) in world.trees.iter().enumerate() {
        if world::on_river(t.x, t.z, &world.river_segments) {
            println!("  Tree[{}] at ({:.1},{:.1}) ON RIVER", i, t.x, t.z);
        }
    }

    println!("\n--- Road-River Crossings ---");
    for (si, seg) in net.segments.iter().enumerate() {
        if seg.tier != state::RoadTier::CarRoad { continue; }
        let mid_x = (seg.x0 + seg.x1) / 2.0;
        let mid_z = (seg.z0 + seg.z1) / 2.0;
        if world::on_river(mid_x, mid_z, &world.river_segments) {
            println!("  Road[{}] midpoint ({:.1},{:.1}) crosses river", si, mid_x, mid_z);
        }
    }
}

fn analyze_parking(world: &state::WorldData, net: &state::RoadNetwork) {
    println!("=== PARKING ANALYSIS ===");
    println!("Total spots: {}", net.parking_spots.len());
    let occupied = net.parking_spots.iter().filter(|p| p.occupied_by.is_some()).count();
    println!("Occupied: {}", occupied);
    println!("Free: {}", net.parking_spots.len() - occupied);
    println!("Total vehicles: {}", world.vehicles.len());

    let parked = world.vehicles.iter().filter(|v| v.parked).count();
    let ai_active = world.vehicles.iter().filter(|v| v.ai_active && !v.parked).count();
    let inactive = world.vehicles.iter().filter(|v| !v.ai_active && !v.parked && !v.occupied).count();

    println!("Vehicles parked: {}", parked);
    println!("Vehicles AI-driving: {}", ai_active);
    println!("Vehicles inactive/stopped: {}", inactive);

    let no_spot: Vec<_> = world.vehicles.iter().enumerate()
        .filter(|(_, v)| v.parked && v.parking_target.is_none())
        .collect();
    println!("Parked vehicles with no parking_target: {}", no_spot.len());
}

fn analyze_vehicles(world: &state::WorldData) {
    println!("=== VEHICLE ANALYSIS ===");
    println!("{:>4} {:>8} {:>8} {:>6} {:>6} {:>6} {:>8} {:>6}",
        "ID", "X", "Z", "Park", "AI", "Occup", "Speed", "Owner");

    for (i, v) in world.vehicles.iter().enumerate() {
        let owner = match v.owner_npc {
            Some(n) => format!("NPC{}", n),
            None => "---".to_string(),
        };
        println!("{:>4} {:>8.1} {:>8.1} {:>6} {:>6} {:>6} {:>8.1} {:>6}",
            i, v.x, v.z, v.parked, v.ai_active, v.occupied, v.speed, owner);
    }

    let on_river: Vec<_> = world.vehicles.iter().enumerate()
        .filter(|(_, v)| world::on_river(v.x, v.z, &world.river_segments))
        .collect();
    if !on_river.is_empty() {
        println!("\n!!! Vehicles ON RIVER: {}", on_river.len());
        for (i, v) in &on_river {
            println!("  Vehicle[{}] at ({:.1},{:.1})", i, v.x, v.z);
        }
    }
}

fn print_summary(world: &state::WorldData, net: &state::RoadNetwork, terrain: &state::Terrain, seed: u64) {
    println!("=== PROBE3D (seed={}) ===", seed);
    println!("World size: {:.0}m x {:.0}m", state::WORLD_SIZE, state::WORLD_SIZE);
    println!("Terrain grid: {}x{}, cell={:.1}m", terrain.grid, terrain.grid, terrain.cell_size);
    println!();
    println!("Buildings: {}", world.buildings.len());
    println!("Trees: {}", world.trees.len());
    println!("Rocks: {}", world.rocks.len());
    println!("Street lights: {}", world.street_lights.len());
    println!("Trash bins: {}", world.trash_bins.len());
    println!("Interactibles: {}", world.interactibles.len());
    println!("Walls: {}", world.walls.len());
    println!("River segments: {}", world.river_segments.len());
    println!("Static tris: {}", world.static_tris.len());
    println!();
    println!("NPCs: {}", world.npcs.len());
    println!("Vehicles: {}", world.vehicles.len());
    println!("Items: {}", world.items.len());
    println!();
    println!("Road segments: {}", net.segments.len());
    println!("Road nodes: {}", net.nodes.len());
    println!("Parking spots: {}", net.parking_spots.len());
    println!();

    let homes_on_river: usize = world.npcs.iter()
        .filter(|n| n.home_idx < world.buildings.len()
            && world::on_river(world.buildings[n.home_idx].x, world.buildings[n.home_idx].z, &world.river_segments))
        .count();
    let vehicles_on_river = world.vehicles.iter().filter(|v| world::on_river(v.x, v.z, &world.river_segments)).count();

    // Check objects on road surface
    let rocks_on_road = world.rocks.iter().filter(|r| world::on_road_surface(r.x, r.z, net)).count();
    let bins_on_road = world.trash_bins.iter().filter(|b| world::on_road_surface(b.x, b.z, net)).count();
    let interactibles_on_road = world.interactibles.iter().filter(|i| world::on_road_surface(i.x, i.z, net)).count();

    println!("--- Health Check ---");
    println!("NPC homes on river: {}", homes_on_river);
    println!("Vehicles on river: {}", vehicles_on_river);
    println!("Parking spots occupied: {}/{}",
        net.parking_spots.iter().filter(|p| p.occupied_by.is_some()).count(),
        net.parking_spots.len());
    println!("Rocks on road: {}/{}", rocks_on_road, world.rocks.len());
    println!("Trash bins on road: {}/{}", bins_on_road, world.trash_bins.len());
    println!("Interactibles on road: {}/{}", interactibles_on_road, world.interactibles.len());
}

// === Reachability Analysis ===
// Walkability grid + flood-fill connected components + building/NPC reachability

const GRID_RES: usize = 500;  // 500x500 = 1m cells for 500m world

struct WalkGrid {
    cells: Vec<u8>,        // 0=blocked, 1=walkable
    component: Vec<u32>,   // connected component ID (0=unassigned)
    dist_to_blocked: Vec<f32>, // distance transform: distance to nearest blocked cell
}

impl WalkGrid {
    fn new() -> Self {
        WalkGrid {
            cells: vec![0; GRID_RES * GRID_RES],
            component: vec![0; GRID_RES * GRID_RES],
            dist_to_blocked: vec![0.0; GRID_RES * GRID_RES],
        }
    }

    fn idx(gx: usize, gz: usize) -> usize { gz * GRID_RES + gx }

    // World coords -> grid coords
    fn to_grid(wx: f32, wz: f32) -> (usize, usize) {
        let gx = ((wx + state::WORLD_HALF) as usize).min(GRID_RES - 1);
        let gz = ((wz + state::WORLD_HALF) as usize).min(GRID_RES - 1);
        (gx, gz)
    }

    // Grid coords -> world coords (cell center)
    fn to_world(gx: usize, gz: usize) -> (f32, f32) {
        let wx = gx as f32 - state::WORLD_HALF + 0.5;
        let wz = gz as f32 - state::WORLD_HALF + 0.5;
        (wx, wz)
    }
}

fn analyze_reachability(world: &state::WorldData, net: &state::RoadNetwork) {
    println!("=== REACHABILITY ANALYSIS (seed from world) ===");

    // Step 1: Build walkability grid
    let mut grid = WalkGrid::new();
    let mut walkable_count = 0u32;
    let mut blocked_count = 0u32;

    for gz in 0..GRID_RES {
        for gx in 0..GRID_RES {
            let (wx, wz) = WalkGrid::to_world(gx, gz);
            let blocked = world::check_walk_collision(world, wx, wz, 0.4, Some(usize::MAX))
                || world::on_river_not_bridge(wx, wz, &world.river_segments, &world.bridges);
            let idx = WalkGrid::idx(gx, gz);
            if blocked {
                grid.cells[idx] = 0;
                blocked_count += 1;
            } else {
                grid.cells[idx] = 1;
                walkable_count += 1;
            }
        }
    }

    println!("Walkability grid: {}x{}, {} walkable, {} blocked",
        GRID_RES, GRID_RES, walkable_count, blocked_count);

    // Step 2: Flood-fill connected components (BFS, 4-connected)
    let mut num_components = 0u32;
    let mut component_sizes: Vec<u32> = Vec::new();
    // Representative position of each component (first cell found)
    let mut component_reps: Vec<(f32, f32)> = Vec::new();

    for start_gz in 0..GRID_RES {
        for start_gx in 0..GRID_RES {
            let start_idx = WalkGrid::idx(start_gx, start_gz);
            if grid.cells[start_idx] == 0 || grid.component[start_idx] != 0 { continue; }

            num_components += 1;
            let comp_id = num_components;
            let mut size = 0u32;
            let (rx, rz) = WalkGrid::to_world(start_gx, start_gz);

            // BFS
            let mut queue = Vec::new();
            queue.push((start_gx, start_gz));
            grid.component[start_idx] = comp_id;

            while let Some((gx, gz)) = queue.pop() {
                size += 1;
                // 4-connected neighbors
                let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dx, dz) in &neighbors {
                    let nx = gx as i32 + dx;
                    let nz = gz as i32 + dz;
                    if nx < 0 || nx >= GRID_RES as i32 || nz < 0 || nz >= GRID_RES as i32 { continue; }
                    let nx = nx as usize;
                    let nz = nz as usize;
                    let ni = WalkGrid::idx(nx, nz);
                    if grid.cells[ni] == 1 && grid.component[ni] == 0 {
                        grid.component[ni] = comp_id;
                        queue.push((nx, nz));
                    }
                }
            }

            component_sizes.push(size);
            component_reps.push((rx, rz));
        }
    }

    // Identify main component: the one containing the most road nodes
    let mut road_nodes_per_comp: Vec<u32> = vec![0; num_components as usize + 1];
    for node in &net.nodes {
        let (gx, gz) = WalkGrid::to_grid(node[0], node[1]);
        let comp = grid.component[WalkGrid::idx(gx, gz)] as usize;
        if comp > 0 && comp <= num_components as usize {
            road_nodes_per_comp[comp] += 1;
        }
    }

    let main_comp = road_nodes_per_comp.iter().enumerate()
        .skip(1) // skip index 0
        .max_by_key(|&(_, count)| count)
        .map(|(i, _)| i as u32)
        .unwrap_or(1);

    println!("Connected components: {}", num_components);
    // Sort components by size descending for display
    let mut comp_display: Vec<(u32, u32, f32, f32, u32)> = (1..=num_components)
        .map(|c| {
            let ci = c as usize - 1;
            (c, component_sizes[ci], component_reps[ci].0, component_reps[ci].1,
             road_nodes_per_comp[c as usize])
        })
        .collect();
    comp_display.sort_by(|a, b| b.1.cmp(&a.1));

    for (c, size, rx, rz, roads) in &comp_display {
        let main_tag = if *c == main_comp { " (MAIN)" } else { "" };
        println!("  Component {}: {} cells, ~({:.0},{:.0}), {} road nodes{}",
            c, size, rx, rz, roads, main_tag);
    }

    // Step 3: Building reachability — sample 8 directions from building center
    let mut building_reachable = vec![false; world.buildings.len()];
    let mut isolated_buildings: Vec<(usize, f32, f32, u32)> = Vec::new();

    for (bi, b) in world.buildings.iter().enumerate() {
        let half_extent = b.w.max(b.d) * 0.5 + 1.5;
        let angles: [f32; 8] = [0.0, std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_2,
            3.0 * std::f32::consts::FRAC_PI_4, std::f32::consts::PI,
            5.0 * std::f32::consts::FRAC_PI_4, 3.0 * std::f32::consts::FRAC_PI_2,
            7.0 * std::f32::consts::FRAC_PI_4];

        for &angle in &angles {
            let sx = b.x + angle.cos() * half_extent;
            let sz = b.z + angle.sin() * half_extent;
            let (gx, gz) = WalkGrid::to_grid(sx, sz);
            if gx < GRID_RES && gz < GRID_RES {
                let comp = grid.component[WalkGrid::idx(gx, gz)];
                if comp == main_comp {
                    building_reachable[bi] = true;
                    break;
                }
            }
        }

        if !building_reachable[bi] {
            // Find which component the building is in (if any)
            let (gx, gz) = WalkGrid::to_grid(b.x, b.z);
            let mut nearby_comp = 0u32;
            // Check surrounding cells for a component
            for dr in 1..=5 {
                for &(ddx, ddz) in &[(-1i32, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)] {
                    let nx = gx as i32 + ddx * dr;
                    let nz = gz as i32 + ddz * dr;
                    if nx >= 0 && nx < GRID_RES as i32 && nz >= 0 && nz < GRID_RES as i32 {
                        let c = grid.component[WalkGrid::idx(nx as usize, nz as usize)];
                        if c != 0 { nearby_comp = c; break; }
                    }
                }
                if nearby_comp != 0 { break; }
            }
            isolated_buildings.push((bi, b.x, b.z, nearby_comp));
        }
    }

    let reachable_count = building_reachable.iter().filter(|&&r| r).count();
    println!("\nBuilding reachability: {}/{} reachable", reachable_count, world.buildings.len());
    for (bi, bx, bz, comp) in &isolated_buildings {
        // Find nearest road distance
        let mut min_road_dist = f32::MAX;
        for node in &net.nodes {
            let d = ((node[0] - bx).powi(2) + (node[1] - bz).powi(2)).sqrt();
            if d < min_road_dist { min_road_dist = d; }
        }
        let comp_str = if *comp == 0 { "no-component".to_string() }
            else { format!("component {}", comp) };
        println!("  ISOLATED: Building[{}] at ({:.1},{:.1}), {}, nearest road {:.1}m",
            bi, bx, bz, comp_str, min_road_dist);
    }

    // Step 4: Distance transform (Chamfer 2-pass approximation)
    // Each walkable cell gets distance to nearest blocked cell
    let big = (GRID_RES * 2) as f32;
    for i in 0..grid.dist_to_blocked.len() {
        grid.dist_to_blocked[i] = if grid.cells[i] == 0 { 0.0 } else { big };
    }

    // Forward pass (top-left to bottom-right)
    for gz in 0..GRID_RES {
        for gx in 0..GRID_RES {
            let idx = WalkGrid::idx(gx, gz);
            if grid.cells[idx] == 0 { continue; }
            let mut d = grid.dist_to_blocked[idx];
            if gx > 0 { d = d.min(grid.dist_to_blocked[WalkGrid::idx(gx - 1, gz)] + 1.0); }
            if gz > 0 { d = d.min(grid.dist_to_blocked[WalkGrid::idx(gx, gz - 1)] + 1.0); }
            if gx > 0 && gz > 0 { d = d.min(grid.dist_to_blocked[WalkGrid::idx(gx - 1, gz - 1)] + 1.414); }
            if gx + 1 < GRID_RES && gz > 0 { d = d.min(grid.dist_to_blocked[WalkGrid::idx(gx + 1, gz - 1)] + 1.414); }
            grid.dist_to_blocked[idx] = d;
        }
    }

    // Backward pass (bottom-right to top-left)
    for gz in (0..GRID_RES).rev() {
        for gx in (0..GRID_RES).rev() {
            let idx = WalkGrid::idx(gx, gz);
            if grid.cells[idx] == 0 { continue; }
            let mut d = grid.dist_to_blocked[idx];
            if gx + 1 < GRID_RES { d = d.min(grid.dist_to_blocked[WalkGrid::idx(gx + 1, gz)] + 1.0); }
            if gz + 1 < GRID_RES { d = d.min(grid.dist_to_blocked[WalkGrid::idx(gx, gz + 1)] + 1.0); }
            if gx + 1 < GRID_RES && gz + 1 < GRID_RES { d = d.min(grid.dist_to_blocked[WalkGrid::idx(gx + 1, gz + 1)] + 1.414); }
            if gx > 0 && gz + 1 < GRID_RES { d = d.min(grid.dist_to_blocked[WalkGrid::idx(gx - 1, gz + 1)] + 1.414); }
            grid.dist_to_blocked[idx] = d;
        }
    }

    // Count narrow passages on main component
    let mut narrow_cells = 0u32;
    let mut narrow_clusters: Vec<(f32, f32, f32)> = Vec::new(); // (x, z, clearance)
    for gz in 0..GRID_RES {
        for gx in 0..GRID_RES {
            let idx = WalkGrid::idx(gx, gz);
            if grid.component[idx] == main_comp && grid.dist_to_blocked[idx] > 0.0
                && grid.dist_to_blocked[idx] <= 1.5 {
                narrow_cells += 1;
                // Sample some for display (every 20th)
                if narrow_cells % 20 == 1 && narrow_clusters.len() < 15 {
                    let (wx, wz) = WalkGrid::to_world(gx, gz);
                    narrow_clusters.push((wx, wz, grid.dist_to_blocked[idx]));
                }
            }
        }
    }

    println!("\nNarrow passages (main component, clearance <= 1.5m): {} cells", narrow_cells);
    for (wx, wz, c) in &narrow_clusters {
        println!("  Narrow at ({:.0},{:.0}) clearance={:.1}m", wx, wz, c);
    }

    // Step 5: NPC home reachability
    let mut npc_reachable = 0u32;
    let mut npc_isolated: Vec<(usize, usize, f32, f32)> = Vec::new();

    for (i, npc) in world.npcs.iter().enumerate() {
        let bi = npc.home_idx;
        if bi < world.buildings.len() && building_reachable[bi] {
            npc_reachable += 1;
        } else {
            let (hx, hz) = if bi < world.buildings.len() {
                (world.buildings[bi].x, world.buildings[bi].z)
            } else { (0.0, 0.0) };
            npc_isolated.push((i, bi, hx, hz));
        }
    }

    println!("\nNPC home reachability: {}/{} reachable", npc_reachable, world.npcs.len());
    for (ni, bi, hx, hz) in &npc_isolated {
        println!("  STUCK-RISK: NPC[{}] home=Building[{}] at ({:.1},{:.1}) — ISOLATED",
            ni, bi, hx, hz);
    }

    // Step 6: Summary statistics
    println!("\n--- Reachability Summary ---");
    println!("Main component: #{} ({} cells, {:.1}% of walkable)",
        main_comp,
        component_sizes.get((main_comp - 1) as usize).copied().unwrap_or(0),
        component_sizes.get((main_comp - 1) as usize).copied().unwrap_or(0) as f64
            / walkable_count.max(1) as f64 * 100.0);
    println!("Isolated buildings: {}", isolated_buildings.len());
    println!("At-risk NPCs: {}", npc_isolated.len());
    println!("Narrow passage cells: {}", narrow_cells);

    // Average clearance for main component
    let mut total_clearance = 0.0f64;
    let mut main_count = 0u64;
    for i in 0..grid.cells.len() {
        if grid.component[i] == main_comp {
            total_clearance += grid.dist_to_blocked[i] as f64;
            main_count += 1;
        }
    }
    if main_count > 0 {
        println!("Average clearance (main component): {:.2}m", total_clearance / main_count as f64);
    }
}

fn analyze_pathfinding(game: &mut state::GameState, args: &[String]) {
    let x0: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let z0: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let x1: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(50.0);
    let z1: f32 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(50.0);

    let straight_dist = ((x1 - x0).powi(2) + (z1 - z0).powi(2)).sqrt();
    println!("=== PATHFINDING ANALYSIS ===");
    println!("From: ({:.1}, {:.1})", x0, z0);
    println!("To:   ({:.1}, {:.1})", x1, z1);
    println!("Straight-line distance: {:.1}m", straight_dist);
    println!();

    // Place a temporary NPC at the start position
    let y0 = game.terrain.height_at(x0, z0);
    let npc_idx = game.world.npcs.len();
    game.world.npcs.push(state::Npc {
        x: x0, y: y0, z: z0,
        rot_y: 0.0, walk_phase: 0.0,
        target_x: x1, target_z: z1,
        shirt_color: 0xFF0000, pants_color: 0x0000FF,
        rng: rng::Rng::new(99999),
        vel_y: 0.0, on_ground: true,
        terrain_normal: [0.0, 1.0, 0.0],
        state: state::NpcState::Working,
        home_idx: 0, car_idx: 0,
        wake_hour: 7.0,
        state_timer: 0.0,
        money: 0.0,
        carrying_item: false,
        carrying_bin: None,
        target_item: None,
        target_bin: None,
        items_deposited_today: 0,
        in_vehicle: false,
        parked_x: x0, parked_z: z0,
        stuck_timer: 0.0,
        nav_path: Vec::new(), nav_path_idx: 0,
        nav_target_x: 0.0, nav_target_z: 0.0,
        job: state::NpcJob::Collector,
        job_timer: 0.0,
        job_target_x: x1, job_target_z: z1,
        interaction_target: None,
        interacting_with: None,
        interaction_timer: 0.0,
        brain_idx: 0,
        fitness_money_earned: 0.0,
        fitness_items_picked: 0,
        fitness_interactions: 0,
        fitness_distance: 0.0,
        fitness_stuck_time: 0.0,
        prev_x: x0, prev_z: z0,
        health: 100.0,
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
        sound: [0.0; 3],
        fitness_sounds_made: 0,
        fitness_npcs_heard: 0,
        fitness_proximity: 0.0,
        ragdoll_active: false,
        ragdoll_points: [[0.0; 3]; 7],
        ragdoll_prev: [[0.0; 3]; 7],
        ragdoll_timer: 0.0,
        skeleton: clauding::skeleton::Skeleton::new_humanoid(),
        body: {
            let shape = clauding::physics::CollisionShape::Capsule { radius: 0.3, half_height: 0.625 };
            let inertia = shape.inertia_diag(75.0);
            clauding::physics::RigidBody::new_dynamic([x0, y0, z0], 75.0, inertia)
        },
        wanted: false,
        bounty: 0.0,
        violation_timer: 0.0,
        police_target: None,
        find_item_failures: 0,
        find_bin_failures: 0,
        stuck_recoveries: 0,
    });

    let dt = 1.0 / 60.0; // simulate at 60 FPS timestep
    let max_ticks = 10000;
    let mut total_distance = 0.0f32;
    let mut max_stuck = 0.0f32;
    let mut arrived = false;

    for tick in 0..max_ticks {
        let prev_x = game.world.npcs[npc_idx].x;
        let prev_z = game.world.npcs[npc_idx].z;

        let remaining = npc::npc_walk_toward(
            &mut game.world, npc_idx, x1, z1,
            &game.road_network, &game.terrain, dt,
            &game.walk_grid,
        );

        let step_dx = game.world.npcs[npc_idx].x - prev_x;
        let step_dz = game.world.npcs[npc_idx].z - prev_z;
        total_distance += (step_dx * step_dx + step_dz * step_dz).sqrt();

        let stuck = game.world.npcs[npc_idx].stuck_timer;
        if stuck > max_stuck { max_stuck = stuck; }

        if remaining < 1.0 {
            arrived = true;
            println!("Arrived after {} ticks ({:.1}s simulated)", tick + 1, (tick + 1) as f32 * dt);
            break;
        }

        // Print progress at intervals
        if (tick + 1) % 2000 == 0 {
            let npc = &game.world.npcs[npc_idx];
            let dist_to_goal = ((npc.x - x1).powi(2) + (npc.z - z1).powi(2)).sqrt();
            println!("  tick {:>5}: pos=({:.1},{:.1}) dist_to_goal={:.1}m stuck={:.1}",
                tick + 1, npc.x, npc.z, dist_to_goal, npc.stuck_timer);
        }
    }

    if !arrived {
        let npc = &game.world.npcs[npc_idx];
        let final_dist = ((npc.x - x1).powi(2) + (npc.z - z1).powi(2)).sqrt();
        println!("DID NOT ARRIVE after {} ticks ({:.1}s)", max_ticks, max_ticks as f32 * dt);
        println!("Final position: ({:.1}, {:.1}), {:.1}m from target", npc.x, npc.z, final_dist);
    }

    println!();
    println!("--- Path Statistics ---");
    println!("Total distance walked: {:.1}m", total_distance);
    println!("Straight-line distance: {:.1}m", straight_dist);
    let efficiency = if total_distance > 0.01 { straight_dist / total_distance } else { 0.0 };
    println!("Path efficiency: {:.1}% (straight/actual)", efficiency * 100.0);
    println!("Max stuck score: {:.1}", max_stuck);
    println!("Arrived: {}", arrived);

    // Remove the temporary NPC
    game.world.npcs.pop();
}

fn analyze_bins(game: &state::GameState) {
    println!("=== TRASH BIN ACCESSIBILITY ANALYSIS ===");
    println!("Total trash bins: {}", game.world.trash_bins.len());
    println!();

    let mut walkable_count = 0u32;
    let mut near_road = 0u32;  // <15m
    let mut far_road = 0u32;   // >30m
    let mut inaccessible: Vec<(usize, f32, f32, &'static str)> = Vec::new();

    println!("{:>4} {:>8} {:>8} {:>7} {:>6} {:>6} {:>6} {:>8} {:>10}",
        "ID", "X", "Z", "Height", "River", "BldCl", "Steep", "RdDist", "Status");

    for (i, bin) in game.world.trash_bins.iter().enumerate() {
        let on_river = world::on_river(bin.x, bin.z, &game.world.river_segments);
        let in_building = world::check_walk_collision(&game.world, bin.x, bin.z, 0.3, None);
        let terrain_normal = game.terrain.normal_at(bin.x, bin.z);
        let slope_deg = terrain_normal[1].clamp(-1.0, 1.0).acos().to_degrees();
        let steep = slope_deg > 40.0;

        // Distance to nearest road node
        let mut min_road_dist = f32::MAX;
        for node in &game.road_network.nodes {
            let d = ((node[0] - bin.x).powi(2) + (node[1] - bin.z).powi(2)).sqrt();
            if d < min_road_dist { min_road_dist = d; }
        }

        let walkable = !on_river && !in_building && !steep;
        if walkable { walkable_count += 1; }
        if min_road_dist < 15.0 { near_road += 1; }
        if min_road_dist > 30.0 { far_road += 1; }

        let status = if on_river { "RIVER" }
            else if in_building { "IN_BLDG" }
            else if steep { "STEEP" }
            else if min_road_dist > 30.0 { "far" }
            else { "ok" };

        let is_problem = on_river || in_building || steep;
        if is_problem {
            let reason = if on_river { "in river" }
                else if in_building { "inside building collision" }
                else { "very steep terrain (>40 deg)" };
            inaccessible.push((i, bin.x, bin.z, reason));
        }

        println!("{:>4} {:>8.1} {:>8.1} {:>7.2} {:>6} {:>6} {:>5.1}° {:>8.1} {:>10}",
            i, bin.x, bin.z, bin.y, on_river, in_building, slope_deg, min_road_dist, status);
    }

    println!();
    println!("--- Summary ---");
    println!("Bins on walkable terrain: {}/{}", walkable_count, game.world.trash_bins.len());
    println!("Bins near roads (<15m): {}", near_road);
    println!("Bins far from roads (>30m): {}", far_road);

    if inaccessible.is_empty() {
        println!("\nNo potentially inaccessible bins found.");
    } else {
        println!("\n--- Potentially Inaccessible Bins ({}) ---", inaccessible.len());
        for (idx, x, z, reason) in &inaccessible {
            println!("  Bin[{}] at ({:.1},{:.1}): {}", idx, x, z, reason);
        }
    }
}

fn nearest_node(x: f32, z: f32, net: &state::RoadNetwork) -> (usize, f32) {
    let mut best_i = 0;
    let mut best_d = f32::MAX;
    for (i, node) in net.nodes.iter().enumerate() {
        let d = dist2(x, z, node[0], node[1]);
        if d < best_d { best_d = d; best_i = i; }
    }
    (best_i, best_d)
}

fn dist2(x0: f32, z0: f32, x1: f32, z1: f32) -> f32 {
    (x0 - x1).powi(2) + (z0 - z1).powi(2)
}

fn analyze_slopes(world: &state::WorldData, terrain: &state::Terrain, step: f32) {
    let half = state::WORLD_HALF;
    println!("=== SLOPE ANALYSIS (step={:.1}m) ===\n", step);

    // 1. Scan terrain for slope distribution
    let mut slope_counts = [0u32; 10]; // 0-5, 5-10, 10-15, ... 45+ degrees
    let mut steepest_angle: f32 = 0.0;
    let mut steepest_pos = (0.0f32, 0.0f32);
    let mut total_samples = 0u32;

    let mut x = -half;
    while x <= half {
        let mut z = -half;
        while z <= half {
            let n = terrain.normal_at(x, z);
            let angle_deg = n[1].clamp(-1.0, 1.0).acos().to_degrees();
            let bucket = (angle_deg / 5.0) as usize;
            slope_counts[bucket.min(9)] += 1;
            if angle_deg > steepest_angle {
                steepest_angle = angle_deg;
                steepest_pos = (x, z);
            }
            total_samples += 1;
            z += step;
        }
        x += step;
    }

    println!("Terrain slope distribution ({} samples):", total_samples);
    let labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45+"];
    for (i, label) in labels.iter().enumerate() {
        let pct = slope_counts[i] as f32 / total_samples as f32 * 100.0;
        let bar_len = (pct * 0.5) as usize;
        let bar: String = (0..bar_len).map(|_| '#').collect();
        println!("  {:>5}°: {:5} ({:5.1}%) {}", label, slope_counts[i], pct, bar);
    }
    println!("  Steepest: {:.1}° at ({:.1}, {:.1})", steepest_angle, steepest_pos.0, steepest_pos.1);
    let sn = terrain.normal_at(steepest_pos.0, steepest_pos.1);
    println!("  Normal there: ({:.3}, {:.3}, {:.3})", sn[0], sn[1], sn[2]);
    println!();

    // 2. Find top 10 steep spots (unique, spaced >20m apart)
    let mut steep_spots: Vec<(f32, f32, f32)> = Vec::new(); // angle, x, z
    x = -half;
    while x <= half {
        let mut z = -half;
        while z <= half {
            let n = terrain.normal_at(x, z);
            let angle_deg = n[1].clamp(-1.0, 1.0).acos().to_degrees();
            if angle_deg > 8.0 {
                let too_close = steep_spots.iter().any(|s| dist2(x, z, s.1, s.2) < 400.0);
                if !too_close {
                    steep_spots.push((angle_deg, x, z));
                }
            }
            z += step;
        }
        x += step;
    }
    steep_spots.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    steep_spots.truncate(10);

    println!("Top {} steep locations (>8°, spaced >20m):", steep_spots.len());
    for (i, s) in steep_spots.iter().enumerate() {
        let h = terrain.height_at(s.1, s.2);
        println!("  #{}: {:.1}° at ({:.0}, {:.0}) height={:.1}m", i + 1, s.0, s.1, s.2, h);
    }
    println!();

    // 3. Entities on slopes — NPCs
    println!("--- NPCs on slopes (tilt > 3°) ---");
    let mut npc_on_slope = 0;
    for (i, npc) in world.npcs.iter().enumerate() {
        let n = npc.terrain_normal;
        let angle = n[1].clamp(-1.0, 1.0).acos().to_degrees();
        if angle > 3.0 {
            npc_on_slope += 1;
            println!("  NPC[{:2}] tilt={:.1}° normal=({:.3},{:.3},{:.3}) pos=({:.0},{:.0}) h={:.1}",
                i, angle, n[0], n[1], n[2], npc.x, npc.z, npc.y);
        }
    }
    if npc_on_slope == 0 { println!("  (none — all on flat terrain at init)"); }
    println!();

    // 4. Entities on slopes — Vehicles
    println!("--- Vehicles on slopes (tilt > 3°) ---");
    let mut veh_on_slope = 0;
    for (i, v) in world.vehicles.iter().enumerate() {
        let n = v.terrain_normal;
        let angle = n[1].clamp(-1.0, 1.0).acos().to_degrees();
        if angle > 3.0 {
            veh_on_slope += 1;
            println!("  Vehicle[{:2}] tilt={:.1}° normal=({:.3},{:.3},{:.3}) pos=({:.0},{:.0}) h={:.1} parked={}",
                i, angle, n[0], n[1], n[2], v.x, v.z, v.y, v.parked);
        }
    }
    if veh_on_slope == 0 { println!("  (none — all on flat terrain at init)"); }
    println!();

    // 5. Entities on slopes — Trash bins
    println!("--- Trash bins on slopes (tilt > 3°) ---");
    let mut bin_on_slope = 0;
    for (i, b) in world.trash_bins.iter().enumerate() {
        let n = b.terrain_normal;
        let angle = n[1].clamp(-1.0, 1.0).acos().to_degrees();
        if angle > 3.0 {
            bin_on_slope += 1;
            println!("  Bin[{:2}] tilt={:.1}° normal=({:.3},{:.3},{:.3}) pos=({:.0},{:.0}) h={:.1}",
                i, angle, n[0], n[1], n[2], b.x, b.z, b.y);
        }
    }
    if bin_on_slope == 0 { println!("  (none — all initialized flat)"); }
    println!();

    // 6. Terrain normal at each entity position (what tilt they SHOULD have)
    println!("--- Expected tilt at entity positions (terrain normal, > 3°) ---");
    let mut expected_count = 0;
    for (i, npc) in world.npcs.iter().enumerate() {
        let n = terrain.normal_at(npc.x, npc.z);
        let angle = n[1].clamp(-1.0, 1.0).acos().to_degrees();
        if angle > 3.0 {
            expected_count += 1;
            println!("  NPC[{:2}] expected_tilt={:.1}° at ({:.0},{:.0})", i, angle, npc.x, npc.z);
        }
    }
    for (i, v) in world.vehicles.iter().enumerate() {
        let n = terrain.normal_at(v.x, v.z);
        let angle = n[1].clamp(-1.0, 1.0).acos().to_degrees();
        if angle > 3.0 {
            expected_count += 1;
            println!("  Vehicle[{:2}] expected_tilt={:.1}° at ({:.0},{:.0})", i, angle, v.x, v.z);
        }
    }
    if expected_count == 0 { println!("  (all entities on flat terrain)"); }
    println!();

    println!("Summary: {} NPCs, {} vehicles, {} bins on tilted ground at init", npc_on_slope, veh_on_slope, bin_on_slope);
}
