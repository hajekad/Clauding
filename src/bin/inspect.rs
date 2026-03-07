// World inspector: generates the world and dumps diagnostic views
// Usage: cargo run --bin inspect -- [seed]
// Outputs: /tmp/clauding_inspect.txt

use clauding::{state, world};
use std::fmt::Write;

const MAP_W: usize = 120;
const MAP_H: usize = 60;

fn main() {
    let _ = std::fs::create_dir_all("debug");
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);

    let mut game = state::GameState::new(1, 1, seed);
    world::generate_world(&mut game);

    let mut out = String::with_capacity(64000);

    // ---- Full-world top-down map (500m, 1 cell ≈ 4.2m) ----
    dump_full_map(&game, &mut out);

    // ---- Zoomed downtown map (200m centered, 1 cell ≈ 1.7m) ----
    dump_zoomed_map(&game, &mut out, 0.0, 40.0, 100.0, "DOWNTOWN + RIVER ZONE (200m, centered on river)");

    // ---- River cross-sections ----
    dump_river_sections(&game, &mut out);

    // ---- Object placement quality ----
    dump_placement_audit(&game, &mut out);

    // ---- Height profile along river ----
    dump_river_height_profile(&game, &mut out);

    // ---- Parking lot details ----
    dump_parking_details(&game, &mut out);

    let path = "debug/inspect.txt";
    std::fs::write(path, &out).unwrap();
    eprintln!("Wrote {} bytes to {}", out.len(), path);
}

fn dump_full_map(game: &state::GameState, out: &mut String) {
    let _ = writeln!(out, "=== FULL WORLD MAP ({}m, seed={}) ===", state::WORLD_SIZE as u32, game.world_seed);
    let _ = writeln!(out, "Legend: #=building .=terrain ~=river ==bridge R=road r=fieldroad P=parking");
    let _ = writeln!(out, "        T=tree O=rock L=lamp B=bin V=vending M=market S=bus_stop W=wall");
    let _ = writeln!(out, "        v=vehicle *=item ^=water_tower X=billboard");
    let _ = writeln!(out);

    let w = &game.world;
    let half = state::WORLD_HALF;
    let cell_x = state::WORLD_SIZE / MAP_W as f32;
    let cell_z = state::WORLD_SIZE / MAP_H as f32;

    let mut grid = vec![vec![b'.'; MAP_W]; MAP_H];

    // Terrain: mark dockyard water
    for row in 0..MAP_H {
        for col in 0..MAP_W {
            let wz = -half + (row as f32 + 0.5) * cell_z;
            if wz > state::DOCK_Z_START + 15.0 { grid[row][col] = b'~'; }
        }
    }

    // Roads
    for seg in &game.road_network.segments {
        let steps = 40;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let x = seg.x0 + (seg.x1 - seg.x0) * t;
            let z = seg.z0 + (seg.z1 - seg.z0) * t;
            let (gx, gz) = w2g(x, z, half, cell_x, cell_z);
            if gx < MAP_W && gz < MAP_H {
                grid[gz][gx] = match seg.tier {
                    state::RoadTier::CarRoad => b'R',
                    state::RoadTier::FieldRoad => b'r',
                };
            }
        }
    }

    // River
    for seg in &w.river_segments {
        let steps = 20;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let x = seg.x1 + (seg.x2 - seg.x1) * t;
            let z = seg.z1 + (seg.z2 - seg.z1) * t;
            // Mark river width
            for offset in -3i32..=3 {
                let ox = x + offset as f32 * 2.0;
                let (gx, gz) = w2g(ox, z, half, cell_x, cell_z);
                if gx < MAP_W && gz < MAP_H && grid[gz][gx] != b'=' {
                    grid[gz][gx] = b'~';
                }
            }
        }
    }

    // Buildings
    for b in &w.buildings {
        let (gx, gz) = w2g(b.x, b.z, half, cell_x, cell_z);
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'#'; }
    }

    // Trees
    for t in &w.trees {
        let (gx, gz) = w2g(t.x, t.z, half, cell_x, cell_z);
        if gx < MAP_W && gz < MAP_H && grid[gz][gx] == b'.' { grid[gz][gx] = b'T'; }
    }

    // Rocks
    for r in &w.rocks {
        let (gx, gz) = w2g(r.x, r.z, half, cell_x, cell_z);
        if gx < MAP_W && gz < MAP_H && grid[gz][gx] == b'.' { grid[gz][gx] = b'O'; }
    }

    // Street lights
    for sl in &w.street_lights {
        let (gx, gz) = w2g(sl.x, sl.z, half, cell_x, cell_z);
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'L'; }
    }

    // Trash bins
    for tb in &w.trash_bins {
        let (gx, gz) = w2g(tb.x, tb.z, half, cell_x, cell_z);
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'B'; }
    }

    // Interactibles
    for inter in &w.interactibles {
        let (gx, gz) = w2g(inter.x, inter.z, half, cell_x, cell_z);
        if gx < MAP_W && gz < MAP_H {
            grid[gz][gx] = match inter.kind {
                state::InteractibleKind::VendingMachine => b'V',
                state::InteractibleKind::ParkBench => b'b',
                state::InteractibleKind::Dumpster => b'D',
                _ => b'i',
            };
        }
    }

    // Walls (fences, stall counters, etc)
    for wall in &w.walls {
        let (gx, gz) = w2g(wall.x, wall.z, half, cell_x, cell_z);
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'W'; }
    }

    // Vehicles
    for v in &w.vehicles {
        let (gx, gz) = w2g(v.x, v.z, half, cell_x, cell_z);
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'v'; }
    }

    // Parking spots (mark unoccupied ones)
    for spot in &game.road_network.parking_spots {
        let (gx, gz) = w2g(spot.x, spot.z, half, cell_x, cell_z);
        if gx < MAP_W && gz < MAP_H && grid[gz][gx] == b'R' { grid[gz][gx] = b'P'; }
    }

    // Road nodes (intersections)
    for node in &game.road_network.nodes {
        let (gx, gz) = w2g(node[0], node[1], half, cell_x, cell_z);
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'+'; }
    }

    // Column labels (every 10)
    let _ = write!(out, "     ");
    for col in 0..MAP_W {
        if col % 10 == 0 {
            let wx = -half + col as f32 * cell_x;
            let _ = write!(out, "{:<10.0}", wx);
        }
    }
    let _ = writeln!(out);

    for row in 0..MAP_H {
        let wz = -half + (row as f32 + 0.5) * cell_z;
        let _ = write!(out, "{:4.0} ", wz);
        for col in 0..MAP_W {
            let _ = write!(out, "{}", grid[row][col] as char);
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(out);
}

fn dump_zoomed_map(game: &state::GameState, out: &mut String, cx: f32, cz: f32, extent: f32, title: &str) {
    let _ = writeln!(out, "=== {} ===", title);
    let _ = writeln!(out, "Center=({:.0},{:.0}) extent={:.0}m", cx, cz, extent);
    let _ = writeln!(out);

    let w = &game.world;
    let half_ext = extent * 0.5;
    let min_x = cx - half_ext;
    let min_z = cz - half_ext;
    let cell = extent / MAP_W as f32;
    let cell_z = extent / MAP_H as f32;

    let mut grid = vec![vec![b'.'; MAP_W]; MAP_H];

    // Terrain height shading
    for row in 0..MAP_H {
        for col in 0..MAP_W {
            let wx = min_x + (col as f32 + 0.5) * cell;
            let wz = min_z + (row as f32 + 0.5) * cell_z;
            let h = game.terrain.height_at(wx, wz);
            grid[row][col] = if h < -2.0 { b'_' } // river bed
            else if h < -0.5 { b',' } // shallow
            else { b'.' };
        }
    }

    // Roads
    for seg in &game.road_network.segments {
        let steps = 80;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let x = seg.x0 + (seg.x1 - seg.x0) * t;
            let z = seg.z0 + (seg.z1 - seg.z0) * t;
            let gx = ((x - min_x) / cell) as usize;
            let gz = ((z - min_z) / cell_z) as usize;
            if gx < MAP_W && gz < MAP_H {
                grid[gz][gx] = match seg.tier {
                    state::RoadTier::CarRoad => b'R',
                    state::RoadTier::FieldRoad => b'r',
                };
            }
        }
    }

    // River
    for seg in &w.river_segments {
        let steps = 30;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let x = seg.x1 + (seg.x2 - seg.x1) * t;
            let z = seg.z1 + (seg.z2 - seg.z1) * t;
            for offset in -6i32..=6 {
                let ox = x + offset as f32 * 1.0;
                let gx = ((ox - min_x) / cell) as usize;
                let gz = ((z - min_z) / cell_z) as usize;
                if gx < MAP_W && gz < MAP_H {
                    grid[gz][gx] = b'~';
                }
            }
        }
    }

    // Buildings
    for b in &w.buildings {
        let gx = ((b.x - min_x) / cell) as usize;
        let gz = ((b.z - min_z) / cell_z) as usize;
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'#'; }
    }

    // Trees, rocks, lights, bins, walls, vehicles, parking
    for t in &w.trees {
        let gx = ((t.x - min_x) / cell) as usize;
        let gz = ((t.z - min_z) / cell_z) as usize;
        if gx < MAP_W && gz < MAP_H && grid[gz][gx] == b'.' { grid[gz][gx] = b'T'; }
    }
    for wall in &w.walls {
        let gx = ((wall.x - min_x) / cell) as usize;
        let gz = ((wall.z - min_z) / cell_z) as usize;
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'W'; }
    }
    for v in &w.vehicles {
        let gx = ((v.x - min_x) / cell) as usize;
        let gz = ((v.z - min_z) / cell_z) as usize;
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'v'; }
    }
    for spot in &game.road_network.parking_spots {
        let gx = ((spot.x - min_x) / cell) as usize;
        let gz = ((spot.z - min_z) / cell_z) as usize;
        if gx < MAP_W && gz < MAP_H && (grid[gz][gx] == b'.' || grid[gz][gx] == b'R') { grid[gz][gx] = b'P'; }
    }
    for sl in &w.street_lights {
        let gx = ((sl.x - min_x) / cell) as usize;
        let gz = ((sl.z - min_z) / cell_z) as usize;
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'L'; }
    }
    for inter in &w.interactibles {
        let gx = ((inter.x - min_x) / cell) as usize;
        let gz = ((inter.z - min_z) / cell_z) as usize;
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'i'; }
    }
    // Nodes
    for node in &game.road_network.nodes {
        let gx = ((node[0] - min_x) / cell) as usize;
        let gz = ((node[1] - min_z) / cell_z) as usize;
        if gx < MAP_W && gz < MAP_H { grid[gz][gx] = b'+'; }
    }

    for row in 0..MAP_H {
        let wz = min_z + (row as f32 + 0.5) * cell_z;
        let _ = write!(out, "{:4.0} ", wz);
        for col in 0..MAP_W {
            let _ = write!(out, "{}", grid[row][col] as char);
        }
        let _ = writeln!(out);
    }
    let _ = writeln!(out);
}

fn dump_river_sections(game: &state::GameState, out: &mut String) {
    let _ = writeln!(out, "=== RIVER CROSS-SECTIONS (height profile, X at fixed points) ===");
    let _ = writeln!(out, "  Shows terrain height at Z positions crossing the river");
    let _ = writeln!(out);

    // Sample at a few X positions
    for &sample_x in &[-150.0f32, -50.0, 0.0, 50.0, 150.0] {
        let river_center_z = 40.0 + 25.0 * (sample_x * 0.02f32).sin();
        let _ = writeln!(out, "--- X={:.0}, river center Z={:.1} ---", sample_x, river_center_z);

        let z_start = river_center_z - 20.0;
        let z_end = river_center_z + 20.0;
        let steps = 40;
        let _ = write!(out, "  Z:   ");
        for i in 0..=steps {
            let z = z_start + (z_end - z_start) * i as f32 / steps as f32;
            if i % 5 == 0 { let _ = write!(out, "{:5.0}", z); }
        }
        let _ = writeln!(out);

        let _ = write!(out, "  H:   ");
        for i in 0..=steps {
            let z = z_start + (z_end - z_start) * i as f32 / steps as f32;
            let h = game.terrain.height_at(sample_x, z);
            if i % 5 == 0 { let _ = write!(out, "{:5.1}", h); }
        }
        let _ = writeln!(out);

        // ASCII cross-section (height = row, Z = col)
        let h_min = -5.0f32;
        let h_max = 5.0f32;
        let rows = 12;
        for row in 0..rows {
            let h_level = h_max - (row as f32 / rows as f32) * (h_max - h_min);
            let _ = write!(out, "  {:5.1} |", h_level);
            for i in 0..=steps {
                let z = z_start + (z_end - z_start) * i as f32 / steps as f32;
                let h = game.terrain.height_at(sample_x, z);
                let on_riv = world::on_river(sample_x, z, &game.world.river_segments);
                if h >= h_level - 0.3 && h < h_level + 0.3 {
                    let _ = write!(out, "#");
                } else if on_riv && h_level < -0.3 && h_level > h - 0.3 {
                    let _ = write!(out, "~");
                } else {
                    let _ = write!(out, " ");
                }
            }
            let _ = writeln!(out, "|");
        }
        let _ = writeln!(out);
    }
}

fn dump_river_height_profile(game: &state::GameState, out: &mut String) {
    let _ = writeln!(out, "=== RIVER HEIGHT PROFILE (water Y vs terrain bank Y) ===");
    let _ = writeln!(out, "  X       | river_center_Z | bank_h (edge) | carved_h (center) | delta");
    let _ = writeln!(out, "  --------|----------------|---------------|-------------------|------");
    for &x in &[-200.0f32, -150.0, -100.0, -50.0, 0.0, 50.0, 100.0, 150.0, 200.0] {
        let rz = 40.0 + 25.0 * (x * 0.02f32).sin();
        let bank_h = game.terrain.height_at(x, rz + state::RIVER_WIDTH * 0.5 + 2.0);
        let center_h = game.terrain.height_at(x, rz);
        let _ = writeln!(out, "  {:7.0} | {:14.1} | {:13.2} | {:17.2} | {:5.2}",
            x, rz, bank_h, center_h, bank_h - center_h);
    }
    let _ = writeln!(out);
}

fn dump_placement_audit(game: &state::GameState, out: &mut String) {
    let _ = writeln!(out, "=== PLACEMENT AUDIT ===");
    let w = &game.world;

    // Buildings on river
    let mut buildings_on_river = 0;
    for b in &w.buildings {
        if world::on_river(b.x, b.z, &w.river_segments) { buildings_on_river += 1; }
    }
    let _ = writeln!(out, "Buildings on river: {}/{}", buildings_on_river, w.buildings.len());

    // Trees on river
    let mut trees_on_river = 0;
    for t in &w.trees {
        if world::on_river(t.x, t.z, &w.river_segments) { trees_on_river += 1; }
    }
    let _ = writeln!(out, "Trees on river: {}/{}", trees_on_river, w.trees.len());

    // Vehicles on river
    let mut vehicles_on_river = 0;
    for v in &w.vehicles {
        if world::on_river(v.x, v.z, &w.river_segments) { vehicles_on_river += 1; }
    }
    let _ = writeln!(out, "Vehicles on river: {}/{}", vehicles_on_river, w.vehicles.len());

    // Parking spots on roads
    let mut spots_on_road = 0;
    let mut spots_on_river = 0;
    for spot in &game.road_network.parking_spots {
        if world::on_river(spot.x, spot.z, &w.river_segments) { spots_on_river += 1; }
        let surf = world::surface_at(spot.x, spot.z, &game.road_network);
        if surf == state::Surface::CarRoad || surf == state::Surface::Sidewalk { spots_on_road += 1; }
    }
    let _ = writeln!(out, "Parking spots on river: {}/{}", spots_on_river, game.road_network.parking_spots.len());
    let _ = writeln!(out, "Parking spots on/near road (expected): {}/{}", spots_on_road, game.road_network.parking_spots.len());

    // Vehicles in parking spots vs not
    let parked_count = w.vehicles.iter().filter(|v| v.parked).count();
    let _ = writeln!(out, "Vehicles parked: {}/{}", parked_count, w.vehicles.len());

    // Wall details
    let _ = writeln!(out, "\nWalls ({}):", w.walls.len());
    for (i, wall) in w.walls.iter().enumerate() {
        let _ = writeln!(out, "  [{:2}] pos=({:6.1},{:6.1}) hw={:.2} hd={:.2} h={:.1}",
            i, wall.x, wall.z, wall.hw, wall.hd, wall.height);
    }

    // Object counts summary
    let _ = writeln!(out, "\nObject counts:");
    let _ = writeln!(out, "  Buildings: {}", w.buildings.len());
    let _ = writeln!(out, "  Trees: {}", w.trees.len());
    let _ = writeln!(out, "  Rocks: {}", w.rocks.len());
    let _ = writeln!(out, "  Street lights: {}", w.street_lights.len());
    let _ = writeln!(out, "  Trash bins: {}", w.trash_bins.len());
    let _ = writeln!(out, "  Interactibles: {}", w.interactibles.len());
    let _ = writeln!(out, "  Walls: {}", w.walls.len());
    let _ = writeln!(out, "  River segments: {}", w.river_segments.len());
    let _ = writeln!(out, "  Vehicles: {} ({} parked)", w.vehicles.len(), parked_count);
    let _ = writeln!(out, "  Parking spots: {} ({} occupied)",
        game.road_network.parking_spots.len(),
        game.road_network.parking_spots.iter().filter(|s| s.occupied_by.is_some()).count());
    let _ = writeln!(out, "  Static tris: {}", w.static_tris.len());
    let _ = writeln!(out);
}

fn dump_parking_details(game: &state::GameState, out: &mut String) {
    let _ = writeln!(out, "=== PARKING LOT DETAILS ===");

    // Group parking spots by proximity to identify lots
    let spots = &game.road_network.parking_spots;
    let mut visited = vec![false; spots.len()];
    let mut lot_id = 0;

    for i in 0..spots.len() {
        if visited[i] { continue; }
        // Find all spots within 20m of this one (cluster = lot)
        let mut cluster = vec![i];
        visited[i] = true;
        let mut j = 0;
        while j < cluster.len() {
            let ci = cluster[j];
            for k in 0..spots.len() {
                if visited[k] { continue; }
                let dx = spots[ci].x - spots[k].x;
                let dz = spots[ci].z - spots[k].z;
                if dx * dx + dz * dz < 20.0 * 20.0 {
                    cluster.push(k);
                    visited[k] = true;
                }
            }
            j += 1;
        }
        if cluster.len() >= 6 {
            let avg_x: f32 = cluster.iter().map(|&i| spots[i].x).sum::<f32>() / cluster.len() as f32;
            let avg_z: f32 = cluster.iter().map(|&i| spots[i].z).sum::<f32>() / cluster.len() as f32;
            let occupied = cluster.iter().filter(|&&i| spots[i].occupied_by.is_some()).count();
            let _ = writeln!(out, "Lot {} at ({:.0},{:.0}): {} spots, {} occupied",
                lot_id, avg_x, avg_z, cluster.len(), occupied);
            lot_id += 1;
        }
    }
    let _ = writeln!(out);
}

fn w2g(x: f32, z: f32, half: f32, cell_x: f32, cell_z: f32) -> (usize, usize) {
    let gx = ((x + half) / cell_x) as usize;
    let gz = ((z + half) / cell_z) as usize;
    (gx.min(MAP_W - 1), gz.min(MAP_H - 1))
}
