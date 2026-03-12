// Simulation observer: runs headless game and dumps behavioral analysis
// Usage: cargo run --bin observe -- [seed] [days] [output_path]
// Outputs: debug/observe_s{seed}.txt (or custom path) with detailed game dynamics

use clauding::{state, world, npc, vehicle, collision, combat};
use std::fmt::Write;

// Use shared headless timestep from state module
const FIXED_DT: f32 = state::HEADLESS_DT;

fn main() {
    let _ = std::fs::create_dir_all("debug");
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let max_days: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);

    eprintln!("Observer: seed={} days={}", seed, max_days);

    let mut game = state::GameState::init(1, 1, seed);

    let mut out = String::with_capacity(128000);
    let _ = writeln!(out, "=== SIMULATION OBSERVER (seed={}, days={}) ===\n", seed, max_days);

    // Track NPC positions over time for stuck detection
    let n_npcs = game.world.npcs.len();
    let n_vehicles = game.world.vehicles.len();
    let mut prev_npc_pos: Vec<(f32, f32)> = game.world.npcs.iter().map(|n| (n.x, n.z)).collect();
    let mut npc_stuck_ticks: Vec<u32> = vec![0; n_npcs];
    let mut npc_river_ticks: Vec<u32> = vec![0; n_npcs];
    let mut npc_total_dist: Vec<f32> = vec![0.0; n_npcs];
    // Peak stuck data captured before midnight reset (survives across days)
    let mut peak_stuck_ticks: Vec<u32> = vec![0; n_npcs];
    let mut peak_river_ticks: Vec<u32> = vec![0; n_npcs];

    let mut prev_veh_pos: Vec<(f32, f32)> = game.world.vehicles.iter().map(|v| (v.x, v.z)).collect();
    let mut veh_stuck_ticks: Vec<u32> = vec![0; n_vehicles];

    // Aggregate counters
    let mut _snapshot_count: u32 = 0;
    let mut state_time_accum = [0u32; 8]; // ticks in each NPC state
    let mut total_items_picked: u32 = 0; // accumulated across all days (survives midnight reset)
    let mut total_items_deposited: u32 = 0;

    // Per-job tracking
    let mut per_job_items_picked = [0u32; state::NPC_JOB_COUNT];
    let mut per_job_items_deposited = [0u32; state::NPC_JOB_COUNT];
    let mut per_job_state_time = [[0u32; 8]; state::NPC_JOB_COUNT]; // [job][state] tick counts

    // Per-NPC productivity
    let mut per_npc_items_picked: Vec<u32> = vec![0; n_npcs];

    // Stuck episode counting (separate from continuous stuck_ticks)
    let mut npc_stuck_episodes: Vec<u32> = vec![0; n_npcs];
    let mut npc_was_stuck: Vec<bool> = vec![false; n_npcs];

    // Job failure diagnostic counters
    let mut total_find_item_failures: u32 = 0;
    let mut total_find_bin_failures: u32 = 0;
    let mut total_stuck_recoveries: u32 = 0;
    let mut per_job_find_item_failures = [0u32; state::NPC_JOB_COUNT];
    let mut per_job_find_bin_failures = [0u32; state::NPC_JOB_COUNT];
    let mut per_job_stuck_recoveries = [0u32; state::NPC_JOB_COUNT];

    // Vehicle-on-river tracking
    let mut veh_river_ticks: Vec<u32> = vec![0; n_vehicles];

    // Per-hour snapshots
    let mut prev_time_of_day: f32 = game.time_of_day;
    let _ = prev_time_of_day;
    let mut last_snapshot_hour: i32 = -1;
    let mut tick: u64 = 0;

    let start = std::time::Instant::now();

    while game.day_count <= max_days {
        prev_time_of_day = game.time_of_day;

        // Advance time
        game.time_of_day += FIXED_DT * 24.0 / state::DAY_LENGTH;
        if game.time_of_day >= 24.0 { game.time_of_day -= 24.0; }

        // Capture pre-reset metrics before midnight clears them
        if prev_time_of_day > 23.5 && game.time_of_day < 0.5 {
            for (i, npc) in game.world.npcs.iter().enumerate() {
                total_items_picked += npc.fitness_items_picked;
                total_items_deposited += npc.items_deposited_today;
                let ji = npc.job.index();
                per_job_items_picked[ji] += npc.fitness_items_picked;
                per_job_items_deposited[ji] += npc.items_deposited_today;
                per_npc_items_picked[i] += npc.fitness_items_picked;
                total_find_item_failures += npc.find_item_failures;
                total_find_bin_failures += npc.find_bin_failures;
                total_stuck_recoveries += npc.stuck_recoveries;
                per_job_find_item_failures[ji] += npc.find_item_failures;
                per_job_find_bin_failures[ji] += npc.find_bin_failures;
                per_job_stuck_recoveries[ji] += npc.stuck_recoveries;
            }
        }
        // Midnight reset
        if npc::sys_midnight_reset(
            &mut game.world, game.time_of_day, prev_time_of_day,
            &mut game.neat_population, &mut game.neat_brains,
        ) {
            game.day_count += 1;
            let _ = writeln!(out, "--- DAY {} RESET (gen {}) ---", game.day_count, game.neat_population.generation);
            // Capture peak stuck/river data before resetting (max across all days)
            for i in 0..n_npcs {
                peak_stuck_ticks[i] = peak_stuck_ticks[i].max(npc_stuck_ticks[i]);
                peak_river_ticks[i] = peak_river_ticks[i].max(npc_river_ticks[i]);
            }
            // Reset per-day observer counters so snapshots reflect current day only
            for t in npc_stuck_ticks.iter_mut() { *t = 0; }
            for t in npc_river_ticks.iter_mut() { *t = 0; }
        }

        // Game systems
        vehicle::sys_vehicle(&mut game, FIXED_DT);
        npc::sys_npc(
            &mut game.world, &mut game.road_network, &game.terrain,
            FIXED_DT, game.time_of_day, &mut game.neat_brains,
            0.0, 0.0, &game.walk_grid,
        );
        npc::sys_night_spawning(
            &mut game.world, &game.terrain, game.time_of_day,
            FIXED_DT, &mut game.spawn_rng, &game.road_network,
        );
        npc::sys_items_update(&mut game.world, FIXED_DT);
        npc::sys_npc_interactions(&mut game.world, FIXED_DT);
        npc::sys_hunger_thirst(&mut game.world, &mut game.player, FIXED_DT);
        collision::sys_collisions_headless(&mut game.world, &game.terrain, FIXED_DT);
        combat::sys_ragdoll_update(&mut game.world, &game.terrain, FIXED_DT);

        // Headless NPC-NPC combat
        combat::sys_combat_headless(&mut game.world, &game.terrain, FIXED_DT);


        tick += 1;
        game.frame_counter += 1;

        // Per-tick tracking (every 10 ticks to reduce overhead)
        if tick % 10 == 0 {
            for i in 0..n_npcs {
                let npc = &game.world.npcs[i];
                let dx = npc.x - prev_npc_pos[i].0;
                let dz = npc.z - prev_npc_pos[i].1;
                let dist = (dx * dx + dz * dz).sqrt();
                npc_total_dist[i] += dist;

                if dist < 0.05 && npc.state == state::NpcState::Working && npc.job.is_mobile() {
                    npc_stuck_ticks[i] += 1;
                    if !npc_was_stuck[i] {
                        npc_stuck_episodes[i] += 1;
                        npc_was_stuck[i] = true;
                    }
                } else {
                    npc_stuck_ticks[i] = 0;
                    npc_was_stuck[i] = false;
                }

                if !npc.in_vehicle && world::on_river_not_bridge(npc.x, npc.z, &game.world.river_segments, &game.world.bridges) {
                    npc_river_ticks[i] += 1;
                }

                prev_npc_pos[i] = (npc.x, npc.z);

                // State accumulator
                let idx = npc.state.index();
                state_time_accum[idx] += 1;
                per_job_state_time[npc.job.index()][idx] += 1;
            }

            for i in 0..n_vehicles {
                let v = &game.world.vehicles[i];
                let dx = v.x - prev_veh_pos[i].0;
                let dz = v.z - prev_veh_pos[i].1;
                let dist = (dx * dx + dz * dz).sqrt();
                if dist < 0.01 && v.ai_active && v.speed.abs() > 0.5 {
                    veh_stuck_ticks[i] += 1;
                } else if v.ai_active {
                    veh_stuck_ticks[i] = 0;
                }
                if world::on_river_not_bridge(v.x, v.z, &game.world.river_segments, &game.world.bridges) {
                    veh_river_ticks[i] += 1;
                }
                prev_veh_pos[i] = (v.x, v.z);
            }
        }

        // Hourly snapshot
        let current_hour = game.time_of_day as i32;
        if current_hour != last_snapshot_hour && current_hour % 3 == 0 {
            last_snapshot_hour = current_hour;
            _snapshot_count += 1;
            dump_snapshot(&game, &mut out, tick, &npc_stuck_ticks, &npc_river_ticks,
                &npc_total_dist, &veh_stuck_ticks, &state_time_accum);
        }
    }

    let elapsed = start.elapsed().as_secs_f32();

    // Final analysis
    let _ = writeln!(out, "\n=== FINAL ANALYSIS ({:.1}s real time, {} ticks) ===\n", elapsed, tick);

    // Merge final day's data into peak
    for i in 0..n_npcs {
        peak_stuck_ticks[i] = peak_stuck_ticks[i].max(npc_stuck_ticks[i]);
        peak_river_ticks[i] = peak_river_ticks[i].max(npc_river_ticks[i]);
    }

    // Stuck NPC analysis — only flag mobile jobs (Collector, GarbageCollector, Delivery, etc.)
    // Jobs like Vendor, Fisherman, Mechanic, Lumberjack, etc. intentionally stand still
    let _ = writeln!(out, "--- STUCK NPC ANALYSIS (peak stuck score across all days, mobile jobs only) ---");
    let mut stuck_npcs = 0;
    for i in 0..n_npcs {
        if peak_stuck_ticks[i] > 50 {
            let npc = &game.world.npcs[i];
            if !npc.job.is_mobile() { continue; }
            let _ = writeln!(out, "  NPC[{:2}] stuck_score={:4} episodes={:2} pos=({:6.1},{:6.1}) total_dist={:6.1}m job={:12}",
                i, peak_stuck_ticks[i], npc_stuck_episodes[i], npc.x, npc.z, npc_total_dist[i], npc.job.name());
            stuck_npcs += 1;
        }
    }
    let total_episodes: u32 = npc_stuck_episodes.iter().sum();
    let _ = writeln!(out, "  Total stuck episodes (all NPCs): {}", total_episodes);
    if stuck_npcs == 0 { let _ = writeln!(out, "  No stuck NPCs detected!"); }

    // River exposure
    let _ = writeln!(out, "\n--- RIVER EXPOSURE (NPCs on river, peak across all days) ---");
    let mut river_npcs = 0;
    for i in 0..n_npcs {
        if peak_river_ticks[i] > 0 {
            let npc = &game.world.npcs[i];
            let _ = writeln!(out, "  NPC[{:2}] river_ticks={:4} pos=({:6.1},{:6.1}) job={:12}",
                i, peak_river_ticks[i], npc.x, npc.z, npc.job.name());
            river_npcs += 1;
        }
    }
    if river_npcs == 0 { let _ = writeln!(out, "  No NPCs entered the river!"); }

    // Stuck vehicles
    let _ = writeln!(out, "\n--- STUCK VEHICLES (ai_active + speed>0.5 but not moving) ---");
    let mut stuck_vehs = 0;
    for i in 0..n_vehicles {
        if veh_stuck_ticks[i] > 20 {
            let v = &game.world.vehicles[i];
            let _ = writeln!(out, "  Vehicle[{:3}] stuck_score={:4} pos=({:6.1},{:6.1}) speed={:.1} owner={:?}",
                i, veh_stuck_ticks[i], v.x, v.z, v.speed, v.owner_npc);
            stuck_vehs += 1;
        }
    }
    if stuck_vehs == 0 { let _ = writeln!(out, "  No stuck vehicles detected!"); }

    // Vehicle-on-river analysis
    let _ = writeln!(out, "\n--- VEHICLE RIVER EXPOSURE ---");
    let mut veh_river_count = 0;
    for i in 0..n_vehicles {
        if veh_river_ticks[i] > 0 {
            let v = &game.world.vehicles[i];
            let _ = writeln!(out, "  Vehicle[{:3}] river_ticks={:4} pos=({:6.1},{:6.1}) owner={:?}",
                i, veh_river_ticks[i], v.x, v.z, v.owner_npc);
            veh_river_count += 1;
        }
    }
    if veh_river_count == 0 { let _ = writeln!(out, "  No vehicles entered the river!"); }

    // NPC movement stats
    let _ = writeln!(out, "\n--- NPC MOVEMENT STATS ---");
    let mut dists: Vec<(usize, f32)> = (0..n_npcs).map(|i| (i, npc_total_dist[i])).collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let _ = writeln!(out, "  Least mobile NPCs:");
    for &(i, d) in dists.iter().take(5) {
        let npc = &game.world.npcs[i];
        let _ = writeln!(out, "    NPC[{:2}] dist={:6.1}m pos=({:6.1},{:6.1}) job={:12} state={:12}",
            i, d, npc.x, npc.z, npc.job.name(), npc.state.name());
    }
    let _ = writeln!(out, "  Most mobile NPCs:");
    for &(i, d) in dists.iter().rev().take(5) {
        let npc = &game.world.npcs[i];
        let _ = writeln!(out, "    NPC[{:2}] dist={:6.1}m pos=({:6.1},{:6.1}) job={:12} state={:12}",
            i, d, npc.x, npc.z, npc.job.name(), npc.state.name());
    }

    // State time distribution
    let total_state_ticks: u32 = state_time_accum.iter().sum();
    let _ = writeln!(out, "\n--- STATE TIME DISTRIBUTION (% of NPC-ticks) ---");
    let names = ["Sleeping", "HomeTask", "GoingToWork", "Working", "GoingHome", "Driving", "Interacting", "KnockedOut"];
    for (i, name) in names.iter().enumerate() {
        let pct = state_time_accum[i] as f32 / total_state_ticks.max(1) as f32 * 100.0;
        let _ = writeln!(out, "  {:12} {:5.1}% ({} ticks)", name, pct, state_time_accum[i]);
    }

    // Per-job state time breakdown
    let _ = writeln!(out, "\n--- STATE TIME BY JOB (% of ticks for that job) ---");
    for ji in 0..state::NPC_JOB_COUNT {
        let job_total: u32 = per_job_state_time[ji].iter().sum();
        if job_total == 0 { continue; }
        let mut parts = String::new();
        for (si, &name) in names.iter().enumerate() {
            let pct = per_job_state_time[ji][si] as f32 / job_total as f32 * 100.0;
            if pct > 0.5 {
                let _ = write!(parts, " {}={:.0}%", name, pct);
            }
        }
        let _ = writeln!(out, "  {:12}{}", state::JOB_NAMES[ji], parts);
    }

    // Building (home) distribution
    let _ = writeln!(out, "\n--- BUILDING/HOME DISTRIBUTION ---");
    let mut bq = [0u32; 4];
    for b in &game.world.buildings { let q = if b.x<0.0 {if b.z<0.0 {0} else {2}} else {if b.z<0.0 {1} else {3}}; bq[q]+=1; }
    let _ = writeln!(out, "  All {} buildings: NW={} NE={} SW={} SE={}", game.world.buildings.len(), bq[0], bq[1], bq[2], bq[3]);
    let mut hq = [0u32; 4];
    let mut home_near = 0u32;
    let mut home_far = 0u32;
    for i in 0..n_npcs {
        let b = &game.world.buildings[game.world.npcs[i].home_idx];
        let q = if b.x<0.0 {if b.z<0.0 {0} else {2}} else {if b.z<0.0 {1} else {3}}; hq[q]+=1;
        let d = (b.x*b.x + b.z*b.z).sqrt();
        if d < 50.0 { home_near += 1; }
        if d > 150.0 { home_far += 1; }
    }
    let _ = writeln!(out, "  NPC homes (first 100): NW={} NE={} SW={} SE={}", hq[0], hq[1], hq[2], hq[3]);
    let _ = writeln!(out, "  NPC homes: near_center(<50m)={} far_out(>150m)={}", home_near, home_far);

    // Item economy — use accumulated totals plus current (incomplete) day
    let items_active = game.world.items.iter().filter(|it| it.active).count();
    let cur_picked: u32 = game.world.npcs.iter().map(|n| n.fitness_items_picked).sum();
    let cur_dep: u32 = game.world.npcs.iter().map(|n| n.items_deposited_today).sum();
    // Add current day's per-job/per-NPC data
    let mut final_per_job_picked = per_job_items_picked;
    let mut final_per_job_deposited = per_job_items_deposited;
    let mut final_per_npc_picked = per_npc_items_picked.clone();
    for (i, npc) in game.world.npcs.iter().enumerate() {
        let ji = npc.job.index();
        final_per_job_picked[ji] += npc.fitness_items_picked;
        final_per_job_deposited[ji] += npc.items_deposited_today;
        final_per_npc_picked[i] += npc.fitness_items_picked;
    }
    let _ = writeln!(out, "\n--- ITEM ECONOMY ---");
    let _ = writeln!(out, "  Active items: {}/{}", items_active, state::NUM_ITEMS);
    let _ = writeln!(out, "  Total picked (all days): {}", total_items_picked + cur_picked);
    let _ = writeln!(out, "  Total deposited (all days): {}", total_items_deposited + cur_dep);
    let total_npc_money: f32 = game.world.npcs.iter().map(|n| n.money).sum();
    let avg_npc_money = total_npc_money / n_npcs.max(1) as f32;
    let _ = writeln!(out, "  Total NPC money: ${:.0} (avg ${:.0})", total_npc_money, avg_npc_money);

    // Per-job item breakdown
    let _ = writeln!(out, "\n  Per-job breakdown:");
    for ji in 0..state::NPC_JOB_COUNT {
        if final_per_job_picked[ji] > 0 || final_per_job_deposited[ji] > 0 {
            let _ = writeln!(out, "    {:12} picked={:4} deposited={:4}",
                state::JOB_NAMES[ji], final_per_job_picked[ji], final_per_job_deposited[ji]);
        }
    }

    // Per-NPC productivity top/bottom 5
    let _ = writeln!(out, "\n  Top 5 productive NPCs:");
    let mut prod: Vec<(usize, u32)> = (0..n_npcs).map(|i| (i, final_per_npc_picked[i])).collect();
    prod.sort_by(|a, b| b.1.cmp(&a.1));
    for &(i, count) in prod.iter().take(5) {
        let npc = &game.world.npcs[i];
        let _ = writeln!(out, "    NPC[{:2}] picked={:3} deposited={:3} job={:12} dist={:.0}m",
            i, count, npc.items_deposited_today, npc.job.name(), npc_total_dist[i]);
    }
    let _ = writeln!(out, "  Bottom 5 productive NPCs (mobile jobs):");
    let mobile_prod: Vec<(usize, u32)> = prod.iter()
        .filter(|&&(i, _)| game.world.npcs[i].job.is_mobile())
        .copied().collect();
    for &(i, count) in mobile_prod.iter().rev().take(5) {
        let npc = &game.world.npcs[i];
        let _ = writeln!(out, "    NPC[{:2}] picked={:3} deposited={:3} job={:12} dist={:.0}m",
            i, count, npc.items_deposited_today, npc.job.name(), npc_total_dist[i]);
    }

    // Also capture current (incomplete) day's failure data
    for npc in &game.world.npcs {
        let ji = npc.job.index();
        total_find_item_failures += npc.find_item_failures;
        total_find_bin_failures += npc.find_bin_failures;
        total_stuck_recoveries += npc.stuck_recoveries;
        per_job_find_item_failures[ji] += npc.find_item_failures;
        per_job_find_bin_failures[ji] += npc.find_bin_failures;
        per_job_stuck_recoveries[ji] += npc.stuck_recoveries;
    }
    let _ = writeln!(out, "\n--- JOB FAILURE DIAGNOSTICS (all days) ---");
    let _ = writeln!(out, "  find_item failures: {}  find_bin failures: {}  stuck recoveries: {}",
        total_find_item_failures, total_find_bin_failures, total_stuck_recoveries);
    let _ = writeln!(out, "\n  Per-job breakdown:");
    for ji in 0..state::NPC_JOB_COUNT {
        if per_job_find_item_failures[ji] > 0 || per_job_find_bin_failures[ji] > 0 || per_job_stuck_recoveries[ji] > 0 {
            let _ = writeln!(out, "    {:12} item_fail={:4} bin_fail={:4} stuck_recover={:4}",
                state::JOB_NAMES[ji], per_job_find_item_failures[ji], per_job_find_bin_failures[ji], per_job_stuck_recoveries[ji]);
        }
    }

    // Survival stats
    let avg_hunger = game.world.npcs.iter().map(|n| n.hunger).sum::<f32>() / n_npcs.max(1) as f32;
    let avg_thirst = game.world.npcs.iter().map(|n| n.thirst).sum::<f32>() / n_npcs.max(1) as f32;
    let starving = game.world.npcs.iter().filter(|n| n.hunger <= 0.0).count();
    let dehydrated = game.world.npcs.iter().filter(|n| n.thirst <= 0.0).count();
    let dead = game.world.npcs.iter().filter(|n| n.starving_dead).count();
    let ko = game.world.npcs.iter().filter(|n| n.state == state::NpcState::KnockedOut).count();
    let _ = writeln!(out, "\n--- SURVIVAL ---");
    let _ = writeln!(out, "  Avg hunger: {:.0}  Avg thirst: {:.0}", avg_hunger, avg_thirst);
    let _ = writeln!(out, "  Starving: {}  Dehydrated: {}  Dead: {}  KO: {}", starving, dehydrated, dead, ko);

    // NPC spatial distribution
    let _ = writeln!(out, "\n--- NPC SPATIAL DISTRIBUTION ---");
    let mut quadrants = [0u32; 4]; // NW, NE, SW, SE
    let mut near_center = 0u32;
    let mut far_out = 0u32;
    for npc in &game.world.npcs {
        let q = if npc.x < 0.0 { if npc.z < 0.0 { 0 } else { 2 } } else { if npc.z < 0.0 { 1 } else { 3 } };
        quadrants[q] += 1;
        let d = (npc.x * npc.x + npc.z * npc.z).sqrt();
        if d < 50.0 { near_center += 1; }
        if d > 150.0 { far_out += 1; }
    }
    let _ = writeln!(out, "  NW={} NE={} SW={} SE={}", quadrants[0], quadrants[1], quadrants[2], quadrants[3]);
    let _ = writeln!(out, "  Near center (<50m): {}  Far out (>150m): {}", near_center, far_out);

    // Slope physics analysis — terrain tilt on entities after simulation
    let _ = writeln!(out, "\n--- SLOPE PHYSICS (entity tilt after simulation) ---");
    let mut npc_tilted = 0u32;
    let mut npc_max_tilt: f32 = 0.0;
    let mut veh_tilted = 0u32;
    let mut veh_max_tilt: f32 = 0.0;
    let mut bin_tilted = 0u32;
    for npc in &game.world.npcs {
        let angle = npc.terrain_normal[1].clamp(-1.0, 1.0).acos().to_degrees();
        if angle > 2.0 { npc_tilted += 1; }
        if angle > npc_max_tilt { npc_max_tilt = angle; }
    }
    for v in &game.world.vehicles {
        let angle = v.terrain_normal[1].clamp(-1.0, 1.0).acos().to_degrees();
        if angle > 2.0 { veh_tilted += 1; }
        if angle > veh_max_tilt { veh_max_tilt = angle; }
    }
    for b in &game.world.trash_bins {
        let angle = b.terrain_normal[1].clamp(-1.0, 1.0).acos().to_degrees();
        if angle > 2.0 { bin_tilted += 1; }
    }
    let _ = writeln!(out, "  NPCs tilted (>2°): {}/{}  max={:.1}°", npc_tilted, n_npcs, npc_max_tilt);
    let _ = writeln!(out, "  Vehicles tilted (>2°): {}/{}  max={:.1}°", veh_tilted, n_vehicles, veh_max_tilt);
    let _ = writeln!(out, "  Bins tilted (>2°): {}/{}", bin_tilted, game.world.trash_bins.len());
    // Check for extreme tilt (clipping risk)
    let extreme_npc = game.world.npcs.iter().filter(|n| n.terrain_normal[1].clamp(-1.0, 1.0).acos().to_degrees() > 30.0).count();
    let extreme_veh = game.world.vehicles.iter().filter(|v| v.terrain_normal[1].clamp(-1.0, 1.0).acos().to_degrees() > 30.0).count();
    if extreme_npc > 0 || extreme_veh > 0 {
        let _ = writeln!(out, "  WARNING: {} NPCs + {} vehicles at >30° tilt (clipping risk)", extreme_npc, extreme_veh);
    }

    // Vehicle activity
    let active_vehicles = game.world.vehicles.iter().filter(|v| v.ai_active || v.occupied).count();
    let moving_vehicles = game.world.vehicles.iter().filter(|v| v.speed.abs() > 0.5).count();
    let parked_vehicles = game.world.vehicles.iter().filter(|v| v.parked).count();
    let _ = writeln!(out, "\n--- VEHICLE ACTIVITY ---");
    let _ = writeln!(out, "  Active (AI/occupied): {}  Moving: {}  Parked: {}", active_vehicles, moving_vehicles, parked_vehicles);

    let path = args.get(3).cloned().unwrap_or_else(|| format!("debug/observe_s{}.txt", seed));
    std::fs::write(&path, &out).unwrap();
    eprintln!("Wrote {} bytes to {}", out.len(), path);
}

fn dump_snapshot(game: &state::GameState, out: &mut String, tick: u64,
    npc_stuck: &[u32], npc_river: &[u32], _npc_dist: &[f32],
    veh_stuck: &[u32], _state_accum: &[u32; 8])
{
    let w = &game.world;
    let hour = game.time_of_day as u32;
    let minute = ((game.time_of_day - hour as f32) * 60.0) as u32;

    let mut states = [0u32; 8];
    for npc in &w.npcs {
        states[npc.state.index()] += 1;
    }

    let active_items = w.items.iter().filter(|it| it.active).count();
    // Only count mobile jobs (Collector, Garbage, Delivery, Mail, Paramedic, Police, Taxi) as stuck
    let stuck_npcs = (0..w.npcs.len()).filter(|&idx| {
        npc_stuck[idx] > 30 && w.npcs[idx].job.is_mobile()
    }).count();
    let river_npcs = npc_river.iter().filter(|&&r| r > 0).count();
    let stuck_vehs = veh_stuck.iter().filter(|&&s| s > 10).count();
    let moving_vehs = w.vehicles.iter().filter(|v| v.speed.abs() > 0.5).count();
    let avg_hunger = w.npcs.iter().map(|n| n.hunger).sum::<f32>() / w.npcs.len().max(1) as f32;
    let avg_thirst = w.npcs.iter().map(|n| n.thirst).sum::<f32>() / w.npcs.len().max(1) as f32;
    let dead = w.npcs.iter().filter(|n| n.starving_dead).count();

    let _ = writeln!(out, "[day{} {:02}:{:02} tick={}] S={} H={} G2W={} W={} GH={} D={} I={} KO={} | items={} | stuck_npc={} river={} | vehs: moving={} stuck={} | hunger={:.0} thirst={:.0} dead={}",
        game.day_count, hour, minute, tick,
        states[0], states[1], states[2], states[3], states[4], states[5], states[6], states[7],
        active_items, stuck_npcs, river_npcs, moving_vehs, stuck_vehs,
        avg_hunger, avg_thirst, dead);
}


