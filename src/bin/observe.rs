// Simulation observer: runs headless game and dumps behavioral analysis
// Usage: cargo run --bin observe -- [seed] [days]
// Outputs: /tmp/clauding_observe.txt with detailed game dynamics

use clauding::{state, world, npc, neat, vehicle, collision, combat};
use std::fmt::Write;

const FIXED_DT: f32 = 1.0 / 30.0; // moderate timestep for accuracy

fn main() {
    let _ = std::fs::create_dir_all("debug");
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let max_days: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);

    eprintln!("Observer: seed={} days={}", seed, max_days);

    let mut game = state::GameState::new(1, 1, seed);
    world::generate_world(&mut game);

    // Load trained brains if available
    if let Some(loaded) = neat::load_population("neat_trained.bin", state::NUM_NPCS) {
        eprintln!("Loaded trained population gen {}", loaded.generation);
        game.neat_population = loaded;
    }
    game.neat_brains = game.neat_population.genomes.iter()
        .map(|g| neat::NeatBrain::compile(g))
        .collect();

    let mut out = String::with_capacity(128000);
    let _ = writeln!(out, "=== SIMULATION OBSERVER (seed={}, days={}) ===\n", seed, max_days);

    // Track NPC positions over time for stuck detection
    let n_npcs = game.world.npcs.len();
    let n_vehicles = game.world.vehicles.len();
    let mut prev_npc_pos: Vec<(f32, f32)> = game.world.npcs.iter().map(|n| (n.x, n.z)).collect();
    let mut npc_stuck_ticks: Vec<u32> = vec![0; n_npcs];
    let mut npc_river_ticks: Vec<u32> = vec![0; n_npcs];
    let mut npc_total_dist: Vec<f32> = vec![0.0; n_npcs];

    let mut prev_veh_pos: Vec<(f32, f32)> = game.world.vehicles.iter().map(|v| (v.x, v.z)).collect();
    let mut veh_stuck_ticks: Vec<u32> = vec![0; n_vehicles];

    // Aggregate counters
    let mut _snapshot_count: u32 = 0;
    let mut state_time_accum = [0u32; 8]; // ticks in each NPC state

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

        // Midnight reset
        if npc::sys_midnight_reset(
            &mut game.world, game.time_of_day, prev_time_of_day,
            &mut game.neat_population, &mut game.neat_brains,
        ) {
            game.day_count += 1;
            // Reset daily counters for new day tracking
            let _ = writeln!(out, "--- DAY {} RESET (gen {}) ---", game.day_count, game.neat_population.generation);
        }

        // Game systems
        vehicle::sys_vehicle(&mut game, FIXED_DT);
        npc::sys_npc(
            &mut game.world, &mut game.road_network, &game.terrain,
            FIXED_DT, game.time_of_day, &mut game.neat_brains,
            0.0, 0.0, // player at origin
        );
        npc::sys_night_spawning(
            &mut game.world, &game.terrain, game.time_of_day,
            FIXED_DT, &mut game.spawn_rng,
        );
        npc::sys_items_update(&mut game.world, FIXED_DT);
        npc::sys_npc_interactions(&mut game.world, FIXED_DT);
        npc::sys_hunger_thirst(&mut game.world, &mut game.player, FIXED_DT);
        collision::sys_collisions_headless(&mut game.world, &game.terrain, FIXED_DT);
        combat::sys_ragdoll_update(&mut game.world, &game.terrain, FIXED_DT);

        // Headless NPC-NPC combat
        headless_combat(&mut game.world, &game.terrain, FIXED_DT);

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

                if dist < 0.05 && npc.state == state::NpcState::Working {
                    npc_stuck_ticks[i] += 1;
                } else {
                    npc_stuck_ticks[i] = 0;
                }

                if world::on_river_not_bridge(npc.x, npc.z, &game.world.river_segments, &game.world.bridges) {
                    npc_river_ticks[i] += 1;
                }

                prev_npc_pos[i] = (npc.x, npc.z);

                // State accumulator
                let idx = match npc.state {
                    state::NpcState::Sleeping => 0,
                    state::NpcState::HomeTask => 1,
                    state::NpcState::GoingToWork => 2,
                    state::NpcState::Working => 3,
                    state::NpcState::GoingHome => 4,
                    state::NpcState::Driving => 5,
                    state::NpcState::Interacting => 6,
                    state::NpcState::KnockedOut => 7,
                };
                state_time_accum[idx] += 1;
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

    // Stuck NPC analysis
    let _ = writeln!(out, "--- STUCK NPC ANALYSIS (stuck = no movement while Working, 10-tick windows) ---");
    let mut stuck_npcs = 0;
    for i in 0..n_npcs {
        if npc_stuck_ticks[i] > 50 {
            let npc = &game.world.npcs[i];
            let _ = writeln!(out, "  NPC[{:2}] stuck_score={:4} pos=({:6.1},{:6.1}) total_dist={:6.1}m job={:12}",
                i, npc_stuck_ticks[i], npc.x, npc.z, npc_total_dist[i], npc_job_name(npc.job));
            stuck_npcs += 1;
        }
    }
    if stuck_npcs == 0 { let _ = writeln!(out, "  No stuck NPCs detected!"); }

    // River exposure
    let _ = writeln!(out, "\n--- RIVER EXPOSURE (NPCs on river) ---");
    let mut river_npcs = 0;
    for i in 0..n_npcs {
        if npc_river_ticks[i] > 0 {
            let npc = &game.world.npcs[i];
            let _ = writeln!(out, "  NPC[{:2}] river_ticks={:4} pos=({:6.1},{:6.1}) job={:12}",
                i, npc_river_ticks[i], npc.x, npc.z, npc_job_name(npc.job));
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

    // NPC movement stats
    let _ = writeln!(out, "\n--- NPC MOVEMENT STATS ---");
    let mut dists: Vec<(usize, f32)> = (0..n_npcs).map(|i| (i, npc_total_dist[i])).collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let _ = writeln!(out, "  Least mobile NPCs:");
    for &(i, d) in dists.iter().take(5) {
        let npc = &game.world.npcs[i];
        let _ = writeln!(out, "    NPC[{:2}] dist={:6.1}m pos=({:6.1},{:6.1}) job={:12} state={:12}",
            i, d, npc.x, npc.z, npc_job_name(npc.job), npc_state_name(npc.state));
    }
    let _ = writeln!(out, "  Most mobile NPCs:");
    for &(i, d) in dists.iter().rev().take(5) {
        let npc = &game.world.npcs[i];
        let _ = writeln!(out, "    NPC[{:2}] dist={:6.1}m pos=({:6.1},{:6.1}) job={:12} state={:12}",
            i, d, npc.x, npc.z, npc_job_name(npc.job), npc_state_name(npc.state));
    }

    // State time distribution
    let total_state_ticks: u32 = state_time_accum.iter().sum();
    let _ = writeln!(out, "\n--- STATE TIME DISTRIBUTION (% of NPC-ticks) ---");
    let names = ["Sleeping", "HomeTask", "GoingToWork", "Working", "GoingHome", "Driving", "Interacting", "KnockedOut"];
    for (i, name) in names.iter().enumerate() {
        let pct = state_time_accum[i] as f32 / total_state_ticks.max(1) as f32 * 100.0;
        let _ = writeln!(out, "  {:12} {:5.1}% ({} ticks)", name, pct, state_time_accum[i]);
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

    // Item economy
    let items_active = game.world.items.iter().filter(|it| it.active).count();
    let total_picked: u32 = game.world.npcs.iter().map(|n| n.fitness_items_picked).sum();
    let total_dep: u32 = game.world.npcs.iter().map(|n| n.items_deposited_today).sum();
    let _ = writeln!(out, "\n--- ITEM ECONOMY ---");
    let _ = writeln!(out, "  Active items: {}/{}", items_active, state::NUM_ITEMS);
    let _ = writeln!(out, "  Total picked (last day): {}", total_picked);
    let _ = writeln!(out, "  Total deposited (last day): {}", total_dep);
    let total_npc_money: f32 = game.world.npcs.iter().map(|n| n.money).sum();
    let avg_npc_money = total_npc_money / n_npcs.max(1) as f32;
    let _ = writeln!(out, "  Total NPC money: ${:.0} (avg ${:.0})", total_npc_money, avg_npc_money);

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

    // Vehicle activity
    let active_vehicles = game.world.vehicles.iter().filter(|v| v.ai_active || v.occupied).count();
    let moving_vehicles = game.world.vehicles.iter().filter(|v| v.speed.abs() > 0.5).count();
    let parked_vehicles = game.world.vehicles.iter().filter(|v| v.parked).count();
    let _ = writeln!(out, "\n--- VEHICLE ACTIVITY ---");
    let _ = writeln!(out, "  Active (AI/occupied): {}  Moving: {}  Parked: {}", active_vehicles, moving_vehicles, parked_vehicles);

    let path = "debug/observe.txt";
    std::fs::write(path, &out).unwrap();
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
        let idx = match npc.state {
            state::NpcState::Sleeping => 0,
            state::NpcState::HomeTask => 1,
            state::NpcState::GoingToWork => 2,
            state::NpcState::Working => 3,
            state::NpcState::GoingHome => 4,
            state::NpcState::Driving => 5,
            state::NpcState::Interacting => 6,
            state::NpcState::KnockedOut => 7,
        };
        states[idx] += 1;
    }

    let active_items = w.items.iter().filter(|it| it.active).count();
    let stuck_npcs = npc_stuck.iter().filter(|&&s| s > 30).count();
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

fn npc_job_name(j: state::NpcJob) -> &'static str {
    match j {
        state::NpcJob::Collector => "Collector",
        state::NpcJob::GarbageCollector => "Garbage",
        state::NpcJob::TaxiDriver => "Taxi",
        state::NpcJob::DeliveryCourier => "Delivery",
        state::NpcJob::MailCarrier => "Mail",
        state::NpcJob::Paramedic => "Paramedic",
        state::NpcJob::Firefighter => "Firefighter",
        state::NpcJob::PolicePatrol => "Police",
        state::NpcJob::StreetVendor => "Vendor",
        state::NpcJob::Mechanic => "Mechanic",
        state::NpcJob::ConstructionWorker => "Construction",
        state::NpcJob::Fisherman => "Fisherman",
        state::NpcJob::Farmer => "Farmer",
        state::NpcJob::Lumberjack => "Lumberjack",
        state::NpcJob::Scavenger => "Scavenger",
    }
}

fn npc_state_name(s: state::NpcState) -> &'static str {
    match s {
        state::NpcState::Sleeping => "Sleeping",
        state::NpcState::HomeTask => "HomeTask",
        state::NpcState::GoingToWork => "GoToWork",
        state::NpcState::Working => "Working",
        state::NpcState::GoingHome => "GoHome",
        state::NpcState::Driving => "Driving",
        state::NpcState::Interacting => "Interacting",
        state::NpcState::KnockedOut => "KnockedOut",
    }
}

/// Headless NPC-NPC combat (same as train.rs)
fn headless_combat(world: &mut state::WorldData, terrain: &state::Terrain, dt: f32) {
    let n = world.npcs.len();
    for npc in &mut world.npcs {
        npc.attack_cooldown = (npc.attack_cooldown - dt).max(0.0);
        npc.attack_phase = (npc.attack_phase - dt).max(0.0);
        npc.hit_flash = (npc.hit_flash - dt).max(0.0);
    }

    for i in 0..n {
        let intent = world.npcs[i].attack_intent;
        world.npcs[i].attack_intent = 0;
        if intent == 0 || world.npcs[i].attack_cooldown > 0.0 { continue; }
        if world.npcs[i].state == state::NpcState::KnockedOut { continue; }
        if world.npcs[i].state == state::NpcState::Sleeping { continue; }

        if intent == 2 {
            let ax = world.npcs[i].x;
            let az = world.npcs[i].z;
            let arot = world.npcs[i].rot_y;
            let mut best_dist = state::ATTACK_RANGE * state::ATTACK_RANGE;
            let mut best_j = None;
            for j in 0..n {
                if j == i { continue; }
                if world.npcs[j].state == state::NpcState::KnockedOut { continue; }
                let dx = world.npcs[j].x - ax;
                let dz = world.npcs[j].z - az;
                let d2 = dx * dx + dz * dz;
                if d2 < best_dist { best_dist = d2; best_j = Some(j); }
            }
            if let Some(j) = best_j {
                let dx = world.npcs[j].x - ax;
                let dz = world.npcs[j].z - az;
                let dist = (dx * dx + dz * dz).sqrt();
                if dist < 0.01 { continue; }
                let (sin_r, cos_r) = arot.sin_cos();
                let fwd_x = -sin_r;
                let fwd_z = -cos_r;
                let dot = (dx / dist) * fwd_x + (dz / dist) * fwd_z;
                if dot < state::ATTACK_CONE_COS { continue; }

                world.npcs[i].attack_cooldown = state::ATTACK_COOLDOWN;
                world.npcs[i].attack_phase = state::ATTACK_ANIM_DURATION;
                world.npcs[i].fitness_hits_landed += 1;

                world.npcs[j].health -= state::NPC_ATTACK_DAMAGE;
                world.npcs[j].hit_flash = state::HIT_FLASH_DURATION;
                world.npcs[j].knockback_vx += dx / dist * state::KNOCKBACK_FORCE * 0.7;
                world.npcs[j].knockback_vz += dz / dist * state::KNOCKBACK_FORCE * 0.7;
                world.npcs[j].vel_y = state::KNOCKBACK_UP * 0.7;

                if world.npcs[j].health <= 0.0 {
                    world.npcs[j].health = 0.0;
                    world.npcs[j].state = state::NpcState::KnockedOut;
                    world.npcs[j].knockout_timer = state::KNOCKOUT_TIME;
                    world.npcs[j].carrying_item = false;
                    world.npcs[j].carrying_bin = None;
                    world.npcs[j].fitness_knockouts += 1;
                    world.npcs[j].sound = [0.0; 3];
                }
            }
        }
    }

    for i in 0..n {
        let npc = &mut world.npcs[i];
        if npc.knockback_vx.abs() > 0.01 || npc.knockback_vz.abs() > 0.01 {
            npc.x += npc.knockback_vx * dt;
            npc.z += npc.knockback_vz * dt;
            npc.x = npc.x.clamp(-state::WORLD_HALF, state::WORLD_HALF);
            npc.z = npc.z.clamp(-state::WORLD_HALF, state::WORLD_HALF);
            let friction = (-state::KNOCKBACK_FRICTION * dt).exp();
            npc.knockback_vx *= friction;
            npc.knockback_vz *= friction;
        }
        if npc.state == state::NpcState::KnockedOut && !npc.starving_dead {
            npc.knockout_timer -= dt;
            if npc.knockout_timer <= 0.0 {
                npc.health = state::KNOCKOUT_REGEN_HP;
                npc.state = state::NpcState::Working;
                npc.knockout_timer = 0.0;
                npc.state_timer = 0.0;
            }
        }
        if npc.state != state::NpcState::KnockedOut && npc.health < state::NPC_HEALTH_MAX {
            npc.health = (npc.health + state::HEALTH_REGEN_RATE * dt).min(state::NPC_HEALTH_MAX);
        }
        let ground = terrain.height_at(npc.x, npc.z);
        if npc.y < ground { npc.y = ground; }
    }
}
