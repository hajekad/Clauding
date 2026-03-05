// Headless NEAT training binary
// Usage: cargo run --bin train -- [seed] [days]
// No Wayland, no rendering, no input. Evolves NPC brains at max tick speed.

use clauding::{state, world, npc, neat, vehicle};
use std::sync::atomic::{AtomicBool, Ordering};

static RUNNING: AtomicBool = AtomicBool::new(true);
static START_TIME: std::sync::LazyLock<std::time::Instant> = std::sync::LazyLock::new(std::time::Instant::now);

const FIXED_DT: f32 = 1.0 / 20.0;  // coarser than game (1/60) — 3x faster training

fn main() {
    // Ctrl+C handler via raw signal
    unsafe {
        libc_signal(2 /* SIGINT */, sigint_handler);
    }

    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let max_days: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);

    eprintln!("NEAT Trainer: seed={} days={}", seed, max_days);
    eprintln!("  {} NPCs, {} items, {}m world", state::NUM_NPCS, state::NUM_ITEMS, state::WORLD_SIZE as u32);
    eprintln!("  Press Ctrl+C to save and exit early\n");

    let mut game = state::GameState::new(1, 1, seed);
    world::generate_world(&mut game);

    // Try to resume from existing trained population
    if let Some(loaded) = neat::load_population("neat_trained.bin", state::NUM_NPCS) {
        eprintln!("Resuming from gen {}, {} species", loaded.generation, loaded.species.len());
        game.neat_population = loaded;
    }

    // Compile brains
    game.neat_brains = game.neat_population.genomes.iter()
        .map(|g| neat::NeatBrain::compile(g))
        .collect();

    let mut prev_time_of_day: f32 = game.time_of_day;
    let _ = prev_time_of_day;
    let _ = *START_TIME; // initialize LazyLock
    let start_gen = game.neat_population.generation;

    while game.day_count <= max_days && RUNNING.load(Ordering::Relaxed) {
        prev_time_of_day = game.time_of_day;

        // Advance time
        game.time_of_day += FIXED_DT * 24.0 / state::DAY_LENGTH;
        if game.time_of_day >= 24.0 { game.time_of_day -= 24.0; }

        // Capture pre-reset stats just before midnight
        if prev_time_of_day > 23.5 && game.time_of_day < 0.5 {
            capture_pre_reset_stats(&game);
        }

        // Midnight reset (evolve + auto-save via npc.rs)
        if npc::sys_midnight_reset(
            &mut game.world, game.time_of_day, prev_time_of_day,
            &mut game.neat_population, &mut game.neat_brains,
        ) {
            game.day_count += 1;

            // Save every 10 generations
            if game.neat_population.generation % 10 == 0 {
                neat::save_population("neat_trained.bin", &game.neat_population);
            }
        }

        // Vehicle AI (needed for NPC driving)
        vehicle::sys_vehicle(&mut game, FIXED_DT);

        // NPC systems
        npc::sys_npc(
            &mut game.world, &game.road_network, &game.terrain,
            FIXED_DT, game.time_of_day, &mut game.neat_brains,
            game.player.x, game.player.z,
        );
        npc::sys_night_spawning(
            &mut game.world, &game.terrain, game.time_of_day,
            FIXED_DT, &mut game.spawn_rng,
        );
        npc::sys_items_update(&mut game.world, FIXED_DT);
        npc::sys_npc_interactions(&mut game.world, FIXED_DT);
        npc::sys_hunger_thirst(&mut game.world, &mut game.player, FIXED_DT);

        // Headless NPC-NPC combat (no particles/rendering needed)
        headless_combat(&mut game.world, &game.terrain, FIXED_DT);

        game.frame_counter += 1;

        // Diagnostic: print NPC state snapshot every 10000 ticks (first 2 days only)
        if game.frame_counter == 10000 || game.frame_counter == 50000 {
            print_diagnostic(&game);
        }
    }

    // Final save
    neat::save_population("neat_trained.bin", &game.neat_population);
    let elapsed = START_TIME.elapsed().as_secs_f32();
    let gens = game.neat_population.generation - start_gen;
    eprintln!("\nTraining complete: {} generations in {:.1}s ({:.1} gen/s)",
        gens, elapsed, gens as f32 / elapsed.max(0.001));
    eprintln!("Saved to neat_trained.bin");
}

fn capture_pre_reset_stats(game: &state::GameState) {
    let w = &game.world;
    let fitnesses: Vec<f32> = w.npcs.iter().map(|n| neat::evaluate_fitness(n)).collect();
    let best = fitnesses.iter().cloned().fold(f32::MIN, f32::max);
    let avg = fitnesses.iter().sum::<f32>() / fitnesses.len().max(1) as f32;
    let worst = fitnesses.iter().cloned().fold(f32::MAX, f32::min);

    let alive = w.npcs.iter().filter(|n| !n.starving_dead).count();
    let avg_hunger = w.npcs.iter().map(|n| n.hunger).sum::<f32>() / w.npcs.len().max(1) as f32;
    let avg_thirst = w.npcs.iter().map(|n| n.thirst).sum::<f32>() / w.npcs.len().max(1) as f32;
    let items_picked: u32 = w.npcs.iter().map(|n| n.fitness_items_picked).sum();
    let deposited: u32 = w.npcs.iter().map(|n| n.items_deposited_today).sum();
    let _food_eaten: u32 = w.npcs.iter().filter(|n| n.hunger > 50.0).count() as u32;
    let elapsed = START_TIME.elapsed().as_secs_f32();

    eprintln!("gen {:4} | fit best={:6.1} avg={:6.1} worst={:6.1} | alive={:3}/{} | hunger={:.0} thirst={:.0} | picked={:3} dep={} | species={} | {:.1}s",
        game.neat_population.generation + 1,
        best, avg, worst,
        alive, w.npcs.len(),
        avg_hunger, avg_thirst,
        items_picked, deposited,
        game.neat_population.species.len(),
        elapsed);
}

fn print_diagnostic(game: &state::GameState) {
    let w = &game.world;
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
    let falling_items = w.items.iter().filter(|it| it.falling).count();

    // Find closest NPC-to-item distance
    let mut min_dist = f32::MAX;
    let mut working_near_item = 0;
    for npc in &w.npcs {
        if npc.state != state::NpcState::Working { continue; }
        for item in &w.items {
            if !item.active { continue; }
            let dx = npc.x - item.x;
            let dz = npc.z - item.z;
            let d = (dx * dx + dz * dz).sqrt();
            if d < min_dist { min_dist = d; }
            if d < 5.0 { working_near_item += 1; }
        }
    }

    eprintln!("  DIAG tick={} time={:.1}h | S={} H={} G2W={} W={} GH={} D={} I={} KO={} | items: active={} falling={} | closest_npc_item={:.1}m working_near_5m={}",
        game.frame_counter, game.time_of_day, states[0], states[1], states[2], states[3], states[4], states[5], states[6], states[7],
        active_items, falling_items, min_dist, working_near_item);

    // Find a working NPC closest to an item and trace its brain
    let mut best_i = 0;
    let mut best_d = f32::MAX;
    for (i, npc) in w.npcs.iter().enumerate() {
        if npc.state != state::NpcState::Working { continue; }
        for item in &w.items {
            if !item.active { continue; }
            let dx = npc.x - item.x;
            let dz = npc.z - item.z;
            let d = (dx * dx + dz * dz).sqrt();
            if d < best_d { best_d = d; best_i = i; }
        }
    }
    let npc = &w.npcs[best_i];
    eprintln!("    Closest working NPC[{}] pos=({:.0},{:.0}) brain={} carry={} bin={:?} dist_to_item={:.1}m",
        best_i, npc.x, npc.z, npc.brain_idx, npc.carrying_item, npc.carrying_bin, best_d);

    // Evaluate brain and print outputs
    if npc.state == state::NpcState::Working {
        let inputs = neat::gather_inputs(w, best_i, &game.road_network, game.time_of_day, game.player.x, game.player.z);
        let bi = npc.brain_idx;
        if bi < game.neat_brains.len() {
            let mut brain = neat::NeatBrain::compile(&game.neat_population.genomes[bi]);
            let outputs = brain.activate(&inputs);
            let pickup = outputs[3];
            let walk_mag = outputs[2];
            eprintln!("    Brain outputs: walk_dx={:.3} walk_dz={:.3} walk_mag={:.3} pickup={:.3} deposit={:.3}",
                outputs[0], outputs[1], walk_mag, pickup, outputs[4]);
            eprintln!("    do_pickup={} (>{:.1}) walk_active={} conns={}",
                pickup > 0.5, 0.5, walk_mag > 0.1, game.neat_population.genomes[bi].connections.iter().filter(|c| c.enabled).count());
            eprintln!("    Key inputs: carry={:.0} item_dx={:.3} item_dz={:.3} near_item={:.0} hunger={:.2} thirst={:.2} bias={:.0}",
                inputs[0], inputs[6], inputs[7], inputs[15], inputs[33], inputs[34], inputs[27]);
        }
    }
}

/// Stripped-down combat for headless training: NPC-NPC attacks only, no particles
fn headless_combat(world: &mut state::WorldData, terrain: &state::Terrain, dt: f32) {
    let n = world.npcs.len();

    // Tick cooldowns
    for npc in &mut world.npcs {
        npc.attack_cooldown = (npc.attack_cooldown - dt).max(0.0);
        npc.attack_phase = (npc.attack_phase - dt).max(0.0);
        npc.hit_flash = (npc.hit_flash - dt).max(0.0);
    }

    // Process NPC attack intents
    for i in 0..n {
        let intent = world.npcs[i].attack_intent;
        world.npcs[i].attack_intent = 0;
        if intent == 0 { continue; }
        if world.npcs[i].attack_cooldown > 0.0 { continue; }
        if world.npcs[i].state == state::NpcState::KnockedOut { continue; }
        if world.npcs[i].state == state::NpcState::Sleeping { continue; }

        if intent == 2 {
            // Attack nearest NPC
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
        // intent == 1 (attack player) is ignored in headless mode
    }

    // Knockback friction + knockout recovery + health regen
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

// --- Raw signal handling (no libc crate) ---

extern "C" fn sigint_handler(_sig: i32) {
    RUNNING.store(false, Ordering::Relaxed);
}

unsafe fn libc_signal(sig: i32, handler: extern "C" fn(i32)) {
    unsafe extern "C" {
        fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    }
    unsafe { signal(sig, handler); }
}
