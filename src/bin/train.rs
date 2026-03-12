// Headless NEAT training binary
// Usage: cargo run --bin train -- [seed] [days]
// No Wayland, no rendering, no input. Evolves NPC brains at max tick speed.

use clauding::{state, world, npc, neat, vehicle, collision, combat};
use std::sync::atomic::{AtomicBool, Ordering};

static RUNNING: AtomicBool = AtomicBool::new(true);
static START_TIME: std::sync::LazyLock<std::time::Instant> = std::sync::LazyLock::new(std::time::Instant::now);

// Intentionally coarser than state::HEADLESS_DT (1/30) — fewer ticks per day = faster training.
// Training doesn't need simulation accuracy, just enough fidelity for brains to learn behaviors.
const FIXED_DT: f32 = 1.0 / 20.0;

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

    let mut game = state::GameState::init(1, 1, seed);
    if game.neat_population.generation > 0 {
        eprintln!("Resuming from gen {}, {} species", game.neat_population.generation, game.neat_population.species.len());
    }

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

            // Multi-seed world rotation — regenerate every 25 generations so brains generalize
            if game.neat_population.generation % 25 == 0 && game.neat_population.generation > 0 {
                let new_seed = seed + (game.neat_population.generation as u64 / 25);
                eprintln!("  Rotating world: new seed={}", new_seed);
                let mut temp = state::GameState::new(1, 1, new_seed);
                world::generate_world(&mut temp);
                game.world = temp.world;
                game.terrain = temp.terrain;
                game.road_network = temp.road_network;
                game.spawn_rng = temp.spawn_rng;
                // Recompile brains for the new world (population/genomes carry over)
                game.neat_brains = game.neat_population.genomes.iter()
                    .map(|g| neat::NeatBrain::compile(g))
                    .collect();
            }
        }

        // Vehicle AI (needed for NPC driving)
        vehicle::sys_vehicle(&mut game, FIXED_DT);

        // NPC systems
        npc::sys_npc(
            &mut game.world, &mut game.road_network, &game.terrain,
            FIXED_DT, game.time_of_day, &mut game.neat_brains,
            game.player.x, game.player.z, &game.walk_grid,
        );
        npc::sys_night_spawning(
            &mut game.world, &game.terrain, game.time_of_day,
            FIXED_DT, &mut game.spawn_rng, &game.road_network,
        );
        npc::sys_items_update(&mut game.world, FIXED_DT);
        npc::sys_npc_interactions(&mut game.world, FIXED_DT);
        npc::sys_hunger_thirst(&mut game.world, &mut game.player, FIXED_DT);

        // Collision + ragdoll physics
        collision::sys_collisions_headless(&mut game.world, &game.terrain, FIXED_DT);
        combat::sys_ragdoll_update(&mut game.world, &game.terrain, FIXED_DT);

        // Headless NPC-NPC combat (no particles/rendering needed)
        combat::sys_combat_headless(&mut game.world, &game.terrain, FIXED_DT);


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
        states[npc.state.index()] += 1;
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
