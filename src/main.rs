use clauding::{state, world, neat, npc, vehicle, player, camera, hud, particle, raster, render, menu, input, combat, collision, player_jobs, telemetry, gpu, platform, math};

use std::time::Instant;

fn bytemuck_cast(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}
fn bytemuck_cast_mut(data: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) }
}

const FIXED_DT: f32 = 1.0 / 60.0;
const MAX_ACCUMULATOR: f32 = 0.25; // cap to prevent death spiral

/// Frame time ring buffer for percentile analysis
struct FrameStats {
    times: Vec<f32>,      // frame times in seconds
    head: usize,
    count: usize,
    report_timer: f32,
}
impl FrameStats {
    fn new(capacity: usize) -> Self {
        FrameStats { times: vec![0.0; capacity], head: 0, count: 0, report_timer: 0.0 }
    }
    fn push(&mut self, dt: f32) {
        self.times[self.head] = dt;
        self.head = (self.head + 1) % self.times.len();
        if self.count < self.times.len() { self.count += 1; }
    }
    fn percentile_fps(&self, pct: f32) -> f32 {
        if self.count < 10 { return 0.0; }
        let mut sorted: Vec<f32> = self.times[..self.count].to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)); // descending (worst first)
        let idx = ((self.count as f32 * pct / 100.0) as usize).min(self.count - 1);
        let worst_dt = sorted[idx];
        if worst_dt > 0.0 { 1.0 / worst_dt } else { 0.0 }
    }
    fn avg_fps(&self) -> f32 {
        if self.count < 2 { return 0.0; }
        let sum: f32 = self.times[..self.count].iter().sum();
        self.count as f32 / sum
    }
}

fn main() {
    // Init GPU compute
    let mut gpu = match gpu::GpuContext::try_new() {
        Some(mut ctx) => {
            eprintln!("Vulkan GPU: {}", ctx.device_name);
            let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let buf = ctx.create_buffer(data.len() * 4);
            ctx.upload(&buf, bytemuck_cast(&data));
            let count = data.len() as u32;
            ctx.dispatch("test_multiply", &[&buf], &count.to_ne_bytes(), count);
            let mut result = vec![0.0f32; data.len()];
            ctx.download(&buf, bytemuck_cast_mut(&mut result));
            eprintln!("GPU test OK: {:?}", result);
            ctx.free_buffer(buf);
            Some(ctx)
        }
        None => { eprintln!("No Vulkan GPU, CPU fallback"); None }
    };

    let mut window = platform::PlatformWindow::new();
    let w = window.width();
    let h = window.height();
    eprintln!("Window: {}x{}", w, h);

    let world_seed: u64 = 42;
    let mut game = state::GameState::new(w, h, world_seed);
    world::generate_world(&mut game);

    // Try to load saved NEAT population, otherwise use fresh one
    if let Some(loaded) = neat::load_population("/tmp/clauding_neat.bin", state::NUM_NPCS) {
        eprintln!("Loaded NEAT population: gen {}, {} genomes", loaded.generation, loaded.genomes.len());
        game.neat_population = loaded;
    } else {
        // Also try the trained binary output
        if let Some(loaded) = neat::load_population("neat_trained.bin", state::NUM_NPCS) {
            eprintln!("Loaded trained NEAT population: gen {}, {} genomes", loaded.generation, loaded.genomes.len());
            game.neat_population = loaded;
        }
    }

    // Compile NEAT brains from population
    game.neat_brains = game.neat_population.genomes.iter()
        .map(|g| neat::NeatBrain::compile(g))
        .collect();

    // Init GPU graphics pipeline
    if let Some(ref mut ctx) = gpu {
        ctx.init_graphics(w as u32, h as u32);
        if ctx.has_graphics() {
            eprintln!("GPU graphics pipeline ready ({}x{})", w, h);
        }
    }

    let mut fb = raster::Framebuffer::new(w, h);
    let mut particles = particle::ParticleSystem::new(&mut gpu, world_seed.wrapping_add(0xBEEF));

    let mut render_scratch: Vec<state::WorldTri> = Vec::with_capacity(16384);
    let mut gpu_static_verts: Vec<gpu::GpuVertex> = Vec::with_capacity(1024 * 1024);
    let mut gpu_dynamic_verts: Vec<gpu::GpuVertex> = Vec::with_capacity(256 * 1024);

    // Upload static GPU vertices once (lighting computed in shader, never needs regen)
    if gpu.as_ref().is_some_and(|g| g.has_graphics()) {
        render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
        gpu.as_mut().unwrap().upload_static_vertices(&gpu_static_verts);
        eprintln!("Static GPU verts: {} ({:.1}MB) — uploaded once", gpu_static_verts.len(),
            gpu_static_verts.len() as f64 * 28.0 / 1_000_000.0);
    }
    let mut last_frame = Instant::now();
    let mut accumulator: f32 = 0.0;
    let mut prev_time_of_day: f32 = game.time_of_day;
    let _ = prev_time_of_day; // suppress initial assignment warning
    let mut frame_stats = FrameStats::new(1000); // last 1000 frames

    loop {
        // Save previous keys before polling
        game.prev_keys = game.keys;

        window.poll_events(&mut game.keys, &mut game.should_quit, &mut game.mouse_dx, &mut game.mouse_dy);
        if game.should_quit { break; }

        // Menu input (always processed, even when game is paused)
        let quit = menu::sys_menu_input(
            &mut game.menu,
            &mut game.keybinds,
            &mut game.mouse_sensitivity,
            &mut game.invert_mouse_x,
            &mut game.invert_mouse_y,
            &game.keys,
            &game.prev_keys,
        );
        if quit { break; }

        let now = Instant::now();
        let frame_dt = now.duration_since(last_frame).as_secs_f32();
        if frame_dt < state::FRAME_TIME_MIN { continue; }
        last_frame = now;

        // Track frame times for percentile analysis
        frame_stats.push(frame_dt);
        frame_stats.report_timer += frame_dt;
        if frame_stats.report_timer >= 5.0 {
            frame_stats.report_timer = 0.0;
            eprintln!("FPS avg={:.0} 1%low={:.0} 0.1%low={:.0} | sverts={} dverts={} | {}x{}",
                frame_stats.avg_fps(), frame_stats.percentile_fps(1.0),
                frame_stats.percentile_fps(0.1),
                gpu_static_verts.len(), gpu_dynamic_verts.len(),
                fb.w, fb.h);
        }

        let nw = window.width();
        let nh = window.height();
        if nw != fb.w || nh != fb.h {
            fb.resize(nw, nh);
            game.width = nw;
            game.height = nh;
        }

        // Only run game logic when menu is closed
        if game.menu.state != menu::MenuState::None {
            // Discard mouse delta while menu is open
            game.mouse_dx = 0.0;
            game.mouse_dy = 0.0;
        }
        if game.menu.state == menu::MenuState::None {
            // Time speed toggle: T key (scancode 20) edge-detect
            if game.keys[20] && !game.prev_keys[20] {
                game.time_speed = match game.time_speed {
                    1 => 10,
                    10 => 100,
                    100 => 1000,
                    _ => 1,
                };
            }

            // Fixed timestep accumulator
            accumulator += frame_dt.min(MAX_ACCUMULATOR);

            // At high speed, run multiple ticks per frame
            let ticks_per_frame = if game.time_speed > 1 {
                (game.time_speed as usize).min(500)
            } else {
                0 // use normal accumulator
            };

            let tick_count = if ticks_per_frame > 0 {
                accumulator = 0.0;
                ticks_per_frame
            } else {
                let mut count = 0;
                while accumulator >= FIXED_DT {
                    accumulator -= FIXED_DT;
                    count += 1;
                }
                count
            };

            for _ in 0..tick_count {

                prev_time_of_day = game.time_of_day;

                // Advance time of day
                game.time_of_day += FIXED_DT * 24.0 / state::DAY_LENGTH;
                if game.time_of_day >= 24.0 { game.time_of_day -= 24.0; }

                // Midnight reset (day counter, bin reset, NPC daily counters, NEAT evolution)
                if npc::sys_midnight_reset(
                    &mut game.world, game.time_of_day, prev_time_of_day,
                    &mut game.neat_population, &mut game.neat_brains,
                ) {
                    game.day_count += 1;
                }

                // Game-logic systems at fixed dt
                player::sys_player(&mut game, FIXED_DT);
                vehicle::sys_vehicle(&mut game, FIXED_DT);
                npc::sys_npc(
                    &mut game.world, &mut game.road_network, &game.terrain,
                    FIXED_DT, game.time_of_day, &mut game.neat_brains,
                    game.player.x, game.player.z,
                );
                npc::sys_night_spawning(
                    &mut game.world, &game.terrain, game.time_of_day,
                    FIXED_DT, &mut game.spawn_rng, &game.road_network,
                );
                npc::sys_items_update(&mut game.world, FIXED_DT);
                npc::sys_npc_interactions(&mut game.world, FIXED_DT);
                collision::sys_collisions(
                    &mut game.world, &mut game.player, &game.terrain, FIXED_DT,
                );
                combat::sys_ragdoll_update(&mut game.world, &game.terrain, FIXED_DT);
                combat::sys_combat(
                    &mut game.world, &mut game.player, &mut particles,
                    &game.terrain, &game.keys, &game.prev_keys,
                    &game.keybinds, FIXED_DT,
                );
                npc::sys_hunger_thirst(&mut game.world, &mut game.player, FIXED_DT);
                player_jobs::sys_interactibles_update(&mut game.world, FIXED_DT);
                player_jobs::sys_player_job(&mut game, FIXED_DT);

                // Player physical interaction (Interact key, edge-detected)
                let interact_now = game.keybinds.is_pressed(input::Action::Interact, &game.keys);
                let interact_prev = game.keybinds.is_pressed(input::Action::Interact, &game.prev_keys);
                let interact_edge = interact_now && !interact_prev;
                // Only fire if not in vehicle (vehicle.rs handles that)
                let do_interact = interact_edge && game.player.in_vehicle.is_none();
                if let Some((sx, sz, sc)) = npc::sys_player_interact(
                    &mut game.world, &mut game.player, &game.terrain, do_interact,
                ) {
                    particle::emit_pickup_sparkle(&mut particles, sx, sz, sc);
                }

                game.frame_counter += 1;
                telemetry::sys_telemetry(&game);
            }

            // Visual systems at variable rate
            camera::sys_camera(
                &mut game.camera, &game.player, &game.terrain,
                game.mouse_dx, game.mouse_dy,
                game.mouse_sensitivity, game.invert_mouse_x, game.invert_mouse_y,
                frame_dt, game.frame_counter,
            );
            game.mouse_dx = 0.0;
            game.mouse_dy = 0.0;
            particle::sys_emit_particles(&mut particles, &game, frame_dt);
            particles.update(&mut gpu, frame_dt);
        }

        // Always render (frozen scene when paused)
        let use_gpu = gpu.as_ref().is_some_and(|g| g.has_graphics());

        if use_gpu {
            // GPU offscreen render path
            let ctx = gpu.as_mut().unwrap();
            ctx.resize_render_target(fb.w as u32, fb.h as u32);

            // Generate dynamic vertices (no CPU lighting — shader handles it)
            render::generate_dynamic_gpu_vertices(
                &game.world, &game.player, &game.camera,
                &mut render_scratch, &mut gpu_dynamic_verts,
                game.time_of_day,
            );

            // Build VP matrix (Vulkan depth [0,1] + Y-flip) + lighting push constants
            let aspect = fb.w as f32 / fb.h as f32;
            let eye = [game.camera.x, game.camera.y, game.camera.z];
            let target = [game.camera.tx, game.camera.ty, game.camera.tz];
            let view = math::m4_look_at(eye, target, [0.0, 1.0, 0.0]);
            let proj = math::m4_perspective_vk(60.0_f32.to_radians(), aspect, 0.1, 500.0);
            let vp = math::m4_mul(&proj, &view);
            let push = render::gpu_push_constants(game.time_of_day, eye, target, &vp);

            let clear = render::sky_color_f32(game.time_of_day);
            ctx.render_frame(&gpu_dynamic_verts, &push, clear, fb.w as u32, fb.h as u32, &mut fb.pixels);

            // CPU overlays on top of GPU output (no zbuf clear needed — 2D overlays)
            particle::sys_render_particles(&mut fb, &particles, &game.camera);
            hud::sys_hud(&mut fb, &game);
            menu::sys_menu_render(
                &mut fb, &game.menu, &game.keybinds,
                game.mouse_sensitivity, game.invert_mouse_x, game.invert_mouse_y,
            );

            window.present(&fb.pixels);
        } else {
            // CPU software rasterizer (fallback)
            fb.clear(render::sky_color(game.time_of_day));
            render::sys_render(&mut fb, &game.world, &game.player, &game.camera, game.time_of_day, &mut render_scratch);

            particle::sys_render_particles(&mut fb, &particles, &game.camera);
            hud::sys_hud(&mut fb, &game);
            menu::sys_menu_render(
                &mut fb, &game.menu, &game.keybinds,
                game.mouse_sensitivity, game.invert_mouse_x, game.invert_mouse_y,
            );

            window.present(&fb.pixels);
        }
    }
}
