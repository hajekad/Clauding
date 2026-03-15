use clauding::{state, npc, vehicle, vehicle_physics, player, camera, hud, particle, raster, render, menu, input, combat, collision, player_jobs, telemetry, gpu, platform, math};
use clauding::gpu::{bytemuck_cast, bytemuck_cast_mut};

use std::time::Instant;

const FIXED_DT: f32 = 1.0 / 60.0;
const MAX_ACCUMULATOR: f32 = 0.25; // cap to prevent death spiral
const SCENIC_CUT_INTERVAL: f32 = 20.0; // seconds between camera cuts

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

/// Generate scenic camera viewpoints from world features (road intersections, buildings).
fn generate_scenic_cameras(game: &state::GameState) -> Vec<([f32; 3], [f32; 3])> {
    let nodes = &game.road_network.nodes;
    let num = 8.min(nodes.len().max(1));
    let mut cameras = Vec::new();

    if nodes.is_empty() {
        cameras.push(([20.0, 15.0, 20.0], [0.0, 0.0, 0.0]));
        return cameras;
    }

    let step = nodes.len() / num;
    for i in 0..num {
        let node = &nodes[i * step];
        let tx = node[0];
        let tz = node[1];
        let ty = game.terrain.height_at(tx, tz);

        // Vary offset angle, distance, and height per camera for diverse views
        let angle = (i as f32) * std::f32::consts::TAU / num as f32;
        let dist = 18.0 + (i % 3) as f32 * 5.0;
        let height = 6.0 + (i % 4) as f32 * 3.0;
        let cx = tx + angle.cos() * dist;
        let cz = tz + angle.sin() * dist;
        let cy = game.terrain.height_at(cx, cz) + height;

        cameras.push(([cx, cy, cz], [tx, ty + 1.5, tz]));
    }

    cameras
}

/// Returns true if we're in a title-origin menu state (title itself, or settings/keybinds opened from title).
fn is_title_phase(menu: &menu::MenuData) -> bool {
    match menu.state {
        menu::MenuState::Title => true,
        menu::MenuState::Settings | menu::MenuState::Keybinds => menu.came_from_title,
        _ => false,
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

    // Init GPU graphics pipeline
    if let Some(ref mut ctx) = gpu {
        ctx.init_graphics(w as u32, h as u32);
        if ctx.has_graphics() {
            eprintln!("GPU graphics pipeline ready ({}x{})", w, h);
        }
    }

    let mut fb = raster::Framebuffer::new(w, h);

    // ── Loading screen (shown while world generates) ────────────────────
    let world_seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(42);

    {
        let mut loading_menu = menu::MenuData::new();
        loading_menu.state = menu::MenuState::Loading;
        loading_menu.loading_seed = world_seed;
        let tmp_keybinds = input::KeyBinds::default_binds();
        menu::sys_menu_render(&mut fb, &loading_menu, &tmp_keybinds, 1.0, false, false, 0, &[]);
        window.present(&fb.pixels);
    }

    // Block on world generation
    eprintln!("World seed: {}", world_seed);
    let mut game = state::GameState::init(fb.w, fb.h, world_seed);

    // Start on title screen with live world behind it
    game.menu.state = menu::MenuState::Title;

    // Hide player during title: position far outside world bounds
    // (player mesh still generated but culled by frustum; no vehicles nearby to interact with)
    game.player.x = -500.0;
    game.player.z = -500.0;
    game.player.y = 0.0;

    // ── Init game resources ─────────────────────────────────────────────
    let mut particles = particle::ParticleSystem::new(&mut gpu, world_seed.wrapping_add(0xBEEF));

    let mut render_scratch: Vec<state::WorldTri> = Vec::with_capacity(16384);
    let mut gpu_static_verts: Vec<gpu::GpuVertex> = Vec::with_capacity(1024 * 1024);
    let mut gpu_dynamic_verts: Vec<gpu::GpuVertex> = Vec::with_capacity(256 * 1024);

    if gpu.as_ref().is_some_and(|g| g.has_graphics()) {
        render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
        gpu.as_mut().unwrap().upload_static_vertices(&gpu_static_verts);
        eprintln!("Static GPU verts: {} ({:.1}MB) — uploaded once", gpu_static_verts.len(),
            gpu_static_verts.len() as f64 * 28.0 / 1_000_000.0);
    }

    // Scenic camera system for title screen
    let mut scenic_cameras = generate_scenic_cameras(&game);
    let mut scenic_idx: usize = 0;
    let mut scenic_timer: f32 = 0.0;

    // Set initial scenic camera
    if let Some(&(pos, target)) = scenic_cameras.first() {
        game.camera.x = pos[0]; game.camera.y = pos[1]; game.camera.z = pos[2];
        game.camera.tx = target[0]; game.camera.ty = target[1]; game.camera.tz = target[2];
    }

    let mut last_frame = Instant::now();
    let mut accumulator: f32 = 0.0;
    let mut prev_time_of_day: f32 = game.time_of_day;
    let _ = prev_time_of_day; // suppress initial assignment warning
    let mut frame_stats = FrameStats::new(1000);

    // ── Main loop (handles title, gameplay, menus) ──────────────────────
    loop {
        game.prev_keys = game.keys;

        window.poll_events(&mut game.keys, &mut game.should_quit, &mut game.mouse_dx, &mut game.mouse_dy);
        if game.should_quit { break; }

        // Menu input (always processed)
        let quit = menu::sys_menu_input(
            &mut game.menu,
            &mut game.keybinds,
            &mut game.mouse_sensitivity,
            &mut game.invert_mouse_x,
            &mut game.invert_mouse_y,
            &game.keys,
            &game.prev_keys,
            game.world_seed,
            &mut game.player.model_index,
            game.character_models.len(),
        );
        if quit { break; }

        // ── Title → gameplay transition ─────────────────────────────────
        if game.menu.start_new_game {
            game.menu.start_new_game = false;
            game.menu.state = menu::MenuState::None;

            // Spawn player at world center
            game.player.x = 0.0;
            game.player.z = 10.0;
            game.player.y = game.terrain.height_at(0.0, 10.0);
            game.player.vel_y = 0.0;
            game.player.on_ground = true;
            game.player.health = 100.0;
            game.player.hunger = 100.0;
            game.player.thirst = 100.0;
            game.player.stamina = 100.0;
            game.player.money = 0.0;
            game.player.bounty = 0.0;
            game.player.rot_y = 0.0;

            accumulator = 0.0;
            last_frame = Instant::now();
            scenic_timer = 0.0;
            continue;
        }

        // ── World regeneration (pause menu → New World) ─────────────────
        if let Some(new_seed) = game.menu.regenerate_seed.take() {
            // Show loading screen before blocking init
            game.menu.state = menu::MenuState::Loading;
            game.menu.loading_seed = new_seed;
            menu::sys_menu_render(
                &mut fb, &game.menu, &game.keybinds,
                game.mouse_sensitivity, game.invert_mouse_x, game.invert_mouse_y,
                game.player.model_index, &game.model_library.character_names,
            );
            window.present(&fb.pixels);

            let saved_keybinds = game.keybinds.clone();
            let saved_sensitivity = game.mouse_sensitivity;
            let saved_invert_x = game.invert_mouse_x;
            let saved_invert_y = game.invert_mouse_y;
            let saved_keys = game.keys;
            let saved_prev_keys = game.prev_keys;

            game = state::GameState::init(fb.w, fb.h, new_seed);

            game.keybinds = saved_keybinds;
            game.mouse_sensitivity = saved_sensitivity;
            game.invert_mouse_x = saved_invert_x;
            game.invert_mouse_y = saved_invert_y;
            game.keys = saved_keys;
            game.prev_keys = saved_prev_keys;
            game.menu.state = menu::MenuState::None;

            if gpu.as_ref().is_some_and(|g| g.has_graphics()) {
                render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
                gpu.as_mut().unwrap().upload_static_vertices(&gpu_static_verts);
            }
            particles = particle::ParticleSystem::new(&mut gpu, new_seed.wrapping_add(0xBEEF));
            scenic_cameras = generate_scenic_cameras(&game);
            accumulator = 0.0;
            last_frame = Instant::now();
            eprintln!("World seed: {}", new_seed);
            continue;
        }

        // ── Frame timing ────────────────────────────────────────────────
        let now = Instant::now();
        let frame_dt = now.duration_since(last_frame).as_secs_f32();
        if frame_dt < state::FRAME_TIME_MIN { continue; }
        last_frame = now;

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

        // ── Title phase: headless simulation + scenic camera ────────────
        let in_title = is_title_phase(&game.menu);
        if in_title {
            game.mouse_dx = 0.0;
            game.mouse_dy = 0.0;

            // Fixed timestep headless simulation (NPC AI, vehicles, time of day)
            accumulator += frame_dt.min(MAX_ACCUMULATOR);
            while accumulator >= FIXED_DT {
                accumulator -= FIXED_DT;
                game.tick_headless(FIXED_DT);
            }

            // Cycle scenic camera
            scenic_timer += frame_dt;
            if scenic_timer >= SCENIC_CUT_INTERVAL {
                scenic_timer = 0.0;
                scenic_idx = (scenic_idx + 1) % scenic_cameras.len().max(1);
            }
            if let Some(&(pos, target)) = scenic_cameras.get(scenic_idx) {
                game.camera.x = pos[0]; game.camera.y = pos[1]; game.camera.z = pos[2];
                game.camera.tx = target[0]; game.camera.ty = target[1]; game.camera.tz = target[2];
            }

            particles.update(&mut gpu, frame_dt);
        }
        // ── Gameplay: full simulation with player ───────────────────────
        else if game.menu.state == menu::MenuState::None {
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

            let ticks_per_frame = if game.time_speed > 1 {
                (game.time_speed as usize).min(500)
            } else {
                0
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

                game.time_of_day += FIXED_DT * 24.0 / state::DAY_LENGTH;
                if game.time_of_day >= 24.0 { game.time_of_day -= 24.0; }

                if npc::sys_midnight_reset(
                    &mut game.world, game.time_of_day, prev_time_of_day,
                    &mut game.neat_population, &mut game.neat_brains,
                ) {
                    game.day_count += 1;
                }

                player::sys_player(&mut game, FIXED_DT);
                vehicle::sys_vehicle(&mut game, FIXED_DT);

                for vi in 0..game.world.vehicles.len() {
                    vehicle_physics::step_vehicle_physics(
                        &mut game.world.vehicles[vi], &game.terrain, &game.road_network, FIXED_DT,
                    );
                }

                for npc in &mut game.world.npcs {
                    npc.skeleton.step_ragdoll(&game.terrain, FIXED_DT);
                    if npc.skeleton.ragdoll_active {
                        npc.ragdoll_points = npc.skeleton.to_ragdoll_points();
                        npc.ragdoll_active = true;
                    }
                }

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

                let interact_now = game.keybinds.is_pressed(input::Action::Interact, &game.keys);
                let interact_prev = game.keybinds.is_pressed(input::Action::Interact, &game.prev_keys);
                let interact_edge = interact_now && !interact_prev;
                let do_interact = interact_edge && game.player.in_vehicle.is_none();
                if let Some((sx, sz, sc)) = npc::sys_player_interact(
                    &mut game.world, &mut game.player, &game.terrain, do_interact,
                ) {
                    particle::emit_pickup_sparkle(&mut particles, sx, sz, sc);
                }

                game.frame_counter += 1;
                telemetry::sys_telemetry(&game);
            }

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
        // ── Paused / in-game menus: frozen scene ────────────────────────
        else {
            game.mouse_dx = 0.0;
            game.mouse_dy = 0.0;
        }

        // ── Render ──────────────────────────────────────────────────────
        let use_gpu = gpu.as_ref().is_some_and(|g| g.has_graphics());

        if use_gpu {
            let ctx = gpu.as_mut().unwrap();
            ctx.resize_render_target(fb.w as u32, fb.h as u32);

            render::generate_dynamic_gpu_vertices(
                &game.world, &game.player, &game.camera,
                &mut render_scratch, &mut gpu_dynamic_verts,
                game.time_of_day,
                &game.character_models,
                game.animation_data.as_ref(),
                &game.model_library.cars,
            );

            let aspect = fb.w as f32 / fb.h as f32;
            let eye = [game.camera.x, game.camera.y, game.camera.z];
            let target = [game.camera.tx, game.camera.ty, game.camera.tz];
            let view = math::m4_look_at(eye, target, [0.0, 1.0, 0.0]);
            let proj = math::m4_perspective_vk(60.0_f32.to_radians(), aspect, 0.1, 500.0);
            let vp = math::m4_mul(&proj, &view);
            let push = render::gpu_push_constants(game.time_of_day, eye, target, &vp);

            let clear = render::sky_color_f32(game.time_of_day);
            ctx.render_frame(&gpu_dynamic_verts, &push, clear, fb.w as u32, fb.h as u32, &mut fb.pixels);

            particle::sys_render_particles(&mut fb, &particles, &game.camera);
            if !in_title { hud::sys_hud(&mut fb, &game); }
            menu::sys_menu_render(
                &mut fb, &game.menu, &game.keybinds,
                game.mouse_sensitivity, game.invert_mouse_x, game.invert_mouse_y,
                game.player.model_index, &game.model_library.character_names,
            );

            window.present(&fb.pixels);
        } else {
            fb.clear(render::sky_color(game.time_of_day));
            render::sys_render(&mut fb, &game.world, &game.player, &game.camera, game.time_of_day, &mut render_scratch, &game.character_models, game.animation_data.as_ref(), &game.model_library.cars);

            particle::sys_render_particles(&mut fb, &particles, &game.camera);
            if !in_title { hud::sys_hud(&mut fb, &game); }
            menu::sys_menu_render(
                &mut fb, &game.menu, &game.keybinds,
                game.mouse_sensitivity, game.invert_mouse_x, game.invert_mouse_y,
                game.player.model_index, &game.model_library.character_names,
            );

            window.present(&fb.pixels);
        }
    }
}
