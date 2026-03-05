mod math;
mod state;
mod raster;
mod render;
mod platform;
mod gpu;
mod gpu_kernels;
mod world;
mod player;
mod vehicle;
mod npc;
mod camera;
mod hud;
mod particle;
mod rng;
mod input;
mod menu;

use std::time::Instant;

fn bytemuck_cast(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}
fn bytemuck_cast_mut(data: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) }
}

const FIXED_DT: f32 = 1.0 / 60.0;
const MAX_ACCUMULATOR: f32 = 0.25; // cap to prevent death spiral

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

    let mut fb = raster::Framebuffer::new(w, h);
    let mut particles = particle::ParticleSystem::new(&mut gpu, world_seed.wrapping_add(0xBEEF));

    let mut render_scratch: Vec<state::WorldTri> = Vec::with_capacity(4096);
    let mut last_frame = Instant::now();
    let mut accumulator: f32 = 0.0;
    let mut prev_time_of_day: f32 = game.time_of_day;
    let _ = prev_time_of_day; // suppress initial assignment warning

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
            // Fixed timestep accumulator
            accumulator += frame_dt.min(MAX_ACCUMULATOR);

            while accumulator >= FIXED_DT {
                accumulator -= FIXED_DT;

                prev_time_of_day = game.time_of_day;

                // Advance time of day
                game.time_of_day += FIXED_DT * 24.0 / state::DAY_LENGTH;
                if game.time_of_day >= 24.0 { game.time_of_day -= 24.0; }

                // Midnight reset (day counter, bin reset, NPC daily counters)
                if npc::sys_midnight_reset(&mut game.world, game.time_of_day, prev_time_of_day) {
                    game.day_count += 1;
                }

                // Game-logic systems at fixed dt
                player::sys_player(&mut game, FIXED_DT);
                vehicle::sys_vehicle(&mut game, FIXED_DT);
                npc::sys_npc(
                    &mut game.world, &game.road_network, &game.terrain,
                    FIXED_DT, game.time_of_day,
                );
                npc::sys_night_spawning(
                    &mut game.world, &game.terrain, game.time_of_day,
                    FIXED_DT, &mut game.spawn_rng,
                );
                npc::sys_items_update(&mut game.world, FIXED_DT);

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
            }

            // Visual systems at variable rate
            camera::sys_camera(
                &mut game.camera, &game.player, &game.terrain,
                game.mouse_dx, game.mouse_dy,
                game.mouse_sensitivity, game.invert_mouse_x, game.invert_mouse_y,
                frame_dt,
            );
            game.mouse_dx = 0.0;
            game.mouse_dy = 0.0;
            particle::sys_emit_particles(&mut particles, &game, frame_dt);
            particles.update(&mut gpu, frame_dt);
        }

        // Always render (frozen scene when paused)
        fb.clear(render::sky_color(game.time_of_day));
        render::sys_render(&mut fb, &game.world, &game.player, &game.camera, game.time_of_day, &mut render_scratch);
        particle::sys_render_particles(&mut fb, &particles, &game.camera);
        hud::sys_hud(&mut fb, &game);

        // Menu overlay on top
        menu::sys_menu_render(
            &mut fb, &game.menu, &game.keybinds,
            game.mouse_sensitivity, game.invert_mouse_x, game.invert_mouse_y,
        );

        window.present(&fb.pixels);
    }
}
