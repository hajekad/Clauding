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

use std::time::Instant;

fn bytemuck_cast(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}
fn bytemuck_cast_mut(data: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) }
}

const KEY_ESC: usize = 1;

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

    let mut game = state::GameState::new(w, h);
    world::generate_world(&mut game.world);

    let mut fb = raster::Framebuffer::new(w, h);
    let mut particles = particle::ParticleSystem::new(&mut gpu);

    let mut last_frame = Instant::now();

    loop {
        window.poll_events(&mut game.keys, &mut game.should_quit);
        if game.should_quit || game.keys[KEY_ESC] { break; }

        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32();
        if dt < state::FRAME_TIME_MIN { continue; }
        last_frame = now;

        let nw = window.width();
        let nh = window.height();
        if nw != fb.w || nh != fb.h {
            fb.resize(nw, nh);
            game.width = nw;
            game.height = nh;
        }

        // Advance time of day
        game.time_of_day += dt * 24.0 / state::DAY_LENGTH;
        if game.time_of_day >= 24.0 { game.time_of_day -= 24.0; }

        player::sys_player(&mut game, dt);
        vehicle::sys_vehicle(&mut game, dt);
        npc::sys_npc(&mut game.world, dt);
        let pickups = npc::sys_items(&mut game.world, &mut game.player, dt);
        for p in &pickups {
            particle::emit_pickup_sparkle(&mut particles, p.x, p.z, p.color);
        }
        camera::sys_camera(&mut game.camera, &game.player, dt);

        // Particle emitters + GPU update
        particle::sys_emit_particles(&mut particles, &game, dt);
        particles.update(&mut gpu, dt);

        fb.clear(render::sky_color(game.time_of_day));
        render::sys_render(&mut fb, &game.world, &game.player, &game.camera, game.time_of_day);
        particle::sys_render_particles(&mut fb, &particles, &game.camera);
        hud::sys_hud(&mut fb, &game);

        window.present(&fb.pixels);
    }
}
