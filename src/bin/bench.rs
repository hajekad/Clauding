// Headless render benchmark — measures raw rasterization throughput
// Usage: cargo run --release --bin bench [seed] [frames] [width] [height]
// Reports: avg FPS, 1% low, 0.1% low, frame time distribution

use clauding::{state, render, raster};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let width: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1920);
    let height: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1080);

    eprintln!("=== RENDER BENCHMARK ===");
    eprintln!("Seed: {}, Frames: {}, Resolution: {}x{}", seed, frames, width, height);

    let mut game = state::GameState::init(width, height, seed);

    let mut fb = raster::Framebuffer::new(width, height);
    let mut scratch: Vec<state::WorldTri> = Vec::with_capacity(4096);

    eprintln!("World: {} static tris, {} buildings, {} NPCs, {} vehicles",
        game.world.static_tris.len(), game.world.buildings.len(),
        game.world.npcs.len(), game.world.vehicles.len());

    // Camera positions for benchmark sweep (different viewpoints)
    let cam_positions: Vec<(f32, f32, f32, f32, f32, f32)> = vec![
        // (eye_x, eye_y, eye_z, look_x, look_y, look_z)
        (0.0, 15.0, -30.0, 0.0, 0.0, 0.0),       // overview center
        (50.0, 8.0, 50.0, 0.0, 0.0, 0.0),         // diagonal approach
        (0.0, 2.0, 0.0, 20.0, 2.0, 0.0),          // street level
        (-100.0, 5.0, 0.0, 0.0, 5.0, 0.0),        // far west looking east
        (0.0, 50.0, 0.0, 0.0, 0.0, 0.0),          // birds eye
        (80.0, 3.0, -80.0, 0.0, 3.0, 0.0),        // corner looking in
        (0.0, 2.0, -5.0, 0.0, 2.0, 50.0),         // street level north
        (-50.0, 10.0, 50.0, 50.0, 0.0, -50.0),    // wide angle
    ];

    let mut frame_times: Vec<f32> = Vec::with_capacity(frames);
    let total_start = Instant::now();

    for f in 0..frames {
        // Cycle through camera positions
        let cam_idx = (f / (frames / cam_positions.len().max(1)).max(1)) % cam_positions.len();
        let (ex, ey, ez, tx, ty, tz) = cam_positions[cam_idx];

        // Slightly vary position per frame for realistic cache behavior
        let wobble = (f as f32 * 0.01).sin() * 2.0;
        game.camera.x = ex + wobble;
        game.camera.y = ey;
        game.camera.z = ez + wobble * 0.5;
        game.camera.tx = tx;
        game.camera.ty = ty;
        game.camera.tz = tz;

        // Vary time of day for lighting changes
        game.time_of_day = (f as f32 * 24.0 / frames as f32) % 24.0;

        let frame_start = Instant::now();
        fb.clear(render::sky_color(game.time_of_day));
        render::sys_render(&mut fb, &game.world, &game.player, &game.camera, game.time_of_day, &mut scratch, &game.character_models, game.animation_data.as_ref(), &game.model_library.cars);
        let frame_dt = frame_start.elapsed().as_secs_f32();

        frame_times.push(frame_dt);

        if f % 100 == 99 {
            eprint!(".");
        }
    }
    let total_elapsed = total_start.elapsed().as_secs_f32();
    eprintln!();

    // Analysis
    frame_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let avg_dt = frame_times.iter().sum::<f32>() / frames as f32;
    let median_dt = frame_times[frames / 2];
    let p99_dt = frame_times[(frames as f32 * 0.99) as usize]; // 99th percentile (1% low)
    let p999_dt = frame_times[(frames as f32 * 0.999) as usize]; // 99.9th (0.1% low)
    let min_dt = frame_times[0];
    let max_dt = frame_times[frames - 1];

    println!("=== BENCHMARK RESULTS ({} frames, {:.1}s) ===", frames, total_elapsed);
    println!();
    println!("Resolution: {}x{} ({:.1}M pixels)", width, height, width as f64 * height as f64 / 1_000_000.0);
    println!("Static tris: {}", game.world.static_tris.len());
    println!("Dynamic tris (last frame): {}", scratch.len());
    println!();
    println!("--- Frame Time ---");
    println!("  Average:    {:.3}ms ({:.0} FPS)", avg_dt * 1000.0, 1.0 / avg_dt);
    println!("  Median:     {:.3}ms ({:.0} FPS)", median_dt * 1000.0, 1.0 / median_dt);
    println!("  Best:       {:.3}ms ({:.0} FPS)", min_dt * 1000.0, 1.0 / min_dt);
    println!("  Worst:      {:.3}ms ({:.0} FPS)", max_dt * 1000.0, 1.0 / max_dt);
    println!();
    println!("--- Percentile FPS ---");
    println!("  Average:      {:.0} FPS", 1.0 / avg_dt);
    println!("  1% low:       {:.0} FPS", 1.0 / p99_dt);
    println!("  0.1% low:     {:.0} FPS", 1.0 / p999_dt);
    println!();
    println!("--- Distribution ---");

    // Histogram buckets (in ms)
    let buckets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0];
    let mut counts = vec![0u32; buckets.len() + 1];
    for &dt in &frame_times {
        let ms = dt * 1000.0;
        let mut placed = false;
        for (bi, &bv) in buckets.iter().enumerate() {
            if ms < bv { counts[bi] += 1; placed = true; break; }
        }
        if !placed { counts[buckets.len()] += 1; }
    }
    let mut prev = 0.0;
    for (bi, &bv) in buckets.iter().enumerate() {
        if counts[bi] > 0 {
            let pct = counts[bi] as f32 / frames as f32 * 100.0;
            let bar: String = std::iter::repeat('#').take((pct * 0.5) as usize).collect();
            println!("  {:>5.0}-{:<5.0}ms: {:>5} ({:>5.1}%) {}", prev, bv, counts[bi], pct, bar);
        }
        prev = bv;
    }
    if counts[buckets.len()] > 0 {
        let pct = counts[buckets.len()] as f32 / frames as f32 * 100.0;
        println!("  {:>5.0}ms+    : {:>5} ({:>5.1}%)", prev, counts[buckets.len()], pct);
    }

    // Target check
    println!();
    let target_fps = 180.0;
    let target_dt = 1.0 / target_fps;
    let frames_above = frame_times.iter().filter(|&&dt| dt <= target_dt).count();
    println!("--- Target: {} FPS ---", target_fps as u32);
    println!("  Frames meeting target: {}/{} ({:.1}%)",
        frames_above, frames, frames_above as f32 / frames as f32 * 100.0);
    println!("  1% low meets target: {}", if p99_dt <= target_dt { "YES" } else { "NO" });
    let headroom = (target_dt / avg_dt - 1.0) * 100.0;
    println!("  Headroom (avg): {:.0}%", headroom);
    println!("  Triangle budget at target: ~{:.0}K", game.world.static_tris.len() as f64 * target_dt as f64 / avg_dt as f64 / 1000.0);
}
