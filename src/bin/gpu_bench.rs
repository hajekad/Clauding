// GPU rendering benchmark — measures actual Vulkan pipeline performance
// Usage: cargo run --release --bin gpu_bench [seed] [frames] [width] [height]
// Reports: avg FPS, median, 1% low, 0.1% low, frame time distribution, vertex counts

use clauding::{state, render, raster, gpu};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let frames: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);
    let width: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1920);
    let height: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1080);

    eprintln!("=== GPU RENDER BENCHMARK ===");
    eprintln!("Seed: {}, Frames: {}, Resolution: {}x{}", seed, frames, width, height);

    // Init GPU
    let mut ctx = gpu::GpuContext::init_or_exit(width as u32, height as u32);

    // Generate world
    let game = state::GameState::init(width, height, seed);

    // Upload static geometry to GPU
    let mut gpu_static_verts: Vec<gpu::GpuVertex> = Vec::with_capacity(1024 * 1024);
    render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
    ctx.upload_static_vertices(&gpu_static_verts);
    let static_vert_count = gpu_static_verts.len();

    eprintln!("World: {} static tris, {} buildings, {} NPCs, {} vehicles",
        game.world.static_tris.len(), game.world.buildings.len(),
        game.world.npcs.len(), game.world.vehicles.len());
    eprintln!("Static GPU verts: {} ({:.1}MB)", static_vert_count,
        static_vert_count as f64 * 28.0 / 1_000_000.0);

    let mut fb = raster::Framebuffer::new(width, height);
    let mut dynamic_verts: Vec<gpu::GpuVertex> = Vec::with_capacity(256 * 1024);
    let mut render_scratch: Vec<state::WorldTri> = Vec::with_capacity(16384);

    // Camera positions for benchmark sweep
    let cam_positions: Vec<([f32; 3], [f32; 3])> = vec![
        // (eye, target)
        ([0.0, 50.0, -50.0], [0.0, 0.0, 0.0]),        // overhead center
        ([50.0, 8.0, 50.0],  [0.0, 0.0, 0.0]),         // diagonal approach
        ([0.0, 2.0, 0.0],    [20.0, 2.0, 0.0]),         // street level east
        ([-100.0, 5.0, 0.0], [0.0, 5.0, 0.0]),          // far west looking east
        ([0.0, 80.0, 0.0],   [0.0, 0.0, 1.0]),          // birds eye
        ([80.0, 3.0, -80.0], [0.0, 3.0, 0.0]),          // corner looking in
        ([0.0, 2.0, -5.0],   [0.0, 2.0, 50.0]),         // street level north
        ([-50.0, 10.0, 50.0],[50.0, 0.0, -50.0]),       // wide angle diagonal
    ];

    // Warmup: render 2 throwaway frames to prime the double-buffered pipeline
    {
        let eye = cam_positions[0].0;
        let target = cam_positions[0].1;
        let fake_cam = state::Camera {
            x: eye[0], y: eye[1], z: eye[2],
            tx: target[0], ty: target[1], tz: target[2],
            yaw: 0.0, pitch: 0.0,
        };
        render::generate_dynamic_gpu_vertices(
            &game.world, &game.player, &fake_cam,
            &mut render_scratch, &mut dynamic_verts, 12.0,
            &game.character_models,
            game.animation_data.as_ref(),
            &game.model_library.cars,
        );
        let (_vp, push, clear) = render::frame_setup(width, height, eye, target, 12.0);

        ctx.render_frame(&dynamic_verts, &push, clear, width as u32, height as u32, &mut fb.pixels);
        ctx.render_frame(&dynamic_verts, &push, clear, width as u32, height as u32, &mut fb.pixels);
        eprintln!("Warmup complete (2 frames)");
    }

    // Benchmark loop
    let mut frame_times: Vec<f32> = Vec::with_capacity(frames);
    let mut last_dynamic_count: usize = 0;
    let mut total_dynamic_count: usize = 0;
    let total_start = Instant::now();

    for f in 0..frames {
        // Cycle through camera positions
        let cam_idx = (f / (frames / cam_positions.len()).max(1)) % cam_positions.len();
        let (eye, target) = cam_positions[cam_idx];

        // Add per-frame wobble for realistic cache behavior
        let wobble = (f as f32 * 0.01).sin() * 2.0;
        let eye = [eye[0] + wobble, eye[1], eye[2] + wobble * 0.5];

        // Sweep time of day 0-24h across all frames
        let time_of_day = (f as f32 * 24.0 / frames as f32) % 24.0;

        let fake_cam = state::Camera {
            x: eye[0], y: eye[1], z: eye[2],
            tx: target[0], ty: target[1], tz: target[2],
            yaw: 0.0, pitch: 0.0,
        };

        let frame_start = Instant::now();

        // Generate dynamic vertices (CPU work)
        render::generate_dynamic_gpu_vertices(
            &game.world, &game.player, &fake_cam,
            &mut render_scratch, &mut dynamic_verts, time_of_day,
            &game.character_models,
            game.animation_data.as_ref(),
            &game.model_library.cars,
        );

        // Build VP matrix + push constants
        let (_vp, push, clear) = render::frame_setup(width, height, eye, target, time_of_day);

        // GPU render + readback (two frames for double-buffered pipeline)
        ctx.render_frame(&dynamic_verts, &push, clear, width as u32, height as u32, &mut fb.pixels);
        ctx.render_frame(&dynamic_verts, &push, clear, width as u32, height as u32, &mut fb.pixels);

        let frame_dt = frame_start.elapsed().as_secs_f32();
        frame_times.push(frame_dt);

        last_dynamic_count = dynamic_verts.len();
        total_dynamic_count += dynamic_verts.len();

        if f % 50 == 49 {
            eprint!(".");
        }
    }
    let total_elapsed = total_start.elapsed().as_secs_f32();
    eprintln!();

    // Analysis
    frame_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let avg_dt = frame_times.iter().sum::<f32>() / frames as f32;
    let median_dt = frame_times[frames / 2];
    let p99_dt = frame_times[((frames as f32 * 0.99) as usize).min(frames - 1)];
    let p999_dt = frame_times[((frames as f32 * 0.999) as usize).min(frames - 1)];
    let min_dt = frame_times[0];
    let max_dt = frame_times[frames - 1];

    let avg_dynamic = total_dynamic_count / frames;

    println!("=== GPU BENCHMARK RESULTS ({} frames, {:.1}s) ===", frames, total_elapsed);
    println!();
    println!("GPU: {}", ctx.device_name);
    println!("Resolution: {}x{} ({:.1}M pixels)", width, height,
        width as f64 * height as f64 / 1_000_000.0);
    println!();
    println!("--- Vertex Counts ---");
    println!("  Static:         {:>7} verts ({:.1}MB)",
        static_vert_count, static_vert_count as f64 * 28.0 / 1_000_000.0);
    println!("  Dynamic (avg):  {:>7} verts ({:.1}MB)",
        avg_dynamic, avg_dynamic as f64 * 28.0 / 1_000_000.0);
    println!("  Dynamic (last): {:>7} verts ({:.1}MB)",
        last_dynamic_count, last_dynamic_count as f64 * 28.0 / 1_000_000.0);
    println!("  Total (avg):    {:>7} verts ({:.1}MB)",
        static_vert_count + avg_dynamic,
        (static_vert_count + avg_dynamic) as f64 * 28.0 / 1_000_000.0);
    println!();
    println!("--- Frame Time ---");
    println!("  Average:    {:.3}ms ({:.0} FPS)", avg_dt * 1000.0, 1.0 / avg_dt);
    println!("  Median:     {:.3}ms ({:.0} FPS)", median_dt * 1000.0, 1.0 / median_dt);
    println!("  Best:       {:.3}ms ({:.0} FPS)", min_dt * 1000.0, 1.0 / min_dt);
    println!("  Worst:      {:.3}ms ({:.0} FPS)", max_dt * 1000.0, 1.0 / max_dt);
    println!();
    println!("--- Percentile FPS ---");
    println!("  Average:      {:.0} FPS", 1.0 / avg_dt);
    println!("  Median:       {:.0} FPS", 1.0 / median_dt);
    println!("  1% low:       {:.0} FPS", 1.0 / p99_dt);
    println!("  0.1% low:     {:.0} FPS", 1.0 / p999_dt);
    println!();
    println!("--- Distribution ---");

    // Histogram buckets (in ms) — finer grained for GPU speeds
    let buckets = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0];
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
            println!("  {:>5.1}-{:<5.1}ms: {:>5} ({:>5.1}%) {}", prev, bv, counts[bi], pct, bar);
        }
        prev = bv;
    }
    if counts[buckets.len()] > 0 {
        let pct = counts[buckets.len()] as f32 / frames as f32 * 100.0;
        println!("  {:>5.1}ms+     : {:>5} ({:>5.1}%)", prev, counts[buckets.len()], pct);
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

    // Throughput metrics
    println!();
    println!("--- Throughput ---");
    let total_verts_per_frame = (static_vert_count + avg_dynamic) as f64;
    let tris_per_frame = total_verts_per_frame / 3.0;
    let tris_per_sec = tris_per_frame / avg_dt as f64;
    println!("  Triangles/frame: {:.0}K", tris_per_frame / 1000.0);
    println!("  Triangles/sec:   {:.1}M", tris_per_sec / 1_000_000.0);
    let pixels_per_sec = (width * height) as f64 / avg_dt as f64;
    println!("  Pixels/sec:      {:.1}G", pixels_per_sec / 1_000_000_000.0);
}
