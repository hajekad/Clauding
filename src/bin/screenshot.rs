// Headless screenshot tool: renders the game world via GPU and saves to PNG.
// No window needed — uses Vulkan offscreen rendering.
//
// Usage:
//   cargo run --release --bin screenshot                     # default overhead view
//   cargo run --release --bin screenshot -- --pos 0 30 -50 --look 0 0 0
//   cargo run --release --bin screenshot -- --orbit 8        # 8 screenshots orbiting
//   cargo run --release --bin screenshot -- --grid           # 4x4 grid of viewpoints
//   cargo run --release --bin screenshot -- --pos 0 2 0 --look 0 2 -10  # street level
//
// Output: /tmp/clauding_screenshot*.png

use clauding::{state, world, render, raster, math, gpu, neat};

const W: usize = 1920;
const H: usize = 1080;

fn bytemuck_cast(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}
fn bytemuck_cast_mut(data: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) }
}

fn save_png(pixels: &[u32], w: usize, h: usize, path: &str) {
    use std::io::Write;

    let row_bytes = 1 + w * 3;
    let mut raw = Vec::with_capacity(row_bytes * h);
    for y in 0..h {
        raw.push(0u8);
        for x in 0..w {
            let c = pixels[y * w + x];
            raw.push(((c >> 16) & 0xFF) as u8);
            raw.push(((c >> 8) & 0xFF) as u8);
            raw.push((c & 0xFF) as u8);
        }
    }

    let mut deflate = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 20);
    deflate.push(0x78);
    deflate.push(0x01);

    let mut offset = 0;
    while offset < raw.len() {
        let remaining = raw.len() - offset;
        let block_len = remaining.min(65535);
        let is_last = offset + block_len >= raw.len();
        deflate.push(if is_last { 1 } else { 0 });
        deflate.push((block_len & 0xFF) as u8);
        deflate.push(((block_len >> 8) & 0xFF) as u8);
        deflate.push((!block_len & 0xFF) as u8);
        deflate.push(((!block_len >> 8) & 0xFF) as u8);
        deflate.extend_from_slice(&raw[offset..offset + block_len]);
        offset += block_len;
    }

    let (mut s1, mut s2): (u32, u32) = (1, 0);
    for &b in &raw {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    deflate.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());

    fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFFFFFF;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB88320 } else { crc >> 1 };
            }
        }
        !crc
    }

    fn write_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(data);
        let mut crc_data = Vec::with_capacity(4 + data.len());
        crc_data.extend_from_slice(tag);
        crc_data.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_data).to_be_bytes());
    }

    let mut png = Vec::with_capacity(deflate.len() + 100);
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.push(8); ihdr.push(2); ihdr.push(0); ihdr.push(0); ihdr.push(0);
    write_chunk(&mut png, b"IHDR", &ihdr);
    write_chunk(&mut png, b"IDAT", &deflate);
    write_chunk(&mut png, b"IEND", &[]);

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&png).unwrap();
    eprintln!("Saved: {} ({}x{}, {:.1}MB)", path, w, h, png.len() as f64 / 1_000_000.0);
}

struct CameraSpec {
    pos: [f32; 3],
    look: [f32; 3],
}

fn render_screenshot(
    ctx: &mut gpu::GpuContext,
    game: &state::GameState,
    cam: &CameraSpec,
    time_of_day: f32,
    fb: &mut raster::Framebuffer,
    dynamic_verts: &mut Vec<gpu::GpuVertex>,
    render_scratch: &mut Vec<state::WorldTri>,
) {
    let eye = cam.pos;
    let target = cam.look;

    // Generate dynamic vertices around camera
    let fake_cam = state::Camera {
        x: eye[0], y: eye[1], z: eye[2],
        tx: target[0], ty: target[1], tz: target[2],
        yaw: 0.0, pitch: 0.0,
    };
    render::generate_dynamic_gpu_vertices(
        &game.world, &game.player, &fake_cam,
        time_of_day, render_scratch, dynamic_verts,
    );

    // Build VP matrix
    let aspect = fb.w as f32 / fb.h as f32;
    let view = math::m4_look_at(eye, target, [0.0, 1.0, 0.0]);
    let proj = math::m4_perspective_vk(60.0_f32.to_radians(), aspect, 0.1, 500.0);
    let vp = math::m4_mul(&proj, &view);

    let clear = render::sky_color_f32(time_of_day);
    ctx.render_frame(dynamic_verts, &vp, clear, fb.w as u32, fb.h as u32, &mut fb.pixels);

    // The GPU pipeline uses double-buffering — render a second frame to get actual output
    ctx.render_frame(dynamic_verts, &vp, clear, fb.w as u32, fb.h as u32, &mut fb.pixels);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse mode
    let mut mode = "default";
    let mut custom_pos: Option<[f32; 3]> = None;
    let mut custom_look: Option<[f32; 3]> = None;
    let mut orbit_count = 8;
    let mut time_of_day = 10.0_f32; // 10am default (good lighting)

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pos" => {
                if i + 3 < args.len() {
                    let x: f32 = args[i+1].parse().unwrap_or(0.0);
                    let y: f32 = args[i+2].parse().unwrap_or(30.0);
                    let z: f32 = args[i+3].parse().unwrap_or(0.0);
                    custom_pos = Some([x, y, z]);
                    mode = "custom";
                    i += 3;
                }
            }
            "--look" => {
                if i + 3 < args.len() {
                    let x: f32 = args[i+1].parse().unwrap_or(0.0);
                    let y: f32 = args[i+2].parse().unwrap_or(0.0);
                    let z: f32 = args[i+3].parse().unwrap_or(0.0);
                    custom_look = Some([x, y, z]);
                    i += 3;
                }
            }
            "--orbit" => {
                mode = "orbit";
                if i + 1 < args.len() {
                    if let Ok(n) = args[i+1].parse::<usize>() {
                        orbit_count = n;
                        i += 1;
                    }
                }
            }
            "--grid" => { mode = "grid"; }
            "--time" => {
                if i + 1 < args.len() {
                    time_of_day = args[i+1].parse().unwrap_or(10.0);
                    i += 1;
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: screenshot [OPTIONS]");
                eprintln!("  --pos X Y Z        Camera position (default: 0 50 -50)");
                eprintln!("  --look X Y Z       Look-at target (default: 0 0 0)");
                eprintln!("  --orbit [N]        Take N shots orbiting center (default: 8)");
                eprintln!("  --grid             4x4 grid of viewpoints");
                eprintln!("  --time HOUR        Time of day 0-24 (default: 10)");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    // Init GPU
    let mut ctx = match gpu::GpuContext::try_new() {
        Some(mut c) => {
            eprintln!("GPU: {}", c.device_name);
            let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let buf = c.create_buffer(data.len() * 4);
            c.upload(&buf, bytemuck_cast(&data));
            c.free_buffer(buf);
            c
        }
        None => {
            eprintln!("Error: No Vulkan GPU available. Screenshot tool requires GPU.");
            return;
        }
    };

    // Init GPU graphics pipeline
    ctx.init_graphics(W as u32, H as u32);
    if !ctx.has_graphics() {
        eprintln!("Error: GPU graphics pipeline failed to initialize.");
        return;
    }
    eprintln!("GPU graphics pipeline ready ({}x{})", W, H);

    // Generate world
    let world_seed: u64 = 42;
    let mut game = state::GameState::new(W, H, world_seed);
    world::generate_world(&mut game);

    // Load NEAT population
    if let Some(loaded) = neat::load_population("/tmp/clauding_neat.bin", state::NUM_NPCS) {
        game.neat_population = loaded;
    } else if let Some(loaded) = neat::load_population("neat_trained.bin", state::NUM_NPCS) {
        game.neat_population = loaded;
    }
    game.neat_brains = game.neat_population.genomes.iter()
        .map(|g| neat::NeatBrain::compile(g))
        .collect();

    game.time_of_day = time_of_day;
    eprintln!("World: {} static tris, time={:.1}h", game.world.static_tris.len(), time_of_day);

    // Upload static geometry to GPU
    let mut gpu_static_verts: Vec<gpu::GpuVertex> = Vec::with_capacity(1024 * 1024);
    let eye_default = [0.0f32, 50.0, -50.0];
    render::generate_static_gpu_vertices(&game.world, eye_default, time_of_day, &mut gpu_static_verts);
    ctx.upload_static_vertices(&gpu_static_verts);
    eprintln!("Static GPU verts: {} ({:.1}MB)", gpu_static_verts.len(),
        gpu_static_verts.len() as f64 * 28.0 / 1_000_000.0);

    let mut fb = raster::Framebuffer::new(W, H);
    let mut dynamic_verts: Vec<gpu::GpuVertex> = Vec::with_capacity(256 * 1024);
    let mut render_scratch: Vec<state::WorldTri> = Vec::with_capacity(16384);

    match mode {
        "custom" => {
            let pos = custom_pos.unwrap_or([0.0, 50.0, -50.0]);
            let look = custom_look.unwrap_or([0.0, 0.0, 0.0]);

            // Re-upload static verts with correct eye for lighting/fog
            render::generate_static_gpu_vertices(&game.world, pos, time_of_day, &mut gpu_static_verts);
            ctx.upload_static_vertices(&gpu_static_verts);

            let cam = CameraSpec { pos, look };
            render_screenshot(&mut ctx, &game, &cam, time_of_day, &mut fb, &mut dynamic_verts, &mut render_scratch);
            save_png(&fb.pixels, W, H, "/tmp/clauding_screenshot.png");
        }
        "orbit" => {
            for i in 0..orbit_count {
                let angle = (i as f32 / orbit_count as f32) * std::f32::consts::TAU;
                let radius = 80.0;
                let height = 40.0;
                let pos = [angle.cos() * radius, height, angle.sin() * radius];
                let look = [0.0, 0.0, 0.0];

                render::generate_static_gpu_vertices(&game.world, pos, time_of_day, &mut gpu_static_verts);
                ctx.upload_static_vertices(&gpu_static_verts);

                let cam = CameraSpec { pos, look };
                render_screenshot(&mut ctx, &game, &cam, time_of_day, &mut fb, &mut dynamic_verts, &mut render_scratch);
                save_png(&fb.pixels, W, H, &format!("/tmp/clauding_screenshot_{:02}.png", i));
            }
        }
        "grid" => {
            // 4x4 grid: different viewpoints covering the world
            let views: Vec<(&str, [f32; 3], [f32; 3])> = vec![
                ("overhead",     [0.0, 80.0, 0.0],     [0.0, 0.0, 1.0]),
                ("north",        [0.0, 20.0, -80.0],   [0.0, 0.0, 0.0]),
                ("south",        [0.0, 20.0, 80.0],    [0.0, 0.0, 0.0]),
                ("east",         [80.0, 20.0, 0.0],    [0.0, 0.0, 0.0]),
                ("west",         [-80.0, 20.0, 0.0],   [0.0, 0.0, 0.0]),
                ("street_n",     [0.0, 2.0, -20.0],    [0.0, 2.0, 0.0]),
                ("street_s",     [0.0, 2.0, 20.0],     [0.0, 2.0, 0.0]),
                ("street_e",     [20.0, 2.0, 0.0],     [0.0, 2.0, 0.0]),
                ("street_w",     [-20.0, 2.0, 0.0],    [0.0, 2.0, 0.0]),
                ("dockyard",     [0.0, 30.0, 170.0],   [0.0, 0.0, 200.0]),
                ("closeup_bldg", [30.0, 8.0, -30.0],   [30.0, 5.0, -20.0]),
                ("bridge_area",  [0.0, 15.0, 40.0],    [20.0, 0.0, 40.0]),
                ("parking",      [40.0, 15.0, 10.0],   [40.0, 0.0, 20.0]),
                ("suburb",       [-80.0, 12.0, -60.0],  [-60.0, 0.0, -50.0]),
                ("night_view",   [0.0, 40.0, -60.0],   [0.0, 0.0, 0.0]),
                ("dawn_view",    [60.0, 25.0, -40.0],  [0.0, 0.0, 0.0]),
            ];
            for (name, pos, look) in &views {
                // Special time for night/dawn views
                let t = match *name {
                    "night_view" => 22.0,
                    "dawn_view" => 6.0,
                    _ => time_of_day,
                };

                render::generate_static_gpu_vertices(&game.world, *pos, t, &mut gpu_static_verts);
                ctx.upload_static_vertices(&gpu_static_verts);

                let cam = CameraSpec { pos: *pos, look: *look };
                render_screenshot(&mut ctx, &game, &cam, t, &mut fb, &mut dynamic_verts, &mut render_scratch);
                save_png(&fb.pixels, W, H, &format!("/tmp/clauding_screenshot_{}.png", name));
            }
        }
        _ => {
            // Default: overhead + street level + closeup
            let shots: Vec<(&str, [f32; 3], [f32; 3])> = vec![
                ("overhead",  [0.0, 60.0, -40.0], [0.0, 0.0, 0.0]),
                ("street",    [5.0, 2.5, -15.0],  [5.0, 2.0, 0.0]),
                ("buildings", [25.0, 10.0, -25.0], [25.0, 5.0, -10.0]),
            ];
            for (name, pos, look) in &shots {
                render::generate_static_gpu_vertices(&game.world, *pos, time_of_day, &mut gpu_static_verts);
                ctx.upload_static_vertices(&gpu_static_verts);

                let cam = CameraSpec { pos: *pos, look: *look };
                render_screenshot(&mut ctx, &game, &cam, time_of_day, &mut fb, &mut dynamic_verts, &mut render_scratch);
                save_png(&fb.pixels, W, H, &format!("/tmp/clauding_screenshot_{}.png", name));
            }
        }
    }

    eprintln!("Done.");
}
