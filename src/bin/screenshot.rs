// Headless screenshot tool: renders the game world via GPU and saves to PNG.
// No window needed — uses Vulkan offscreen rendering.
//
// Usage:
//   cargo run --release --bin screenshot                     # default overhead view
//   cargo run --release --bin screenshot -- --pos 0 30 -50 --look 0 0 0
//   cargo run --release --bin screenshot -- --orbit 8        # 8 screenshots orbiting
//   cargo run --release --bin screenshot -- --grid           # 4x4 grid of viewpoints
//   cargo run --release --bin screenshot -- --pos 0 2 0 --look 0 2 -10  # street level
//   cargo run --release --bin screenshot -- --model npc      # 16-angle model orbit (npc/vehicle/player/item/bin)
//   cargo run --release --bin screenshot -- --timelapse      # 12 shots across a full day cycle
//   cargo run --release --bin screenshot -- --around 30 5 -20  # 16-angle orbit around a world point
//   cargo run --release --bin screenshot -- --seed 123       # use different world seed
//   cargo run --release --bin screenshot -- --follow-npc 5   # camera behind NPC #5
//   cargo run --release --bin screenshot -- --follow-vehicle 3  # camera behind vehicle #3
//   cargo run --release --bin screenshot -- --player         # orbit around the player
//
// Output: debug/screenshot*.png

use clauding::{state, render, raster, gpu, npc, vehicle, collision, combat};
use clauding::image::save_png;

const W: usize = 1920;
const H: usize = 1080;

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
        render_scratch, dynamic_verts,
        game.time_of_day,
    );

    // Build VP matrix + lighting push constants
    let (_vp, push, clear) = render::frame_setup(fb.w, fb.h, eye, target, time_of_day);
    ctx.render_frame(dynamic_verts, &push, clear, fb.w as u32, fb.h as u32, &mut fb.pixels);

    // The GPU pipeline uses double-buffering — render a second frame to get actual output
    ctx.render_frame(dynamic_verts, &push, clear, fb.w as u32, fb.h as u32, &mut fb.pixels);
}

fn main() {
    let _ = std::fs::create_dir_all("debug");
    let args: Vec<String> = std::env::args().collect();

    // Parse mode
    let mut mode = "default";
    let mut custom_pos: Option<[f32; 3]> = None;
    let mut custom_look: Option<[f32; 3]> = None;
    let mut orbit_count = 8;
    let mut time_of_day = 10.0_f32; // 10am default (good lighting)
    let mut sim_ticks: u32 = 0;
    let mut custom_name: Option<String> = None;
    let mut model_type = String::new();
    let mut around_pos: Option<[f32; 3]> = None;
    let mut world_seed: u64 = 42;
    let mut follow_npc: Option<usize> = None;
    let mut follow_vehicle: Option<usize> = None;
    let mut show_player: bool = false;

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
            "--model" => {
                mode = "model";
                if i + 1 < args.len() {
                    model_type = args[i+1].clone();
                    i += 1;
                }
            }
            "--timelapse" => { mode = "timelapse"; }
            "--bulk" => { mode = "bulk"; }
            "--around" => {
                mode = "around";
                if i + 3 < args.len() {
                    let x: f32 = args[i+1].parse().unwrap_or(0.0);
                    let y: f32 = args[i+2].parse().unwrap_or(5.0);
                    let z: f32 = args[i+3].parse().unwrap_or(0.0);
                    around_pos = Some([x, y, z]);
                    i += 3;
                }
            }
            "--time" => {
                if i + 1 < args.len() {
                    time_of_day = args[i+1].parse().unwrap_or(10.0);
                    i += 1;
                }
            }
            "--sim" => {
                if i + 1 < args.len() {
                    sim_ticks = args[i+1].parse().unwrap_or(30);
                    i += 1;
                }
            }
            "--seed" => {
                if i + 1 < args.len() {
                    world_seed = args[i+1].parse().unwrap_or(42);
                    i += 1;
                }
            }
            "--follow-npc" => {
                if i + 1 < args.len() {
                    follow_npc = Some(args[i+1].parse().unwrap_or(0));
                    i += 1;
                }
            }
            "--follow-vehicle" => {
                if i + 1 < args.len() {
                    follow_vehicle = Some(args[i+1].parse().unwrap_or(0));
                    i += 1;
                }
            }
            "--player" => {
                show_player = true;
                mode = "custom";
            }
            "--name" => {
                if i + 1 < args.len() {
                    custom_name = Some(args[i+1].clone());
                    i += 1;
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: screenshot [OPTIONS]");
                eprintln!("  --pos X Y Z        Camera position (default: 0 50 -50)");
                eprintln!("  --look X Y Z       Look-at target (default: 0 0 0)");
                eprintln!("  --orbit [N]        Take N shots orbiting center (default: 8)");
                eprintln!("  --grid             4x4 grid of viewpoints");
                eprintln!("  --model TYPE       16-angle orbit of entity (npc/vehicle/player/item/bin)");
                eprintln!("  --timelapse        12 shots across full day cycle (2h intervals)");
                eprintln!("  --around X Y Z     16-angle orbit around world position");
                eprintln!("  --time HOUR        Time of day 0-24 (default: 10)");
                eprintln!("  --sim TICKS        Simulate N ticks before render (default: 0)");
                eprintln!("  --seed N           World seed (default: 42)");
                eprintln!("  --follow-npc N     Center camera 4m behind NPC #N at eye level");
                eprintln!("  --follow-vehicle N Center camera 8m from vehicle #N");
                eprintln!("  --player           Orbit camera around the player");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    // Init GPU
    let mut ctx = gpu::GpuContext::init_or_exit(W as u32, H as u32);

    // Generate world
    eprintln!("Seed: {}", world_seed);
    let mut game = state::GameState::init(W, H, world_seed);

    game.time_of_day = time_of_day;

    // Simulate ticks to warm up physics (terrain normals, slope effects)
    if sim_ticks > 0 {
        eprintln!("Simulating {} ticks...", sim_ticks);
        let dt = 1.0 / 30.0;
        for _ in 0..sim_ticks {
            vehicle::sys_vehicle(&mut game, dt);
            npc::sys_npc(
                &mut game.world, &mut game.road_network, &game.terrain,
                dt, game.time_of_day, &mut game.neat_brains, 0.0, 0.0,
            );
            npc::sys_night_spawning(
                &mut game.world, &game.terrain, game.time_of_day,
                dt, &mut game.spawn_rng, &game.road_network,
            );
            collision::sys_collisions_headless(&mut game.world, &game.terrain, dt);
            combat::sys_ragdoll_update(&mut game.world, &game.terrain, dt);
            game.frame_counter += 1;
        }
    }

    // --follow-npc: override camera to look at specific NPC from behind
    if let Some(idx) = follow_npc {
        if idx < game.world.npcs.len() {
            let npc = &game.world.npcs[idx];
            let angle = npc.rot_y + std::f32::consts::PI; // behind the NPC
            custom_pos = Some([npc.x + angle.sin() * 4.0, npc.y + 1.7, npc.z + angle.cos() * 4.0]);
            custom_look = Some([npc.x, npc.y + 0.9, npc.z]);
            mode = "custom";
            eprintln!("Following NPC #{} at ({:.1}, {:.1}, {:.1})", idx, npc.x, npc.y, npc.z);
        } else {
            eprintln!("Error: NPC #{} not found (only {} NPCs)", idx, game.world.npcs.len());
            return;
        }
    }

    // --follow-vehicle: override camera to look at specific vehicle
    if let Some(idx) = follow_vehicle {
        if idx < game.world.vehicles.len() {
            let v = &game.world.vehicles[idx];
            let angle = v.rot_y + std::f32::consts::PI; // behind the vehicle
            custom_pos = Some([v.x + angle.sin() * 8.0, v.y + 3.0, v.z + angle.cos() * 8.0]);
            custom_look = Some([v.x, v.y + 0.8, v.z]);
            mode = "custom";
            eprintln!("Following vehicle #{} at ({:.1}, {:.1}, {:.1})", idx, v.x, v.y, v.z);
        } else {
            eprintln!("Error: Vehicle #{} not found (only {} vehicles)", idx, game.world.vehicles.len());
            return;
        }
    }

    // --player: orbit camera around the player
    if show_player {
        let px = game.player.x;
        let py = game.player.y;
        let pz = game.player.z;
        custom_pos = Some([px + 4.0, py + 2.0, pz + 4.0]);
        custom_look = Some([px, py + 0.9, pz]);
        eprintln!("Showing player at ({:.1}, {:.1}, {:.1})", px, py, pz);
    }

    eprintln!("World: {} static tris, time={:.1}h", game.world.static_tris.len(), time_of_day);

    // Upload static geometry to GPU
    let mut gpu_static_verts: Vec<gpu::GpuVertex> = Vec::with_capacity(1024 * 1024);
    let _eye_default = [0.0f32, 50.0, -50.0];
    render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
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
            render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
            ctx.upload_static_vertices(&gpu_static_verts);

            let cam = CameraSpec { pos, look };
            render_screenshot(&mut ctx, &game, &cam, time_of_day, &mut fb, &mut dynamic_verts, &mut render_scratch);
            let path = match &custom_name {
                Some(n) => format!("debug/screenshot_{}.png", n),
                None => "debug/screenshot.png".to_string(),
            };
            save_png(&fb.pixels, W, H, &path);
        }
        "orbit" => {
            for i in 0..orbit_count {
                let angle = (i as f32 / orbit_count as f32) * std::f32::consts::TAU;
                let radius = 80.0;
                let height = 40.0;
                let pos = [angle.cos() * radius, height, angle.sin() * radius];
                let look = [0.0, 0.0, 0.0];

                render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
                ctx.upload_static_vertices(&gpu_static_verts);

                let cam = CameraSpec { pos, look };
                render_screenshot(&mut ctx, &game, &cam, time_of_day, &mut fb, &mut dynamic_verts, &mut render_scratch);
                save_png(&fb.pixels, W, H, &format!("debug/screenshot_{:02}.png", i));
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

                render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
                ctx.upload_static_vertices(&gpu_static_verts);

                let cam = CameraSpec { pos: *pos, look: *look };
                render_screenshot(&mut ctx, &game, &cam, t, &mut fb, &mut dynamic_verts, &mut render_scratch);
                save_png(&fb.pixels, W, H, &format!("debug/screenshot_{}.png", name));
            }
        }
        "model" => {
            // 16-angle orbit around a specific entity type — renders through GPU pipeline
            // to show actual backface culling behavior
            let mt = model_type.as_str();
            let entity_center: [f32; 3];
            let orbit_radius: f32;
            let orbit_height_offset: f32;

            match mt {
                "npc" => {
                    orbit_radius = 4.0;
                    orbit_height_offset = 1.0;
                    if game.world.npcs.is_empty() {
                        eprintln!("No NPCs found");
                        return;
                    }
                    // Find the road node with maximum clearance from buildings,
                    // then move the first visible NPC there so the orbit camera
                    // never ends up inside geometry.
                    let min_clearance = orbit_radius + 2.0;
                    let mut clear_pos: Option<(f32, f32)> = None;
                    let mut clear_best = 0.0_f32;
                    for node in &game.road_network.nodes {
                        let nx = node[0];
                        let nz = node[1];
                        let mut min_dist = f32::MAX;
                        for b in &game.world.buildings {
                            let cx = b.x + b.w * 0.5;
                            let cz = b.z + b.d * 0.5;
                            let dx = (nx - cx).abs() - b.w * 0.5;
                            let dz = (nz - cz).abs() - b.d * 0.5;
                            let dist = dx.max(0.0).hypot(dz.max(0.0));
                            if dist < min_dist { min_dist = dist; }
                        }
                        if min_dist > clear_best {
                            clear_best = min_dist;
                            clear_pos = Some((nx, nz));
                        }
                    }
                    // Pick first visible NPC and move it to the clear spot
                    let npc_idx = game.world.npcs.iter().position(|n| {
                        n.state != state::NpcState::Sleeping && !n.in_vehicle
                    }).unwrap_or(0);
                    if let Some((cx, cz)) = clear_pos {
                        if clear_best >= min_clearance {
                            let cy = game.terrain.height_at(cx, cz);
                            game.world.npcs[npc_idx].x = cx;
                            game.world.npcs[npc_idx].y = cy;
                            game.world.npcs[npc_idx].z = cz;
                            eprintln!("  Moved NPC #{} to clear road node ({:.1}, {:.1}) clearance={:.1}m",
                                npc_idx, cx, cz, clear_best);
                        }
                    }
                    let npc = &game.world.npcs[npc_idx];
                    entity_center = [npc.x, npc.y + 1.0, npc.z];
                }
                "vehicle" => {
                    if let Some(v) = game.world.vehicles.first() {
                        entity_center = [v.x, v.y + 0.8, v.z];
                    } else {
                        eprintln!("No vehicles found");
                        return;
                    }
                    orbit_radius = 8.0;
                    orbit_height_offset = 1.5;
                }
                "player" => {
                    entity_center = [game.player.x, game.player.y + 1.0, game.player.z];
                    orbit_radius = 4.0;
                    orbit_height_offset = 1.0;
                }
                "item" => {
                    if let Some(item) = game.world.items.first() {
                        entity_center = [item.x, item.y + 0.3, item.z];
                    } else {
                        eprintln!("No items found");
                        return;
                    }
                    orbit_radius = 2.0;
                    orbit_height_offset = 0.3;
                }
                "bin" => {
                    if let Some(bin) = game.world.trash_bins.first() {
                        entity_center = [bin.x, bin.y + 0.5, bin.z];
                    } else {
                        eprintln!("No bins found");
                        return;
                    }
                    orbit_radius = 3.0;
                    orbit_height_offset = 0.5;
                }
                _ => {
                    eprintln!("Unknown model type '{}'. Use: npc, vehicle, player, item, bin", mt);
                    return;
                }
            }

            eprintln!("Model orbit: {} at ({:.1}, {:.1}, {:.1}), radius={:.1}",
                mt, entity_center[0], entity_center[1], entity_center[2], orbit_radius);

            // 16 angles: 4 rows × 4 cols in output grid
            // Row 0: eye-level orbit (4 angles)
            // Row 1: slightly elevated orbit (4 angles)
            // Row 2: elevated 45° orbit (4 angles)
            // Row 3: top-down + bottom-up + front close + back close
            let tile_w = W / 4;
            let tile_h = H / 4;
            let mut grid_pixels = vec![0x00111111u32; W * H]; // dark grey background

            let elevations: [(f32, &str); 4] = [
                (0.0, "eye-level"),
                (0.3, "low-angle"),
                (0.7, "mid-angle"),
                (1.2, "high-angle"),
            ];

            let mut shot = 0;
            for (row, &(elev, label)) in elevations.iter().enumerate() {
                // Offset each row's starting angle so all 16 shots have unique
                // horizontal angles (avoids near-duplicate framing between rows)
                let row_offset = (row as f32 / 16.0) * std::f32::consts::TAU;
                for col in 0..4 {
                    let angle = (col as f32 / 4.0) * std::f32::consts::TAU + row_offset;
                    let cam_x = entity_center[0] + angle.cos() * orbit_radius;
                    let cam_z = entity_center[2] + angle.sin() * orbit_radius;
                    let cam_y = entity_center[1] + orbit_height_offset + elev * orbit_radius;
                    let pos = [cam_x, cam_y, cam_z];
                    let look = entity_center;

                    render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
                    ctx.upload_static_vertices(&gpu_static_verts);

                    let cam = CameraSpec { pos, look };
                    render_screenshot(&mut ctx, &game, &cam, time_of_day, &mut fb, &mut dynamic_verts, &mut render_scratch);

                    // Copy tile into grid
                    let ox = col * tile_w;
                    let oy = row * tile_h;
                    for ty in 0..tile_h {
                        for tx in 0..tile_w {
                            let src_x = tx * W / tile_w;
                            let src_y = ty * H / tile_h;
                            grid_pixels[(oy + ty) * W + (ox + tx)] = fb.pixels[src_y * W + src_x];
                        }
                    }

                    // Also save individual shot
                    save_png(&fb.pixels, W, H, &format!("debug/model_{}_{:02}.png", mt, shot));
                    shot += 1;
                }
                eprintln!("  Row {}: {} (4 angles)", row, label);
            }

            // Save the 4×4 composite grid
            save_png(&grid_pixels, W, H, &format!("debug/model_{}_grid.png", mt));
            eprintln!("Saved 4x4 grid: debug/model_{}_grid.png", mt);
        }
        "timelapse" => {
            // 12 shots across a full day cycle (every 2 hours)
            let pos = custom_pos.unwrap_or([0.0, 40.0, -60.0]);
            let look = custom_look.unwrap_or([0.0, 0.0, 0.0]);

            for hour_idx in 0..12 {
                let t = hour_idx as f32 * 2.0; // 0, 2, 4, ..., 22

                render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
                ctx.upload_static_vertices(&gpu_static_verts);

                let cam = CameraSpec { pos, look };
                render_screenshot(&mut ctx, &game, &cam, t, &mut fb, &mut dynamic_verts, &mut render_scratch);

                let label = match hour_idx {
                    0 => "00_midnight",
                    1 => "02_late_night",
                    2 => "04_pre_dawn",
                    3 => "06_dawn",
                    4 => "08_morning",
                    5 => "10_mid_morning",
                    6 => "12_noon",
                    7 => "14_afternoon",
                    8 => "16_late_afternoon",
                    9 => "18_sunset",
                    10 => "20_dusk",
                    _ => "22_night",
                };
                save_png(&fb.pixels, W, H, &format!("debug/timelapse_{}.png", label));
            }
        }
        "around" => {
            // 16-angle orbit around an arbitrary world position
            let center = around_pos.unwrap_or([0.0, 5.0, 0.0]);
            let orbit_r = 15.0;

            eprintln!("Orbiting around ({:.1}, {:.1}, {:.1}), radius={:.1}, 16 angles",
                center[0], center[1], center[2], orbit_r);

            for idx in 0..16 {
                let angle = (idx as f32 / 16.0) * std::f32::consts::TAU;
                let elev = if idx < 8 { 0.3 } else { 0.8 }; // first 8: low, next 8: high
                let cam_x = center[0] + angle.cos() * orbit_r;
                let cam_z = center[2] + angle.sin() * orbit_r;
                let cam_y = center[1] + elev * orbit_r;

                render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
                ctx.upload_static_vertices(&gpu_static_verts);

                let cam = CameraSpec { pos: [cam_x, cam_y, cam_z], look: center };
                render_screenshot(&mut ctx, &game, &cam, time_of_day, &mut fb, &mut dynamic_verts, &mut render_scratch);
                save_png(&fb.pixels, W, H, &format!("debug/around_{:02}.png", idx));
            }
        }
        "bulk" => {
            // Generate 100+ screenshots from many world locations, angles, and times
            let mut bulk_shots: Vec<(&str, [f32; 3], [f32; 3], f32)> = Vec::new();

            // Overhead survey (8 shots at different positions)
            for (i, &(x, z)) in [(0.0,0.0), (50.0,0.0), (-50.0,0.0), (0.0,50.0),
                (0.0,-50.0), (80.0,80.0), (-80.0,-80.0), (0.0,150.0)].iter().enumerate() {
                let name: &str = match i {
                    0 => "bulk_overhead_center",
                    1 => "bulk_overhead_east",
                    2 => "bulk_overhead_west",
                    3 => "bulk_overhead_south",
                    4 => "bulk_overhead_north",
                    5 => "bulk_overhead_se",
                    6 => "bulk_overhead_nw",
                    7 => "bulk_overhead_docks",
                    _ => "bulk_overhead",
                };
                bulk_shots.push((name, [x, 60.0, z - 40.0], [x, 0.0, z], 10.0));
            }

            // Street-level walks (16 shots)
            let street_views: Vec<(&str, [f32;3], [f32;3])> = vec![
                ("bulk_street_center_n", [0.0, 2.0, -5.0], [0.0, 2.0, 20.0]),
                ("bulk_street_center_s", [0.0, 2.0, 5.0], [0.0, 2.0, -20.0]),
                ("bulk_street_center_e", [5.0, 2.0, 0.0], [30.0, 2.0, 0.0]),
                ("bulk_street_center_w", [-5.0, 2.0, 0.0], [-30.0, 2.0, 0.0]),
                ("bulk_street_road1", [30.0, 2.0, 10.0], [50.0, 2.0, 10.0]),
                ("bulk_street_road2", [-30.0, 2.0, -10.0], [-50.0, 2.0, 0.0]),
                ("bulk_street_road3", [10.0, 2.0, 40.0], [30.0, 2.0, 50.0]),
                ("bulk_street_alley", [15.0, 2.0, -15.0], [25.0, 2.0, -5.0]),
                ("bulk_street_suburb1", [-60.0, 2.0, -50.0], [-40.0, 2.0, -40.0]),
                ("bulk_street_suburb2", [60.0, 2.0, -60.0], [40.0, 2.0, -50.0]),
                ("bulk_street_docks1", [0.0, 2.0, 180.0], [20.0, 2.0, 200.0]),
                ("bulk_street_docks2", [-20.0, 2.0, 190.0], [0.0, 2.0, 200.0]),
                ("bulk_street_bridge1", [10.0, 3.0, 35.0], [20.0, 2.0, 45.0]),
                ("bulk_street_bridge2", [-10.0, 3.0, 45.0], [-20.0, 2.0, 35.0]),
                ("bulk_street_river1", [30.0, 2.0, 30.0], [20.0, 1.0, 40.0]),
                ("bulk_street_river2", [-30.0, 2.0, 50.0], [-20.0, 1.0, 40.0]),
            ];
            for (name, pos, look) in &street_views {
                bulk_shots.push((name, *pos, *look, 10.0));
            }

            // Mid-level views (12 shots)
            let mid_views: Vec<(&str, [f32;3], [f32;3])> = vec![
                ("bulk_mid_downtown", [0.0, 15.0, -30.0], [0.0, 0.0, 0.0]),
                ("bulk_mid_intersection", [20.0, 12.0, 10.0], [10.0, 0.0, 20.0]),
                ("bulk_mid_parking_lot", [40.0, 10.0, 15.0], [35.0, 0.0, 25.0]),
                ("bulk_mid_buildings_e", [60.0, 15.0, 0.0], [40.0, 5.0, 10.0]),
                ("bulk_mid_buildings_w", [-60.0, 15.0, 0.0], [-40.0, 5.0, 10.0]),
                ("bulk_mid_river_cross", [0.0, 20.0, 40.0], [0.0, 0.0, 45.0]),
                ("bulk_mid_bridge_side", [25.0, 8.0, 40.0], [10.0, 2.0, 40.0]),
                ("bulk_mid_dockyard", [20.0, 20.0, 180.0], [0.0, 5.0, 200.0]),
                ("bulk_mid_outer_ne", [80.0, 15.0, -80.0], [60.0, 0.0, -60.0]),
                ("bulk_mid_outer_sw", [-80.0, 15.0, 80.0], [-60.0, 0.0, 60.0]),
                ("bulk_mid_panorama_n", [0.0, 25.0, -100.0], [0.0, 0.0, 0.0]),
                ("bulk_mid_panorama_s", [0.0, 25.0, 100.0], [0.0, 0.0, 0.0]),
            ];
            for (name, pos, look) in &mid_views {
                bulk_shots.push((name, *pos, *look, 10.0));
            }

            // Close-ups of specific features (12 shots)
            let closeup_views: Vec<(&str, [f32;3], [f32;3])> = vec![
                ("bulk_closeup_bldg1", [20.0, 5.0, -15.0], [25.0, 4.0, -10.0]),
                ("bulk_closeup_bldg2", [-20.0, 5.0, 15.0], [-15.0, 4.0, 20.0]),
                ("bulk_closeup_vehicle1", [15.0, 2.0, -2.0], [15.0, 1.0, 2.0]),
                ("bulk_closeup_vehicle2", [35.0, 3.0, 8.0], [40.0, 1.0, 12.0]),
                ("bulk_closeup_tree", [10.0, 3.0, -8.0], [12.0, 3.0, -5.0]),
                ("bulk_closeup_streetlight", [5.0, 4.0, -3.0], [5.0, 6.0, 0.0]),
                ("bulk_closeup_npc1", [2.0, 2.0, 3.0], [0.0, 1.5, 5.0]),
                ("bulk_closeup_npc2", [-5.0, 2.0, -8.0], [-3.0, 1.5, -6.0]),
                ("bulk_closeup_items", [8.0, 1.5, 6.0], [10.0, 0.5, 8.0]),
                ("bulk_closeup_bin", [-8.0, 2.0, 4.0], [-6.0, 1.0, 6.0]),
                ("bulk_closeup_roof", [25.0, 12.0, -20.0], [25.0, 8.0, -18.0]),
                ("bulk_closeup_ground", [0.0, 1.0, 0.0], [3.0, 0.0, 3.0]),
            ];
            for (name, pos, look) in &closeup_views {
                bulk_shots.push((name, *pos, *look, 10.0));
            }

            // Time-of-day variations at the same downtown viewpoint (8 shots)
            let tod_times = [0.0, 3.0, 5.5, 7.0, 12.0, 16.0, 19.0, 21.0];
            let tod_names = ["bulk_tod_midnight", "bulk_tod_3am", "bulk_tod_predawn",
                "bulk_tod_sunrise", "bulk_tod_noon", "bulk_tod_afternoon",
                "bulk_tod_sunset", "bulk_tod_9pm"];
            for (name, t) in tod_names.iter().zip(tod_times.iter()) {
                bulk_shots.push((name, [0.0, 20.0, -40.0], [0.0, 0.0, 0.0], *t));
            }

            // Dramatic angles (8 shots)
            let dramatic: Vec<(&str, [f32;3], [f32;3], f32)> = vec![
                ("bulk_dramatic_low_angle", [5.0, 0.5, -10.0], [0.0, 8.0, 0.0], 10.0),
                ("bulk_dramatic_birds_eye", [0.0, 120.0, 0.0], [0.0, 0.0, 10.0], 10.0),
                ("bulk_dramatic_sunset_silhouette", [0.0, 5.0, -80.0], [0.0, 5.0, 0.0], 18.5),
                ("bulk_dramatic_dawn_glow", [-40.0, 3.0, -20.0], [0.0, 5.0, 0.0], 6.0),
                ("bulk_dramatic_night_street", [10.0, 2.5, -5.0], [20.0, 2.0, 10.0], 22.0),
                ("bulk_dramatic_dusk_river", [0.0, 5.0, 35.0], [0.0, 1.0, 45.0], 19.5),
                ("bulk_dramatic_long_road", [0.0, 3.0, -50.0], [0.0, 2.0, 50.0], 14.0),
                ("bulk_dramatic_horizon", [0.0, 2.0, -120.0], [0.0, 2.0, 0.0], 10.0),
            ];
            for (name, pos, look, t) in &dramatic {
                bulk_shots.push((name, *pos, *look, *t));
            }

            // Quarter-circle sweeps around center (16 shots)
            for i in 0..16 {
                let angle = (i as f32 / 16.0) * std::f32::consts::TAU;
                let r = 50.0;
                let pos = [angle.cos() * r, 15.0, angle.sin() * r];
                let look = [0.0, 2.0, 0.0];
                // Use leaked string so it lives long enough
                let name_str = format!("bulk_sweep_{:02}", i);
                let name_leaked: &'static str = Box::leak(name_str.into_boxed_str());
                bulk_shots.push((name_leaked, pos, look, 10.0));
            }

            // Distant fog views (4 shots)
            let fog_views: Vec<(&str, [f32;3], [f32;3], f32)> = vec![
                ("bulk_fog_north", [0.0, 10.0, -200.0], [0.0, 5.0, 0.0], 10.0),
                ("bulk_fog_south", [0.0, 10.0, 200.0], [0.0, 5.0, 0.0], 10.0),
                ("bulk_fog_east", [200.0, 10.0, 0.0], [0.0, 5.0, 0.0], 10.0),
                ("bulk_fog_west", [-200.0, 10.0, 0.0], [0.0, 5.0, 0.0], 10.0),
            ];
            for (name, pos, look, t) in &fog_views {
                bulk_shots.push((name, *pos, *look, *t));
            }

            // High altitude views (4 shots)
            let high_views: Vec<(&str, [f32;3], [f32;3], f32)> = vec![
                ("bulk_high_center", [0.0, 150.0, -50.0], [0.0, 0.0, 30.0], 10.0),
                ("bulk_high_docks", [0.0, 100.0, 100.0], [0.0, 0.0, 180.0], 10.0),
                ("bulk_high_suburbs", [-60.0, 80.0, -80.0], [-40.0, 0.0, -40.0], 10.0),
                ("bulk_high_overview", [50.0, 200.0, 50.0], [0.0, 0.0, 50.0], 10.0),
            ];
            for (name, pos, look, t) in &high_views {
                bulk_shots.push((name, *pos, *look, *t));
            }

            // Night city shots (8 shots)
            let night_views: Vec<(&str, [f32;3], [f32;3])> = vec![
                ("bulk_night_overhead", [0.0, 60.0, -30.0], [0.0, 0.0, 0.0]),
                ("bulk_night_street1", [0.0, 2.0, -5.0], [0.0, 2.0, 20.0]),
                ("bulk_night_street2", [20.0, 2.0, 5.0], [40.0, 2.0, 15.0]),
                ("bulk_night_buildings", [30.0, 8.0, -20.0], [20.0, 5.0, -10.0]),
                ("bulk_night_river", [0.0, 5.0, 35.0], [10.0, 1.0, 45.0]),
                ("bulk_night_docks", [0.0, 15.0, 180.0], [10.0, 5.0, 200.0]),
                ("bulk_night_suburb", [-50.0, 10.0, -40.0], [-30.0, 2.0, -30.0]),
                ("bulk_night_panorama", [0.0, 30.0, -80.0], [0.0, 5.0, 0.0]),
            ];
            for (name, pos, look) in &night_views {
                bulk_shots.push((name, *pos, *look, 1.0));
            }

            eprintln!("Bulk mode: rendering {} screenshots...", bulk_shots.len());
            for (idx, (name, pos, look, t)) in bulk_shots.iter().enumerate() {
                render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
                ctx.upload_static_vertices(&gpu_static_verts);

                let cam = CameraSpec { pos: *pos, look: *look };
                render_screenshot(&mut ctx, &game, &cam, *t, &mut fb, &mut dynamic_verts, &mut render_scratch);
                let path = format!("debug/{}.png", name);
                save_png(&fb.pixels, W, H, &path);
                if (idx + 1) % 10 == 0 {
                    eprintln!("  {}/{} rendered...", idx + 1, bulk_shots.len());
                }
            }
            eprintln!("Saved {} bulk screenshots to debug/bulk_*.png", bulk_shots.len());
        }
        _ => {
            // Default: overhead + street level + closeup
            let shots: Vec<(&str, [f32; 3], [f32; 3])> = vec![
                ("overhead",  [0.0, 60.0, -40.0], [0.0, 0.0, 0.0]),
                ("street",    [5.0, 2.5, -15.0],  [5.0, 2.0, 0.0]),
                ("buildings", [25.0, 10.0, -25.0], [25.0, 5.0, -10.0]),
            ];
            for (name, pos, look) in &shots {
                render::generate_static_gpu_vertices(&game.world, &mut gpu_static_verts);
                ctx.upload_static_vertices(&gpu_static_verts);

                let cam = CameraSpec { pos: *pos, look: *look };
                render_screenshot(&mut ctx, &game, &cam, time_of_day, &mut fb, &mut dynamic_verts, &mut render_scratch);
                save_png(&fb.pixels, W, H, &format!("debug/screenshot_{}.png", name));
            }
        }
    }

    eprintln!("Done.");
}
