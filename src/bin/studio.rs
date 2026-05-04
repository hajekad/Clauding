// Studio: unified game server for LLM agents.
// Stdin/stdout text protocol. One command per line, response ends with "---".
// Diagnostics to stderr only.
// Absorbs: screenshot, observe, probe3d, render_map, inspect, winding_check, debug_render.

use clauding::{
    state, npc, vehicle, vehicle_physics, player, camera, particle, raster, render, gpu, input,
    combat, collision, player_jobs, telemetry, image, world, skeleton, gltf_loader,
    skeleton_anim,
};

use std::io::{self, BufRead, Write as IoWrite};
use std::fmt::Write as FmtWrite;

const FIXED_DT: f32 = 1.0 / 60.0;
const W: usize = 1920;
const H: usize = 1080;


/// Direct analog input override (bypasses keyboard simulation)
struct DirectInput {
    active: bool,
    throttle: f32,  // -1..1 (negative = reverse)
    brake: f32,     // 0..1
    steer: f32,     // -1..1
    handbrake: bool,
}

impl DirectInput {
    fn new() -> Self {
        DirectInput { active: false, throttle: 0.0, brake: 0.0, steer: 0.0, handbrake: false }
    }
}

struct Studio {
    game: state::GameState,
    gpu_ctx: Option<gpu::GpuContext>,
    fb: raster::Framebuffer,
    static_verts: Vec<gpu::GpuVertex>,
    dynamic_verts: Vec<gpu::GpuVertex>,
    scratch: Vec<state::WorldTri>,
    particles: particle::ParticleSystem,
    screenshot_counter: u32,
    speed_factor: u32,
    pending_oneshot: Vec<usize>,
    direct_input: DirectInput,
    mark_pos: Option<[f32; 3]>,
    cam_dist: f32,
}

impl Studio {
    fn new(seed: u64) -> Self {
        let _ = std::fs::create_dir_all("debug");

        eprintln!("studio: init seed={}", seed);
        let game = state::GameState::init(W, H, seed);

        let mut gpu_ctx = gpu::GpuContext::try_new();
        if let Some(ref mut ctx) = gpu_ctx {
            eprintln!("studio: GPU {}", ctx.device_name);
            ctx.init_graphics(W as u32, H as u32);
        } else {
            eprintln!("studio: no GPU, CPU fallback");
        }

        let fb = raster::Framebuffer::new(W, H);
        let mut static_verts: Vec<gpu::GpuVertex> = Vec::with_capacity(1024 * 1024);

        if gpu_ctx.as_ref().is_some_and(|g| g.has_graphics()) {
            render::generate_static_gpu_vertices(&game.world, &mut static_verts);
            gpu_ctx.as_mut().unwrap().upload_static_vertices(&static_verts);
            eprintln!("studio: static verts={}", static_verts.len());
        }

        let particles = particle::ParticleSystem::new(&mut gpu_ctx, seed.wrapping_add(0xBEEF));

        Studio {
            game,
            gpu_ctx,
            fb,
            static_verts,
            dynamic_verts: Vec::with_capacity(256 * 1024),
            scratch: Vec::with_capacity(16384),
            particles,
            screenshot_counter: 0,
            speed_factor: 1,
            pending_oneshot: Vec::new(),
            direct_input: DirectInput::new(),
            mark_pos: None,
            cam_dist: 8.0,
        }
    }

    fn run_full_tick(&mut self) {
        let game = &mut self.game;

        // prev_keys MUST be first (edge detection)
        game.prev_keys = game.keys;

        // For one-shot keys: clear from prev_keys so edge detection fires
        for &sc in &self.pending_oneshot {
            game.prev_keys[sc] = false;
        }

        let prev_time_of_day = game.time_of_day;

        // Advance time
        game.time_of_day += FIXED_DT * 24.0 / state::DAY_LENGTH;
        if game.time_of_day >= 24.0 { game.time_of_day -= 24.0; }

        // Midnight reset
        if npc::sys_midnight_reset(
            &mut game.world, game.time_of_day, prev_time_of_day,
            &mut game.neat_population, &mut game.neat_brains,
        ) {
            game.day_count += 1;
        }

        // Game systems at fixed dt
        player::sys_player(game, FIXED_DT);
        vehicle::sys_vehicle(game, FIXED_DT);

        // Direct input override: apply analog inputs to driven vehicle after sys_vehicle
        if self.direct_input.active {
            if let Some(vi) = game.player.in_vehicle {
                let v = &mut game.world.vehicles[vi];
                v.drivetrain.throttle = self.direct_input.throttle;
                v.drivetrain.brake = self.direct_input.brake;
                v.drivetrain.steer_input = self.direct_input.steer;
                v.drivetrain.handbrake = self.direct_input.handbrake;
            }
        }

        // Vehicle rigid body physics
        for vi in 0..game.world.vehicles.len() {
            vehicle_physics::step_vehicle_physics(
                &mut game.world.vehicles[vi], &game.terrain, &game.road_network, FIXED_DT,
            );
        }

        // Skeleton ragdoll step
        for n in &mut game.world.npcs {
            n.skeleton.step_ragdoll(&game.terrain, FIXED_DT);
            if n.skeleton.ragdoll_active {
                n.ragdoll_points = n.skeleton.to_ragdoll_points();
                n.ragdoll_active = true;
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
            &mut game.world, &mut game.player, &mut self.particles,
            &game.terrain, &game.keys, &game.prev_keys,
            &game.keybinds, FIXED_DT,
        );
        npc::sys_hunger_thirst(&mut game.world, &mut game.player, FIXED_DT);
        player_jobs::sys_interactibles_update(&mut game.world, FIXED_DT);
        player_jobs::sys_player_job(game, FIXED_DT);

        // Edge-detected interact
        let interact_now = game.keybinds.is_pressed(input::Action::Interact, &game.keys);
        let interact_prev = game.keybinds.is_pressed(input::Action::Interact, &game.prev_keys);
        let interact_edge = interact_now && !interact_prev;
        let do_interact = interact_edge && game.player.in_vehicle.is_none();
        if let Some((sx, sz, sc)) = npc::sys_player_interact(
            &mut game.world, &mut game.player, &game.terrain, do_interact,
        ) {
            particle::emit_pickup_sparkle(&mut self.particles, sx, sz, sc);
        }

        game.frame_counter += 1;
        telemetry::sys_telemetry(game);

        // Camera update
        camera::sys_camera(
            &mut game.camera, &game.player, &game.terrain,
            0.0, 0.0, game.mouse_sensitivity, game.invert_mouse_x, game.invert_mouse_y,
            FIXED_DT, game.frame_counter,
        );
        particle::sys_emit_particles(&mut self.particles, game, FIXED_DT);
        self.particles.update(&mut self.gpu_ctx, FIXED_DT);

        // Clear one-shot scancodes
        for &sc in &self.pending_oneshot {
            self.game.keys[sc] = false;
        }
        self.pending_oneshot.clear();
    }

    fn snap_camera(&mut self) {
        let p = &self.game.player;
        let cam = &mut self.game.camera;
        let dist = if p.in_vehicle.is_some() { 14.0 } else { self.cam_dist };
        let cos_p = cam.pitch.cos();
        let sin_p = cam.pitch.sin();
        cam.x = p.x + cam.yaw.sin() * cos_p * dist;
        cam.y = p.y + sin_p * dist + 1.5;
        cam.z = p.z + cam.yaw.cos() * cos_p * dist;
        let terrain_y = self.game.terrain.height_at(cam.x, cam.z) + 1.0;
        if cam.y < terrain_y { cam.y = terrain_y; }
        cam.tx = p.x;
        cam.ty = p.y + 1.2;
        cam.tz = p.z;
    }

    fn take_screenshot(&mut self, path: &str) -> String {
        self.take_screenshot_wh(W, H, path)
    }

    fn take_screenshot_wh(&mut self, w: usize, h: usize, path: &str) -> String {
        if w != self.fb.w || h != self.fb.h {
            self.fb = raster::Framebuffer::new(w, h);
        }
        let use_gpu = false; // GPU readback broken — use CPU rasterizer for screenshots
        if use_gpu {
            let ctx = self.gpu_ctx.as_mut().unwrap();
            ctx.resize_render_target(w as u32, h as u32);

            render::generate_static_gpu_vertices(&self.game.world, &mut self.static_verts);
            ctx.upload_static_vertices(&self.static_verts);

            render::generate_dynamic_gpu_vertices(
                &self.game.world, &self.game.player, &self.game.camera,
                &mut self.scratch, &mut self.dynamic_verts,
                self.game.time_of_day,
                &self.game.character_models,
                self.game.animation_data.as_ref(),
                &self.game.model_library.cars,
            );
            let eye = [self.game.camera.x, self.game.camera.y, self.game.camera.z];
            let target = [self.game.camera.tx, self.game.camera.ty, self.game.camera.tz];
            let (_vp, push, clear) = render::frame_setup(w, h, eye, target, self.game.time_of_day);
            // Triple render: submit frame 0, submit frame 1 (reads back 0), submit frame 2 (reads back 1 into output)
            ctx.render_frame(&self.dynamic_verts, &push, clear, w as u32, h as u32, &mut self.fb.pixels);
            ctx.render_frame(&self.dynamic_verts, &push, clear, w as u32, h as u32, &mut self.fb.pixels);
            ctx.render_frame(&self.dynamic_verts, &push, clear, w as u32, h as u32, &mut self.fb.pixels);
        } else {
            self.fb.clear(render::sky_color(self.game.time_of_day));
            render::sys_render(
                &mut self.fb, &self.game.world, &self.game.player, &self.game.camera,
                self.game.time_of_day, &mut self.scratch,
                &self.game.character_models,
                self.game.animation_data.as_ref(),
                &self.game.model_library.cars,
            );
        }
        image::save_png(&self.fb.pixels, w, h, path);
        format!("{} {}x{}", path, w, h)
    }

    fn dispatch(&mut self, line: &str) -> String {
        let line = line.trim();
        if line.is_empty() { return String::new(); }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let cmd = parts[0].to_ascii_lowercase();
        let args = &parts[1..];

        match cmd.as_str() {
            "help" => self.cmd_help(),
            "quit" | "exit" => std::process::exit(0),
            // Simulation
            "tick" => self.cmd_tick(args),
            "time" => self.cmd_time(args),
            "speed" => self.cmd_speed(args),
            // Player input
            "move" => self.cmd_move(args),
            "sprint" => self.cmd_sprint(args),
            "jump" => self.cmd_jump(),
            "attack" => self.cmd_attack(),
            "interact" => self.cmd_interact(),
            "look" => self.cmd_look(args),
            "face" => self.cmd_face(args),
            "teleport" | "tp" => self.cmd_teleport(args),
            // Queries
            "player" => self.cmd_player(),
            "npcs" => self.cmd_npcs(args),
            "vehicles" | "vehs" => self.cmd_vehicles(args),
            "items" => self.cmd_items(args),
            "nearby" => self.cmd_nearby(args),
            "npc" => self.cmd_npc(args),
            "vehicle" | "veh" => self.cmd_vehicle(args),
            "surface" => self.cmd_surface(args),
            "world" => self.cmd_world(),
            "buildings" => self.cmd_buildings(args),
            "observe" => self.cmd_observe(),
            // World diagnostics (from probe3d)
            "probe" => self.cmd_probe(args),
            "roads" => self.cmd_roads(),
            "river" => self.cmd_river(),
            "parking" => self.cmd_parking(),
            // Physics diagnostics
            "skeleton" | "skel" => self.cmd_skeleton(args),
            "gait" => self.cmd_gait(args),
            "vphysics" | "vphys" => self.cmd_vphysics(args),
            // Vehicle testing
            "set" => self.cmd_set(args),
            "throttle" => self.cmd_throttle(args),
            "brake" => self.cmd_brake(args),
            "handbrake" => self.cmd_handbrake(args),
            "steer" => self.cmd_steer(args),
            "inputs" => self.cmd_inputs(args),
            "mark" => self.cmd_mark(),
            "distance" | "dist" => self.cmd_distance(),
            "log" => self.cmd_log(args),
            "test" => self.cmd_test(args),
            "testcourse" => self.cmd_testcourse(args),
            "enter" => self.cmd_enter_vehicle(args),
            "nude" => {
                let on = !render::NUDE_MODE.load(std::sync::atomic::Ordering::Relaxed);
                render::NUDE_MODE.store(on, std::sync::atomic::Ordering::Relaxed);
                format!("nude_mode={}", on)
            }
            "zoom" => {
                if let Some(d) = args.first().and_then(|s| s.parse::<f32>().ok()) {
                    self.cam_dist = d.clamp(1.0, 50.0);
                    self.snap_camera();
                    format!("ok dist={:.1}", self.cam_dist)
                } else {
                    format!("dist={:.1}", self.cam_dist)
                }
            }
            // Rendering
            "screenshot" | "ss" => self.cmd_screenshot(args),
            "render" => self.cmd_render(args),
            "export" => self.cmd_export(args),
            // Map/analysis (from render_map, inspect, winding_check, debug_render, observe)
            "map" => self.cmd_map(args),
            "inspect" => self.cmd_inspect(args),
            "winding" => self.cmd_winding(args),
            "compare" => self.cmd_compare(args),
            "sim" => self.cmd_sim(args),
            // Model library
            "models" => self.cmd_models(args),
            "model" => self.cmd_model(args),
            "fbx" => self.cmd_fbx(args),
            // Build
            "build" => self.cmd_build(),
            _ => format!("err: unknown command '{}'. try 'help'", cmd),
        }
    }

    // ---- Commands ----

    fn cmd_help(&self) -> String {
        [
            "=== Simulation ===",
            "tick [N]          run N ticks (default 1)",
            "time [HOUR]       get/set time (0-24)",
            "speed [N]         get/set tick multiplier",
            "=== Player Input ===",
            "move DIR          set movement: n/s/e/w/ne/nw/se/sw/stop",
            "sprint on|off     toggle sprint",
            "jump              one-shot jump",
            "attack            one-shot attack",
            "interact          one-shot interact",
            "look BEARING PITCH  set camera (degrees, 0=N, pitch up=positive)",
            "face BEARING      set player facing (degrees, 0=N)",
            "teleport X Y Z    move player to position",
            "=== Queries ===",
            "player            query player state",
            "npcs [R]          NPCs within R meters (default 50)",
            "vehicles [R]      vehicles within R meters (default 50)",
            "items [R]         items within R meters (default 50)",
            "nearby [R]        all entities within R meters (default 30)",
            "npc ID            detailed NPC info",
            "vehicle ID        detailed vehicle info",
            "surface [X Z]     surface type at position (default: player pos)",
            "world             world statistics",
            "buildings [R]     buildings within R meters (default 50)",
            "observe           structured scene snapshot",
            "=== World Diagnostics ===",
            "probe X Z         detailed point query (height, surface, collision, nearby)",
            "roads             road network summary",
            "river             river segment analysis",
            "parking           parking spot analysis",
            "=== Physics Diagnostics ===",
            "skeleton player|npc ID  per-bone positions, ragdoll state",
            "gait player|npc ID      gait, walk_phase, feet, CoM",
            "vphysics ID       vehicle physics (wheels, slip, ABS, surface, drivetrain)",
            "=== Vehicle Testing ===",
            "set speed N       set vehicle speed (m/s) along forward direction",
            "set surface TYPE  force surface: asphalt|grass|gravel|dirt|ice|wet|metal|none",
            "throttle N        direct throttle (-1.0 to 1.0, negative=reverse)",
            "brake N           direct brake (0.0 to 1.0)",
            "handbrake on|off  direct handbrake",
            "steer N           direct steer (-1.0 to 1.0)",
            "inputs off        disable direct inputs, return to keyboard control",
            "mark              set reference position at current vehicle/player pos",
            "distance          measure distance from mark",
            "log N [FILE]      dump N ticks of vehicle CSV (default: debug/vlog.csv)",
            "test accel        0-100-200 km/h acceleration benchmark on asphalt",
            "test brake        100-0 km/h braking distance benchmark",
            "test topspeed     terminal velocity benchmark",
            "testcourse [SURF] generate flat test track (default: asphalt)",
            "=== Rendering ===",
            "screenshot [PATH] save screenshot (default: debug/studio_NNNN.png)",
            "render W H PATH   render at custom resolution",
            "export TYPE PATH  OBJ export (player|npc N|vehicle N|world|scene)",
            "=== Analysis (writes to file) ===",
            "map [PATH]        top-down PPM world map (default: debug/map.ppm)",
            "inspect [PATH]    ASCII diagnostic report (default: debug/inspect.txt)",
            "winding [FILTER]  triangle winding audit (all|static|npc|vehicle|item|bin|player)",
            "compare [PATH]    CPU vs GPU render comparison (default: debug/render_compare.png)",
            "sim DAYS [PATH]   multi-day headless simulation analysis",
            "=== Character Models ===",
            "model list        list loaded character models with index and name",
            "model view N      render side-by-side bind/animated comparison → debug/model_<name>.png",
            "model viewall     render all character models",
            "model obj N PATH  export character model to OBJ file",
            "=== Other ===",
            "build             run cargo build --release",
            "help              this message",
            "quit              exit",
        ].join("\n")
    }

    fn cmd_tick(&mut self, args: &[&str]) -> String {
        let n: u32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1);
        let n = n.min(100_000);
        for _ in 0..n {
            self.run_full_tick();
        }
        format!("t={} time={:.1}", n, self.game.time_of_day)
    }

    fn cmd_time(&mut self, args: &[&str]) -> String {
        if let Some(h) = args.first().and_then(|s| s.parse::<f32>().ok()) {
            self.game.time_of_day = h.clamp(0.0, 24.0);
            if self.game.time_of_day >= 24.0 { self.game.time_of_day = 0.0; }
            "ok".into()
        } else {
            format!("time={:.1}", self.game.time_of_day)
        }
    }

    fn cmd_speed(&mut self, args: &[&str]) -> String {
        if let Some(n) = args.first().and_then(|s| s.parse::<u32>().ok()) {
            self.speed_factor = n.max(1);
            "ok".into()
        } else {
            format!("speed={}", self.speed_factor)
        }
    }

    fn cmd_move(&mut self, args: &[&str]) -> String {
        let dir = args.first().copied().unwrap_or("stop");
        let kb = &self.game.keybinds;
        let fwd = kb.key_for(input::Action::MoveForward);
        let back = kb.key_for(input::Action::MoveBack);
        let left = kb.key_for(input::Action::MoveLeft);
        let right = kb.key_for(input::Action::MoveRight);

        self.game.keys[fwd] = false;
        self.game.keys[back] = false;
        self.game.keys[left] = false;
        self.game.keys[right] = false;

        match dir.to_ascii_lowercase().as_str() {
            "n" | "north" | "forward" | "fwd" => { self.game.keys[fwd] = true; }
            "s" | "south" | "back" | "backward" => { self.game.keys[back] = true; }
            "e" | "east" | "right" => { self.game.keys[right] = true; }
            "w" | "west" | "left" => { self.game.keys[left] = true; }
            "ne" => { self.game.keys[fwd] = true; self.game.keys[right] = true; }
            "nw" => { self.game.keys[fwd] = true; self.game.keys[left] = true; }
            "se" => { self.game.keys[back] = true; self.game.keys[right] = true; }
            "sw" => { self.game.keys[back] = true; self.game.keys[left] = true; }
            "stop" | "none" => {}
            _ => return format!("err: unknown direction '{}'. use n/s/e/w/ne/nw/se/sw/stop", dir),
        }
        "ok".into()
    }

    fn cmd_sprint(&mut self, args: &[&str]) -> String {
        let sc = self.game.keybinds.key_for(input::Action::Sprint);
        match args.first().copied().unwrap_or("on") {
            "on" | "1" | "true" => { self.game.keys[sc] = true; }
            "off" | "0" | "false" => { self.game.keys[sc] = false; }
            _ => return "err: sprint on|off".into(),
        }
        "ok".into()
    }

    fn cmd_jump(&mut self) -> String {
        let sc = self.game.keybinds.key_for(input::Action::Jump);
        self.game.keys[sc] = true;
        self.pending_oneshot.push(sc);
        self.run_full_tick();
        "ok".into()
    }

    fn cmd_attack(&mut self) -> String {
        let sc = self.game.keybinds.key_for(input::Action::Attack);
        self.game.keys[sc] = true;
        self.pending_oneshot.push(sc);
        self.run_full_tick();
        "ok".into()
    }

    fn cmd_interact(&mut self) -> String {
        let sc = self.game.keybinds.key_for(input::Action::Interact);
        self.game.keys[sc] = true;
        self.pending_oneshot.push(sc);
        self.run_full_tick();
        "ok".into()
    }

    fn cmd_look(&mut self, args: &[&str]) -> String {
        if args.len() < 2 {
            let bearing = (-self.game.camera.yaw).to_degrees().rem_euclid(360.0);
            let pitch = self.game.camera.pitch.to_degrees();
            return format!("bearing={:.1} pitch={:.1}", bearing, pitch);
        }
        let bearing: f32 = match args[0].parse() { Ok(v) => v, Err(_) => return "err: bad bearing".into() };
        let pitch: f32 = match args[1].parse() { Ok(v) => v, Err(_) => return "err: bad pitch".into() };
        self.game.camera.yaw = -bearing.to_radians();
        self.game.camera.pitch = pitch.to_radians().clamp(0.05, 1.2);
        self.snap_camera();
        "ok".into()
    }

    fn cmd_face(&mut self, args: &[&str]) -> String {
        if args.is_empty() {
            let deg = (-self.game.player.rot_y).to_degrees().rem_euclid(360.0);
            return format!("facing={:.1}", deg);
        }
        let bearing: f32 = match args[0].parse() { Ok(v) => v, Err(_) => return "err: bad bearing".into() };
        let yaw = -bearing.to_radians();
        self.game.player.rot_y = yaw;
        // Also rotate vehicle when driving
        if let Some(vi) = self.game.player.in_vehicle {
            let v = &mut self.game.world.vehicles[vi];
            v.rot_y = yaw;
            v.body.quat = clauding::math::quat_from_rot_y(yaw);
            v.body.update_inertia();
        }
        "ok".into()
    }

    fn cmd_teleport(&mut self, args: &[&str]) -> String {
        if args.len() < 3 { return "err: teleport X Y Z".into(); }
        let x: f32 = match args[0].parse() { Ok(v) => v, Err(_) => return "err: bad X".into() };
        let y: f32 = match args[1].parse() { Ok(v) => v, Err(_) => return "err: bad Y".into() };
        let z: f32 = match args[2].parse() { Ok(v) => v, Err(_) => return "err: bad Z".into() };
        // If driving, teleport the vehicle too
        if let Some(vi) = self.game.player.in_vehicle {
            let v = &mut self.game.world.vehicles[vi];
            v.x = x; v.y = y; v.z = z;
            v.body.pos = [x, y, z];
            v.body.vel = [0.0, 0.0, 0.0];
            v.body.ang_vel = [0.0, 0.0, 0.0];
            v.speed = 0.0;
        }
        let p = &mut self.game.player;
        p.x = x; p.y = y; p.z = z;
        p.vel_y = 0.0;
        p.body.pos = [x, y, z];
        p.body.vel = [0.0, 0.0, 0.0];
        p.skeleton.bones[0].world_pos = [x, y, z];
        self.snap_camera();
        "ok".into()
    }

    fn cmd_player(&self) -> String {
        let p = &self.game.player;
        let veh = match p.in_vehicle {
            Some(i) => format!("{}", i),
            None => "none".into(),
        };
        format!(
            "pos={:.1},{:.1},{:.1} vel_y={:.1} hp={:.0} sta={:.1} ground={} rot={:.1} veh={} hunger={:.1} thirst={:.1} money={:.1} job={}",
            p.x, p.y, p.z, p.vel_y, p.health, p.stamina,
            p.on_ground as u8, p.rot_y, veh, p.hunger, p.thirst, p.money,
            p.active_job.job_type.name(),
        )
    }

    fn cmd_npcs(&self, args: &[&str]) -> String {
        let radius: f32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(50.0);
        let r2 = radius * radius;
        let px = self.game.player.x;
        let pz = self.game.player.z;
        let mut lines = Vec::new();
        for (i, n) in self.game.world.npcs.iter().enumerate() {
            let dx = n.x - px;
            let dz = n.z - pz;
            let d2 = dx * dx + dz * dz;
            if d2 > r2 { continue; }
            let dist = d2.sqrt();
            let bearing = bearing_deg(px, pz, n.x, n.z);
            lines.push(format!(
                "[{}] pos={:.1},{:.1},{:.1} hp={:.0} job={} state={} dist={:.1} bearing={:.0}",
                i, n.x, n.y, n.z, n.health, n.job.name(), n.state.name(), dist, bearing,
            ));
        }
        if lines.is_empty() { "none".into() } else { lines.join("\n") }
    }

    fn cmd_vehicles(&self, args: &[&str]) -> String {
        let radius: f32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(50.0);
        let r2 = radius * radius;
        let px = self.game.player.x;
        let pz = self.game.player.z;
        let mut lines = Vec::new();
        for (i, v) in self.game.world.vehicles.iter().enumerate() {
            let dx = v.x - px;
            let dz = v.z - pz;
            let d2 = dx * dx + dz * dz;
            if d2 > r2 { continue; }
            let dist = d2.sqrt();
            let bearing = bearing_deg(px, pz, v.x, v.z);
            let st = if v.parked { "parked" } else if v.occupied { "occupied" } else { "moving" };
            lines.push(format!(
                "[{}] pos={:.1},{:.1},{:.1} spd={:.1} rot={:.1} {} dist={:.1} bearing={:.0}",
                i, v.x, v.y, v.z, v.speed, v.rot_y, st, dist, bearing,
            ));
        }
        if lines.is_empty() { "none".into() } else { lines.join("\n") }
    }

    fn cmd_items(&self, args: &[&str]) -> String {
        let radius: f32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(50.0);
        let r2 = radius * radius;
        let px = self.game.player.x;
        let pz = self.game.player.z;
        let mut lines = Vec::new();
        for (i, item) in self.game.world.items.iter().enumerate() {
            if !item.active { continue; }
            let dx = item.x - px;
            let dz = item.z - pz;
            let d2 = dx * dx + dz * dz;
            if d2 > r2 { continue; }
            let dist = d2.sqrt();
            let bearing = bearing_deg(px, pz, item.x, item.z);
            let kind = item_kind_str(item.kind);
            lines.push(format!(
                "[{}] pos={:.1},{:.1},{:.1} kind={} dist={:.1} bearing={:.0}",
                i, item.x, item.y, item.z, kind, dist, bearing,
            ));
        }
        if lines.is_empty() { "none".into() } else { lines.join("\n") }
    }

    fn cmd_nearby(&self, args: &[&str]) -> String {
        let radius: f32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(30.0);
        let r2 = radius * radius;
        let px = self.game.player.x;
        let pz = self.game.player.z;
        let mut lines = Vec::new();

        for (i, n) in self.game.world.npcs.iter().enumerate() {
            let d2 = (n.x - px) * (n.x - px) + (n.z - pz) * (n.z - pz);
            if d2 > r2 { continue; }
            let dist = d2.sqrt();
            let b = bearing_deg(px, pz, n.x, n.z);
            lines.push(format!("npc[{}] dist={:.1} dir={} job={} state={} hp={:.0}",
                i, dist, compass(b), n.job.name(), n.state.name(), n.health));
        }
        for (i, v) in self.game.world.vehicles.iter().enumerate() {
            let d2 = (v.x - px) * (v.x - px) + (v.z - pz) * (v.z - pz);
            if d2 > r2 { continue; }
            let dist = d2.sqrt();
            let b = bearing_deg(px, pz, v.x, v.z);
            let st = if v.parked { "parked" } else if v.occupied { "occupied" } else { "moving" };
            lines.push(format!("veh[{}] dist={:.1} dir={} {} spd={:.1}",
                i, dist, compass(b), st, v.speed));
        }
        for (i, item) in self.game.world.items.iter().enumerate() {
            if !item.active { continue; }
            let d2 = (item.x - px) * (item.x - px) + (item.z - pz) * (item.z - pz);
            if d2 > r2 { continue; }
            let dist = d2.sqrt();
            let b = bearing_deg(px, pz, item.x, item.z);
            lines.push(format!("item[{}] dist={:.1} dir={} kind={}",
                i, dist, compass(b), item_kind_str(item.kind)));
        }

        lines.sort_by(|a, b| {
            extract_dist(a).partial_cmp(&extract_dist(b)).unwrap_or(std::cmp::Ordering::Equal)
        });

        if lines.is_empty() { "none".into() } else { lines.join("\n") }
    }

    fn cmd_npc(&self, args: &[&str]) -> String {
        let id: usize = match args.first().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return "err: npc ID".into(),
        };
        if id >= self.game.world.npcs.len() {
            return format!("err: npc {} not found (max {})", id, self.game.world.npcs.len() - 1);
        }
        let n = &self.game.world.npcs[id];
        let px = self.game.player.x;
        let pz = self.game.player.z;
        let dist = ((n.x - px) * (n.x - px) + (n.z - pz) * (n.z - pz)).sqrt();
        let bearing = bearing_deg(px, pz, n.x, n.z);
        let veh = if n.in_vehicle { format!("car={}", n.car_idx) } else { "on_foot".into() };
        format!(
            "pos={:.1},{:.1},{:.1} hp={:.0} job={} state={} dist={:.1} bearing={:.0} rot={:.1} {} home={} money={:.1} hunger={:.1} thirst={:.1} items_today={} ragdoll={} wanted={}",
            n.x, n.y, n.z, n.health, n.job.name(), n.state.name(), dist, bearing,
            n.rot_y, veh, n.home_idx, n.money, n.hunger, n.thirst,
            n.items_deposited_today, n.ragdoll_active, n.wanted,
        )
    }

    fn cmd_vehicle(&self, args: &[&str]) -> String {
        let id: usize = match args.first().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return "err: vehicle ID".into(),
        };
        if id >= self.game.world.vehicles.len() {
            return format!("err: vehicle {} not found (max {})", id, self.game.world.vehicles.len() - 1);
        }
        let v = &self.game.world.vehicles[id];
        let px = self.game.player.x;
        let pz = self.game.player.z;
        let dist = ((v.x - px) * (v.x - px) + (v.z - pz) * (v.z - pz)).sqrt();
        let bearing = bearing_deg(px, pz, v.x, v.z);
        let owner = match v.owner_npc { Some(i) => format!("{}", i), None => "none".into() };
        let state_str = if v.parked { "parked" } else if v.occupied { "occupied" } else { "moving" };
        format!(
            "pos={:.1},{:.1},{:.1} spd={:.1} rot={:.1} {} owner={} dist={:.1} bearing={:.0} ai={} cruise={:.1}",
            v.x, v.y, v.z, v.speed, v.rot_y, state_str, owner, dist, bearing,
            v.ai_active, v.cruise_speed,
        )
    }

    fn cmd_surface(&self, args: &[&str]) -> String {
        let (x, z) = if args.len() >= 2 {
            let x: f32 = match args[0].parse() { Ok(v) => v, Err(_) => return "err: bad X".into() };
            let z: f32 = match args[1].parse() { Ok(v) => v, Err(_) => return "err: bad Z".into() };
            (x, z)
        } else {
            (self.game.player.x, self.game.player.z)
        };
        let surface = world::surface_at(x, z, &self.game.road_network);
        let y = self.game.terrain.height_at(x, z);
        format!("surface={} height={:.1} at={:.1},{:.1}", surf_str(surface), y, x, z)
    }

    fn cmd_world(&self) -> String {
        let w = &self.game.world;
        format!(
            "buildings={} npcs={} vehicles={} items={} trees={} rocks={} bins={} lights={} tris={} day={} time={:.1} frame={} seed={}",
            w.buildings.len(), w.npcs.len(), w.vehicles.len(),
            w.items.iter().filter(|i| i.active).count(),
            w.trees.len(), w.rocks.len(), w.trash_bins.len(), w.street_lights.len(),
            w.static_tris.len(), self.game.day_count, self.game.time_of_day,
            self.game.frame_counter, self.game.world_seed,
        )
    }

    fn cmd_buildings(&self, args: &[&str]) -> String {
        let radius: f32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(50.0);
        let r2 = radius * radius;
        let px = self.game.player.x;
        let pz = self.game.player.z;
        let mut lines = Vec::new();
        for (i, b) in self.game.world.buildings.iter().enumerate() {
            let cx = b.x + b.w * 0.5;
            let cz = b.z + b.d * 0.5;
            let d2 = (cx - px) * (cx - px) + (cz - pz) * (cz - pz);
            if d2 > r2 { continue; }
            let dist = d2.sqrt();
            let bearing = bearing_deg(px, pz, cx, cz);
            lines.push(format!(
                "[{}] pos={:.1},{:.1} size={:.1}x{:.1} h={:.1} dist={:.1} dir={}",
                i, b.x, b.z, b.w, b.d, b.h, dist, compass(bearing),
            ));
        }
        if lines.is_empty() { "none".into() } else { lines.join("\n") }
    }

    fn cmd_observe(&self) -> String {
        let p = &self.game.player;
        let px = p.x;
        let pz = p.z;
        let surf_name = surf_str(world::surface_at(px, pz, &self.game.road_network));
        let facing_deg = (-p.rot_y).to_degrees().rem_euclid(360.0);

        let mut lines = Vec::new();
        lines.push(format!("time={:.1} day={} frame={}", self.game.time_of_day, self.game.day_count, self.game.frame_counter));
        lines.push(format!("player: pos={:.1},{:.1},{:.1} facing={} ground={} hp={:.0} sta={:.1} hunger={:.1} thirst={:.1}",
            px, p.y, pz, compass(facing_deg), surf_name, p.health, p.stamina, p.hunger, p.thirst));

        let npc_r = 30.0_f32;
        let npc_r2 = npc_r * npc_r;
        let mut nearby_npcs: Vec<(usize, f32, f32)> = Vec::new();
        for (i, n) in self.game.world.npcs.iter().enumerate() {
            let d2 = (n.x - px) * (n.x - px) + (n.z - pz) * (n.z - pz);
            if d2 <= npc_r2 { nearby_npcs.push((i, d2.sqrt(), bearing_deg(px, pz, n.x, n.z))); }
        }
        nearby_npcs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut nearby_vehs: Vec<(usize, f32, f32)> = Vec::new();
        for (i, v) in self.game.world.vehicles.iter().enumerate() {
            let d2 = (v.x - px) * (v.x - px) + (v.z - pz) * (v.z - pz);
            if d2 <= npc_r2 { nearby_vehs.push((i, d2.sqrt(), bearing_deg(px, pz, v.x, v.z))); }
        }
        nearby_vehs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        lines.push(format!("nearby_npcs: {} within {}m", nearby_npcs.len(), npc_r as u32));
        lines.push(format!("nearby_vehicles: {} within {}m", nearby_vehs.len(), npc_r as u32));

        for (idx, dist, b) in &nearby_npcs {
            let n = &self.game.world.npcs[*idx];
            lines.push(format!("[npc{}] dist={:.1} dir={} job={} state={} hp={:.0}",
                idx, dist, compass(*b), n.job.name(), n.state.name(), n.health));
        }
        for (idx, dist, b) in &nearby_vehs {
            let v = &self.game.world.vehicles[*idx];
            let st = if v.parked { "parked" } else if v.occupied { "occupied" } else { "moving" };
            lines.push(format!("[veh{}] dist={:.1} dir={} {} spd={:.1}",
                idx, dist, compass(*b), st, v.speed));
        }

        let mut nearby_bldgs: Vec<(usize, f32, f32)> = Vec::new();
        for (i, b) in self.game.world.buildings.iter().enumerate() {
            let cx = b.x + b.w * 0.5;
            let cz = b.z + b.d * 0.5;
            let d2 = (cx - px) * (cx - px) + (cz - pz) * (cz - pz);
            if d2 <= npc_r2 { nearby_bldgs.push((i, d2.sqrt(), bearing_deg(px, pz, cx, cz))); }
        }
        nearby_bldgs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if !nearby_bldgs.is_empty() {
            let bldg_strs: Vec<String> = nearby_bldgs.iter().take(5).map(|(i, d, b)| {
                format!("bldg{}@{:.0}m({})", i, d, compass(*b))
            }).collect();
            lines.push(format!("buildings: {}", bldg_strs.join(", ")));
        }

        lines.join("\n")
    }

    // ---- World Diagnostics (from probe3d) ----

    fn cmd_probe(&self, args: &[&str]) -> String {
        if args.len() < 2 { return "err: probe X Z".into(); }
        let x: f32 = match args[0].parse() { Ok(v) => v, Err(_) => return "err: bad X".into() };
        let z: f32 = match args[1].parse() { Ok(v) => v, Err(_) => return "err: bad Z".into() };

        let h = self.game.terrain.height_at(x, z);
        let normal = self.game.terrain.normal_at(x, z);
        let surface = world::surface_at(x, z, &self.game.road_network);
        let on_riv = world::on_river(x, z, &self.game.world.river_segments);
        let bld_col = world::check_walk_collision(&self.game.world, x, z, 0.4, None);
        let on_road = world::on_any_road(x, z, &self.game.road_network);

        let mut lines = Vec::new();
        lines.push(format!("pos={:.1},{:.1} height={:.2} normal={:.3},{:.3},{:.3} surface={} river={} collision={} road={}",
            x, z, h, normal[0], normal[1], normal[2], surf_str(surface), on_riv, bld_col, on_road));

        // Nearby objects within 15m
        let r2 = 15.0 * 15.0;
        let w = &self.game.world;
        for (i, b) in w.buildings.iter().enumerate() {
            let d2 = (x - b.x) * (x - b.x) + (z - b.z) * (z - b.z);
            if d2 < r2 {
                lines.push(format!("bldg[{}] at={:.1},{:.1} size={:.1}x{:.1}x{:.1} dist={:.1}",
                    i, b.x, b.z, b.w * 2.0, b.d * 2.0, b.h, d2.sqrt()));
            }
        }
        for (i, t) in w.trees.iter().enumerate() {
            let d2 = (x - t.x) * (x - t.x) + (z - t.z) * (z - t.z);
            if d2 < r2 { lines.push(format!("tree[{}] at={:.1},{:.1} r={:.2} dist={:.1}", i, t.x, t.z, t.trunk_radius, d2.sqrt())); }
        }
        for (i, r) in w.rocks.iter().enumerate() {
            let d2 = (x - r.x) * (x - r.x) + (z - r.z) * (z - r.z);
            if d2 < r2 { lines.push(format!("rock[{}] at={:.1},{:.1} size={:.2} dist={:.1}", i, r.x, r.z, r.size, d2.sqrt())); }
        }
        for (i, n) in w.npcs.iter().enumerate() {
            let d2 = (x - n.x) * (x - n.x) + (z - n.z) * (z - n.z);
            if d2 < r2 { lines.push(format!("npc[{}] at={:.1},{:.1} job={} state={} dist={:.1}", i, n.x, n.z, n.job.name(), n.state.name(), d2.sqrt())); }
        }
        for (i, v) in w.vehicles.iter().enumerate() {
            let d2 = (x - v.x) * (x - v.x) + (z - v.z) * (z - v.z);
            if d2 < r2 { lines.push(format!("veh[{}] at={:.1},{:.1} parked={} dist={:.1}", i, v.x, v.z, v.parked, d2.sqrt())); }
        }

        lines.join("\n")
    }

    fn cmd_roads(&self) -> String {
        let net = &self.game.road_network;
        let mut car_roads = 0u32;
        let mut field_roads = 0u32;
        let mut car_len = 0.0f32;
        let mut field_len = 0.0f32;
        for seg in &net.segments {
            let len = ((seg.x1 - seg.x0) * (seg.x1 - seg.x0) + (seg.z1 - seg.z0) * (seg.z1 - seg.z0)).sqrt();
            match seg.tier {
                state::RoadTier::CarRoad => { car_roads += 1; car_len += len; }
                state::RoadTier::FieldRoad => { field_roads += 1; field_len += len; }
            }
        }
        let occupied = net.parking_spots.iter().filter(|p| p.occupied_by.is_some()).count();
        format!("segments={} nodes={} car_roads={}/{:.0}m field_roads={}/{:.0}m parking={}/{} occupied",
            net.segments.len(), net.nodes.len(), car_roads, car_len, field_roads, field_len,
            net.parking_spots.len(), occupied)
    }

    fn cmd_river(&self) -> String {
        let segs = &self.game.world.river_segments;
        let mut lines = Vec::new();
        lines.push(format!("segments={}", segs.len()));
        for (i, seg) in segs.iter().enumerate() {
            let len = ((seg.x2 - seg.x1) * (seg.x2 - seg.x1) + (seg.z2 - seg.z1) * (seg.z2 - seg.z1)).sqrt();
            lines.push(format!("[{}] ({:.0},{:.0})->({:.0},{:.0}) len={:.0} w={:.1}",
                i, seg.x1, seg.z1, seg.x2, seg.z2, len, seg.width));
        }
        // Count objects on river
        let bldgs_on = self.game.world.buildings.iter().filter(|b| world::near_river(b.x, b.z, segs, b.w.max(b.d) * 0.5)).count();
        let vehs_on = self.game.world.vehicles.iter().filter(|v| world::on_river(v.x, v.z, segs)).count();
        lines.push(format!("buildings_on_river={} vehicles_on_river={}", bldgs_on, vehs_on));
        lines.join("\n")
    }

    fn cmd_parking(&self) -> String {
        let net = &self.game.road_network;
        let w = &self.game.world;
        let occupied = net.parking_spots.iter().filter(|p| p.occupied_by.is_some()).count();
        let parked = w.vehicles.iter().filter(|v| v.parked).count();
        let ai_driving = w.vehicles.iter().filter(|v| v.ai_active && !v.parked).count();
        format!("spots={} occupied={} free={} vehicles={} parked={} ai_driving={}",
            net.parking_spots.len(), occupied, net.parking_spots.len() - occupied,
            w.vehicles.len(), parked, ai_driving)
    }

    // ---- Physics Diagnostics ----

    fn cmd_skeleton(&self, args: &[&str]) -> String {
        if args.is_empty() { return "err: skeleton player|npc ID".into(); }
        let skel = match args[0].to_ascii_lowercase().as_str() {
            "player" => &self.game.player.skeleton,
            "npc" => {
                let id: usize = match args.get(1).and_then(|s| s.parse().ok()) {
                    Some(v) => v, None => return "err: skeleton npc ID".into(),
                };
                if id >= self.game.world.npcs.len() { return format!("err: npc {} not found", id); }
                &self.game.world.npcs[id].skeleton
            }
            _ => return "err: skeleton player|npc ID".into(),
        };

        let bone_names = ["Hips","Spine","Chest","Neck","Head",
            "LUpperArm","LForearm","RUpperArm","RForearm",
            "LUpperLeg","LLowerLeg","LFoot","RUpperLeg","RLowerLeg","RFoot"];

        let mut lines = Vec::new();
        lines.push(format!("ragdoll={} timer={:.2} blend={:.2} jump_phase={:.2} jump_crouch={:.3}",
            skel.ragdoll_active, skel.ragdoll_timer, skel.ragdoll_blend,
            skel.jump_phase, skel.jump_crouch));

        for (i, bone) in skel.bones.iter().enumerate() {
            let p = bone.world_pos;
            let v = bone.vel;
            let speed = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
            let active = skel.bone_active[i];
            lines.push(format!("[{:2}] {:10} pos={:.2},{:.2},{:.2} vel={:.1} len={:.3} mass={:.1} active={}",
                i, bone_names[i], p[0], p[1], p[2], speed, bone.length, bone.mass, active));
        }

        if let Some(ref vc) = skel.vehicle_contact {
            lines.push(format!("vehicle_contact: veh={} time={:.2} friction={:.2} normal={:.2},{:.2},{:.2}",
                vc.vehicle_idx, vc.time, vc.friction, vc.surface_normal[0], vc.surface_normal[1], vc.surface_normal[2]));
            let contacts: Vec<usize> = vc.bone_contacts.iter().enumerate().filter(|&(_, &c)| c).map(|(i, _)| i).collect();
            lines.push(format!("  contact_bones={:?}", contacts));
        }

        lines.join("\n")
    }

    fn cmd_gait(&self, args: &[&str]) -> String {
        if args.is_empty() { return "err: gait player|npc ID".into(); }
        let skel = match args[0].to_ascii_lowercase().as_str() {
            "player" => &self.game.player.skeleton,
            "npc" => {
                let id: usize = match args.get(1).and_then(|s| s.parse().ok()) {
                    Some(v) => v, None => return "err: gait npc ID".into(),
                };
                if id >= self.game.world.npcs.len() { return format!("err: npc {} not found", id); }
                &self.game.world.npcs[id].skeleton
            }
            _ => return "err: gait player|npc ID".into(),
        };

        let gait_name = match skel.gait {
            skeleton::Gait::Idle => "idle",
            skeleton::Gait::Walk => "walk",
            skeleton::Gait::Run => "run",
            skeleton::Gait::Sprint => "sprint",
        };

        let mut lines = Vec::new();
        lines.push(format!("gait={} blend={:.2} walk_phase={:.3} landing_speed={:.2}",
            gait_name, skel.gait_blend, skel.walk_phase, skel.landing_speed));
        lines.push(format!("com_offset={:.3},{:.3},{:.3} com_world={:.2},{:.2},{:.2}",
            skel.com_offset[0], skel.com_offset[1], skel.com_offset[2],
            skel.com_world[0], skel.com_world[1], skel.com_world[2]));
        lines.push(format!("com_lean={:.3},{:.3},{:.3} push_force={:.1},{:.1},{:.1}",
            skel.com_lean[0], skel.com_lean[1], skel.com_lean[2],
            skel.total_push_force[0], skel.total_push_force[1], skel.total_push_force[2]));
        lines.push(format!("stumble_timer={:.2} stumble_brake={:.2}",
            skel.stumble_timer, skel.stumble_brake));

        for (fi, foot) in skel.feet.iter().enumerate() {
            let side = if fi == 0 { "L" } else { "R" };
            lines.push(format!("foot_{}: grounded={} ground_y={:.2} lift={:.3} target={:.2},{:.2},{:.2} push={:.1},{:.1},{:.1}",
                side, foot.grounded, foot.ground_y, foot.lift_height,
                foot.target_pos[0], foot.target_pos[1], foot.target_pos[2],
                foot.push_force[0], foot.push_force[1], foot.push_force[2]));
        }

        if skel.anticipation_timer > 0.0 {
            lines.push(format!("anticipation: timer={:.2} dir={:.2},{:.2},{:.2}",
                skel.anticipation_timer, skel.anticipation_dir[0], skel.anticipation_dir[1], skel.anticipation_dir[2]));
        }

        lines.join("\n")
    }

    fn cmd_vphysics(&self, args: &[&str]) -> String {
        let id: usize = match args.first().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return "err: vphysics ID".into(),
        };
        if id >= self.game.world.vehicles.len() {
            return format!("err: vehicle {} not found", id);
        }
        let v = &self.game.world.vehicles[id];
        let mut lines = Vec::new();

        // Rigid body
        let b = &v.body;
        lines.push(format!("body: pos={:.2},{:.2},{:.2} vel={:.2},{:.2},{:.2} mass={:.0}",
            b.pos[0], b.pos[1], b.pos[2], b.vel[0], b.vel[1], b.vel[2], b.mass));
        lines.push(format!("  ang_vel={:.3},{:.3},{:.3} quat={:.3},{:.3},{:.3},{:.3}",
            b.ang_vel[0], b.ang_vel[1], b.ang_vel[2],
            b.quat[0], b.quat[1], b.quat[2], b.quat[3]));

        // Drivetrain
        let d = &v.drivetrain;
        lines.push(format!("drivetrain: throttle={:.2} brake={:.2} steer={:.2} handbrake={} engine_torque={:.0} gear={:.2}",
            d.throttle, d.brake, d.steer_input, d.handbrake, d.engine_torque, d.gear_ratio));

        // Speed
        let spd = v.speed;
        let spd_kmh = spd * 3.6;
        lines.push(format!("speed: {:.2} m/s ({:.1} km/h)", spd, spd_kmh));

        // Surface
        let surface = v.surface_override
            .unwrap_or_else(|| world::surface_at(v.body.pos[0], v.body.pos[2], &self.game.road_network));
        let surface_mat = clauding::material::material_for_surface(surface);
        lines.push(format!("surface: {} (friction={:.2} rolling={:.3}){}",
            surf_str(surface), surface_mat.dynamic_friction, surface_mat.rolling_resistance,
            if v.surface_override.is_some() { " [override]" } else { "" }));

        // Wheels
        let wheel_names = ["FL", "FR", "RL", "RR"];
        for (i, wh) in v.wheels.iter().enumerate() {
            lines.push(format!("wheel_{}: ground={} slip={:.3} abs={} brake_t={:.0} ang_vel={:.1} compress={:.3} contact_f={:.1},{:.1},{:.1}",
                wheel_names[i], wh.on_ground, wh.slip_ratio, wh.abs,
                wh.brake_torque, wh.ang_vel, wh.compression,
                wh.contact_force[0], wh.contact_force[1], wh.contact_force[2]));
        }

        // Suspension
        for (i, sus) in v.suspension.iter().enumerate() {
            lines.push(format!("susp_{}: compression={:.3} prev={:.3}",
                wheel_names[i], sus.compression, sus.prev_compression));
        }

        // Deformation
        let def = &v.deformation;
        let total_deform: f32 = def.offsets.iter().sum();
        if total_deform > 0.01 {
            let offsets_str: Vec<String> = def.offsets.iter().map(|o| format!("{:.2}", o)).collect();
            lines.push(format!("deformation: total={:.2} max={:.2} points=[{}]",
                total_deform, def.max_deform, offsets_str.join(",")));
        } else {
            lines.push("deformation: none".into());
        }

        lines.join("\n")
    }

    // ---- Rendering ----

    fn cmd_screenshot(&mut self, args: &[&str]) -> String {
        let path = if let Some(p) = args.first() {
            p.to_string()
        } else {
            self.screenshot_counter += 1;
            format!("debug/studio_{:04}.png", self.screenshot_counter)
        };
        self.take_screenshot(&path)
    }

    fn cmd_render(&mut self, args: &[&str]) -> String {
        if args.len() < 3 { return "err: render W H PATH".into(); }
        let w: usize = match args[0].parse() { Ok(v) => v, Err(_) => return "err: bad W".into() };
        let h: usize = match args[1].parse() { Ok(v) => v, Err(_) => return "err: bad H".into() };
        let path = args[2];
        if w == 0 || h == 0 || w > 7680 || h > 4320 { return "err: dimensions out of range".into(); }
        self.take_screenshot_wh(w, h, path)
    }

    fn cmd_export(&mut self, args: &[&str]) -> String {
        if args.len() < 2 { return "err: export TYPE PATH (types: player, npc N, vehicle N, world, scene)".into(); }
        let kind = args[0].to_ascii_lowercase();
        match kind.as_str() {
            "player" => {
                let path = args[1];
                let mut tris = Vec::new();
                render::gen_player_mesh(&self.game.player, &mut tris);
                write_obj(&tris, path)
            }
            "nude" => {
                let path = args[1];
                let is_female = args.get(2).map_or(false, |s| *s == "f");
                let mut tris = Vec::new();
                render::gen_nude_player_body_export(&mut tris, is_female);
                write_obj(&tris, path)
            }
            "npc" => {
                if args.len() < 3 { return "err: export npc ID PATH".into(); }
                let id: usize = match args[1].parse() { Ok(v) => v, Err(_) => return "err: bad ID".into() };
                let path = args[2];
                if id >= self.game.world.npcs.len() { return format!("err: npc {} not found", id); }
                let mut tris = Vec::new();
                render::gen_npc_mesh(&self.game.world.npcs[id], &mut tris);
                write_obj(&tris, path)
            }
            "vehicle" | "veh" => {
                if args.len() < 3 { return "err: export vehicle ID PATH".into(); }
                let id: usize = match args[1].parse() { Ok(v) => v, Err(_) => return "err: bad ID".into() };
                let path = args[2];
                if id >= self.game.world.vehicles.len() { return format!("err: vehicle {} not found", id); }
                let mut tris = Vec::new();
                render::gen_vehicle_mesh(&self.game.world.vehicles[id], &mut tris, true, false);
                write_obj(&tris, path)
            }
            "world" => {
                let path = args[1];
                write_obj(&self.game.world.static_tris, path)
            }
            "scene" => {
                let path = args[1];
                let mut tris = self.game.world.static_tris.clone();
                render::gen_player_mesh(&self.game.player, &mut tris);
                for n in &self.game.world.npcs { render::gen_npc_mesh(n, &mut tris); }
                for v in &self.game.world.vehicles { render::gen_vehicle_mesh(v, &mut tris, false, false); }
                for item in &self.game.world.items {
                    if item.active { render::gen_item_mesh(item, &mut tris); }
                }
                write_obj(&tris, path)
            }
            _ => "err: export types: player, npc N, vehicle N, world, scene".into(),
        }
    }

    // ---- Analysis commands (write to file) ----

    fn cmd_map(&self, args: &[&str]) -> String {
        let path = args.first().copied().unwrap_or("debug/map.ppm");
        let img_size: usize = 2048;
        let world_size = state::WORLD_SIZE;
        let half = state::WORLD_HALF;

        let mut buf = vec![0u8; img_size * img_size * 3];

        // Terrain heightmap
        for pz in 0..img_size {
            for px in 0..img_size {
                let wx = px as f32 / img_size as f32 * world_size - half;
                let wz = pz as f32 / img_size as f32 * world_size - half;
                let h = self.game.terrain.height_at(wx, wz);
                let t = ((h + 2.0) / 8.0).clamp(0.0, 1.0);
                let idx = (pz * img_size + px) * 3;
                buf[idx]     = (60.0 + t * 80.0) as u8;
                buf[idx + 1] = (80.0 + t * 60.0) as u8;
                buf[idx + 2] = (40.0 + t * 30.0) as u8;
            }
        }

        // River
        for seg in &self.game.world.river_segments {
            ppm_draw_line(&mut buf, img_size, half, world_size, seg.x1, seg.z1, seg.x2, seg.z2, seg.width, 40, 100, 180);
        }

        // Roads
        for seg in &self.game.road_network.segments {
            let (r, g, b, w) = match seg.tier {
                state::RoadTier::CarRoad => (60, 60, 65, state::CAR_ROAD_WIDTH + state::SIDEWALK_WIDTH * 2.0),
                state::RoadTier::FieldRoad => (110, 95, 70, state::FIELD_ROAD_WIDTH),
            };
            ppm_draw_line(&mut buf, img_size, half, world_size, seg.x0, seg.z0, seg.x1, seg.z1, w, r, g, b);
        }

        // Buildings
        for b in &self.game.world.buildings {
            ppm_fill_rect(&mut buf, img_size, half, world_size, b.x, b.z, b.w, b.d, 140, 130, 120);
        }

        // Trees
        for t in &self.game.world.trees {
            ppm_fill_circle(&mut buf, img_size, half, world_size, t.x, t.z, 1.5, 30, 130, 30);
        }

        // NPCs (colored by state)
        for npc in &self.game.world.npcs {
            let (r, g, b) = match npc.state {
                state::NpcState::Working => (50, 255, 50),
                state::NpcState::Sleeping => (100, 100, 150),
                state::NpcState::Driving => (255, 255, 255),
                state::NpcState::KnockedOut => (255, 0, 0),
                _ => (200, 200, 50),
            };
            ppm_fill_circle(&mut buf, img_size, half, world_size, npc.x, npc.z, 1.5, r, g, b);
        }

        // Vehicles
        for v in &self.game.world.vehicles {
            let (r, g, b) = ((v.color >> 16 & 0xFF) as u8, (v.color >> 8 & 0xFF) as u8, (v.color & 0xFF) as u8);
            ppm_fill_circle(&mut buf, img_size, half, world_size, v.x, v.z, 1.2, r, g, b);
        }

        // Items
        for item in &self.game.world.items {
            if !item.active { continue; }
            let (r, g, b) = match item.kind {
                state::ItemKind::Health => (255, 50, 50),
                state::ItemKind::Money => (255, 215, 0),
                state::ItemKind::Stamina => (255, 255, 50),
                state::ItemKind::Food => (255, 160, 50),
                state::ItemKind::Water => (50, 200, 255),
            };
            ppm_fill_circle(&mut buf, img_size, half, world_size, item.x, item.z, 0.8, r, g, b);
        }

        // Write PPM
        let mut ppm = Vec::with_capacity(img_size * img_size * 3 + 50);
        ppm.extend_from_slice(format!("P6\n{} {}\n255\n", img_size, img_size).as_bytes());
        ppm.extend_from_slice(&buf);
        match std::fs::write(path, &ppm) {
            Ok(_) => format!("{} {}x{} {:.1}MB", path, img_size, img_size, ppm.len() as f32 / 1e6),
            Err(e) => format!("err: {}", e),
        }
    }

    /// Render every model: original (full detail) on left, game (decimated) on right.
    /// Loads original fresh from disk for comparison.
    /// Usage: models [arch|tree|all] [dir]
    fn cmd_models(&self, args: &[&str]) -> String {
        let category = args.first().copied().unwrap_or("all");
        let out_dir = args.get(1).copied().unwrap_or("debug");
        let bg = 0xFF404050u32;
        let vw = 384usize;
        let vh = 384usize;

        // Render a model from 2 angles (front + perspective) side by side
        let render_pair = |tris: &[state::WorldTri], label: &str, img_w: usize, img_h: usize,
                           combined: &mut Vec<u32>, col_offset: usize| {
            let (bw, bh, bd) = gltf_loader::measure_bounds(tris);
            let radius = (bw * bw + bh * bh + bd * bd).sqrt() * 0.5;
            let dist = (radius * 2.5).max(0.1);
            let views: [(f32, f32); 2] = [
                (30.0, 20.0),  // perspective
                (0.0, 10.0),   // front
            ];
            for (vi, &(bearing, pitch)) in views.iter().enumerate() {
                let pixels = render::render_model_to_pixels(tris, bearing, pitch, dist, vw, vh, bg);
                let ox = col_offset;
                let oy = vi * vh;
                for y in 0..vh {
                    for x in 0..vw {
                        if ox + x < img_w && oy + y < img_h {
                            combined[(oy + y) * img_w + (ox + x)] = pixels[y * vw + x];
                        }
                    }
                }
            }
            let _ = label;
        };

        let render_comparison = |original_tris: &[state::WorldTri], game_tris: &[state::WorldTri],
                                  name: &str, path: &str| {
            let img_w = vw * 2; // left = original, right = game
            let img_h = vh * 2; // 2 rows (perspective + front)
            let mut combined = vec![bg; img_w * img_h];
            render_pair(original_tris, "ORIGINAL", img_w, img_h, &mut combined, 0);
            render_pair(game_tris, "GAME", img_w, img_h, &mut combined, vw);
            let (ow, oh, od) = gltf_loader::measure_bounds(original_tris);
            let (gw, gh, gd) = gltf_loader::measure_bounds(game_tris);
            image::save_png(&combined, img_w, img_h, path);
            eprintln!("  {} -> {} (orig: {} tris {:.1}x{:.1}x{:.1}m | game: {} tris {:.1}x{:.1}x{:.1}m)",
                name, path, original_tris.len(), ow, oh, od, game_tris.len(), gw, gh, gd);
        };

        let mut count = 0;
        if category == "all" || category == "arch" {
            // Load originals fresh from disk for comparison
            let arch_dirs = gltf_loader::discover_model_dirs("models/v1/architecture");
            for (i, entry) in self.game.model_library.architecture.iter().enumerate() {
                // Find matching original directory by name
                let orig_dir = arch_dirs.iter().find(|d| d.ends_with(&entry.name));
                let original = if let Some(dir) = orig_dir {
                    gltf_loader::try_load_gltf_scaled(dir, &entry.name, 0xFFCCBBAA, 8.0)
                        .map(|m| m.tris)
                } else { None };
                let orig_tris = original.as_deref().unwrap_or(&entry.tris);
                let path = format!("{}/model_arch_{:02}_{}.png", out_dir, i, entry.name.replace(' ', "_"));
                render_comparison(orig_tris, &entry.tris, &entry.name, &path);
                count += 1;
            }
        }
        if category == "all" || category == "tree" {
            let max_trees = 30;
            for (i, entry) in self.game.model_library.trees.iter().take(max_trees).enumerate() {
                // Trees are split islands — no original dir to load from
                // Just render game version from 2 angles
                let path = format!("{}/model_tree_{:02}_{}.png", out_dir, i,
                    entry.name.replace(' ', "_").chars().take(40).collect::<String>());
                render_comparison(&entry.tris, &entry.tris, &entry.name, &path);
                count += 1;
            }
        }
        format!("rendered {} models to {}/", count, out_dir)
    }

    // ---- Character model commands ----

    fn cmd_model(&self, args: &[&str]) -> String {
        if args.is_empty() { return "err: model list|view|viewall|obj".into(); }
        let sub = args[0].to_ascii_lowercase();
        let sub_args = &args[1..];
        match sub.as_str() {
            "list" => self.cmd_model_list(),
            "view" => self.cmd_model_view(sub_args),
            "viewall" => self.cmd_model_viewall(),
            "obj" => self.cmd_model_obj(sub_args),
            _ => format!("err: unknown model subcommand '{}'. try: list, view, viewall, obj", sub),
        }
    }

    fn cmd_model_list(&self) -> String {
        let names = &self.game.model_library.character_names;
        let models = &self.game.character_models;
        if models.is_empty() {
            return "no character models loaded".into();
        }
        let mut out = String::new();
        for (i, tris) in models.iter().enumerate() {
            let name = names.get(i).map(|s| s.as_str()).unwrap_or("?");
            let has_anim = self.game.animation_data.as_ref()
                .map_or(false, |ad| ad.is_ready()
                    && ad.model_bone_assignments.get(i).map_or(false, |a| !a.is_empty()));
            let _ = write!(out, "[{}] {} ({} tris{})\n",
                i, name, tris.len(),
                if has_anim { ", rigged" } else { "" });
        }
        out.trim_end().to_string()
    }

    fn resolve_model_index(&self, key: &str) -> Option<usize> {
        // Try as numeric index first
        if let Ok(idx) = key.parse::<usize>() {
            if idx < self.game.character_models.len() { return Some(idx); }
        }
        // Try as name (case-insensitive substring match)
        let key_lower = key.to_ascii_lowercase();
        for (i, name) in self.game.model_library.character_names.iter().enumerate() {
            if name.to_ascii_lowercase().contains(&key_lower) { return Some(i); }
        }
        None
    }

    fn render_model_sidebyside(&self, model_idx: usize) -> (String, Vec<u32>, usize, usize) {
        let names = &self.game.model_library.character_names;
        let bind_tris = &self.game.character_models[model_idx];
        let name = names.get(model_idx).map(|s| s.as_str()).unwrap_or("unknown");

        let half_w = 800usize;
        let img_h = 1000usize;
        let img_w = half_w * 2; // 1600
        let bg = 0xFF1A1A2Eu32;
        let bearing = 45.0f32;
        let pitch = 10.0f32;
        let dist = 4.0f32;

        // Left half: bind pose
        let left_pixels = render::render_model_to_pixels(bind_tris, bearing, pitch, dist, half_w, img_h, bg);

        // Right half: animated pose (walking at t=0.5) if rigged, else bind pose again
        let right_pixels = if let Some(anim_data) = self.game.animation_data.as_ref() {
            let has_assignments = anim_data.is_ready()
                && anim_data.model_bone_assignments.get(model_idx).map_or(false, |a| !a.is_empty());
            if has_assignments {
                let mut animated = Vec::new();
                anim_data.generate_animated_mesh(bind_tris, model_idx, skeleton_anim::CLIP_WALKING, 0.5, &mut animated);
                render::render_model_to_pixels(&animated, bearing, pitch, dist, half_w, img_h, bg)
            } else {
                render::render_model_to_pixels(bind_tris, bearing, pitch, dist, half_w, img_h, bg)
            }
        } else {
            render::render_model_to_pixels(bind_tris, bearing, pitch, dist, half_w, img_h, bg)
        };

        // Composite side by side
        let mut combined = vec![bg; img_w * img_h];
        for y in 0..img_h {
            for x in 0..half_w {
                combined[y * img_w + x] = left_pixels[y * half_w + x];
                combined[y * img_w + half_w + x] = right_pixels[y * half_w + x];
            }
        }

        // Draw label bar at top (simple text: write model name + tri count as a colored strip)
        let label_h = 24usize;
        let label_bg = 0xFF000000u32;
        for y in 0..label_h.min(img_h) {
            for x in 0..img_w {
                combined[y * img_w + x] = label_bg;
            }
        }
        // Render a simple text indicator via colored blocks for "Bind" and "Anim" labels
        // Left half center: small cyan block at (half_w/2 - 20, 8)
        for x in (half_w / 2 - 20)..(half_w / 2 + 20) {
            for y in 8..16 {
                if y < label_h && x < half_w {
                    combined[y * img_w + x] = 0xFF88CCFF; // cyan = bind pose
                }
            }
        }
        // Right half center: small green block
        for x in (half_w + half_w / 2 - 20)..(half_w + half_w / 2 + 20) {
            for y in 8..16 {
                if y < label_h && x < img_w {
                    combined[y * img_w + x] = 0xFF88FF88; // green = animated
                }
            }
        }

        let info = format!("{} ({} tris)", name, bind_tris.len());
        (info, combined, img_w, img_h)
    }

    fn cmd_model_view(&self, args: &[&str]) -> String {
        if args.is_empty() { return "err: model view <index|name>".into(); }
        let key = args[0];
        let Some(idx) = self.resolve_model_index(key) else {
            return format!("err: model '{}' not found. use 'model list' to see available models", key);
        };
        let name = self.game.model_library.character_names.get(idx).map(|s| s.as_str()).unwrap_or("unknown");
        let path = format!("debug/model_{}.png", name);
        let (info, pixels, w, h) = self.render_model_sidebyside(idx);
        image::save_png(&pixels, w, h, &path);
        format!("{} -> {} ({}x{})", info, path, w, h)
    }

    fn cmd_model_viewall(&self) -> String {
        if self.game.character_models.is_empty() {
            return "no character models loaded".into();
        }
        let mut out = String::new();
        for i in 0..self.game.character_models.len() {
            let name = self.game.model_library.character_names.get(i).map(|s| s.as_str()).unwrap_or("unknown");
            let path = format!("debug/model_{}.png", name);
            let (info, pixels, w, h) = self.render_model_sidebyside(i);
            image::save_png(&pixels, w, h, &path);
            let _ = writeln!(out, "{} -> {} ({}x{})", info, path, w, h);
        }
        out.trim_end().to_string()
    }

    fn cmd_model_obj(&self, args: &[&str]) -> String {
        if args.len() < 2 { return "err: model obj <index|name> <path>".into(); }
        let key = args[0];
        let path = args[1];
        let Some(idx) = self.resolve_model_index(key) else {
            return format!("err: model '{}' not found. use 'model list' to see available models", key);
        };
        let tris = &self.game.character_models[idx];
        write_obj(tris, path)
    }

    fn cmd_fbx(&self, args: &[&str]) -> String {
        if args.is_empty() { return "err: fbx inspect <path>|skin <path>".into(); }
        let sub = args[0].to_ascii_lowercase();
        match sub.as_str() {
            "inspect" => {
                let path = args.get(1).copied().unwrap_or("models/v1/animations/walking.fbx");
                clauding::fbx_anim::inspect_fbx(path)
            }
            "skin" => {
                let path = args.get(1).copied().unwrap_or("models/v1/animations/walking.fbx");
                match clauding::fbx_anim::extract_skin_data(path) {
                    Some(skin) => {
                        let mut out = format!("Skin data: {} clusters, {} mesh verts\n",
                            skin.clusters.len(), skin.vertex_count);
                        for (i, c) in skin.clusters.iter().enumerate() {
                            out.push_str(&format!("  [{}] bone='{}' verts={} weights={}\n",
                                i, c.bone_name, c.vertex_indices.len(), c.weights.len()));
                            if !c.vertex_indices.is_empty() {
                                let max_idx = c.vertex_indices.iter().max().unwrap_or(&0);
                                let min_w = c.weights.iter().cloned().fold(f64::MAX, f64::min);
                                let max_w = c.weights.iter().cloned().fold(f64::MIN, f64::max);
                                out.push_str(&format!("    idx_range=[0..{}] weight_range=[{:.4}..{:.4}]\n",
                                    max_idx, min_w, max_w));
                            }
                        }
                        out
                    }
                    None => "No skin data found in FBX".to_string(),
                }
            }
            _ => format!("err: unknown fbx subcommand '{}'. try: inspect, skin", sub),
        }
    }

    fn cmd_inspect(&self, args: &[&str]) -> String {
        let path = args.first().copied().unwrap_or("debug/inspect.txt");
        let mut out = String::with_capacity(32000);
        let w = &self.game.world;

        let _ = writeln!(out, "=== WORLD INSPECT (seed={}) ===", self.game.world_seed);

        // Object counts
        let _ = writeln!(out, "buildings={} trees={} rocks={} lights={} bins={} walls={} interactibles={}",
            w.buildings.len(), w.trees.len(), w.rocks.len(), w.street_lights.len(),
            w.trash_bins.len(), w.walls.len(), w.interactibles.len());
        let _ = writeln!(out, "npcs={} vehicles={} items={}/{} river_segments={}",
            w.npcs.len(), w.vehicles.len(),
            w.items.iter().filter(|i| i.active).count(), w.items.len(),
            w.river_segments.len());
        let _ = writeln!(out, "roads={} nodes={} parking={} static_tris={}",
            self.game.road_network.segments.len(), self.game.road_network.nodes.len(),
            self.game.road_network.parking_spots.len(), w.static_tris.len());

        // Placement audit
        let _ = writeln!(out, "\n=== PLACEMENT AUDIT ===");
        let bldgs_on_river = w.buildings.iter().filter(|b| world::on_river(b.x, b.z, &w.river_segments)).count();
        let trees_on_river = w.trees.iter().filter(|t| world::on_river(t.x, t.z, &w.river_segments)).count();
        let vehs_on_river = w.vehicles.iter().filter(|v| world::on_river(v.x, v.z, &w.river_segments)).count();
        let _ = writeln!(out, "on_river: buildings={} trees={} vehicles={}", bldgs_on_river, trees_on_river, vehs_on_river);

        let parked = w.vehicles.iter().filter(|v| v.parked).count();
        let _ = writeln!(out, "vehicles: parked={}/{}", parked, w.vehicles.len());

        // Walls
        let _ = writeln!(out, "\nwalls ({}):", w.walls.len());
        for (i, wall) in w.walls.iter().enumerate() {
            let _ = writeln!(out, "  [{}] pos={:.1},{:.1} hw={:.2} hd={:.2} h={:.1}", i, wall.x, wall.z, wall.hw, wall.hd, wall.height);
        }

        // NPC homes
        let _ = writeln!(out, "\n=== NPC HOMES ===");
        let mut building_use = vec![0u32; w.buildings.len()];
        for npc in &w.npcs {
            if npc.home_idx < building_use.len() { building_use[npc.home_idx] += 1; }
        }
        let used = building_use.iter().filter(|&&c| c > 0).count();
        let multi = building_use.iter().filter(|&&c| c > 1).count();
        let _ = writeln!(out, "used={}/{} multi_npc={}", used, w.buildings.len(), multi);

        // Parking lot clusters
        let _ = writeln!(out, "\n=== PARKING LOTS ===");
        let spots = &self.game.road_network.parking_spots;
        let mut visited = vec![false; spots.len()];
        let mut lot_id = 0u32;
        for i in 0..spots.len() {
            if visited[i] { continue; }
            let mut cluster = vec![i];
            visited[i] = true;
            let mut j = 0;
            while j < cluster.len() {
                let ci = cluster[j];
                for k in 0..spots.len() {
                    if visited[k] { continue; }
                    let dx = spots[ci].x - spots[k].x;
                    let dz = spots[ci].z - spots[k].z;
                    if dx * dx + dz * dz < 400.0 {
                        cluster.push(k);
                        visited[k] = true;
                    }
                }
                j += 1;
            }
            if cluster.len() >= 6 {
                let avg_x: f32 = cluster.iter().map(|&i| spots[i].x).sum::<f32>() / cluster.len() as f32;
                let avg_z: f32 = cluster.iter().map(|&i| spots[i].z).sum::<f32>() / cluster.len() as f32;
                let occ = cluster.iter().filter(|&&i| spots[i].occupied_by.is_some()).count();
                let _ = writeln!(out, "lot{} at=({:.0},{:.0}) spots={} occupied={}", lot_id, avg_x, avg_z, cluster.len(), occ);
                lot_id += 1;
            }
        }

        match std::fs::write(path, &out) {
            Ok(_) => format!("{} {} bytes", path, out.len()),
            Err(e) => format!("err: {}", e),
        }
    }

    fn cmd_winding(&self, args: &[&str]) -> String {
        let filter = args.first().copied().unwrap_or("all");
        let do_static = matches!(filter, "all" | "static");
        let do_npc = matches!(filter, "all" | "npc");
        let do_vehicle = matches!(filter, "all" | "vehicle");
        let do_item = matches!(filter, "all" | "item");
        let do_bin = matches!(filter, "all" | "bin");
        let do_player = matches!(filter, "all" | "player");

        if !matches!(filter, "all" | "static" | "npc" | "vehicle" | "item" | "bin" | "player") {
            return format!("err: unknown filter '{}'. use all|static|npc|vehicle|item|bin|player", filter);
        }

        let mut results: Vec<(String, u32, u32, u32, u32)> = Vec::new(); // (name, total, match, mismatch, degen)

        if do_static {
            let (t, m, mm, d) = audit_winding(&self.game.world.static_tris);
            results.push(("static".into(), t, m, mm, d));
        }
        if do_npc {
            let mut total = (0u32, 0u32, 0u32, 0u32);
            for npc in &self.game.world.npcs {
                let mut tris = Vec::new();
                render::gen_npc_mesh(npc, &mut tris);
                let (t, m, mm, d) = audit_winding(&tris);
                total.0 += t; total.1 += m; total.2 += mm; total.3 += d;
            }
            results.push((format!("npc({})", self.game.world.npcs.len()), total.0, total.1, total.2, total.3));
        }
        if do_vehicle {
            let mut total = (0u32, 0u32, 0u32, 0u32);
            for v in &self.game.world.vehicles {
                let mut tris = Vec::new();
                render::gen_vehicle_mesh(v, &mut tris, false, false);
                let (t, m, mm, d) = audit_winding(&tris);
                total.0 += t; total.1 += m; total.2 += mm; total.3 += d;
            }
            results.push((format!("vehicle({})", self.game.world.vehicles.len()), total.0, total.1, total.2, total.3));
        }
        if do_item {
            let mut total = (0u32, 0u32, 0u32, 0u32);
            for item in &self.game.world.items {
                let mut tris = Vec::new();
                render::gen_item_mesh(item, &mut tris);
                let (t, m, mm, d) = audit_winding(&tris);
                total.0 += t; total.1 += m; total.2 += mm; total.3 += d;
            }
            results.push((format!("item({})", self.game.world.items.len()), total.0, total.1, total.2, total.3));
        }
        if do_bin {
            let mut total = (0u32, 0u32, 0u32, 0u32);
            for bin in &self.game.world.trash_bins {
                let mut tris = Vec::new();
                render::gen_trash_bin_mesh(bin, &mut tris);
                let (t, m, mm, d) = audit_winding(&tris);
                total.0 += t; total.1 += m; total.2 += mm; total.3 += d;
            }
            results.push((format!("bin({})", self.game.world.trash_bins.len()), total.0, total.1, total.2, total.3));
        }
        if do_player {
            let mut tris = Vec::new();
            render::gen_player_mesh(&self.game.player, &mut tris);
            let (t, m, mm, d) = audit_winding(&tris);
            results.push(("player".into(), t, m, mm, d));
        }

        let mut lines = Vec::new();
        let mut grand = (0u32, 0u32, 0u32, 0u32);
        for (name, t, m, mm, d) in &results {
            lines.push(format!("{}: total={} match={} mismatch={} degen={}", name, t, m, mm, d));
            grand.0 += t; grand.1 += m; grand.2 += mm; grand.3 += d;
        }
        if grand.0 > 0 {
            let match_pct = grand.1 as f64 / grand.0 as f64 * 100.0;
            lines.push(format!("TOTAL: {} tris, {:.1}% match, {} mismatch, {} degen",
                grand.0, match_pct, grand.2, grand.3));
        }
        lines.join("\n")
    }

    fn cmd_compare(&mut self, args: &[&str]) -> String {
        let path = args.first().copied().unwrap_or("debug/render_compare.png");
        let cw: usize = 960;
        let ch: usize = 540;

        let eye = [self.game.camera.x, self.game.camera.y, self.game.camera.z];
        let target = [self.game.camera.tx, self.game.camera.ty, self.game.camera.tz];
        let time = self.game.time_of_day;

        // CPU render
        let mut cpu_fb = raster::Framebuffer::new(cw, ch);
        cpu_fb.clear(render::sky_color(time));
        let mut scratch: Vec<state::WorldTri> = Vec::new();
        let cam = state::Camera {
            x: eye[0], y: eye[1], z: eye[2],
            tx: target[0], ty: target[1], tz: target[2],
            yaw: self.game.camera.yaw, pitch: self.game.camera.pitch,
        };
        render::sys_render(&mut cpu_fb, &self.game.world, &self.game.player, &cam, time, &mut scratch, &self.game.character_models, self.game.animation_data.as_ref(), &self.game.model_library.cars);

        // GPU render
        let mut gpu_fb = raster::Framebuffer::new(cw, ch);
        let gpu_ok = self.gpu_ctx.as_ref().is_some_and(|g| g.has_graphics());
        if gpu_ok {
            let ctx = self.gpu_ctx.as_mut().unwrap();
            ctx.resize_render_target(cw as u32, ch as u32);
            render::generate_static_gpu_vertices(&self.game.world, &mut self.static_verts);
            ctx.upload_static_vertices(&self.static_verts);
            render::generate_dynamic_gpu_vertices(
                &self.game.world, &self.game.player, &cam,
                &mut scratch, &mut self.dynamic_verts, time,
                &self.game.character_models,
                self.game.animation_data.as_ref(),
                &self.game.model_library.cars,
            );
            let (_vp, push, clear) = render::frame_setup(cw, ch, eye, target, time);
            ctx.render_frame(&self.dynamic_verts, &push, clear, cw as u32, ch as u32, &mut gpu_fb.pixels);
            ctx.render_frame(&self.dynamic_verts, &push, clear, cw as u32, ch as u32, &mut gpu_fb.pixels);
        }

        // Composite side-by-side
        let comp_w = cw * 2 + 4;
        let mut composite = vec![0xFF222222u32; comp_w * ch];
        for y in 0..ch {
            for x in 0..cw { composite[y * comp_w + x] = cpu_fb.pixels[y * cw + x]; }
            for dx in 0..4 { composite[y * comp_w + cw + dx] = 0xFFFFFFFF; }
            for x in 0..cw { composite[y * comp_w + cw + 4 + x] = gpu_fb.pixels[y * cw + x]; }
        }

        image::save_png(&composite, comp_w, ch, path);

        // Pixel diff analysis
        if gpu_ok {
            let mut diff_pixels = 0u32;
            let mut max_diff = 0u32;
            let pixel_count = cw * ch;
            for i in 0..pixel_count {
                let c = cpu_fb.pixels[i];
                let g = gpu_fb.pixels[i];
                let dr = ((c >> 16) & 0xFF) as i32 - ((g >> 16) & 0xFF) as i32;
                let dg = ((c >> 8) & 0xFF) as i32 - ((g >> 8) & 0xFF) as i32;
                let db = (c & 0xFF) as i32 - (g & 0xFF) as i32;
                let d = (dr.unsigned_abs() + dg.unsigned_abs() + db.unsigned_abs()) as u32;
                if d > max_diff { max_diff = d; }
                if d > 0 { diff_pixels += 1; }
            }
            let diff_pct = diff_pixels as f64 / pixel_count as f64 * 100.0;
            format!("{} {}x{} diff_pixels={} ({:.1}%) max_diff={}", path, comp_w, ch, diff_pixels, diff_pct, max_diff)
        } else {
            format!("{} {}x{} (no GPU, right panel blank)", path, comp_w, ch)
        }
    }

    fn cmd_sim(&mut self, args: &[&str]) -> String {
        let days: u32 = match args.first().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return "err: sim DAYS [PATH]".into(),
        };
        let days = days.min(30); // safety cap
        let path = args.get(1).copied().unwrap_or("debug/sim.txt");

        eprintln!("studio: sim {} days...", days);
        let start = std::time::Instant::now();

        // Use headless tick for speed
        let dt = state::HEADLESS_DT;
        let n_npcs = self.game.world.npcs.len();
        let mut prev_pos: Vec<(f32, f32)> = self.game.world.npcs.iter().map(|n| (n.x, n.z)).collect();
        let mut npc_dist: Vec<f32> = vec![0.0; n_npcs];
        let mut tick: u64 = 0;
        let start_day = self.game.day_count;

        while self.game.day_count < start_day + days {
            self.game.tick_headless(dt);
            tick += 1;

            if tick % 10 == 0 {
                for i in 0..n_npcs {
                    let n = &self.game.world.npcs[i];
                    let dx = n.x - prev_pos[i].0;
                    let dz = n.z - prev_pos[i].1;
                    npc_dist[i] += (dx * dx + dz * dz).sqrt();
                    prev_pos[i] = (n.x, n.z);
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f32();

        // Generate report
        let mut out = String::with_capacity(16000);
        let _ = writeln!(out, "=== SIM REPORT ({} days, {:.1}s, {} ticks) ===", days, elapsed, tick);

        // State distribution
        let state_names = ["Sleeping","HomeTask","GoingToWork","Working","GoingHome","Driving","Interacting","KnockedOut"];
        let mut states = [0u32; 8];
        for n in &self.game.world.npcs { states[n.state.index()] += 1; }
        let _ = write!(out, "states:");
        for (i, name) in state_names.iter().enumerate() {
            if states[i] > 0 { let _ = write!(out, " {}={}", name, states[i]); }
        }
        let _ = writeln!(out);

        // Survival
        let avg_hunger = self.game.world.npcs.iter().map(|n| n.hunger).sum::<f32>() / n_npcs.max(1) as f32;
        let avg_thirst = self.game.world.npcs.iter().map(|n| n.thirst).sum::<f32>() / n_npcs.max(1) as f32;
        let dead = self.game.world.npcs.iter().filter(|n| n.starving_dead).count();
        let ko = self.game.world.npcs.iter().filter(|n| n.state == state::NpcState::KnockedOut).count();
        let _ = writeln!(out, "survival: hunger_avg={:.0} thirst_avg={:.0} dead={} ko={}", avg_hunger, avg_thirst, dead, ko);

        // Item economy
        let items_active = self.game.world.items.iter().filter(|i| i.active).count();
        let total_money: f32 = self.game.world.npcs.iter().map(|n| n.money).sum();
        let _ = writeln!(out, "items: active={}/{} npc_money_total={:.0}", items_active, state::NUM_ITEMS, total_money);

        // Movement stats (top/bottom 5)
        let mut dists: Vec<(usize, f32)> = (0..n_npcs).map(|i| (i, npc_dist[i])).collect();
        dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let _ = writeln!(out, "most_mobile:");
        for &(i, d) in dists.iter().take(5) {
            let n = &self.game.world.npcs[i];
            let _ = writeln!(out, "  npc[{}] dist={:.0}m job={} state={}", i, d, n.job.name(), n.state.name());
        }
        let _ = writeln!(out, "least_mobile:");
        for &(i, d) in dists.iter().rev().take(5) {
            let n = &self.game.world.npcs[i];
            let _ = writeln!(out, "  npc[{}] dist={:.0}m job={} state={}", i, d, n.job.name(), n.state.name());
        }

        // Vehicles
        let vehs_moving = self.game.world.vehicles.iter().filter(|v| v.speed.abs() > 0.5).count();
        let vehs_parked = self.game.world.vehicles.iter().filter(|v| v.parked).count();
        let _ = writeln!(out, "vehicles: moving={} parked={}", vehs_moving, vehs_parked);

        match std::fs::write(path, &out) {
            Ok(_) => format!("{} {} days {:.1}s {} ticks", path, days, elapsed, tick),
            Err(e) => format!("err: {}", e),
        }
    }

    // ---- Vehicle Testing Commands ----

    fn cmd_set(&mut self, args: &[&str]) -> String {
        if args.len() < 2 { return "err: set speed N | set surface TYPE".into(); }
        match args[0] {
            "speed" => {
                let spd: f32 = match args[1].parse() {
                    Ok(v) => v,
                    Err(_) => return "err: set speed N (m/s)".into(),
                };
                let vi = match self.game.player.in_vehicle {
                    Some(v) => v,
                    None => return "err: not in vehicle".into(),
                };
                let v = &mut self.game.world.vehicles[vi];
                let fwd = clauding::math::quat_forward(v.body.quat);
                v.body.vel = clauding::math::v3_scale(fwd, spd);
                v.speed = spd;
                // Sync wheel angular velocity to match ground speed
                for w in &mut v.wheels {
                    w.ang_vel = spd / w.radius;
                }
                format!("ok speed={:.2} m/s ({:.1} km/h)", spd, spd * 3.6)
            }
            "surface" => {
                let vi = match self.game.player.in_vehicle {
                    Some(v) => v,
                    None => return "err: not in vehicle".into(),
                };
                let surface = match args[1].to_ascii_lowercase().as_str() {
                    "asphalt" | "road" => Some(state::Surface::CarRoad),
                    "concrete" | "sidewalk" => Some(state::Surface::Sidewalk),
                    "grass" | "terrain" => Some(state::Surface::Terrain),
                    "gravel" | "dirt" | "field" => Some(state::Surface::FieldRoad),
                    "none" | "off" | "auto" => None,
                    _ => return format!("err: unknown surface '{}'. use asphalt|grass|gravel|sidewalk|none", args[1]),
                };
                self.game.world.vehicles[vi].surface_override = surface;
                match surface {
                    Some(s) => format!("ok surface={}", surf_str(s)),
                    None => "ok surface=auto".into(),
                }
            }
            _ => format!("err: unknown set target '{}'. use speed|surface", args[0]),
        }
    }

    fn cmd_throttle(&mut self, args: &[&str]) -> String {
        let val: f32 = match args.first().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return "err: throttle N (-1.0 to 1.0)".into(),
        };
        self.direct_input.active = true;
        self.direct_input.throttle = val.clamp(-1.0, 1.0);
        format!("ok throttle={:.2}", self.direct_input.throttle)
    }

    fn cmd_brake(&mut self, args: &[&str]) -> String {
        let val: f32 = match args.first().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return "err: brake N (0.0 to 1.0)".into(),
        };
        self.direct_input.active = true;
        self.direct_input.brake = val.clamp(0.0, 1.0);
        format!("ok brake={:.2}", self.direct_input.brake)
    }

    fn cmd_handbrake(&mut self, args: &[&str]) -> String {
        match args.first().copied().unwrap_or("on") {
            "on" | "1" | "true" => {
                self.direct_input.active = true;
                self.direct_input.handbrake = true;
            }
            "off" | "0" | "false" => {
                self.direct_input.handbrake = false;
            }
            _ => return "err: handbrake on|off".into(),
        }
        format!("ok handbrake={}", self.direct_input.handbrake)
    }

    fn cmd_steer(&mut self, args: &[&str]) -> String {
        let val: f32 = match args.first().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return "err: steer N (-1.0 to 1.0)".into(),
        };
        self.direct_input.active = true;
        self.direct_input.steer = val.clamp(-1.0, 1.0);
        format!("ok steer={:.2}", self.direct_input.steer)
    }

    fn cmd_inputs(&mut self, args: &[&str]) -> String {
        match args.first().copied().unwrap_or("off") {
            "off" | "0" | "reset" => {
                self.direct_input = DirectInput::new();
                "ok inputs=keyboard".into()
            }
            _ => "err: inputs off".into(),
        }
    }

    fn cmd_mark(&mut self) -> String {
        let pos = if let Some(vi) = self.game.player.in_vehicle {
            let v = &self.game.world.vehicles[vi];
            [v.x, v.y, v.z]
        } else {
            [self.game.player.x, self.game.player.y, self.game.player.z]
        };
        self.mark_pos = Some(pos);
        format!("ok mark={:.2},{:.2},{:.2}", pos[0], pos[1], pos[2])
    }

    fn cmd_distance(&self) -> String {
        let mark = match self.mark_pos {
            Some(m) => m,
            None => return "err: no mark set. use 'mark' first".into(),
        };
        let pos = if let Some(vi) = self.game.player.in_vehicle {
            let v = &self.game.world.vehicles[vi];
            [v.x, v.y, v.z]
        } else {
            [self.game.player.x, self.game.player.y, self.game.player.z]
        };
        let dx = pos[0] - mark[0];
        let dy = pos[1] - mark[1];
        let dz = pos[2] - mark[2];
        let dist_3d = (dx * dx + dy * dy + dz * dz).sqrt();
        let dist_2d = (dx * dx + dz * dz).sqrt();
        format!("dist_2d={:.2}m dist_3d={:.2}m dx={:.2} dy={:.2} dz={:.2}", dist_2d, dist_3d, dx, dy, dz)
    }

    fn cmd_log(&mut self, args: &[&str]) -> String {
        let n: u32 = match args.first().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => return "err: log N [FILE]".into(),
        };
        let n = n.min(100_000);
        let path = args.get(1).copied().unwrap_or("debug/vlog.csv");
        let vi = match self.game.player.in_vehicle {
            Some(v) => v,
            None => return "err: not in vehicle".into(),
        };

        let mut csv = String::with_capacity(n as usize * 200);
        csv.push_str("tick,speed_ms,speed_kmh,x,z,throttle,brake,handbrake,slip_FL,slip_FR,slip_RL,slip_RR,abs_FL,abs_FR,abs_RL,abs_RR,ang_FL,ang_FR,ang_RL,ang_RR\n");

        for tick in 0..n {
            self.run_full_tick();
            let v = &self.game.world.vehicles[vi];
            let _ = write!(csv, "{},{:.3},{:.1},{:.3},{:.3},{:.2},{:.2},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{:.1},{:.1},{:.1},{:.1}\n",
                tick, v.speed, v.speed * 3.6, v.x, v.z,
                v.drivetrain.throttle, v.drivetrain.brake, v.drivetrain.handbrake as u8,
                v.wheels[0].slip_ratio, v.wheels[1].slip_ratio, v.wheels[2].slip_ratio, v.wheels[3].slip_ratio,
                v.wheels[0].abs as u8, v.wheels[1].abs as u8, v.wheels[2].abs as u8, v.wheels[3].abs as u8,
                v.wheels[0].ang_vel, v.wheels[1].ang_vel, v.wheels[2].ang_vel, v.wheels[3].ang_vel,
            );
        }

        match std::fs::write(path, &csv) {
            Ok(_) => format!("ok {} ticks -> {}", n, path),
            Err(e) => format!("err: {}", e),
        }
    }

    fn cmd_test(&mut self, args: &[&str]) -> String {
        let test = args.first().copied().unwrap_or("");
        let vi = match self.ensure_in_vehicle() {
            Ok(v) => v,
            Err(e) => return e,
        };

        match test {
            "accel" => self.test_accel(vi),
            "brake" => self.test_brake(vi),
            "topspeed" => self.test_topspeed(vi),
            _ => "err: test accel|brake|topspeed".into(),
        }
    }

    /// Reset vehicle to testcourse start: pos=(0, Y, -200), facing south (+Z), zero velocity
    /// Y is computed from suspension equilibrium so springs don't launch/drop the car
    fn reset_vehicle_to_start(&mut self, vi: usize) {
        let v = &mut self.game.world.vehicles[vi];
        v.x = 0.0;
        v.z = -200.0;
        v.body.vel = [0.0; 3];
        v.body.ang_vel = [0.0; 3];
        v.body.quat = clauding::math::quat_from_rot_y(std::f32::consts::PI);
        v.rot_y = std::f32::consts::PI;
        v.speed = 0.0;
        for w in &mut v.wheels { w.ang_vel = 0.0; }

        // Compute correct body height from suspension equilibrium (same as cold-start)
        let ground_y = self.game.terrain.height_at(0.0, -200.0);
        let wr = v.wheels[0].radius;
        let local_y = v.wheels[0].local_pos[1];
        let equil_comp = (v.body.mass * 9.81 * 0.25) / v.suspension[0].params.spring_rate;
        let equil_length = v.suspension[0].params.rest_length - equil_comp;
        let body_y = ground_y + wr + equil_length - local_y;
        v.y = body_y;
        v.body.pos = [0.0, body_y, -200.0];

        // Set suspension to equilibrium so cold-start detection doesn't fire
        for s in &mut v.suspension {
            let eq = (v.body.mass * 9.81 * 0.25) / s.params.spring_rate;
            s.compression = eq;
            s.prev_compression = eq;
        }
        // Clear crash damage so tests are repeatable
        v.deformation = clauding::deform::VehicleDeformation::new();
    }

    /// Clear the test area: deactivate AI vehicles, move NPCs out, flatten terrain
    fn setup_test_env(&mut self, vi: usize) {
        self.game.world.vehicles[vi].surface_override = Some(state::Surface::CarRoad);
        flatten_terrain_around(&mut self.game.terrain, 0.0, 0.0, 600.0);
        // Move NPCs and other vehicles completely out of the way
        for (i, v) in self.game.world.vehicles.iter_mut().enumerate() {
            if i != vi {
                v.ai_active = false;
                v.parked = true;
                v.speed = 0.0;
                // Teleport to corner so no collision with test vehicle
                v.x = 245.0; v.z = 245.0;
                v.body.pos = [245.0, 0.0, 245.0];
            }
        }
        for npc in &mut self.game.world.npcs {
            npc.x = 245.0; npc.z = 245.0;
            npc.body.pos = [245.0, 0.0, 245.0];
        }
        // Move all static obstacles out of the test area
        for b in &mut self.game.world.buildings {
            b.x = 245.0; b.z = 245.0;
        }
        for w in &mut self.game.world.walls {
            w.x = 245.0; w.z = 245.0;
        }
        for sl in &mut self.game.world.street_lights {
            sl.x = 245.0; sl.z = 245.0;
        }
        // Reset player health and vehicle deformation
        self.game.player.health = 100.0;
        self.game.world.vehicles[vi].deformation = clauding::deform::VehicleDeformation::new();
        // Settle suspension
        self.direct_input = DirectInput { active: true, throttle: 0.0, brake: 0.0, steer: 0.0, handbrake: false };
        for _ in 0..30 { self.run_full_tick(); }
    }

    fn test_accel(&mut self, vi: usize) -> String {
        self.reset_vehicle_to_start(vi);
        self.setup_test_env(vi);

        self.direct_input = DirectInput { active: true, throttle: 1.0, brake: 0.0, steer: 0.0, handbrake: false };

        let mut time_100 = 0.0f32;
        let mut time_200 = 0.0f32;
        let mut found_100 = false;
        let mut found_200 = false;

        for tick in 0..18000u32 { // 5 min max
            self.run_full_tick();

            // Warp vehicle back to center if approaching world boundary
            let v = &mut self.game.world.vehicles[vi];
            if v.body.pos[0].abs() > 200.0 || v.body.pos[2].abs() > 200.0 {
                v.body.pos = [0.0, v.body.pos[1], -200.0];
                v.x = 0.0;
                v.z = -200.0;
            }

            let spd_kmh = self.game.world.vehicles[vi].speed * 3.6;
            let t = (tick + 1) as f32 * FIXED_DT;

            if !found_100 && spd_kmh >= 100.0 {
                time_100 = t;
                found_100 = true;
            }
            if !found_200 && spd_kmh >= 200.0 {
                time_200 = t;
                found_200 = true;
            }
            if found_200 { break; }
        }

        self.direct_input = DirectInput::new();
        self.game.world.vehicles[vi].surface_override = None;

        let final_spd = self.game.world.vehicles[vi].speed * 3.6;
        let mut out = format!("0-100 km/h: {}", if found_100 { format!("{:.2}s", time_100) } else { "not reached".into() });
        let _ = write!(out, "\n0-200 km/h: {}", if found_200 { format!("{:.2}s", time_200) } else { "not reached".into() });
        let _ = write!(out, "\nfinal: {:.1} km/h", final_spd);
        out
    }

    fn test_brake(&mut self, vi: usize) -> String {
        // Teleport to testcourse start, settle suspension, then set 100 km/h
        self.reset_vehicle_to_start(vi);
        self.setup_test_env(vi);

        // Now set 100 km/h velocity
        let v = &mut self.game.world.vehicles[vi];
        let target_spd = 100.0 / 3.6; // 27.78 m/s
        let fwd = clauding::math::quat_forward(v.body.quat);
        v.body.vel = clauding::math::v3_scale(fwd, target_spd);
        v.speed = target_spd;
        for w in &mut v.wheels { w.ang_vel = target_spd / w.radius; }

        let start_x = v.x;
        let start_z = v.z;

        self.direct_input = DirectInput { active: true, throttle: 0.0, brake: 1.0, steer: 0.0, handbrake: false };

        let mut stop_time = 0.0f32;
        let mut stop_dist = 0.0f32;
        let mut stopped = false;

        for tick in 0..3600u32 { // 60s max
            self.run_full_tick();
            let v = &self.game.world.vehicles[vi];
            let spd = v.speed.abs();
            let t = (tick + 1) as f32 * FIXED_DT;

            if !stopped && spd < 0.1 {
                stop_time = t;
                let dx = v.x - start_x;
                let dz = v.z - start_z;
                stop_dist = (dx * dx + dz * dz).sqrt();
                stopped = true;
                break;
            }
        }

        self.direct_input = DirectInput::new();
        self.game.world.vehicles[vi].surface_override = None;

        if stopped {
            format!("100-0 km/h: {:.2}s {:.1}m\ndecel: {:.2}g", stop_time, stop_dist,
                (100.0 / 3.6) / (stop_time * 9.81))
        } else {
            "err: vehicle did not stop within 60s".into()
        }
    }

    fn test_topspeed(&mut self, vi: usize) -> String {
        self.reset_vehicle_to_start(vi);
        self.setup_test_env(vi);

        self.direct_input = DirectInput { active: true, throttle: 1.0, brake: 0.0, steer: 0.0, handbrake: false };

        let mut max_spd = 0.0f32;
        let mut stable_count = 0u32;
        let mut stable_spd = 0.0f32;

        for _ in 0..36000u32 { // 10 min max
            self.run_full_tick();
            // Warp vehicle back to center if approaching world boundary (preserves velocity)
            let v = &mut self.game.world.vehicles[vi];
            if v.body.pos[0].abs() > 200.0 || v.body.pos[2].abs() > 200.0 {
                v.body.pos = [0.0, v.body.pos[1], -200.0];
                v.x = 0.0;
                v.z = -200.0;
            }

            let spd = self.game.world.vehicles[vi].speed * 3.6;
            if spd > max_spd { max_spd = spd; }

            // Check if speed has stabilized (within 0.5 km/h for 120 ticks = 2s)
            if (spd - stable_spd).abs() < 0.5 {
                stable_count += 1;
                if stable_count >= 120 {
                    self.direct_input = DirectInput::new();
                    self.game.world.vehicles[vi].surface_override = None;
                    return format!("top speed: {:.1} km/h (stabilized)\nmax: {:.1} km/h", spd, max_spd);
                }
            } else {
                stable_spd = spd;
                stable_count = 0;
            }
        }

        self.direct_input = DirectInput::new();
        self.game.world.vehicles[vi].surface_override = None;
        format!("top speed: {:.1} km/h (max, not fully stabilized)", max_spd)
    }

    /// Force player into a vehicle (by index, or nearest)
    fn cmd_enter_vehicle(&mut self, args: &[&str]) -> String {
        if self.game.player.in_vehicle.is_some() {
            return "already in vehicle".into();
        }
        let vi = if let Some(idx_str) = args.first() {
            match idx_str.parse::<usize>() {
                Ok(i) if i < self.game.world.vehicles.len() => i,
                _ => return "err: bad vehicle index".into(),
            }
        } else {
            // Find nearest vehicle
            let px = self.game.player.x;
            let pz = self.game.player.z;
            let mut best = 0;
            let mut best_d = f32::MAX;
            for (i, v) in self.game.world.vehicles.iter().enumerate() {
                let dx = v.x - px;
                let dz = v.z - pz;
                let d = dx * dx + dz * dz;
                if d < best_d { best_d = d; best = i; }
            }
            best
        };
        // Force enter
        let v = &mut self.game.world.vehicles[vi];
        v.occupied = true;
        v.parked = false;
        self.game.player.in_vehicle = Some(vi);
        self.game.player.x = v.x;
        self.game.player.z = v.z;
        format!("entered vehicle {}", vi)
    }

    /// Ensure player is in a vehicle for testing; returns vehicle index or error
    fn ensure_in_vehicle(&mut self) -> Result<usize, String> {
        if let Some(vi) = self.game.player.in_vehicle {
            return Ok(vi);
        }
        // Auto-enter nearest vehicle
        let px = self.game.player.x;
        let pz = self.game.player.z;
        let mut best = 0;
        let mut best_d = f32::MAX;
        for (i, v) in self.game.world.vehicles.iter().enumerate() {
            let dx = v.x - px;
            let dz = v.z - pz;
            let d = dx * dx + dz * dz;
            if d < best_d { best_d = d; best = i; }
        }
        let v = &mut self.game.world.vehicles[best];
        v.occupied = true;
        v.parked = false;
        self.game.player.in_vehicle = Some(best);
        self.game.player.x = v.x;
        self.game.player.z = v.z;
        Ok(best)
    }

    fn cmd_testcourse(&mut self, args: &[&str]) -> String {
        let surface_name = args.first().copied().unwrap_or("asphalt");
        let base_surface = match surface_name.to_ascii_lowercase().as_str() {
            "asphalt" | "road" => state::Surface::CarRoad,
            "grass" | "terrain" => state::Surface::Terrain,
            "gravel" | "dirt" => state::Surface::FieldRoad,
            _ => return format!("err: unknown surface '{}'. use asphalt|grass|gravel", surface_name),
        };

        let terrain = &mut self.game.terrain;
        let grid = terrain.grid;
        let stride = grid + 1;
        let cell = terrain.cell_size;
        let half = state::WORLD_HALF;

        // Clear terrain to flat at Y=0
        for h in terrain.heights.iter_mut() {
            *h = 0.0;
        }

        // Build test features using terrain heightmap:
        // Layout (viewed from above, Z goes down, X goes right):
        //
        //  Z=-200: ========== 400m straight at Y=0 (main straight) ==========
        //  Z=-150 to -100: slope section (5°, 10°, 15°, 20° grades side by side)
        //  Z=-50: speed bumps (periodic 0.15m humps every 5m)
        //  Z=0 to Z=200: ========== 400m straight at Y=0 (braking straight) ==========

        // Slope section: Z=-150 to Z=-100, different angles at different X ranges
        let slopes = [
            (-50.0, -20.0, 5.0f32),   // X -50 to -20: 5° slope
            (-15.0, 15.0, 10.0),       // X -15 to 15: 10° slope
            (20.0, 50.0, 15.0),        // X 20 to 50: 15° slope
            (55.0, 85.0, 20.0),        // X 55 to 85: 20° slope
        ];
        for &(x_min, x_max, angle_deg) in &slopes {
            let rise_per_m = angle_deg.to_radians().tan(); // height per meter of Z travel
            for iz in 0..=grid {
                let z = iz as f32 * cell - half;
                if z < -150.0 || z > -100.0 { continue; }
                let t = (z - (-150.0)) / 50.0; // 0 at Z=-150, 1 at Z=-100
                let height = t * 50.0 * rise_per_m;
                for ix in 0..=grid {
                    let x = ix as f32 * cell - half;
                    if x >= x_min && x <= x_max {
                        terrain.heights[iz * stride + ix] = height;
                    }
                }
            }
        }

        // Speed bumps: Z=-55 to Z=-45, full width, every 5m a 0.15m hump
        for iz in 0..=grid {
            let z = iz as f32 * cell - half;
            if z < -55.0 || z > -45.0 { continue; }
            // Sinusoidal bumps every 5m in Z direction
            let bump = ((z * std::f32::consts::PI / 2.5).sin()).max(0.0) * 0.15;
            for ix in 0..=grid {
                let x = ix as f32 * cell - half;
                if x.abs() < 30.0 {
                    terrain.heights[iz * stride + ix] = bump;
                }
            }
        }

        // Auto-enter nearest vehicle if not in one
        let vi = match self.ensure_in_vehicle() {
            Ok(v) => v,
            Err(e) => return e,
        };

        // Set vehicle surface override to requested surface
        self.game.world.vehicles[vi].surface_override = Some(base_surface);

        // Deactivate all AI vehicles and move NPCs out of the way
        for v in &mut self.game.world.vehicles {
            v.ai_active = false;
            v.parked = true;
            v.speed = 0.0;
        }
        for npc in &mut self.game.world.npcs {
            npc.x = 240.0;
            npc.z = 240.0;
            npc.body.pos = [240.0, 0.0, 240.0];
        }
        // Re-activate player's vehicle
        self.game.world.vehicles[vi].parked = false;

        // Regenerate static GPU vertices for the flattened terrain
        if self.gpu_ctx.as_ref().is_some_and(|g| g.has_graphics()) {
            self.static_verts.clear();
            render::generate_static_gpu_vertices(&self.game.world, &mut self.static_verts);
            self.gpu_ctx.as_mut().unwrap().upload_static_vertices(&self.static_verts);
        }

        // Teleport player vehicle to start of main straight, facing south (+Z)
        // so the 400m straight from Z=-200 to Z=+200 is ahead
        self.reset_vehicle_to_start(vi);

        let mut out = format!("ok testcourse surface={}", surf_str(base_surface));
        let _ = write!(out, "\nlayout:");
        let _ = write!(out, "\n  Z=-200..200: main straights (flat, 400m each direction)");
        let _ = write!(out, "\n  Z=-150..-100: slope section (5/10/15/20 degrees at X=-35/-0/+35/+70)");
        let _ = write!(out, "\n  Z=-55..-45: speed bumps (0.15m humps every 5m)");
        let _ = write!(out, "\n  all AI vehicles deactivated");
        let _ = write!(out, "\n  vehicle at (0, 0.5, -200) facing south (+Z)");
        out
    }

    fn cmd_build(&self) -> String {
        match std::process::Command::new("cargo")
            .args(["build", "--release"])
            .output()
        {
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if output.status.success() {
                    let warnings = stderr.matches("warning:").count();
                    format!("ok {} warnings", warnings)
                } else {
                    let first_err = stderr.lines()
                        .find(|l| l.contains("error"))
                        .unwrap_or("unknown error");
                    format!("err: build failed: {}", first_err)
                }
            }
            Err(e) => format!("err: {}", e),
        }
    }
}

// ---- Utility functions ----

fn surf_str(s: state::Surface) -> &'static str {
    match s {
        state::Surface::Sidewalk => "sidewalk",
        state::Surface::CarRoad => "car_road",
        state::Surface::FieldRoad => "field_road",
        state::Surface::Terrain => "terrain",
    }
}

fn item_kind_str(k: state::ItemKind) -> &'static str {
    match k {
        state::ItemKind::Health => "health",
        state::ItemKind::Money => "money",
        state::ItemKind::Stamina => "stamina",
        state::ItemKind::Food => "food",
        state::ItemKind::Water => "water",
    }
}

fn bearing_deg(ax: f32, az: f32, bx: f32, bz: f32) -> f32 {
    let dx = bx - ax;
    let dz = bz - az;
    let rad = dx.atan2(-dz);
    rad.to_degrees().rem_euclid(360.0)
}

/// Flatten terrain to Y=0 within a radius of (cx, cz)
fn flatten_terrain_around(terrain: &mut state::Terrain, cx: f32, cz: f32, radius: f32) {
    let grid = terrain.grid;
    let stride = grid + 1;
    let cell = terrain.cell_size;
    let half = state::WORLD_HALF;
    let r2 = radius * radius;

    for iz in 0..=grid {
        let z = iz as f32 * cell - half;
        let dz = z - cz;
        for ix in 0..=grid {
            let x = ix as f32 * cell - half;
            let dx = x - cx;
            if dx * dx + dz * dz < r2 {
                terrain.heights[iz * stride + ix] = 0.0;
            }
        }
    }
}

fn compass(deg: f32) -> &'static str {
    let d = deg.rem_euclid(360.0);
    if d < 22.5 { "N" }
    else if d < 67.5 { "NE" }
    else if d < 112.5 { "E" }
    else if d < 157.5 { "SE" }
    else if d < 202.5 { "S" }
    else if d < 247.5 { "SW" }
    else if d < 292.5 { "W" }
    else if d < 337.5 { "NW" }
    else { "N" }
}

fn extract_dist(s: &str) -> f32 {
    if let Some(pos) = s.find("dist=") {
        let rest = &s[pos + 5..];
        let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.').unwrap_or(rest.len());
        rest[..end].parse().unwrap_or(f32::MAX)
    } else {
        f32::MAX
    }
}

fn write_obj(tris: &[state::WorldTri], path: &str) -> String {
    use std::io::Write as W;
    let file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => return format!("err: {}", e),
    };
    let mut w = std::io::BufWriter::new(file);

    for tri in tris {
        for v in &tri.v { let _ = writeln!(w, "v {} {} {}", v[0], v[1], v[2]); }
    }
    for tri in tris {
        let _ = writeln!(w, "vn {} {} {}", tri.normal[0], tri.normal[1], tri.normal[2]);
    }
    for (i, _) in tris.iter().enumerate() {
        let vi = i * 3 + 1;
        let ni = i + 1;
        let _ = writeln!(w, "f {}//{} {}//{} {}//{}", vi, ni, vi + 1, ni, vi + 2, ni);
    }

    format!("{} {} tris {} verts", path, tris.len(), tris.len() * 3)
}

// ---- Winding check helpers ----

fn audit_winding(tris: &[state::WorldTri]) -> (u32, u32, u32, u32) {
    let mut total = 0u32;
    let mut matched = 0u32;
    let mut mismatched = 0u32;
    let mut degenerate = 0u32;
    for tri in tris {
        total += 1;
        let e1 = [tri.v[1][0]-tri.v[0][0], tri.v[1][1]-tri.v[0][1], tri.v[1][2]-tri.v[0][2]];
        let e2 = [tri.v[2][0]-tri.v[0][0], tri.v[2][1]-tri.v[0][1], tri.v[2][2]-tri.v[0][2]];
        let cx = e1[1]*e2[2] - e1[2]*e2[1];
        let cy = e1[2]*e2[0] - e1[0]*e2[2];
        let cz = e1[0]*e2[1] - e1[1]*e2[0];
        let len = (cx*cx + cy*cy + cz*cz).sqrt();
        if len < 1e-6 { degenerate += 1; continue; }
        let inv = 1.0 / len;
        let cn = [cx*inv, cy*inv, cz*inv];
        let sn = tri.normal;
        let sl = (sn[0]*sn[0] + sn[1]*sn[1] + sn[2]*sn[2]).sqrt();
        let d = if sl > 1e-6 {
            (cn[0]*sn[0] + cn[1]*sn[1] + cn[2]*sn[2]) / sl
        } else { 0.0 };
        if d > 0.0 { matched += 1; } else { mismatched += 1; }
    }
    (total, matched, mismatched, degenerate)
}

// ---- PPM map helpers ----

fn ppm_world_to_px(wx: f32, wz: f32, img_size: usize, half: f32, world_size: f32) -> (usize, usize) {
    let px = ((wx + half) / world_size * img_size as f32) as usize;
    let pz = ((wz + half) / world_size * img_size as f32) as usize;
    (px.min(img_size - 1), pz.min(img_size - 1))
}

fn ppm_set_pixel(buf: &mut [u8], img_size: usize, x: usize, z: usize, r: u8, g: u8, b: u8) {
    if x < img_size && z < img_size {
        let idx = (z * img_size + x) * 3;
        buf[idx] = r; buf[idx+1] = g; buf[idx+2] = b;
    }
}

fn ppm_fill_rect(buf: &mut [u8], img_size: usize, half: f32, world_size: f32, cx: f32, cz: f32, hw: f32, hd: f32, r: u8, g: u8, b: u8) {
    let (x0, z0) = ppm_world_to_px(cx - hw, cz - hd, img_size, half, world_size);
    let (x1, z1) = ppm_world_to_px(cx + hw, cz + hd, img_size, half, world_size);
    for z in z0..=z1 {
        for x in x0..=x1 {
            ppm_set_pixel(buf, img_size, x, z, r, g, b);
        }
    }
}

fn ppm_fill_circle(buf: &mut [u8], img_size: usize, half: f32, world_size: f32, wx: f32, wz: f32, radius: f32, r: u8, g: u8, b: u8) {
    let px_radius = (radius / world_size * img_size as f32).max(1.0) as i32;
    let (cx, cz) = ppm_world_to_px(wx, wz, img_size, half, world_size);
    let cx = cx as i32;
    let cz = cz as i32;
    for dz in -px_radius..=px_radius {
        for dx in -px_radius..=px_radius {
            if dx*dx + dz*dz <= px_radius*px_radius {
                ppm_set_pixel(buf, img_size, (cx+dx) as usize, (cz+dz) as usize, r, g, b);
            }
        }
    }
}

fn ppm_draw_line(buf: &mut [u8], img_size: usize, half: f32, world_size: f32,
                 x0: f32, z0: f32, x1: f32, z1: f32, width: f32, r: u8, g: u8, b: u8) {
    let dx = x1 - x0;
    let dz = z1 - z0;
    let len = (dx*dx + dz*dz).sqrt();
    if len < 0.01 { return; }
    let steps = (len / world_size * img_size as f32 * 2.0) as usize + 1;
    let hw = width * 0.5;
    let px = -dz / len;
    let pz = dx / len;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let cx = x0 + dx * t;
        let cz = z0 + dz * t;
        let w_steps = (hw / world_size * img_size as f32).max(1.0) as i32;
        for w in -w_steps..=w_steps {
            let wf = w as f32 / w_steps as f32 * hw;
            let (ppx, ppz) = ppm_world_to_px(cx + px * wf, cz + pz * wf, img_size, half, world_size);
            ppm_set_pixel(buf, img_size, ppx, ppz, r, g, b);
        }
    }
}

// ---- Banner and main ----

fn print_banner() {
    println!("studio ready");
    println!("commands: tick time speed move sprint jump attack interact look face teleport player npcs vehicles items nearby npc vehicle surface world buildings observe probe roads river parking skeleton gait vphysics screenshot render export map inspect winding compare sim build help quit");
    println!("---");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seed: u64 = 42;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--seed" && i + 1 < args.len() {
            seed = args[i + 1].parse().unwrap_or(42);
            i += 1;
        }
        i += 1;
    }

    let mut studio = Studio::new(seed);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    print_banner();
    let _ = out.flush();

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let response = studio.dispatch(&line);
        if !response.is_empty() {
            let _ = writeln!(out, "{}", response);
        }
        let _ = writeln!(out, "---");
        let _ = out.flush();
    }
}
