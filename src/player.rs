// sys_player: Elden Ring style movement — camera-relative, mouse-driven direction
// W = forward (camera direction), A/D = strafe, S = disabled
// Character always faces camera forward direction, smooth rotation

use crate::state::*;
use crate::world::{check_walk_collision, on_river_not_bridge, surface_at};
use crate::input::Action;

const TURN_RATE: f32 = 10.0; // radians/sec for character rotation toward movement dir
const IDLE_TURN_RATE: f32 = 5.0; // slower rotation when idle (following camera)

pub fn sys_player(state: &mut GameState, dt: f32) {
    // Job menu input
    if state.player.job_menu_open {
        let up = state.keybinds.is_pressed(Action::MoveForward, &state.keys)
            && !state.keybinds.is_pressed(Action::MoveForward, &state.prev_keys);
        let down = state.keybinds.is_pressed(Action::MoveBack, &state.keys)
            && !state.keybinds.is_pressed(Action::MoveBack, &state.prev_keys);
        let enter = state.keybinds.is_pressed(Action::Interact, &state.keys)
            && !state.keybinds.is_pressed(Action::Interact, &state.prev_keys);
        let esc = state.keys[9]; // ESC scancode

        if up && state.player.job_menu_cursor > 0 {
            state.player.job_menu_cursor -= 1;
        }
        if down && state.player.job_menu_cursor < 13 {
            state.player.job_menu_cursor += 1;
        }
        if enter {
            crate::player_jobs::accept_job(state);
            state.player.job_menu_open = false;
        }
        if esc {
            state.player.job_menu_open = false;
        }
        return; // no movement while job menu open
    }

    // Bench sitting — regen hp/stamina, cancel on movement
    if state.player.sitting {
        state.player.health = (state.player.health + 5.0 * dt).min(100.0);
        state.player.stamina = (state.player.stamina + 15.0 * dt).min(100.0);
        let any_move = state.keybinds.is_pressed(Action::MoveForward, &state.keys)
            || state.keybinds.is_pressed(Action::MoveLeft, &state.keys)
            || state.keybinds.is_pressed(Action::MoveRight, &state.keys)
            || state.keybinds.is_pressed(Action::Jump, &state.keys);
        if any_move {
            state.player.sitting = false;
        }
        return; // no movement while sitting
    }

    // Skip walking controls when driving
    if state.player.in_vehicle.is_some() { return; }

    // Player ragdoll: skeleton ragdoll runs, player has no movement control
    if state.player.skeleton.ragdoll_active {
        let p = &mut state.player;
        // Run ragdoll physics
        p.skeleton.step_ragdoll(&state.terrain, dt);
        // Sync position to hips
        let hips = p.skeleton.bones[0].world_pos;
        p.body.pos = hips;
        p.x = hips[0];
        p.y = hips[1];
        p.z = hips[2];
        // Blend recovery when timer expired
        p.skeleton.blend_from_ragdoll(p.body.pos, p.rot_y, dt);
        // Timer is decremented by step_ragdoll() — no double-decrement here
        return;
    }

    // Camera forward/right projected to XZ plane
    let cam_yaw = state.camera.yaw;
    let fwd_x = -cam_yaw.sin();
    let fwd_z = -cam_yaw.cos();
    let right_x = cam_yaw.cos();
    let right_z = -cam_yaw.sin();

    // Input: W=forward, A/D=strafe, S=nothing
    let mut move_x = 0.0f32;
    let mut move_z = 0.0f32;
    if state.keybinds.is_pressed(Action::MoveForward, &state.keys) {
        move_x += fwd_x;
        move_z += fwd_z;
    }
    if state.keybinds.is_pressed(Action::MoveLeft, &state.keys) {
        move_x -= right_x;
        move_z -= right_z;
    }
    if state.keybinds.is_pressed(Action::MoveRight, &state.keys) {
        move_x += right_x;
        move_z += right_z;
    }

    let len = (move_x * move_x + move_z * move_z).sqrt();
    let moving = len > 0.01;

    let p = &mut state.player;

    p.sprinting = state.keybinds.is_pressed(Action::Sprint, &state.keys) && p.stamina > 0.0 && moving;
    // Gait selected from input intent — speed emerges from gait parameters, not a lookup table
    let desired_gait = if !moving {
        crate::skeleton::Gait::Idle
    } else if p.sprinting {
        crate::skeleton::Gait::Sprint
    } else {
        crate::skeleton::Gait::Run // default walk key = run gait
    };

    // Jump: two-phase leg compression → extension ground reaction force
    // Phase 0→1 = compression (crouch, ~0.1s), Phase 1→2 = extension (launch force applied over ~0.05s)
    let jump_now = state.keybinds.is_pressed(Action::Jump, &state.keys);
    let jump_prev = state.keybinds.is_pressed(Action::Jump, &state.prev_keys);
    if jump_now && !jump_prev && p.on_ground && p.skeleton.jump_phase <= 0.0 {
        p.skeleton.jump_phase = 0.001; // begin compression
    }
    // Advance jump phases
    if p.skeleton.jump_phase > 0.0 && p.skeleton.jump_phase < 1.0 {
        // Compression phase: legs bend, body lowers (~0.1s duration)
        let compress_speed = 10.0; // phase units/sec (0→1 in 0.1s)
        p.skeleton.jump_phase += compress_speed * dt;
        p.skeleton.jump_crouch = p.skeleton.jump_phase.min(1.0) * 0.08; // up to 8cm crouch
        if p.skeleton.jump_phase >= 1.0 {
            p.skeleton.jump_phase = 1.0; // transition to extension
        }
    }
    if p.skeleton.jump_phase >= 1.0 && p.skeleton.jump_phase < 2.0 {
        // Extension phase: single impulse at transition, then visual unwind
        let extend_speed = 20.0; // phase units/sec (1→2 in 0.05s)

        // Apply single impulse exactly once at extension start (timestep-independent)
        // Use < 1.01 instead of == 1.0 to avoid float equality fragility
        if p.skeleton.jump_phase < 1.01 && p.on_ground {
            let ground_n = state.terrain.normal_at(p.body.pos[0], p.body.pos[2]);
            let launch_dir = crate::math::v3_normalize([
                ground_n[0] * 0.3,
                ground_n[1].max(0.7),
                ground_n[2] * 0.3,
            ]);
            let impulse_mag = JUMP_VELOCITY * p.body.mass;
            p.body.apply_impulse([
                launch_dir[0] * impulse_mag,
                launch_dir[1] * impulse_mag,
                launch_dir[2] * impulse_mag,
            ]);
        }

        p.skeleton.jump_phase += extend_speed * dt;
        p.skeleton.jump_crouch = (2.0 - p.skeleton.jump_phase.min(2.0)) * 0.08;

        if p.skeleton.jump_phase >= 2.0 {
            p.skeleton.jump_phase = 0.0;
            p.skeleton.jump_crouch = 0.0;
            p.on_ground = false;
        }
    }

    // Query surface material for friction
    let surface = surface_at(p.body.pos[0], p.body.pos[2], &state.road_network);
    let surface_friction = crate::material::material_for_surface(surface).dynamic_friction;

    // Landing detection: check for ragdoll/stumble BEFORE animation consumes landing_speed
    if p.on_ground && p.skeleton.should_ragdoll_from_fall() {
        // Player ragdoll from high fall (landing speed < -10 m/s)
        let impulse = [p.body.vel[0] * 0.5, 0.0, p.body.vel[2] * 0.5];
        p.skeleton.activate_ragdoll([p.x, p.y, p.z], p.rot_y, impulse);
        p.skeleton.ragdoll_timer = RAGDOLL_DURATION;
    }

    if moving {
        let dx = move_x / len;
        let dz = move_z / len;

        // Smooth rotation toward movement direction
        let target_rot = (-dx).atan2(-dz);
        smooth_rotate(&mut p.rot_y, target_rot, TURN_RATE, dt);

        // Locomotion: legs push against ground — speed emerges from gait stride parameters
        let desired_dir = [dx, 0.0, dz];
        let walk_force = p.skeleton.compute_locomotion_force(
            desired_dir, desired_gait, p.body.vel, p.body.mass, surface_friction, f32::MAX,
        );
        p.body.apply_force(walk_force);
    } else {
        // Idle: rotate character to face camera forward direction
        let target_rot = (-fwd_x).atan2(-fwd_z);
        smooth_rotate(&mut p.rot_y, target_rot, IDLE_TURN_RATE, dt);

        // Deceleration: legs actively brake by pushing against ground
        let decel_force = p.skeleton.compute_locomotion_force(
            [0.0, 0.0, 0.0], crate::skeleton::Gait::Idle, p.body.vel, p.body.mass, surface_friction, f32::MAX,
        );
        p.body.apply_force(decel_force);
    }

    // Slope sliding force — steeper slope + lower friction = more slide
    if p.on_ground {
        let raw_n = state.terrain.normal_at(p.body.pos[0], p.body.pos[2]);
        let slope = (1.0 - raw_n[1]).max(0.0);
        if slope > 0.15 {
            let mass = p.body.mass;
            // Slide force inversely proportional to friction: ice slides hard, asphalt barely
            let slide_mag = slope * slope * 40.0 * mass * (1.0 - surface_friction).max(0.0);
            p.body.apply_force([-raw_n[0] * slide_mag, 0.0, -raw_n[2] * slide_mag]);
        }
    }

    // Save pre-integration position for collision rollback
    let prev_pos = p.body.pos;

    // Integrate rigid body (gravity + walk force + slope force + damping)
    crate::physics::integrate(&mut p.body, dt);

    // Ground contact enforcement
    let ground_y = state.terrain.height_at(p.body.pos[0], p.body.pos[2]);
    if p.body.pos[1] <= ground_y {
        p.body.pos[1] = ground_y;
        if p.body.vel[1] < 0.0 { p.body.vel[1] = 0.0; }
        p.on_ground = true;
    } else {
        p.on_ground = false;
    }

    // Character-on-vehicle stacking: check if standing on a vehicle roof
    {
        let feet_pos = p.body.pos;
        let mut found_vehicle = None;
        for (vi, v) in state.world.vehicles.iter().enumerate() {
            let half_w = 0.93 * v.scale;
            let half_d = 2.3 * v.scale;
            let roof_h = 1.2 * v.scale;
            if let Some((_normal, surface_y)) = crate::physics::point_on_vehicle_surface(
                feet_pos, v.body.pos, v.rot_y, half_w, half_d, roof_h,
            ) {
                // Vehicle surface is higher than terrain → stand on vehicle
                if surface_y > ground_y {
                    p.body.pos[1] = surface_y;
                    if p.body.vel[1] < 0.0 { p.body.vel[1] = 0.0; }
                    p.on_ground = true;
                    // Transfer vehicle velocity via friction (metal surface, μ ≈ 0.4)
                    let friction = 0.4;
                    let vvel = v.body.vel;
                    p.body.vel[0] += (vvel[0] - p.body.vel[0]) * friction * dt.min(0.1);
                    p.body.vel[2] += (vvel[2] - p.body.vel[2]) * friction * dt.min(0.1);
                    found_vehicle = Some(vi);
                    p.standing_on_vehicle_timer = 0.15; // hysteresis
                    break;
                }
            }
        }
        if found_vehicle.is_some() {
            p.standing_on_vehicle = found_vehicle;
        } else if p.standing_on_vehicle_timer > 0.0 {
            p.standing_on_vehicle_timer -= dt;
            if p.standing_on_vehicle_timer <= 0.0 {
                p.standing_on_vehicle = None;
            }
        } else {
            p.standing_on_vehicle = None;
        }
    }

    // Building collision (axis-separated sliding)
    if check_walk_collision(&state.world, p.body.pos[0], prev_pos[2], PLAYER_RADIUS, None) {
        p.body.pos[0] = prev_pos[0];
        p.body.vel[0] = 0.0;
    }
    if check_walk_collision(&state.world, p.body.pos[0], p.body.pos[2], PLAYER_RADIUS, None) {
        p.body.pos[2] = prev_pos[2];
        p.body.vel[2] = 0.0;
    }

    // World bounds
    p.body.pos[0] = p.body.pos[0].clamp(-WORLD_HALF, WORLD_HALF);
    p.body.pos[2] = p.body.pos[2].clamp(-WORLD_HALF, WORLD_HALF);

    // Sync body → legacy fields
    p.x = p.body.pos[0];
    p.y = p.body.pos[1]; // player body is clamped to ground directly (no capsule offset)
    p.z = p.body.pos[2];
    p.vel_y = p.body.vel[1];

    // River: current push + drowning damage
    if on_river_not_bridge(p.x, p.z, &state.world.river_segments, &state.world.bridges) {
        p.body.vel[0] += RIVER_CURRENT * dt;
        p.body.pos[0] += RIVER_CURRENT * dt;
        p.x = p.body.pos[0];
        p.health = (p.health - DROWN_DAMAGE * dt).max(0.0);
    }

    // Smooth terrain normal for slope tilting
    let raw_n = state.terrain.normal_at(p.x, p.z);
    let target_n = crate::math::clamp_normal_tilt(raw_n, 25.0);
    let lerp_rate = 8.0 * dt;
    let old_n = p.terrain_normal;
    p.terrain_normal = crate::math::v3_normalize(crate::math::v3_lerp(old_n, target_n, lerp_rate.min(1.0)));

    if p.sprinting {
        p.stamina = (p.stamina - 20.0 * dt).max(0.0);
    } else {
        p.stamina = (p.stamina + 10.0 * dt).min(100.0);
    }

    // Procedural animation: skeleton IK + walk cycle from physics velocity
    let vel = p.body.vel;
    let pos = p.body.pos;
    let rot_y = p.rot_y;
    let on_ground = p.on_ground;
    let mass = p.body.mass;
    p.skeleton.step_animation(vel, pos, rot_y, &state.terrain, on_ground, mass, dt);

    // Sync skeleton walk_phase to legacy walk_phase for renderer
    p.walk_phase = p.skeleton.walk_phase;
}

fn smooth_rotate(rot: &mut f32, target: f32, rate: f32, dt: f32) {
    let mut diff = target - *rot;
    while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
    while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
    *rot += diff * (rate * dt).min(1.0);
}
