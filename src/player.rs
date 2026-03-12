// sys_player: Elden Ring style movement — camera-relative, mouse-driven direction
// W = forward (camera direction), A/D = strafe, S = disabled
// Character always faces camera forward direction, smooth rotation

use crate::state::*;
use crate::world::{check_walk_collision, on_river_not_bridge};
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

    // Jump: vertical velocity on body
    let jump_now = state.keybinds.is_pressed(Action::Jump, &state.keys);
    let jump_prev = state.keybinds.is_pressed(Action::Jump, &state.prev_keys);
    if jump_now && !jump_prev && state.player.on_ground {
        state.player.body.vel[1] = JUMP_VELOCITY;
        state.player.on_ground = false;
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
    // S (MoveBack) intentionally does nothing
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
    let speed = if p.sprinting { SPRINT_SPEED } else { PLAYER_SPEED };

    if moving {
        let dx = move_x / len;
        let dz = move_z / len;

        // Smooth rotation toward movement direction
        let target_rot = (-dx).atan2(-dz);
        smooth_rotate(&mut p.rot_y, target_rot, TURN_RATE, dt);

        // walk_phase now driven by skeleton.step_animation()

        // Walk force: accelerate toward desired velocity
        let mass = p.body.mass;
        p.body.apply_force([
            mass * (dx * speed - p.body.vel[0]) / 0.1,
            0.0,
            mass * (dz * speed - p.body.vel[2]) / 0.1,
        ]);
    } else {
        // Idle: rotate character to face camera forward direction
        let target_rot = (-fwd_x).atan2(-fwd_z);
        smooth_rotate(&mut p.rot_y, target_rot, IDLE_TURN_RATE, dt);

        // Deceleration force
        let mass = p.body.mass;
        p.body.apply_force([
            mass * -p.body.vel[0] / 0.1,
            0.0,
            mass * -p.body.vel[2] / 0.1,
        ]);
    }

    // Slope sliding force
    if p.on_ground {
        let raw_n = state.terrain.normal_at(p.body.pos[0], p.body.pos[2]);
        let slope = (1.0 - raw_n[1]).max(0.0);
        if slope > 0.15 {
            let mass = p.body.mass;
            let slide_mag = slope * slope * 40.0 * mass;
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
    p.y = p.body.pos[1];
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
    p.skeleton.step_animation(vel, pos, rot_y, &state.terrain, on_ground, dt);

    // Sync skeleton walk_phase to legacy walk_phase for renderer
    p.walk_phase = p.skeleton.walk_phase;
}

fn smooth_rotate(rot: &mut f32, target: f32, rate: f32, dt: f32) {
    let mut diff = target - *rot;
    while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
    while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
    *rot += diff * (rate * dt).min(1.0);
}
