// sys_player: Elden Ring style movement — camera-relative, mouse-driven direction
// W = forward (camera direction), A/D = strafe, S = disabled
// Character always faces camera forward direction, smooth rotation

use crate::state::*;
use crate::world::check_building_collision;
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

    // Gravity + jump physics
    let jump_now = state.keybinds.is_pressed(Action::Jump, &state.keys);
    let jump_prev = state.keybinds.is_pressed(Action::Jump, &state.prev_keys);
    if jump_now && !jump_prev && state.player.on_ground {
        state.player.vel_y = JUMP_VELOCITY;
        state.player.on_ground = false;
    }

    state.player.vel_y -= GRAVITY * dt;
    state.player.y += state.player.vel_y * dt;

    let ground_y = state.terrain.height_at(state.player.x, state.player.z);
    if state.player.y <= ground_y {
        state.player.y = ground_y;
        state.player.vel_y = 0.0;
        state.player.on_ground = true;
    } else {
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

        p.walk_phase += dt * speed * 2.5;

        let new_x = p.x + dx * speed * dt;
        let new_z = p.z + dz * speed * dt;

        if !check_building_collision(&state.world, new_x, p.z, PLAYER_RADIUS) {
            p.x = new_x;
        }
        if !check_building_collision(&state.world, p.x, new_z, PLAYER_RADIUS) {
            p.z = new_z;
        }
    } else {
        // Idle: rotate character to face camera forward direction
        let target_rot = (-fwd_x).atan2(-fwd_z);
        smooth_rotate(&mut p.rot_y, target_rot, IDLE_TURN_RATE, dt);
    }

    p.x = p.x.clamp(-WORLD_HALF, WORLD_HALF);
    p.z = p.z.clamp(-WORLD_HALF, WORLD_HALF);

    // Re-snap to terrain after XZ movement (walking on slopes)
    if p.on_ground {
        let new_ground = state.terrain.height_at(p.x, p.z);
        p.y = new_ground;
    }

    if p.sprinting {
        p.stamina = (p.stamina - 20.0 * dt).max(0.0);
    } else {
        p.stamina = (p.stamina + 10.0 * dt).min(100.0);
    }
}

fn smooth_rotate(rot: &mut f32, target: f32, rate: f32, dt: f32) {
    let mut diff = target - *rot;
    while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
    while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
    *rot += diff * (rate * dt).min(1.0);
}
