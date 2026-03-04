// sys_player: Elden Ring style movement — camera-relative, mouse-driven direction
// W = forward (camera direction), A/D = strafe, S = disabled
// Character always faces camera forward direction, smooth rotation

use crate::state::*;
use crate::world::check_building_collision;
use crate::input::Action;

const TURN_RATE: f32 = 10.0; // radians/sec for character rotation toward movement dir
const IDLE_TURN_RATE: f32 = 5.0; // slower rotation when idle (following camera)

pub fn sys_player(state: &mut GameState, dt: f32) {
    // Skip walking controls when driving
    if state.player.in_vehicle.is_some() { return; }

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
