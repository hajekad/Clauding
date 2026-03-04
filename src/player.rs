// sys_player: movement, sprint, stamina, collision, walking animation

use crate::state::*;
use crate::world::check_building_collision;

// Linux keycodes
const KEY_W: usize = 17;
const KEY_A: usize = 30;
const KEY_S: usize = 31;
const KEY_D: usize = 32;
const KEY_SPACE: usize = 57;
const KEY_UP: usize = 103;
const KEY_DOWN: usize = 108;
const KEY_LEFT: usize = 105;
const KEY_RIGHT: usize = 106;

pub fn sys_player(state: &mut GameState, dt: f32) {
    // Skip walking controls when driving
    if state.player.in_vehicle.is_some() { return; }

    let p = &mut state.player;

    let mut dx = 0.0f32;
    let mut dz = 0.0f32;
    if state.keys[KEY_W] || state.keys[KEY_UP] { dz -= 1.0; }
    if state.keys[KEY_S] || state.keys[KEY_DOWN] { dz += 1.0; }
    if state.keys[KEY_A] || state.keys[KEY_LEFT] { dx -= 1.0; }
    if state.keys[KEY_D] || state.keys[KEY_RIGHT] { dx += 1.0; }

    let len = (dx * dx + dz * dz).sqrt();
    let moving = len > 0.01;

    p.sprinting = state.keys[KEY_SPACE] && p.stamina > 0.0 && moving;
    let speed = if p.sprinting { SPRINT_SPEED } else { PLAYER_SPEED };

    if moving {
        dx /= len;
        dz /= len;
        p.rot_y = (-dx).atan2(-dz);
        p.walk_phase += dt * speed * 2.5;
    }

    let new_x = p.x + dx * speed * dt;
    let new_z = p.z + dz * speed * dt;

    if !check_building_collision(&state.world, new_x, p.z, PLAYER_RADIUS) {
        p.x = new_x;
    }
    if !check_building_collision(&state.world, p.x, new_z, PLAYER_RADIUS) {
        p.z = new_z;
    }

    p.x = p.x.clamp(-WORLD_HALF, WORLD_HALF);
    p.z = p.z.clamp(-WORLD_HALF, WORLD_HALF);

    if p.sprinting {
        p.stamina = (p.stamina - 20.0 * dt).max(0.0);
    } else {
        p.stamina = (p.stamina + 10.0 * dt).min(100.0);
    }
}
