// sys_vehicle: AI road-following for parked/roaming cars, player driving controls
// Player enters/exits with E key, drives with WASD

use crate::state::*;
use crate::world::check_building_collision;

const KEY_W: usize = 17;
const KEY_A: usize = 30;
const KEY_S: usize = 31;
const KEY_D: usize = 32;
const KEY_E: usize = 18;
const KEY_UP: usize = 103;
const KEY_DOWN: usize = 108;
const KEY_LEFT: usize = 105;
const KEY_RIGHT: usize = 106;

pub fn sys_vehicle(state: &mut GameState, dt: f32) {
    // Handle enter/exit toggle (E key, with edge detection)
    let e_pressed = state.keys[KEY_E];
    if e_pressed && !state.prev_key_e {
        if let Some(vi) = state.player.in_vehicle {
            // Exit vehicle
            let v = &state.world.vehicles[vi];
            let exit_x = v.x + v.rot_y.sin() * 2.5;
            let exit_z = v.z + v.rot_y.cos() * 2.5;
            state.player.x = exit_x;
            state.player.z = exit_z;
            state.player.rot_y = v.rot_y;
            state.world.vehicles[vi].occupied = false;
            state.world.vehicles[vi].speed = 0.0;
            state.player.in_vehicle = None;
        } else {
            // Try to enter nearest vehicle
            let mut best_dist = VEHICLE_ENTER_DIST * VEHICLE_ENTER_DIST;
            let mut best_idx = None;
            for (i, v) in state.world.vehicles.iter().enumerate() {
                let dx = state.player.x - v.x;
                let dz = state.player.z - v.z;
                let d2 = dx * dx + dz * dz;
                if d2 < best_dist {
                    best_dist = d2;
                    best_idx = Some(i);
                }
            }
            if let Some(vi) = best_idx {
                state.world.vehicles[vi].occupied = true;
                state.world.vehicles[vi].speed = 0.0;
                state.player.in_vehicle = Some(vi);
            }
        }
    }
    state.prev_key_e = e_pressed;

    // Update driven vehicle
    if let Some(vi) = state.player.in_vehicle {
        drive_vehicle(state, vi, dt);
    }

    // AI for unoccupied vehicles
    let n = state.world.vehicles.len();
    for i in 0..n {
        if state.world.vehicles[i].occupied { continue; }
        if !state.world.vehicles[i].ai_active { continue; }
        ai_drive(i, &mut state.world, &state.road_positions, dt);
    }
}

fn drive_vehicle(state: &mut GameState, vi: usize, dt: f32) {
    let fwd = state.keys[KEY_W] || state.keys[KEY_UP];
    let back = state.keys[KEY_S] || state.keys[KEY_DOWN];
    let left = state.keys[KEY_A] || state.keys[KEY_LEFT];
    let right = state.keys[KEY_D] || state.keys[KEY_RIGHT];

    let v = &mut state.world.vehicles[vi];

    // Accelerate / brake
    if fwd {
        v.speed = (v.speed + VEHICLE_ACCEL * dt).min(VEHICLE_SPEED);
    } else if back {
        v.speed = (v.speed - VEHICLE_BRAKE * dt).max(-VEHICLE_SPEED * 0.4);
    } else {
        // Coast to stop
        if v.speed > 0.0 {
            v.speed = (v.speed - VEHICLE_BRAKE * 0.5 * dt).max(0.0);
        } else {
            v.speed = (v.speed + VEHICLE_BRAKE * 0.5 * dt).min(0.0);
        }
    }

    // Steering (only when moving)
    if v.speed.abs() > 0.5 {
        let turn = VEHICLE_TURN_SPEED * dt * (v.speed / VEHICLE_SPEED).signum();
        if left { v.rot_y += turn; }
        if right { v.rot_y -= turn; }
    }

    // Move — copy values to avoid borrow conflict
    let cur_x = v.x;
    let cur_z = v.z;
    let spd = v.speed;
    let rot = v.rot_y;
    let new_x = cur_x - rot.sin() * spd * dt;
    let new_z = cur_z - rot.cos() * spd * dt;

    // Collision (vehicle radius ~1.5)
    if !check_building_collision(&state.world, new_x, cur_z, 1.5) {
        state.world.vehicles[vi].x = new_x;
    } else {
        state.world.vehicles[vi].speed *= -0.3; // bounce
    }
    let cur_x = state.world.vehicles[vi].x;
    if !check_building_collision(&state.world, cur_x, new_z, 1.5) {
        state.world.vehicles[vi].z = new_z;
    } else {
        state.world.vehicles[vi].speed *= -0.3;
    }

    let v = &mut state.world.vehicles[vi];
    v.x = v.x.clamp(-WORLD_HALF, WORLD_HALF);
    v.z = v.z.clamp(-WORLD_HALF, WORLD_HALF);

    // Sync player position to vehicle
    state.player.x = v.x;
    state.player.z = v.z;
    state.player.rot_y = v.rot_y;
}

fn ai_drive(vi: usize, world: &mut WorldData, road_positions: &[f32], dt: f32) {
    let v = &world.vehicles[vi];
    // Simple: drive toward AI target on road, pick new target when close
    let dx = v.ai_target_x - v.x;
    let dz = v.ai_target_z - v.z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist < 5.0 {
        pick_ai_target(&mut world.vehicles[vi], road_positions);
    }

    let v = &mut world.vehicles[vi];

    // Turn toward target
    let dx = v.ai_target_x - v.x;
    let dz = v.ai_target_z - v.z;
    let desired = (-dx).atan2(-dz);
    let mut diff = desired - v.rot_y;
    while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
    while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
    v.rot_y += diff.clamp(-2.0 * dt, 2.0 * dt);

    // Drive at half speed
    let speed = VEHICLE_SPEED * 0.4;
    let new_x = v.x - v.rot_y.sin() * speed * dt;
    let new_z = v.z - v.rot_y.cos() * speed * dt;

    let collides = world.buildings.iter().any(|b| {
        new_x + 1.5 > b.x - b.w * 0.5 && new_x - 1.5 < b.x + b.w * 0.5
        && new_z + 1.5 > b.z - b.d * 0.5 && new_z - 1.5 < b.z + b.d * 0.5
    }) || world.rocks.iter().any(|r| {
        let rdx = new_x - r.x;
        let rdz = new_z - r.z;
        rdx * rdx + rdz * rdz < (r.size + 1.5) * (r.size + 1.5)
    });

    if !collides {
        world.vehicles[vi].x = new_x;
        world.vehicles[vi].z = new_z;
    } else {
        pick_ai_target(&mut world.vehicles[vi], road_positions);
    }

    world.vehicles[vi].x = world.vehicles[vi].x.clamp(-WORLD_HALF, WORLD_HALF);
    world.vehicles[vi].z = world.vehicles[vi].z.clamp(-WORLD_HALF, WORLD_HALF);
}

fn pick_ai_target(v: &mut Vehicle, road_positions: &[f32]) {
    // Find closest road axis and drive along it
    let mut best_road = 0.0f32;
    let mut best_dist = f32::MAX;
    let mut is_x_road = true; // road runs along X axis (at Z = road_pos)

    for &r in road_positions {
        let dz = (v.z - r).abs();
        if dz < best_dist { best_dist = dz; best_road = r; is_x_road = true; }
        let dx = (v.x - r).abs();
        if dx < best_dist { best_dist = dx; best_road = r; is_x_road = false; }
    }

    // Drive to a deterministic random point along that road
    let t = v.rng.range(-80.0, 80.0);
    if is_x_road {
        v.ai_target_x = t;
        v.ai_target_z = best_road;
    } else {
        v.ai_target_x = best_road;
        v.ai_target_z = t;
    }
}
