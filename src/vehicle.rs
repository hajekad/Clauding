// sys_vehicle: AI road-following for parked/roaming cars, player driving controls
// Player enters/exits with Interact key, drives with movement keys

use crate::state::*;
use crate::world::check_building_collision;
use crate::input::Action;

pub fn sys_vehicle(state: &mut GameState, dt: f32) {
    // Handle enter/exit toggle (Interact key, with edge detection)
    let interact_now = state.keybinds.is_pressed(Action::Interact, &state.keys);
    let interact_prev = state.keybinds.is_pressed(Action::Interact, &state.prev_keys);
    if interact_now && !interact_prev {
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
        } else if state.player.carrying_bin.is_none() {
            // Try to enter nearest vehicle (can't enter while carrying bin)
            let mut best_dist = VEHICLE_ENTER_DIST * VEHICLE_ENTER_DIST;
            let mut best_idx = None;
            for (i, v) in state.world.vehicles.iter().enumerate() {
                if v.occupied { continue; }
                let npc_driving = state.world.npcs.iter().any(|npc| npc.in_vehicle && npc.car_idx == i);
                if npc_driving { continue; }
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

    // Update driven vehicle
    if let Some(vi) = state.player.in_vehicle {
        drive_vehicle(state, vi, dt);
    }

    // AI for unoccupied vehicles (ambient + NPC-driven)
    let n = state.world.vehicles.len();
    for i in 0..n {
        if state.world.vehicles[i].occupied { continue; }
        if !state.world.vehicles[i].ai_active { continue; }
        ai_drive(i, &mut state.world, &state.road_network, &state.terrain, dt);
    }
}

fn drive_vehicle(state: &mut GameState, vi: usize, dt: f32) {
    let fwd = state.keybinds.is_pressed(Action::MoveForward, &state.keys);
    let back = state.keybinds.is_pressed(Action::MoveBack, &state.keys);
    let left = state.keybinds.is_pressed(Action::MoveLeft, &state.keys);
    let right = state.keybinds.is_pressed(Action::MoveRight, &state.keys);

    let v = &mut state.world.vehicles[vi];

    if fwd {
        v.speed = (v.speed + VEHICLE_ACCEL * dt).min(VEHICLE_SPEED);
    } else if back {
        v.speed = (v.speed - VEHICLE_BRAKE * dt).max(-VEHICLE_SPEED * 0.4);
    } else {
        if v.speed > 0.0 {
            v.speed = (v.speed - VEHICLE_BRAKE * 0.5 * dt).max(0.0);
        } else {
            v.speed = (v.speed + VEHICLE_BRAKE * 0.5 * dt).min(0.0);
        }
    }

    if v.speed.abs() > 0.5 {
        let turn = VEHICLE_TURN_SPEED * dt * (v.speed / VEHICLE_SPEED).signum();
        if left { v.rot_y += turn; }
        if right { v.rot_y -= turn; }
    }

    let cur_x = v.x;
    let cur_z = v.z;
    let spd = v.speed;
    let rot = v.rot_y;
    let new_x = cur_x - rot.sin() * spd * dt;
    let new_z = cur_z - rot.cos() * spd * dt;

    if !check_building_collision(&state.world, new_x, cur_z, 1.5) {
        state.world.vehicles[vi].x = new_x;
    } else {
        state.world.vehicles[vi].speed *= -0.3;
    }
    let cur_x = state.world.vehicles[vi].x;
    if !check_building_collision(&state.world, cur_x, new_z, 1.5) {
        state.world.vehicles[vi].z = new_z;
    } else {
        state.world.vehicles[vi].speed *= -0.3;
    }

    state.world.vehicles[vi].x = state.world.vehicles[vi].x.clamp(-WORLD_HALF, WORLD_HALF);
    state.world.vehicles[vi].z = state.world.vehicles[vi].z.clamp(-WORLD_HALF, WORLD_HALF);
    let vx = state.world.vehicles[vi].x;
    let vz = state.world.vehicles[vi].z;
    state.world.vehicles[vi].y = state.terrain.height_at(vx, vz);

    state.player.x = state.world.vehicles[vi].x;
    state.player.y = state.world.vehicles[vi].y;
    state.player.z = state.world.vehicles[vi].z;
    state.player.rot_y = state.world.vehicles[vi].rot_y;
}

fn ai_drive(vi: usize, world: &mut WorldData, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    let v = &world.vehicles[vi];
    let dx = v.ai_target_x - v.x;
    let dz = v.ai_target_z - v.z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist < 5.0 {
        if let Some(_npc_owner) = world.vehicles[vi].owner_npc {
            world.vehicles[vi].ai_active = false;
            world.vehicles[vi].speed = 0.0;
            return;
        }
        pick_ai_target(&mut world.vehicles[vi], net);
    }

    // Soft road-following: if drifting off CarRoad, gentle perpendicular correction
    {
        let v = &world.vehicles[vi];
        let vx = v.x;
        let vz = v.z;
        let mut best_dist = f32::MAX;
        let mut best_proj_x = vx;
        let mut best_proj_z = vz;
        for seg in &net.segments {
            if seg.tier != RoadTier::CarRoad { continue; }
            let sdx = seg.x1 - seg.x0;
            let sdz = seg.z1 - seg.z0;
            let len_sq = sdx * sdx + sdz * sdz;
            if len_sq < 1e-8 { continue; }
            let t = ((vx - seg.x0) * sdx + (vz - seg.z0) * sdz) / len_sq;
            let t = t.clamp(0.0, 1.0);
            let px = seg.x0 + t * sdx;
            let pz = seg.z0 + t * sdz;
            let ex = vx - px;
            let ez = vz - pz;
            let d = ex * ex + ez * ez;
            if d < best_dist {
                best_dist = d;
                best_proj_x = px;
                best_proj_z = pz;
            }
        }
        let road_offset = best_dist.sqrt();
        if road_offset > CAR_ROAD_WIDTH * 0.3 && road_offset < CAR_ROAD_WIDTH * 2.0 {
            let correction = 0.5 * dt;
            let cdx = best_proj_x - vx;
            let cdz = best_proj_z - vz;
            let cd = (cdx * cdx + cdz * cdz).sqrt().max(0.01);
            world.vehicles[vi].x += cdx / cd * correction;
            world.vehicles[vi].z += cdz / cd * correction;
        }
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

    let is_npc = v.owner_npc.is_some();
    let speed = if is_npc { VEHICLE_SPEED * 0.5 } else { VEHICLE_SPEED * 0.4 };
    let new_x = v.x - v.rot_y.sin() * speed * dt;
    let new_z = v.z - v.rot_y.cos() * speed * dt;

    let collides = world.buildings.iter().any(|b| {
        new_x + 1.5 > b.x - b.w * 0.5 && new_x - 1.5 < b.x + b.w * 0.5
        && new_z + 1.5 > b.z - b.d * 0.5 && new_z - 1.5 < b.z + b.d * 0.5
    }) || world.rocks.iter().any(|r| {
        let rdx = new_x - r.x;
        let rdz = new_z - r.z;
        rdx * rdx + rdz * rdz < (r.size + 1.5) * (r.size + 1.5)
    }) || world.trees.iter().any(|t| {
        let tdx = new_x - t.x;
        let tdz = new_z - t.z;
        tdx * tdx + tdz * tdz < (t.trunk_radius + 1.5) * (t.trunk_radius + 1.5)
    });

    if !collides {
        world.vehicles[vi].x = new_x;
        world.vehicles[vi].z = new_z;
        world.vehicles[vi].speed = speed; // track actual movement
    } else {
        // Rotate to try a different angle on next frame
        world.vehicles[vi].rot_y += 0.8;
        world.vehicles[vi].speed = 0.0;
        if !is_npc {
            pick_ai_target(&mut world.vehicles[vi], net);
        }
    }

    world.vehicles[vi].x = world.vehicles[vi].x.clamp(-WORLD_HALF, WORLD_HALF);
    world.vehicles[vi].z = world.vehicles[vi].z.clamp(-WORLD_HALF, WORLD_HALF);
    world.vehicles[vi].y = terrain.height_at(world.vehicles[vi].x, world.vehicles[vi].z);
}

fn pick_ai_target(v: &mut Vehicle, net: &RoadNetwork) {
    if net.nodes.is_empty() { return; }
    // Find nearest node, then pick a random different node as target
    let mut best_dist = f32::MAX;
    let mut _best_node = 0;
    for (i, node) in net.nodes.iter().enumerate() {
        let dx = v.x - node[0];
        let dz = v.z - node[1];
        let d = dx * dx + dz * dz;
        if d < best_dist { best_dist = d; _best_node = i; }
    }
    // Pick a random node as destination (vehicles drive between intersections)
    let target_idx = v.rng.next() as usize % net.nodes.len();
    v.ai_target_x = net.nodes[target_idx][0];
    v.ai_target_z = net.nodes[target_idx][1];
}
