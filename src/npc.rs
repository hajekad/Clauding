// NPC life simulation: state machine, physics, pathfinding, item pickup/deposit, night spawning

use crate::state::*;
use crate::world::{check_npc_walk_collision, surface_at, point_to_segment_dist};
use crate::rng::Rng;

// Home task duration: 4 game-hours = 4 * 60 = 240 real seconds
const HOME_TASK_DURATION: f32 = 240.0;
// Work duration: 12 game-hours = 720 real seconds
const WORK_DURATION: f32 = 720.0;
// Night spawn interval: ~1 item every 4.8 seconds
const NIGHT_SPAWN_INTERVAL: f32 = 4.8;
// Bin relocation threshold: if nearest bin > this distance from item cluster, relocate
const BIN_RELOCATE_DIST: f32 = 20.0;

pub fn sys_npc(
    world: &mut WorldData, road_network: &RoadNetwork, terrain: &Terrain,
    dt: f32, time_of_day: f32,
) {
    let n = world.npcs.len();
    for i in 0..n {
        // Physics: gravity + ground snap
        npc_physics(world, i, terrain, dt);

        // State machine
        let state = world.npcs[i].state;
        match state {
            NpcState::Sleeping => npc_sleeping(world, i, time_of_day),
            NpcState::HomeTask => npc_home_task(world, i, terrain, dt),
            NpcState::GoingToWork => npc_going_to_work(world, i, road_network, terrain, dt),
            NpcState::Working => npc_working(world, i, road_network, terrain, dt),
            NpcState::GoingHome => npc_going_home(world, i, road_network, terrain, dt),
            NpcState::Driving => npc_driving(world, i, terrain, dt),
        }
    }
}

fn npc_physics(world: &mut WorldData, i: usize, terrain: &Terrain, dt: f32) {
    let npc = &mut world.npcs[i];
    if npc.in_vehicle || npc.state == NpcState::Sleeping { return; }

    npc.vel_y -= GRAVITY * dt;
    npc.y += npc.vel_y * dt;

    let ground = terrain.height_at(npc.x, npc.z);
    if npc.y <= ground {
        npc.y = ground;
        npc.vel_y = 0.0;
        npc.on_ground = true;
    } else {
        npc.on_ground = false;
    }
}

/// Walk NPC toward (target_x, target_z) with collision avoidance, pathfinding, and surface-aware speed
fn npc_walk_toward(world: &mut WorldData, i: usize, tx: f32, tz: f32, net: &RoadNetwork, terrain: &Terrain, dt: f32) -> f32 {
    let npc = &world.npcs[i];
    if npc.in_vehicle { return 0.0; }

    let (actual_tx, actual_tz) = if npc.detouring {
        (npc.detour_x, npc.detour_z)
    } else {
        (tx, tz)
    };

    let dx = actual_tx - npc.x;
    let dz = actual_tz - npc.z;
    let dist = (dx * dx + dz * dz).sqrt();
    if dist < 0.5 {
        if world.npcs[i].detouring {
            world.npcs[i].detouring = false;
            world.npcs[i].stuck_timer = 0.0;
        }
        return dist;
    }

    // Turn toward target
    let desired = (-dx).atan2(-dz);
    let npc = &mut world.npcs[i];
    let mut diff = desired - npc.rot_y;
    while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
    while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
    npc.rot_y += diff.clamp(-4.0 * dt, 4.0 * dt);

    // Surface-aware speed
    let surface = surface_at(npc.x, npc.z, net);
    let speed = match surface {
        Surface::Sidewalk => NPC_SPEED_SIDEWALK,
        Surface::CarRoad => NPC_SPEED_CAR_ROAD,
        Surface::FieldRoad => NPC_SPEED_FIELD_ROAD,
        Surface::Terrain => {
            // Check slope for steep terrain
            let normal = terrain.normal_at(npc.x, npc.z);
            let steepness = 1.0 - normal[1]; // 0 = flat, 1 = vertical
            let t = (steepness * 3.0).clamp(0.0, 1.0);
            NPC_SPEED_TERRAIN * (1.0 - t) + NPC_SPEED_STEEP * t
        }
    };

    // Car road avoidance: if on a car road and not detouring, drift toward nearest sidewalk
    if surface == Surface::CarRoad && !npc.detouring {
        // Find nearest CarRoad segment and compute perpendicular drift toward sidewalk
        let nx = npc.x;
        let nz = npc.z;
        let mut best_seg_dist = f32::MAX;
        let mut best_perp_x = 0.0f32;
        let mut best_perp_z = 0.0f32;
        for seg in &net.segments {
            if seg.tier != RoadTier::CarRoad { continue; }
            let d = point_to_segment_dist(nx, nz, seg.x0, seg.z0, seg.x1, seg.z1);
            if d < best_seg_dist {
                best_seg_dist = d;
                let sdx = seg.x1 - seg.x0;
                let sdz = seg.z1 - seg.z0;
                let len = (sdx * sdx + sdz * sdz).sqrt().max(0.01);
                // Perpendicular direction
                let px = -sdz / len;
                let pz = sdx / len;
                // Which side is the NPC on? Project position onto perpendicular
                let sdx2 = seg.x1 - seg.x0;
                let sdz2 = seg.z1 - seg.z0;
                let len_sq = sdx2 * sdx2 + sdz2 * sdz2;
                let t = ((nx - seg.x0) * sdx2 + (nz - seg.z0) * sdz2) / len_sq.max(0.01);
                let t = t.clamp(0.0, 1.0);
                let proj_x = seg.x0 + t * sdx2;
                let proj_z = seg.z0 + t * sdz2;
                let to_npc_x = nx - proj_x;
                let to_npc_z = nz - proj_z;
                let dot = to_npc_x * px + to_npc_z * pz;
                let sign = if dot >= 0.0 { 1.0 } else { -1.0 };
                best_perp_x = px * sign;
                best_perp_z = pz * sign;
            }
        }
        // Drift toward sidewalk edge
        npc.x += best_perp_x * 0.5 * dt;
        npc.z += best_perp_z * 0.5 * dt;
    }

    // Try to move forward
    let rot = npc.rot_y;
    let new_x = npc.x - rot.sin() * speed * dt;
    let new_z = npc.z - rot.cos() * speed * dt;

    let home_idx = npc.home_idx;
    let collides_x = check_npc_walk_collision(world, new_x, world.npcs[i].z, 0.4, home_idx);
    let collides_z = check_npc_walk_collision(world, world.npcs[i].x, new_z, 0.4, home_idx);

    let npc = &mut world.npcs[i];
    let old_x = npc.x;
    let old_z = npc.z;

    if !collides_x {
        npc.x = new_x;
    }
    if !collides_z {
        npc.z = new_z;
    }

    let moved = ((npc.x - old_x).abs() + (npc.z - old_z).abs()) > speed * dt * 0.3;

    if moved {
        npc.stuck_timer = 0.0;
        npc.walk_phase += dt * speed * 2.5;
    } else {
        npc.stuck_timer += dt;

        if npc.stuck_timer > 0.5 && !npc.detouring {
            let perp_x = -dz / dist.max(0.01) * 5.0;
            let perp_z = dx / dist.max(0.01) * 5.0;
            let sign = if npc.rng.next() % 2 == 0 { 1.0 } else { -1.0 };
            npc.detour_x = npc.x + perp_x * sign;
            npc.detour_z = npc.z + perp_z * sign;
            npc.detouring = true;
        }

        if npc.stuck_timer > 1.0 && npc.stuck_timer < 1.2 && npc.on_ground {
            npc.vel_y = NPC_JUMP_VELOCITY;
            npc.on_ground = false;
        }

        if npc.stuck_timer > 3.0 {
            let angle = npc.rng.range(0.0, std::f32::consts::TAU);
            npc.detour_x = npc.x + angle.cos() * 10.0;
            npc.detour_z = npc.z + angle.sin() * 10.0;
            npc.stuck_timer = 0.5;
        }
    }

    npc.x = npc.x.clamp(-WORLD_HALF, WORLD_HALF);
    npc.z = npc.z.clamp(-WORLD_HALF, WORLD_HALF);
    npc.y = terrain.height_at(npc.x, npc.z).max(npc.y);

    let dx2 = tx - world.npcs[i].x;
    let dz2 = tz - world.npcs[i].z;
    (dx2 * dx2 + dz2 * dz2).sqrt()
}

fn npc_sleeping(world: &mut WorldData, i: usize, time_of_day: f32) {
    let npc = &mut world.npcs[i];
    // Position NPC inside home building (hidden)
    let home = &world.buildings[npc.home_idx];
    npc.x = home.x;
    npc.z = home.z;
    npc.y = home.ground_y;
    npc.walk_phase = 0.0; // no animation

    // Check wake condition (handle midnight wrap)
    let should_wake = if npc.wake_hour < 12.0 {
        time_of_day >= npc.wake_hour && time_of_day < npc.wake_hour + 16.0
    } else {
        time_of_day >= npc.wake_hour || time_of_day < npc.wake_hour - 8.0
    };

    if should_wake {
        npc.state = NpcState::HomeTask;
        npc.state_timer = 0.0;
    }
}

fn npc_home_task(world: &mut WorldData, i: usize, terrain: &Terrain, dt: f32) {
    let npc = &world.npcs[i];
    let home = &world.buildings[npc.home_idx];
    let hx = home.x;
    let hz = home.z;
    let hw = home.w * 0.4;
    let hd = home.d * 0.4;

    world.npcs[i].state_timer += dt;

    // Wander inside building footprint
    let npc = &world.npcs[i];
    let dx = npc.target_x - npc.x;
    let dz = npc.target_z - npc.z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist < 1.5 {
        // Pick new target inside building
        let npc = &mut world.npcs[i];
        npc.target_x = (hx + npc.rng.range(-hw, hw)).clamp(hx - hw, hx + hw);
        npc.target_z = (hz + npc.rng.range(-hd, hd)).clamp(hz - hd, hz + hd);
    }

    // Simple walk (no collision check inside building)
    let npc = &mut world.npcs[i];
    let speed = NPC_SPEED * 0.5;
    if dist > 0.5 {
        let desired = (-dx).atan2(-dz);
        let mut diff = desired - npc.rot_y;
        while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
        while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
        npc.rot_y += diff.clamp(-4.0 * dt, 4.0 * dt);
        npc.x -= npc.rot_y.sin() * speed * dt;
        npc.z -= npc.rot_y.cos() * speed * dt;
        npc.walk_phase += dt * speed * 2.5;
    }
    // Clamp inside building
    npc.x = npc.x.clamp(hx - hw, hx + hw);
    npc.z = npc.z.clamp(hz - hd, hz + hd);
    npc.y = terrain.height_at(npc.x, npc.z);

    if npc.state_timer >= HOME_TASK_DURATION {
        npc.state = NpcState::GoingToWork;
        npc.state_timer = 0.0;
        npc.detouring = false;
        npc.stuck_timer = 0.0;
    }
}

fn npc_going_to_work(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Find nearest unclaimed item to set as work target area
    let npc = &world.npcs[i];
    if npc.target_item.is_none() {
        let item_idx = find_best_item(world, i);
        if let Some(idx) = item_idx {
            world.npcs[i].target_x = world.items[idx].x;
            world.npcs[i].target_z = world.items[idx].z;
            world.npcs[i].target_item = Some(idx);
            world.items[idx].claimed_by = Some(i);
        }
    }

    let npc = &world.npcs[i];
    let tx = npc.target_x;
    let tz = npc.target_z;

    // Check if should drive
    let dx = tx - npc.x;
    let dz = tz - npc.z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist > NPC_DRIVE_THRESHOLD && !npc.carrying_item && npc.carrying_bin.is_none() {
        // Drive to work
        npc_enter_car(world, i, terrain);
        return;
    }

    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    if remaining < 3.0 {
        world.npcs[i].state = NpcState::Working;
        world.npcs[i].state_timer = 0.0;
    }
}

fn npc_working(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    world.npcs[i].state_timer += dt;

    let npc = &world.npcs[i];

    if npc.state_timer >= WORK_DURATION {
        // Work day done, go home
        world.npcs[i].state = NpcState::GoingHome;
        world.npcs[i].state_timer = 0.0;
        world.npcs[i].target_item = None;
        world.npcs[i].target_bin = None;
        world.npcs[i].detouring = false;
        // Unclaim any item
        unclaim_item(world, i);
        // Set home as target
        let home = &world.buildings[world.npcs[i].home_idx];
        world.npcs[i].target_x = home.x;
        world.npcs[i].target_z = home.z;
        return;
    }

    if npc.carrying_item {
        // Find nearest bin to deposit
        if npc.target_bin.is_none() {
            let bin_idx = find_nearest_bin(world, npc.x, npc.z);
            world.npcs[i].target_bin = bin_idx;
        }
        if let Some(bi) = world.npcs[i].target_bin {
            let bx = world.trash_bins[bi].x;
            let bz = world.trash_bins[bi].z;
            let remaining = npc_walk_toward(world, i, bx, bz, net, terrain, dt);
            if remaining < NPC_BIN_DIST {
                // Deposit item
                world.npcs[i].carrying_item = false;
                world.npcs[i].target_bin = None;
                world.npcs[i].items_deposited_today += 1;
                world.npcs[i].money += 1.0;
                world.trash_bins[bi].items_held += 1;
            }
        } else {
            // No bin available, wander
            pick_npc_wander_target(&mut world.npcs[i]);
            let tx = world.npcs[i].target_x;
            let tz = world.npcs[i].target_z;
            npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        }
    } else if world.npcs[i].carrying_bin.is_some() {
        // Carrying bin: walk to item-dense area, then set down
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        if remaining < 2.0 {
            // Set bin down at current position
            if let Some(bi) = world.npcs[i].carrying_bin {
                world.trash_bins[bi].x = world.npcs[i].x;
                world.trash_bins[bi].z = world.npcs[i].z;
                world.trash_bins[bi].y = terrain.height_at(world.npcs[i].x, world.npcs[i].z);
                world.trash_bins[bi].carried_by = None;
                world.npcs[i].carrying_bin = None;
            }
        }
    } else {
        // Not carrying anything — find an item
        if world.npcs[i].target_item.is_none() {
            let item_idx = find_best_item(world, i);
            if let Some(idx) = item_idx {
                world.npcs[i].target_item = Some(idx);
                world.items[idx].claimed_by = Some(i);
                world.npcs[i].target_x = world.items[idx].x;
                world.npcs[i].target_z = world.items[idx].z;

                // Check if we should relocate a bin closer
                let nearest_bin = find_nearest_bin(world, world.items[idx].x, world.items[idx].z);
                if let Some(bi) = nearest_bin {
                    let bdx = world.trash_bins[bi].x - world.items[idx].x;
                    let bdz = world.trash_bins[bi].z - world.items[idx].z;
                    let bin_dist = (bdx * bdx + bdz * bdz).sqrt();
                    if bin_dist > BIN_RELOCATE_DIST && world.trash_bins[bi].carried_by.is_none() {
                        // Relocate bin: pick it up and carry closer
                        world.npcs[i].carrying_bin = Some(bi);
                        world.trash_bins[bi].carried_by = Some(i);
                        world.npcs[i].target_x = world.items[idx].x;
                        world.npcs[i].target_z = world.items[idx].z;
                        // Unclaim item — we'll pick it up after setting bin down
                        world.items[idx].claimed_by = None;
                        world.npcs[i].target_item = None;
                        // First walk to the bin
                        world.npcs[i].target_x = world.trash_bins[bi].x;
                        world.npcs[i].target_z = world.trash_bins[bi].z;
                        return;
                    }
                }
            } else {
                // No items — wander
                pick_npc_wander_target(&mut world.npcs[i]);
            }
        }

        if let Some(item_idx) = world.npcs[i].target_item {
            // Check if item is still valid
            if item_idx < world.items.len() && world.items[item_idx].active && !world.items[item_idx].falling {
                let ix = world.items[item_idx].x;
                let iz = world.items[item_idx].z;
                let remaining = npc_walk_toward(world, i, ix, iz, net, terrain, dt);
                if remaining < NPC_PICKUP_DIST {
                    // Pick up item
                    world.items[item_idx].active = false;
                    world.items[item_idx].claimed_by = None;
                    world.npcs[i].carrying_item = true;
                    world.npcs[i].target_item = None;
                }
            } else {
                // Item no longer available
                if item_idx < world.items.len() {
                    world.items[item_idx].claimed_by = None;
                }
                world.npcs[i].target_item = None;
            }
        } else {
            let tx = world.npcs[i].target_x;
            let tz = world.npcs[i].target_z;
            npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        }
    }
}

fn npc_going_home(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    let npc = &world.npcs[i];
    let home = &world.buildings[npc.home_idx];
    let hx = home.x;
    let hz = home.z;

    let dx = hx - npc.x;
    let dz = hz - npc.z;
    let dist = (dx * dx + dz * dz).sqrt();

    // Drive if far and not carrying anything
    if dist > NPC_DRIVE_THRESHOLD && !npc.carrying_item && npc.carrying_bin.is_none() && !npc.in_vehicle {
        npc_enter_car(world, i, terrain);
        return;
    }

    let remaining = npc_walk_toward(world, i, hx, hz, net, terrain, dt);
    if remaining < 3.0 {
        world.npcs[i].state = NpcState::Sleeping;
        world.npcs[i].state_timer = 0.0;
        world.npcs[i].walk_phase = 0.0;
    }
}

fn npc_driving(world: &mut WorldData, i: usize, terrain: &Terrain, _dt: f32) {
    let car_idx = world.npcs[i].car_idx;
    if car_idx >= world.vehicles.len() {
        // Invalid car, exit driving
        world.npcs[i].in_vehicle = false;
        world.npcs[i].state = NpcState::Working;
        return;
    }

    // Sync NPC position to vehicle
    world.npcs[i].x = world.vehicles[car_idx].x;
    world.npcs[i].y = world.vehicles[car_idx].y;
    world.npcs[i].z = world.vehicles[car_idx].z;

    // Check if we've arrived at target
    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    let dx = tx - world.vehicles[car_idx].x;
    let dz = tz - world.vehicles[car_idx].z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist < 8.0 {
        // Park and exit vehicle
        npc_exit_car(world, i, terrain);
    } else {
        // Drive toward target using AI
        // Set vehicle AI target
        world.vehicles[car_idx].ai_target_x = tx;
        world.vehicles[car_idx].ai_target_z = tz;
        world.vehicles[car_idx].ai_active = true;
    }
}

fn npc_enter_car(world: &mut WorldData, i: usize, _terrain: &Terrain) {
    let car_idx = world.npcs[i].car_idx;
    if car_idx >= world.vehicles.len() { return; }
    if world.vehicles[car_idx].occupied { return; } // player is in it

    // Check distance to car
    let dx = world.vehicles[car_idx].x - world.npcs[i].x;
    let dz = world.vehicles[car_idx].z - world.npcs[i].z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist > 5.0 {
        // Walk to car first — set car position as intermediate target
        world.npcs[i].target_x = world.vehicles[car_idx].x;
        world.npcs[i].target_z = world.vehicles[car_idx].z;
        return;
    }

    // Enter car
    world.npcs[i].parked_x = world.npcs[i].x;
    world.npcs[i].parked_z = world.npcs[i].z;
    world.npcs[i].in_vehicle = true;
    world.npcs[i].state = NpcState::Driving;
    world.vehicles[car_idx].ai_active = true;
    world.vehicles[car_idx].occupied = false; // NPC driving, not player
}

fn npc_exit_car(world: &mut WorldData, i: usize, terrain: &Terrain) {
    let car_idx = world.npcs[i].car_idx;
    if car_idx >= world.vehicles.len() { return; }

    // Exit to side of vehicle
    let v = &world.vehicles[car_idx];
    let exit_x = v.x + v.rot_y.sin() * 2.5;
    let exit_z = v.z + v.rot_y.cos() * 2.5;
    let exit_y = terrain.height_at(exit_x, exit_z);

    world.npcs[i].x = exit_x;
    world.npcs[i].y = exit_y;
    world.npcs[i].z = exit_z;
    world.npcs[i].in_vehicle = false;
    world.vehicles[car_idx].ai_active = false;
    world.vehicles[car_idx].speed = 0.0;

    // Return to previous intent state
    // Check what we were doing before driving
    let npc = &world.npcs[i];
    if npc.state_timer < WORK_DURATION {
        world.npcs[i].state = NpcState::Working;
    } else {
        world.npcs[i].state = NpcState::GoingHome;
    }
    world.npcs[i].detouring = false;
    world.npcs[i].stuck_timer = 0.0;
}

fn find_best_item(world: &WorldData, npc_idx: usize) -> Option<usize> {
    let npc = &world.npcs[npc_idx];
    let mut best_dist = f32::MAX;
    let mut best_idx = None;

    for (idx, item) in world.items.iter().enumerate() {
        if !item.active { continue; }
        if item.falling { continue; }
        if let Some(claimer) = item.claimed_by {
            if claimer != npc_idx { continue; }
        }
        let dx = item.x - npc.x;
        let dz = item.z - npc.z;
        let dist = dx * dx + dz * dz;
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(idx);
        }
    }
    best_idx
}

fn find_nearest_bin(world: &WorldData, x: f32, z: f32) -> Option<usize> {
    let mut best_dist = f32::MAX;
    let mut best_idx = None;

    for (idx, bin) in world.trash_bins.iter().enumerate() {
        if bin.carried_by.is_some() { continue; }
        let dx = bin.x - x;
        let dz = bin.z - z;
        let dist = dx * dx + dz * dz;
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(idx);
        }
    }
    best_idx
}

fn unclaim_item(world: &mut WorldData, npc_idx: usize) {
    for item in &mut world.items {
        if item.claimed_by == Some(npc_idx) {
            item.claimed_by = None;
        }
    }
}

fn pick_npc_wander_target(npc: &mut Npc) {
    npc.target_x = npc.x + npc.rng.range(-15.0, 15.0);
    npc.target_z = npc.z + npc.rng.range(-15.0, 15.0);
    npc.target_x = npc.target_x.clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
    npc.target_z = npc.target_z.clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
}

// Night sky spawning system
pub fn sys_night_spawning(
    world: &mut WorldData, terrain: &Terrain, time_of_day: f32,
    dt: f32, rng: &mut Rng,
) {
    // Active during night hours (20:00–04:00)
    let is_night = time_of_day >= NIGHT_SPAWN_START || time_of_day < NIGHT_SPAWN_END;

    // Update falling items
    for item in &mut world.items {
        if item.falling {
            item.vel_y -= 15.0 * dt;
            item.y += item.vel_y * dt;
            let ground = terrain.height_at(item.x, item.z);
            if item.y <= ground {
                item.y = ground;
                item.vel_y = 0.0;
                item.falling = false;
                item.active = true;
            }
        }
    }

    if !is_night { return; }

    // Count active items — only spawn if below threshold
    let active_count = world.items.iter().filter(|it| it.active || it.falling).count();
    if active_count >= NUM_ITEMS { return; }

    // Spawn items from sky at random intervals
    // Use a probabilistic approach: chance per frame
    let spawn_chance = dt / NIGHT_SPAWN_INTERVAL;
    if (rng.next() as f32 / u64::MAX as f32) < spawn_chance {
        let x = rng.range(-WORLD_HALF + 10.0, WORLD_HALF - 10.0);
        let z = rng.range(-WORLD_HALF + 10.0, WORLD_HALF - 10.0);
        let y = 40.0 + rng.range(0.0, 20.0);
        let kinds = [ItemKind::Health, ItemKind::Money, ItemKind::Stamina];
        let kind = kinds[rng.next() as usize % 3];

        // Find an inactive item slot to reuse, or push new
        let mut found = false;
        for item in &mut world.items {
            if !item.active && !item.falling {
                item.x = x;
                item.y = y;
                item.z = z;
                item.kind = kind;
                item.falling = true;
                item.vel_y = 0.0;
                item.active = false;
                item.claimed_by = None;
                item.spin_phase = 0.0;
                found = true;
                break;
            }
        }
        if !found {
            world.items.push(Item {
                x, y, z, kind, active: false, respawn_timer: 0.0,
                spin_phase: 0.0, falling: true, vel_y: 0.0, claimed_by: None,
            });
        }
    }
}

/// Reset daily counters at midnight
pub fn sys_midnight_reset(world: &mut WorldData, time_of_day: f32, prev_time: f32) -> bool {
    // Detect midnight crossing
    if prev_time > 23.5 && time_of_day < 0.5 {
        // Reset trash bins
        for bin in &mut world.trash_bins {
            bin.items_held = 0;
        }
        // Reset NPC daily counters
        for npc in &mut world.npcs {
            npc.items_deposited_today = 0;
        }
        return true; // day changed
    }
    false
}

/// Update item spin animation
pub fn sys_items_update(world: &mut WorldData, dt: f32) {
    for item in &mut world.items {
        if item.active {
            item.spin_phase += dt * 3.0;
        }
    }
}

/// Context-sensitive player interaction (Interact key / E):
/// - Near item + not carrying → pick up item
/// - Near bin + carrying item → deposit item in bin (earns $1)
/// - Near bin + not carrying anything → pick up bin
/// - Carrying bin + press E → set bin down
/// Returns (sparkle_x, sparkle_z, sparkle_color) for particle effects
pub fn sys_player_interact(
    world: &mut WorldData, player: &mut Player, terrain: &Terrain,
    interact_pressed: bool,
) -> Option<(f32, f32, u32)> {
    if !interact_pressed { return None; }
    if player.in_vehicle.is_some() { return None; } // vehicle enter/exit handled in vehicle.rs

    let px = player.x;
    let pz = player.z;

    // Priority 1: Carrying item → deposit at nearest bin
    if player.carrying_item {
        let mut best_dist = NPC_BIN_DIST * NPC_BIN_DIST;
        let mut best_bi = None;
        for (bi, bin) in world.trash_bins.iter().enumerate() {
            if bin.carried_by.is_some() { continue; }
            let dx = px - bin.x;
            let dz = pz - bin.z;
            let d2 = dx * dx + dz * dz;
            if d2 < best_dist {
                best_dist = d2;
                best_bi = Some(bi);
            }
        }
        if let Some(bi) = best_bi {
            player.carrying_item = false;
            player.money += 1.0;
            world.trash_bins[bi].items_held += 1;
            return Some((world.trash_bins[bi].x, world.trash_bins[bi].z, 0xFFFFDD33));
        }
        return None; // carrying but no bin nearby
    }

    // Priority 2: Carrying bin → set it down
    if let Some(bi) = player.carrying_bin {
        if bi < world.trash_bins.len() {
            world.trash_bins[bi].x = px;
            world.trash_bins[bi].z = pz;
            world.trash_bins[bi].y = terrain.height_at(px, pz);
            world.trash_bins[bi].carried_by = None;
        }
        player.carrying_bin = None;
        return None;
    }

    // Priority 3: Near item → pick it up
    {
        let mut best_dist = NPC_PICKUP_DIST * NPC_PICKUP_DIST;
        let mut best_ii = None;
        for (ii, item) in world.items.iter().enumerate() {
            if !item.active || item.falling { continue; }
            let dx = px - item.x;
            let dz = pz - item.z;
            let d2 = dx * dx + dz * dz;
            if d2 < best_dist {
                best_dist = d2;
                best_ii = Some(ii);
            }
        }
        if let Some(ii) = best_ii {
            let color = match world.items[ii].kind {
                ItemKind::Health => 0xFFFF3333,
                ItemKind::Money => 0xFFFFDD33,
                ItemKind::Stamina => 0xFF33FF33,
            };
            world.items[ii].active = false;
            world.items[ii].claimed_by = None;
            player.carrying_item = true;
            return Some((world.items[ii].x, world.items[ii].z, color));
        }
    }

    // Priority 4: Near bin → pick it up
    {
        let mut best_dist = NPC_BIN_DIST * NPC_BIN_DIST;
        let mut best_bi = None;
        for (bi, bin) in world.trash_bins.iter().enumerate() {
            if bin.carried_by.is_some() { continue; }
            let dx = px - bin.x;
            let dz = pz - bin.z;
            let d2 = dx * dx + dz * dz;
            if d2 < best_dist {
                best_dist = d2;
                best_bi = Some(bi);
            }
        }
        if let Some(bi) = best_bi {
            player.carrying_bin = Some(bi);
            world.trash_bins[bi].carried_by = Some(usize::MAX); // special marker for player
            return None;
        }
    }

    None
}
