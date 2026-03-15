// NPC life simulation: state machine, physics, pathfinding, item pickup/deposit, night spawning

use crate::state::*;
use crate::math::dist_sq_2d;
use crate::world::{check_walk_collision, surface_at, on_river_not_bridge, on_any_road};
use crate::rng::Rng;

// Home task duration: 4 game-hours = 4 * 60 = 240 real seconds
const HOME_TASK_DURATION: f32 = 240.0;
// Item spawn interval: ~1 item every 1.5 seconds (items are collected efficiently)
const NIGHT_SPAWN_INTERVAL: f32 = 1.0;

pub fn sys_npc(
    world: &mut WorldData, road_network: &mut RoadNetwork, terrain: &Terrain,
    dt: f32, time_of_day: f32, brains: &mut [crate::neat::NeatBrain],
    player_x: f32, player_z: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    let n = world.npcs.len();
    for i in 0..n {
        // Physics: gravity + ground snap
        npc_physics(world, i, terrain, road_network, dt);

        // State machine
        let prev_state = world.npcs[i].state;
        match prev_state {
            NpcState::Sleeping => npc_sleeping(world, i, time_of_day),
            NpcState::HomeTask => npc_home_task(world, i, terrain, dt),
            NpcState::GoingToWork => npc_going_to_work(world, i, road_network, terrain, dt, walk_grid),
            NpcState::Working => npc_working(world, i, road_network, terrain, dt, brains, time_of_day, player_x, player_z, walk_grid),
            NpcState::GoingHome => npc_going_home(world, i, road_network, terrain, dt, walk_grid),
            NpcState::Driving => npc_driving(world, i, terrain, road_network, dt),
            NpcState::Interacting => {} // handled by sys_npc_interactions
            NpcState::KnockedOut => {}  // recovery handled by combat.rs
        }

        // NN picks job when transitioning to GoingToWork
        if prev_state == NpcState::HomeTask && world.npcs[i].state == NpcState::GoingToWork {
            let bi = world.npcs[i].brain_idx;
            if bi < brains.len() {
                let inputs = crate::neat::gather_inputs(world, i, road_network, time_of_day, player_x, player_z);
                let outputs = brains[bi].activate(&inputs);
                let job_val = outputs[13]; // 0.0–1.0 sigmoid
                let job_idx = (job_val * NPC_JOB_COUNT as f32).min(NPC_JOB_COUNT as f32 - 1.0) as usize;
                world.npcs[i].job = crate::neat::ALL_JOBS[job_idx];
                world.npcs[i].job_timer = 0.0;
                world.npcs[i].interaction_target = None;
            }
        }
    }
}

fn npc_physics(world: &mut WorldData, i: usize, terrain: &Terrain, road_network: &RoadNetwork, dt: f32) {
    if world.npcs[i].in_vehicle || world.npcs[i].state == NpcState::Sleeping || world.npcs[i].state == NpcState::Interacting {
        // Keep body in sync for inactive NPCs
        world.npcs[i].body.pos = [world.npcs[i].x, world.npcs[i].y, world.npcs[i].z];
        world.npcs[i].body.quat = crate::math::quat_from_rot_y(world.npcs[i].rot_y);
        world.npcs[i].body.vel = [0.0; 3];
        return;
    }

    // Slope sliding force — steeper slope + lower friction = more slide
    if world.npcs[i].on_ground {
        let raw_n = terrain.normal_at(world.npcs[i].body.pos[0], world.npcs[i].body.pos[2]);
        let slope = (1.0 - raw_n[1]).max(0.0);
        if slope > 0.12 {
            let surface = crate::world::surface_at(
                world.npcs[i].body.pos[0], world.npcs[i].body.pos[2], road_network,
            );
            let friction = crate::material::material_for_surface(surface).dynamic_friction;
            let mass = world.npcs[i].body.mass;
            // Slide force inversely proportional to friction: ice slides hard, asphalt barely
            let slide_mag = slope * slope * 40.0 * mass * (1.0 - friction).max(0.0);
            world.npcs[i].body.apply_force([-raw_n[0] * slide_mag, 0.0, -raw_n[2] * slide_mag]);
        }
    }

    // Save pre-integration position for collision rollback
    let prev_pos = world.npcs[i].body.pos;

    // Integrate rigid body (gravity + walk force + slope force + damping)
    crate::physics::integrate(&mut world.npcs[i].body, dt);

    // Ground contact enforcement — capsule vs terrain with proper normal response
    let surface = crate::world::surface_at(
        world.npcs[i].body.pos[0], world.npcs[i].body.pos[2], road_network,
    );
    let surface_mat = *crate::material::material_for_surface(surface);
    if let Some(contact) = crate::physics::capsule_ground_contact(
        i, &world.npcs[i].body, 0.3, 0.6, terrain, surface_mat,
    ) {
        // Push character out of ground along terrain normal
        world.npcs[i].body.pos = crate::math::v3_add(
            world.npcs[i].body.pos,
            crate::math::v3_scale(contact.normal, contact.penetration),
        );
        // Kill velocity into the ground (project out the component along normal)
        let vel_into = crate::math::v3_dot(world.npcs[i].body.vel, contact.normal);
        if vel_into < 0.0 {
            world.npcs[i].body.vel = crate::math::v3_sub(
                world.npcs[i].body.vel,
                crate::math::v3_scale(contact.normal, vel_into),
            );
        }
        world.npcs[i].on_ground = true;
    } else {
        world.npcs[i].on_ground = false;
    }

    // Character-on-vehicle stacking: NPC standing on vehicle roof
    if !world.npcs[i].ragdoll_active {
        let feet_pos = world.npcs[i].body.pos;
        let ground = terrain.height_at(feet_pos[0], feet_pos[2]);
        for vi in 0..world.vehicles.len() {
            let v = &world.vehicles[vi];
            let dx = feet_pos[0] - v.x;
            let dz = feet_pos[2] - v.z;
            if dx.abs() > 5.0 || dz.abs() > 5.0 { continue; }
            let half_w = 0.93 * v.scale;
            let half_d = 2.3 * v.scale;
            let roof_h = 1.2 * v.scale;
            if let Some((_normal, surface_y)) = crate::physics::point_on_vehicle_surface(
                feet_pos, v.body.pos, v.rot_y, half_w, half_d, roof_h,
            ) {
                if surface_y > ground {
                    world.npcs[i].body.pos[1] = surface_y;
                    if world.npcs[i].body.vel[1] < 0.0 { world.npcs[i].body.vel[1] = 0.0; }
                    world.npcs[i].on_ground = true;
                    // Transfer vehicle velocity via friction
                    let friction = 0.4;
                    let vvel = v.body.vel;
                    world.npcs[i].body.vel[0] += (vvel[0] - world.npcs[i].body.vel[0]) * friction * dt.min(0.1);
                    world.npcs[i].body.vel[2] += (vvel[2] - world.npcs[i].body.vel[2]) * friction * dt.min(0.1);
                    break;
                }
            }
        }
    }

    // Building collision (axis-separated sliding)
    let home_idx = world.npcs[i].home_idx;
    if check_walk_collision(world, world.npcs[i].body.pos[0], prev_pos[2], 0.4, Some(home_idx)) {
        world.npcs[i].body.pos[0] = prev_pos[0];
        world.npcs[i].body.vel[0] = 0.0;
    }
    if check_walk_collision(world, world.npcs[i].body.pos[0], world.npcs[i].body.pos[2], 0.4, Some(home_idx)) {
        world.npcs[i].body.pos[2] = prev_pos[2];
        world.npcs[i].body.vel[2] = 0.0;
    }

    // World bounds
    world.npcs[i].body.pos[0] = world.npcs[i].body.pos[0].clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
    world.npcs[i].body.pos[2] = world.npcs[i].body.pos[2].clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);

    // Sync body → legacy fields
    // y = feet position (body center - capsule half_height - radius)
    world.npcs[i].x = world.npcs[i].body.pos[0];
    world.npcs[i].y = world.npcs[i].body.pos[1] - 0.9; // capsule bottom offset
    world.npcs[i].z = world.npcs[i].body.pos[2];
    world.npcs[i].vel_y = world.npcs[i].body.vel[1];

    // Smooth terrain normal for slope tilting
    let raw_n = terrain.normal_at(world.npcs[i].x, world.npcs[i].z);
    let target_n = crate::math::clamp_normal_tilt(raw_n, 25.0);
    let lerp_rate = 8.0 * dt;
    world.npcs[i].terrain_normal = crate::math::v3_normalize(
        crate::math::v3_lerp(world.npcs[i].terrain_normal, target_n, lerp_rate.min(1.0))
    );

    // Keep body quaternion in sync with rendering rotation
    world.npcs[i].body.quat = crate::math::quat_from_rot_y(world.npcs[i].rot_y);

    // Landing detection: trigger ragdoll from high falls BEFORE animation consumes landing_speed
    let vel = world.npcs[i].body.vel;
    let pos = world.npcs[i].body.pos;
    let rot_y = world.npcs[i].rot_y;
    let on_ground = world.npcs[i].on_ground;
    if on_ground && !world.npcs[i].ragdoll_active
        && world.npcs[i].skeleton.should_ragdoll_from_fall()
    {
        let impulse = [vel[0] * 0.5, 0.0, vel[2] * 0.5];
        crate::collision::init_ragdoll(&mut world.npcs[i], impulse[0], impulse[1], impulse[2]);
    }

    // Procedural animation: skeleton IK + walk cycle from physics velocity
    let mass = world.npcs[i].body.mass;
    world.npcs[i].skeleton.step_animation(vel, pos, rot_y, terrain, on_ground, mass, dt);

    // Ragdoll blend recovery (when ragdoll timer expired)
    if world.npcs[i].skeleton.ragdoll_active {
        world.npcs[i].skeleton.blend_from_ragdoll(pos, rot_y, dt);
    }

    // Sync skeleton walk_phase to legacy walk_phase for renderer
    world.npcs[i].walk_phase = world.npcs[i].skeleton.walk_phase;
}

/// Walk NPC toward (target_x, target_z) using A* navmesh pathfinding.
/// Computes path on demand, follows waypoints, handles collision sliding.
pub fn npc_walk_toward(
    world: &mut WorldData, i: usize, tx: f32, tz: f32,
    net: &RoadNetwork, terrain: &Terrain, dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) -> f32 {
    let npc = &world.npcs[i];
    if npc.in_vehicle { return 0.0; }

    let dx_to_goal = tx - npc.x;
    let dz_to_goal = tz - npc.z;
    let dist_to_goal = (dx_to_goal * dx_to_goal + dz_to_goal * dz_to_goal).sqrt();
    if dist_to_goal < 0.5 {
        world.npcs[i].nav_path.clear();
        world.npcs[i].nav_path_idx = 0;
        return dist_to_goal;
    }

    // Recompute path if target changed significantly or path is empty/exhausted
    let target_changed = {
        let ntx = world.npcs[i].nav_target_x;
        let ntz = world.npcs[i].nav_target_z;
        (tx - ntx) * (tx - ntx) + (tz - ntz) * (tz - ntz) > 4.0 // >2m change
    };
    let path_exhausted = world.npcs[i].nav_path_idx >= world.npcs[i].nav_path.len();

    if target_changed || (path_exhausted && dist_to_goal > 2.0) {
        let path = walk_grid.find_path(world.npcs[i].x, world.npcs[i].z, tx, tz);
        world.npcs[i].nav_path = path;
        world.npcs[i].nav_path_idx = 0;
        world.npcs[i].nav_target_x = tx;
        world.npcs[i].nav_target_z = tz;

        // If path is empty (unreachable), release claimed items
        if world.npcs[i].nav_path.is_empty() {
            if let Some(item_idx) = world.npcs[i].target_item {
                if item_idx < world.items.len() {
                    world.items[item_idx].claimed_by = None;
                }
                world.npcs[i].target_item = None;
            }
            if let Some(bin_idx) = world.npcs[i].target_bin {
                if bin_idx < world.trash_bins.len() {
                    world.trash_bins[bin_idx].carried_by = None;
                }
                world.npcs[i].target_bin = None;
            }
            return dist_to_goal;
        }
    }

    // Determine immediate walk target: next path waypoint, or final goal if close
    let (walk_tx, walk_tz) = if world.npcs[i].nav_path_idx < world.npcs[i].nav_path.len() {
        let wp = world.npcs[i].nav_path[world.npcs[i].nav_path_idx];
        // Advance waypoint if we're close enough
        let wp_dx = wp[0] - world.npcs[i].x;
        let wp_dz = wp[1] - world.npcs[i].z;
        if wp_dx * wp_dx + wp_dz * wp_dz < 1.5 * 1.5 {
            world.npcs[i].nav_path_idx += 1;
            if world.npcs[i].nav_path_idx < world.npcs[i].nav_path.len() {
                let next = world.npcs[i].nav_path[world.npcs[i].nav_path_idx];
                (next[0], next[1])
            } else {
                (tx, tz) // path exhausted, walk directly to goal
            }
        } else {
            (wp[0], wp[1])
        }
    } else {
        (tx, tz) // no path, walk directly
    };

    // Walk direction: toward waypoint (normalized)
    let dx = walk_tx - world.npcs[i].x;
    let dz = walk_tz - world.npcs[i].z;
    let walk_dist = (dx * dx + dz * dz).sqrt();
    let walk_dir = if walk_dist > 0.1 {
        [dx / walk_dist, 0.0, dz / walk_dist]
    } else {
        [0.0, 0.0, 0.0]
    };

    // Face the direction we're actually moving (snap rot_y to movement direction)
    let npc = &mut world.npcs[i];
    if walk_dist > 0.1 {
        let desired = (-dx).atan2(-dz);
        let mut diff = desired - npc.rot_y;
        while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
        while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
        npc.rot_y += diff.clamp(-8.0 * dt, 8.0 * dt);
    }

    // Gait + speed selection — surface-dependent speed cap (terrain/steep = slower)
    let surface = surface_at(npc.x, npc.z, net);
    let desired_gait = crate::skeleton::Gait::Walk;
    let terrain_ny = terrain.normal_at(npc.x, npc.z)[1];
    let max_speed = crate::state::npc_speed_for_surface(surface, terrain_ny) * npc.walk_speed_mult;

    // Locomotion: walk TOWARD target, not in facing direction
    let surface_friction = crate::material::material_for_surface(surface).dynamic_friction;
    let walk_force = npc.skeleton.compute_locomotion_force(
        walk_dir, desired_gait, npc.body.vel, npc.body.mass, surface_friction, max_speed,
    );
    npc.body.apply_force(walk_force);

    let dx2 = tx - world.npcs[i].x;
    let dz2 = tz - world.npcs[i].z;
    (dx2 * dx2 + dz2 * dz2).sqrt()
}

fn find_nearest_vending_machine(world: &WorldData, x: f32, z: f32) -> Option<(f32, f32)> {
    let mut best_dist = f32::MAX;
    let mut best = None;
    for inter in &world.interactibles {
        if inter.kind != InteractibleKind::VendingMachine { continue; }
        let d = dist_sq_2d(inter.x, inter.z, x, z);
        if d < best_dist { best_dist = d; best = Some((inter.x, inter.z)); }
    }
    best
}

fn npc_sleeping(world: &mut WorldData, i: usize, time_of_day: f32) {
    let npc = &mut world.npcs[i];
    // Position NPC inside home building (hidden)
    let home = &world.buildings[npc.home_idx];
    npc.x = home.x;
    npc.z = home.z;
    npc.y = home.ground_y;
    npc.walk_phase = 0.0; // no animation
    npc.sound = [0.0; 3];

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

    // Walk toward target using physics forces — indoor walking gait
    let npc = &mut world.npcs[i];
    if dist > 0.5 {
        let desired = (-dx).atan2(-dz);
        let mut diff = desired - npc.rot_y;
        while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
        while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
        npc.rot_y += diff.clamp(-4.0 * dt, 4.0 * dt);
        let desired_dir = [-npc.rot_y.sin(), 0.0, -npc.rot_y.cos()];
        let walk_force = npc.skeleton.compute_locomotion_force(
            desired_dir, crate::skeleton::Gait::Walk, npc.body.vel, npc.body.mass,
            crate::material::MAT_CONCRETE.dynamic_friction, // indoor floor
            crate::state::NPC_SPEED_SIDEWALK * npc.walk_speed_mult, // indoor = smooth floor, scaled by height
        );
        npc.body.apply_force(walk_force);
    }
    // Clamp inside building
    npc.x = npc.x.clamp(hx - hw, hx + hw);
    npc.z = npc.z.clamp(hz - hd, hz + hd);
    npc.y = terrain.height_at(npc.x, npc.z);

    if npc.state_timer >= HOME_TASK_DURATION {
        npc.state = NpcState::GoingToWork;
        npc.state_timer = 0.0;
        npc.stuck_timer = 0.0;
        npc.nav_path.clear();
    }
}

fn npc_going_to_work(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32, walk_grid: &crate::navmesh::WalkGrid) {
    world.npcs[i].state_timer += dt;

    // Timeout: if can't reach item in 60s, just start working where you are
    if world.npcs[i].state_timer > 60.0 {
        world.npcs[i].state = NpcState::Working;
        world.npcs[i].state_timer = 0.0;
        world.npcs[i].target_item = None;
        unclaim_item(world, i);
        return;
    }

    // Find nearest unclaimed item to set as work target area
    let npc = &world.npcs[i];
    if npc.target_item.is_none() {
        let item_idx = find_closest_item(world, i);
        if let Some(idx) = item_idx {
            world.npcs[i].target_x = world.items[idx].x;
            world.npcs[i].target_z = world.items[idx].z;
            world.npcs[i].target_item = Some(idx);
            world.items[idx].claimed_by = Some(i);
        } else {
            // No items available — just start working immediately
            world.npcs[i].state = NpcState::Working;
            world.npcs[i].state_timer = 0.0;
            return;
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
        if npc_enter_car(world, i, terrain, net) { return; }
        // Car too far — walk toward car (target was redirected by npc_enter_car)
    }

    let tx = world.npcs[i].target_x; // re-read in case redirected to car
    let tz = world.npcs[i].target_z;
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
    if remaining < 3.0 {
        world.npcs[i].state = NpcState::Working;
        world.npcs[i].state_timer = 0.0;
        world.npcs[i].stuck_timer = 0.0;
    }
}

fn npc_working(
    world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain,
    dt: f32, brains: &mut [crate::neat::NeatBrain], time_of_day: f32,
    player_x: f32, player_z: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    world.npcs[i].state_timer += dt;

    if world.npcs[i].state_timer >= WORK_DURATION {
        // Work day done, go home
        world.npcs[i].state = NpcState::GoingHome;
        world.npcs[i].state_timer = 0.0;
        world.npcs[i].target_item = None;
        world.npcs[i].target_bin = None;
        world.npcs[i].interaction_target = None;
        unclaim_item(world, i);
        let home = &world.buildings[world.npcs[i].home_idx];
        world.npcs[i].target_x = home.x;
        world.npcs[i].target_z = home.z;
        return;
    }

    // Survival autopilot: traditional AI overrides NN when hungry/thirsty
    let needs_food = world.npcs[i].hunger < HUNGER_AUTOPILOT && world.npcs[i].money >= VENDING_FOOD_COST;
    let needs_water = world.npcs[i].thirst < THIRST_AUTOPILOT && world.npcs[i].money >= VENDING_DRINK_COST;
    if needs_food || needs_water {
        if let Some((vm_x, vm_z)) = find_nearest_vending_machine(world, world.npcs[i].x, world.npcs[i].z) {
            let dx = vm_x - world.npcs[i].x;
            let dz = vm_z - world.npcs[i].z;
            let dist = (dx * dx + dz * dz).sqrt();
            if dist < INTERACT_DIST {
                // At vending machine — buy
                if needs_water {
                    world.npcs[i].thirst = (world.npcs[i].thirst + VENDING_WATER_RESTORE).min(100.0);
                    world.npcs[i].money -= VENDING_DRINK_COST;
                } else if needs_food {
                    world.npcs[i].hunger = (world.npcs[i].hunger + VENDING_FOOD_RESTORE).min(100.0);
                    world.npcs[i].money -= VENDING_FOOD_COST;
                }
            } else {
                // Walk to vending machine
                npc_walk_toward(world, i, vm_x, vm_z, net, terrain, dt, walk_grid);
            }
            return;
        }
    }

    // Job AI handles work behavior and money earning (mut net for vehicle parking)
    crate::jobs::npc_work_dispatch(world, i, net, terrain, dt, walk_grid);

    // NN observation: fitness tracking + combat/sound outputs only
    let bi = world.npcs[i].brain_idx;
    if bi < brains.len() {
        let dx = world.npcs[i].x - world.npcs[i].prev_x;
        let dz = world.npcs[i].z - world.npcs[i].prev_z;
        let dist = (dx * dx + dz * dz).sqrt();
        world.npcs[i].fitness_distance += dist;
        if dist < 0.01 * dt {
            world.npcs[i].fitness_stuck_time += dt;
        }
        world.npcs[i].prev_x = world.npcs[i].x;
        world.npcs[i].prev_z = world.npcs[i].z;

        let inputs = crate::neat::gather_inputs(world, i, net, time_of_day, player_x, player_z);
        let outputs = brains[bi].activate(&inputs);

        // Combat + sound only (job AI handles movement/pickup/deposit)
        world.npcs[i].attack_intent = if outputs[8] > 0.5 { 1 } else if outputs[9] > 0.5 { 2 } else { 0 };
        world.npcs[i].sound = [outputs[10], outputs[11], outputs[12]];
        if outputs[10] > 0.1 || outputs[11] > 0.1 || outputs[12] > 0.1 {
            world.npcs[i].fitness_sounds_made += 1;
        }

        // Hearing fitness
        let any_heard = inputs[45] > 0.01 || inputs[46] > 0.01 || inputs[47] > 0.01
            || inputs[50] > 0.01 || inputs[51] > 0.01 || inputs[52] > 0.01;
        if any_heard {
            world.npcs[i].fitness_npcs_heard += 1;
        }
    }
}

fn npc_going_home(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32, walk_grid: &crate::navmesh::WalkGrid) {
    let npc = &world.npcs[i];
    let home = &world.buildings[npc.home_idx];
    let hx = home.x;
    let hz = home.z;

    let dx = hx - npc.x;
    let dz = hz - npc.z;
    let dist = (dx * dx + dz * dz).sqrt();

    // Drive if far and not carrying anything
    if dist > NPC_DRIVE_THRESHOLD && !npc.carrying_item && npc.carrying_bin.is_none() && !npc.in_vehicle {
        if npc_enter_car(world, i, terrain, net) { return; }
        // Car too far — walk toward car (target was redirected by npc_enter_car)
    }

    let hx = world.buildings[world.npcs[i].home_idx].x;
    let hz = world.buildings[world.npcs[i].home_idx].z;
    // Walk toward home (or toward car if redirected)
    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
    // Check if at home OR at car (to re-attempt driving next frame)
    let to_home = ((hx - world.npcs[i].x).powi(2) + (hz - world.npcs[i].z).powi(2)).sqrt();
    if to_home < 3.0 {
        world.npcs[i].state = NpcState::Sleeping;
        world.npcs[i].state_timer = 0.0;
        world.npcs[i].walk_phase = 0.0;
    } else if remaining < 3.0 {
        // Reached intermediate target (car or waypoint) — re-check next frame
        world.npcs[i].target_x = hx;
        world.npcs[i].target_z = hz;
    }
}

fn npc_driving(world: &mut WorldData, i: usize, terrain: &Terrain, net: &mut RoadNetwork, _dt: f32) {
    let car_idx = world.npcs[i].car_idx;
    if car_idx >= world.vehicles.len() {
        world.npcs[i].in_vehicle = false;
        world.npcs[i].state = NpcState::Working;
        return;
    }

    // Sync NPC position to vehicle — driver bounces with suspension
    world.npcs[i].x = world.vehicles[car_idx].x;
    world.npcs[i].z = world.vehicles[car_idx].z;
    // Cabin Y offset: average suspension compression shifts driver up/down relative to body
    let avg_comp = world.vehicles[car_idx].suspension.iter()
        .map(|s| s.compression)
        .sum::<f32>() * 0.25;
    // When suspension compresses positively, body drops → driver drops
    // rest_length is baseline; deviation from rest = movement
    let rest = world.vehicles[car_idx].suspension[0].params.rest_length;
    let seat_offset = (avg_comp - rest * 0.5) * 0.3; // damped fraction of suspension travel
    world.npcs[i].y = world.vehicles[car_idx].y + seat_offset;

    // Animate driver skeleton based on suspension bounce
    let susp_comp = [
        world.vehicles[car_idx].suspension[0].compression,
        world.vehicles[car_idx].suspension[1].compression,
        world.vehicles[car_idx].suspension[2].compression,
        world.vehicles[car_idx].suspension[3].compression,
    ];
    let veh_speed = world.vehicles[car_idx].speed;
    let steer = world.vehicles[car_idx].drivetrain.steer_input;
    world.npcs[i].skeleton.step_driving_animation(&susp_comp, veh_speed, steer, _dt);

    // Steering wheel IK
    {
        let v = &world.vehicles[car_idx];
        let local_wheel = [0.35 * v.scale, 0.65 * v.scale, -0.8 * v.scale];
        let (sin_r, cos_r) = v.rot_y.sin_cos();
        let wheel_world = [
            v.x + local_wheel[0] * cos_r - local_wheel[2] * sin_r,
            v.y + local_wheel[1],
            v.z + local_wheel[0] * sin_r + local_wheel[2] * cos_r,
        ];
        let steer_angle = steer * 35.0_f32.to_radians();
        let root_rot = crate::math::quat_from_rot_y(world.vehicles[car_idx].rot_y);
        let skel_pos = [world.npcs[i].x, world.npcs[i].y, world.npcs[i].z];
        world.npcs[i].skeleton.compute_world_transforms(skel_pos, root_rot);
        world.npcs[i].skeleton.apply_steering_wheel_ik(wheel_world, steer_angle);
    }

    // Check if we've arrived near target
    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    let dx = tx - world.vehicles[car_idx].x;
    let dz = tz - world.vehicles[car_idx].z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist < NPC_DRIVE_THRESHOLD {
        // Search for nearest free parking spot within 50m
        if world.vehicles[car_idx].parking_target.is_none() {
            if let Some(si) = crate::vehicle::find_nearest_parking_spot(net, world.vehicles[car_idx].x, world.vehicles[car_idx].z, 50.0) {
                world.vehicles[car_idx].parking_target = Some(si);
                world.vehicles[car_idx].ai_target_x = net.parking_spots[si].x;
                world.vehicles[car_idx].ai_target_z = net.parking_spots[si].z;
            }
        }

        // Check if at parking spot
        if let Some(si) = world.vehicles[car_idx].parking_target {
            if si < net.parking_spots.len() {
                let pdx = net.parking_spots[si].x - world.vehicles[car_idx].x;
                let pdz = net.parking_spots[si].z - world.vehicles[car_idx].z;
                let park_dist = (pdx * pdx + pdz * pdz).sqrt();
                if park_dist < 3.0 {
                    // Parked — exit vehicle
                    npc_exit_car(world, i, terrain, net);
                    return;
                }
            }
        }

        // If close enough but no parking spot found, just stop
        if dist < 8.0 && world.vehicles[car_idx].parking_target.is_none() {
            npc_exit_car(world, i, terrain, net);
            return;
        }
    }

    // Drive toward target
    world.vehicles[car_idx].ai_target_x = if world.vehicles[car_idx].parking_target.is_some() {
        world.vehicles[car_idx].ai_target_x // already set to parking spot
    } else { tx };
    world.vehicles[car_idx].ai_target_z = if world.vehicles[car_idx].parking_target.is_some() {
        world.vehicles[car_idx].ai_target_z
    } else { tz };
    world.vehicles[car_idx].ai_active = true;
    world.vehicles[car_idx].parked = false;
}

/// Try to enter NPC's assigned vehicle. Returns true if entered (state → Driving),
/// false if car is too far (target redirected to car, caller should walk there).
pub fn npc_enter_car(world: &mut WorldData, i: usize, _terrain: &Terrain, net: &mut RoadNetwork) -> bool {
    let car_idx = world.npcs[i].car_idx;
    if car_idx >= world.vehicles.len() { return false; }
    if world.vehicles[car_idx].occupied { return false; }

    let dx = world.vehicles[car_idx].x - world.npcs[i].x;
    let dz = world.vehicles[car_idx].z - world.npcs[i].z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist > 5.0 {
        // Redirect NPC to walk toward car — caller should fall through to walk code
        world.npcs[i].target_x = world.vehicles[car_idx].x;
        world.npcs[i].target_z = world.vehicles[car_idx].z;
        return false;
    }

    // Un-park: release parking spot
    if let Some(si) = world.vehicles[car_idx].parking_target {
        if si < net.parking_spots.len() {
            net.parking_spots[si].occupied_by = None;
        }
    }
    world.vehicles[car_idx].parking_target = None;
    world.vehicles[car_idx].parked = false;
    world.vehicles[car_idx].path.clear();

    world.npcs[i].parked_x = world.npcs[i].x;
    world.npcs[i].parked_z = world.npcs[i].z;
    world.npcs[i].in_vehicle = true;
    world.npcs[i].state = NpcState::Driving;
    world.vehicles[car_idx].ai_active = true;
    world.vehicles[car_idx].ai_target_x = world.npcs[i].target_x;
    world.vehicles[car_idx].ai_target_z = world.npcs[i].target_z;
    world.vehicles[car_idx].occupied = false;
    true
}

pub fn npc_exit_car(world: &mut WorldData, i: usize, terrain: &Terrain, net: &mut RoadNetwork) {
    let car_idx = world.npcs[i].car_idx;
    if car_idx >= world.vehicles.len() { return; }

    // Exit to side of vehicle (try both sides, avoid river)
    let v = &world.vehicles[car_idx];
    let side_x = v.rot_y.sin() * 2.5;
    let side_z = v.rot_y.cos() * 2.5;
    let try1_x = v.x + side_x;
    let try1_z = v.z + side_z;
    let try2_x = v.x - side_x;
    let try2_z = v.z - side_z;
    let (exit_x, exit_z) = if !on_river_not_bridge(try1_x, try1_z, &world.river_segments, &world.bridges) {
        (try1_x, try1_z)
    } else if !on_river_not_bridge(try2_x, try2_z, &world.river_segments, &world.bridges) {
        (try2_x, try2_z)
    } else {
        (v.x, v.z) // fallback to vehicle position
    };
    let exit_y = terrain.height_at(exit_x, exit_z);

    world.npcs[i].x = exit_x;
    world.npcs[i].y = exit_y;
    world.npcs[i].z = exit_z;
    world.npcs[i].in_vehicle = false;
    world.vehicles[car_idx].ai_active = false;
    world.vehicles[car_idx].speed = 0.0;
    world.vehicles[car_idx].parked = true;
    world.vehicles[car_idx].path.clear();

    // Mark parking spot as occupied by this vehicle
    if let Some(si) = world.vehicles[car_idx].parking_target {
        if si < net.parking_spots.len() {
            net.parking_spots[si].occupied_by = Some(car_idx);
        }
    } else {
        // No parking target — snap vehicle to nearest parking spot or road edge
        crate::vehicle::snap_to_parking(car_idx, world, net, terrain);
    }

    let npc = &world.npcs[i];
    if npc.state_timer < WORK_DURATION {
        world.npcs[i].state = NpcState::Working;
    } else {
        world.npcs[i].state = NpcState::GoingHome;
    }
    world.npcs[i].nav_path.clear();
    world.npcs[i].stuck_timer = 0.0;
}

fn find_closest_item(world: &WorldData, npc_idx: usize) -> Option<usize> {
    let npc = &world.npcs[npc_idx];
    let mut best_dist = f32::MAX;
    let mut best_idx = None;

    for (idx, item) in world.items.iter().enumerate() {
        if !item.active { continue; }
        if item.falling { continue; }
        if item.skip_until > 0.0 { continue; }
        if let Some(claimer) = item.claimed_by {
            if claimer != npc_idx { continue; }
        }
        let dist = dist_sq_2d(item.x, item.z, npc.x, npc.z);
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

// Night sky spawning system
pub fn sys_night_spawning(
    world: &mut WorldData, terrain: &Terrain, time_of_day: f32,
    dt: f32, rng: &mut Rng, road_network: &RoadNetwork,
) {
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

    // Count active items — only spawn if below threshold
    let active_count = world.items.iter().filter(|it| it.active || it.falling).count();
    if active_count >= NUM_ITEMS { return; }

    // Spawn rate: full speed at night, slightly reduced during day
    let is_night = time_of_day >= NIGHT_SPAWN_START || time_of_day < NIGHT_SPAWN_END;
    let spawn_interval = if is_night { NIGHT_SPAWN_INTERVAL } else { NIGHT_SPAWN_INTERVAL * 1.5 };

    // Spawn multiple items per tick based on deficit
    let deficit = NUM_ITEMS - active_count;
    let max_spawns = (deficit / 5).max(1).min(10);
    for _ in 0..max_spawns {
    if (rng.next() as f32 / u64::MAX as f32) < (dt / spawn_interval) {
        // Spawn near roads/walkable areas so NPCs can actually reach them
        let mut x;
        let mut z;
        let mut attempts = 0;
        loop {
            // After many failures, spawn near a road node but off the road itself
            if attempts > 30 && !road_network.nodes.is_empty() {
                let ni = rng.next() as usize % road_network.nodes.len();
                // Offset 10-20m from node to land on terrain, not road surface
                let angle = rng.range(0.0, std::f32::consts::TAU);
                let dist = rng.range(10.0, 20.0);
                x = road_network.nodes[ni][0] + angle.cos() * dist;
                z = road_network.nodes[ni][1] + angle.sin() * dist;
                if !on_any_road(x, z, road_network) { break; }
                // If still on road, try once more with bigger offset
                x = road_network.nodes[ni][0] + angle.cos() * 25.0;
                z = road_network.nodes[ni][1] + angle.sin() * 25.0;
                break;
            }
            // 70% near town, 30% anywhere
            if rng.next() % 10 < 7 {
                x = rng.range(-150.0, 150.0);
                z = rng.range(-150.0, 150.0);
            } else {
                x = rng.range(-WORLD_HALF + 10.0, WORLD_HALF - 10.0);
                z = rng.range(-WORLD_HALF + 10.0, WORLD_HALF - 10.0);
            }
            attempts += 1;
            if on_any_road(x, z, road_network) { continue; }
            if check_walk_collision(world, x, z, 0.5, Some(usize::MAX)) { continue; }
            if on_river_not_bridge(x, z, &world.river_segments, &world.bridges) { continue; }
            break;
        }
        let y = 40.0 + rng.range(0.0, 20.0);
        let kinds = [ItemKind::Health, ItemKind::Money, ItemKind::Stamina, ItemKind::Food, ItemKind::Water];
        let kind = kinds[rng.next() as usize % kinds.len()];

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
                item.skip_until = 0.0;
                found = true;
                break;
            }
        }
        if !found {
            world.items.push(Item {
                x, y, z, kind, active: false,
                spin_phase: 0.0, falling: true, vel_y: 0.0, claimed_by: None, skip_until: 0.0,
            });
        }
    } // if spawn_chance
    } // for max_spawns
}

/// Reset daily counters at midnight and evolve NEAT population
pub fn sys_midnight_reset(
    world: &mut WorldData, time_of_day: f32, prev_time: f32,
    population: &mut crate::neat::Population, brains: &mut Vec<crate::neat::NeatBrain>,
) -> bool {
    // Detect midnight crossing
    if prev_time > 23.5 && time_of_day < 0.5 {
        // Evaluate fitness and evolve
        let fitnesses: Vec<f32> = world.npcs.iter()
            .map(|npc| crate::neat::evaluate_fitness(npc))
            .collect();
        population.evolve(&fitnesses);

        // Auto-save evolved population
        crate::neat::save_population("/tmp/clauding_neat.bin", population);

        // Recompile brains from evolved genomes
        *brains = population.genomes.iter()
            .map(|g| crate::neat::NeatBrain::compile(g))
            .collect();

        // Reset ALL NPCs for fair evaluation each generation
        for (i, npc) in world.npcs.iter_mut().enumerate() {
            // Full reset: every NPC starts fresh each day
            npc.health = NPC_HEALTH_MAX;
            npc.hunger = 100.0;
            npc.thirst = 100.0;
            npc.starving_dead = false;
            npc.state = NpcState::Sleeping;
            npc.knockout_timer = 0.0;
            npc.carrying_item = false;
            npc.carrying_bin = None;
            // Release vehicle if NPC was driving
            if npc.in_vehicle {
                let ci = npc.car_idx;
                if ci < world.vehicles.len() {
                    world.vehicles[ci].ai_active = false;
                    world.vehicles[ci].speed = 0.0;
                    world.vehicles[ci].occupied = false;
                    world.vehicles[ci].parked = true;
                }
            }
            npc.in_vehicle = false;
            // Money persists between days — NPCs keep what they earned
            npc.brain_idx = i;
            npc.job_timer = 0.0;
            npc.interaction_target = None;
            npc.target_item = None;
            npc.target_bin = None;
            npc.fitness_money_earned = 0.0;
            npc.fitness_items_picked = 0;
            npc.fitness_interactions = 0;
            npc.fitness_distance = 0.0;
            npc.fitness_stuck_time = 0.0;
            npc.fitness_knockouts = 0;
            npc.fitness_hits_landed = 0;
            npc.fitness_starve_time = 0.0;
            npc.fitness_sounds_made = 0;
            npc.fitness_npcs_heard = 0;
            npc.fitness_proximity = 0.0;
            npc.find_item_failures = 0;
            npc.find_bin_failures = 0;
            npc.stuck_recoveries = 0;
            npc.sound = [0.0; 3];
            npc.ragdoll_active = false;
            npc.ragdoll_timer = 0.0;
            npc.wanted = false;
            npc.bounty = 0.0;
            npc.violation_timer = 0.0;
            npc.police_target = None;
            npc.prev_x = npc.x;
            npc.prev_z = npc.z;
        }

        // Teleport river-stuck NPCs back to their home building
        for npc in world.npcs.iter_mut() {
            if on_river_not_bridge(npc.x, npc.z, &world.river_segments, &world.bridges) {
                let home = &world.buildings[npc.home_idx];
                npc.x = home.x;
                npc.z = home.z;
                npc.prev_x = npc.x;
                npc.prev_z = npc.z;
            }
        }

        // Reset trash bins
        for bin in &mut world.trash_bins {
            bin.items_held = 0;
            bin.carried_by = None;
        }
        // Clear stale item claims — NPCs no longer target these after reset
        for item in &mut world.items {
            item.claimed_by = None;
            item.skip_until = 0.0;
        }
        // Reset NPC daily counters
        for npc in &mut world.npcs {
            npc.items_deposited_today = 0;
            npc.stuck_timer = 0.0;
            npc.nav_path.clear();
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
        if item.skip_until > 0.0 {
            item.skip_until -= dt;
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
            let d2 = dist_sq_2d(px, pz, bin.x, bin.z);
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
            world.trash_bins[bi].terrain_normal = terrain.normal_at(px, pz);
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
            let d2 = dist_sq_2d(px, pz, item.x, item.z);
            if d2 < best_dist {
                best_dist = d2;
                best_ii = Some(ii);
            }
        }
        if let Some(ii) = best_ii {
            let kind = world.items[ii].kind;
            let color = match kind {
                ItemKind::Health => 0xFFFF3333,
                ItemKind::Money => 0xFFFFDD33,
                ItemKind::Stamina => 0xFFFFFF33,
                ItemKind::Food => 0xFFDD8833,
                ItemKind::Water => 0xFF3388FF,
            };
            world.items[ii].active = false;
            world.items[ii].claimed_by = None;
            // Food/Water: auto-consume, no carrying
            match kind {
                ItemKind::Food => { player.hunger = (player.hunger + FOOD_RESTORE).min(100.0); }
                ItemKind::Water => { player.thirst = (player.thirst + WATER_RESTORE).min(100.0); }
                _ => { player.carrying_item = true; }
            }
            return Some((world.items[ii].x, world.items[ii].z, color));
        }
    }

    // Priority 4: Near bin → pick it up
    {
        let mut best_dist = NPC_BIN_DIST * NPC_BIN_DIST;
        let mut best_bi = None;
        for (bi, bin) in world.trash_bins.iter().enumerate() {
            if bin.carried_by.is_some() { continue; }
            let d2 = dist_sq_2d(px, pz, bin.x, bin.z);
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

    // Priority 5: Near interactible
    {
        let interact_dist_sq = INTERACT_DIST * INTERACT_DIST;
        let mut best_dist = interact_dist_sq;
        let mut best_ii = None;
        for (ii, inter) in world.interactibles.iter().enumerate() {
            if inter.cooldown > 0.0 { continue; }
            let d2 = dist_sq_2d(px, pz, inter.x, inter.z);
            if d2 < best_dist {
                best_dist = d2;
                best_ii = Some(ii);
            }
        }
        if let Some(ii) = best_ii {
            let kind = world.interactibles[ii].kind;
            match kind {
                InteractibleKind::VendingMachine => {
                    if player.money >= 2.0 {
                        player.money -= 2.0;
                        player.stamina = (player.stamina + 20.0).min(100.0);
                        player.thirst = (player.thirst + VENDING_WATER_RESTORE).min(100.0);
                        world.interactibles[ii].cooldown = 3.0;
                        return Some((world.interactibles[ii].x, world.interactibles[ii].z, 0xFF33DDFF));
                    }
                }
                InteractibleKind::ParkBench => {
                    player.sitting = true;
                    return None;
                }
                InteractibleKind::Dumpster => {
                    world.interactibles[ii].cooldown = 5.0;
                    let roll = (player.x.to_bits() ^ player.z.to_bits()) % 3;
                    if roll == 0 {
                        player.money += 1.0 + (player.x.to_bits() % 3) as f32;
                        return Some((world.interactibles[ii].x, world.interactibles[ii].z, 0xFFFFDD33));
                    } else if roll == 1 {
                        player.health = (player.health + 10.0).min(100.0);
                        return Some((world.interactibles[ii].x, world.interactibles[ii].z, 0xFFFF3333));
                    }
                    return None;
                }
                InteractibleKind::Atm => {
                    world.interactibles[ii].cooldown = 2.0;
                    if player.carrying_item {
                        // Deposit earnings
                        player.bank_balance += player.money;
                        player.money = 0.0;
                    } else {
                        // Withdraw
                        if player.bank_balance >= 50.0 {
                            player.bank_balance -= 50.0;
                            player.money += 50.0;
                        }
                    }
                    return Some((world.interactibles[ii].x, world.interactibles[ii].z, 0xFF88BBFF));
                }
                InteractibleKind::PhoneBooth => {
                    player.job_menu_open = true;
                    player.job_menu_cursor = 0;
                    return None;
                }
                InteractibleKind::FireHydrant => {
                    world.interactibles[ii].state_val = 5.0;
                    world.interactibles[ii].cooldown = 10.0;
                    return Some((world.interactibles[ii].x, world.interactibles[ii].z, 0xFF3388FF));
                }
                InteractibleKind::NewspaperStand => {
                    if player.money >= 1.0 {
                        player.money -= 1.0;
                        player.stamina = 100.0; // full stamina refill
                        player.health = (player.health + 5.0).min(100.0);
                        player.hunger = (player.hunger + NEWSPAPER_FOOD_RESTORE).min(100.0);
                        world.interactibles[ii].cooldown = 5.0;
                        return Some((world.interactibles[ii].x, world.interactibles[ii].z, 0xFFDDDDDD));
                    }
                }
                InteractibleKind::Mailbox => {
                    // Complete delivery job if carrying
                    if player.carrying_item && player.active_job.job_type == PlayerJobType::MailCarrier {
                        player.carrying_item = false;
                        player.active_job.items_done += 1;
                        return Some((world.interactibles[ii].x, world.interactibles[ii].z, 0xFF44DDFF));
                    }
                }
                InteractibleKind::Payphone => {
                    world.interactibles[ii].cooldown = 30.0;
                    // Call a taxi — teleport nearest unoccupied vehicle to player
                    let mut best_dist = f32::MAX;
                    let mut best_vi = None;
                    for (vi, v) in world.vehicles.iter().enumerate() {
                        if v.occupied { continue; }
                        if v.ai_active { continue; }
                        let vdx = v.x - px;
                        let vdz = v.z - pz;
                        let d = vdx * vdx + vdz * vdz;
                        if d < best_dist {
                            best_dist = d;
                            best_vi = Some(vi);
                        }
                    }
                    if let Some(vi) = best_vi {
                        // Move vehicle near player
                        let angle = player.rot_y;
                        world.vehicles[vi].x = px + angle.sin() * 5.0;
                        world.vehicles[vi].z = pz + angle.cos() * 5.0;
                        world.vehicles[vi].y = terrain.height_at(world.vehicles[vi].x, world.vehicles[vi].z);
                        world.vehicles[vi].rot_y = angle;
                        world.vehicles[vi].speed = 0.0;
                        return Some((world.vehicles[vi].x, world.vehicles[vi].z, 0xFFFFFF44));
                    }
                    return None;
                }
            }
        }
    }

    None
}

/// Hunger/thirst drain and starvation damage for all characters
pub fn sys_hunger_thirst(world: &mut WorldData, player: &mut Player, dt: f32) {
    // NPCs
    for npc in &mut world.npcs {
        if npc.state == NpcState::Sleeping || npc.starving_dead { continue; }

        npc.hunger = (npc.hunger - HUNGER_DRAIN_RATE * dt).max(0.0);
        npc.thirst = (npc.thirst - THIRST_DRAIN_RATE * dt).max(0.0);

        // Track starvation time for fitness
        if npc.hunger <= 0.0 || npc.thirst <= 0.0 {
            npc.fitness_starve_time += dt;
        }

        // Starvation/dehydration damage
        let mut dmg = 0.0;
        if npc.hunger <= 0.0 { dmg += STARVATION_DAMAGE * dt; }
        if npc.thirst <= 0.0 { dmg += DEHYDRATION_DAMAGE * dt; }
        if dmg > 0.0 {
            npc.health -= dmg;
            if npc.health <= 0.0 {
                npc.health = 0.0;
                npc.state = NpcState::KnockedOut;
                npc.starving_dead = true;
                npc.knockout_timer = f32::MAX;
                npc.carrying_item = false;
                npc.carrying_bin = None;
            }
        }
    }

    // Player (god mode: drain but clamp health)
    player.hunger = (player.hunger - HUNGER_DRAIN_RATE * dt).max(0.0);
    player.thirst = (player.thirst - THIRST_DRAIN_RATE * dt).max(0.0);

    let mut dmg = 0.0;
    if player.hunger <= 0.0 { dmg += STARVATION_DAMAGE * dt; }
    if player.thirst <= 0.0 { dmg += DEHYDRATION_DAMAGE * dt; }
    if dmg > 0.0 {
        player.health = (player.health - dmg).max(PLAYER_MIN_HEALTH_STARVE);
    }
}

/// NPC-NPC social interactions
pub fn sys_npc_interactions(world: &mut WorldData, dt: f32) {
    let n = world.npcs.len();

    // Update existing interactions
    for i in 0..n {
        if world.npcs[i].state != NpcState::Interacting { continue; }
        world.npcs[i].interaction_timer -= dt;
        if world.npcs[i].interaction_timer <= 0.0 {
            world.npcs[i].state = NpcState::Working;
            world.npcs[i].interacting_with = None;
            world.npcs[i].interaction_timer = 30.0; // 30s cooldown before next interaction
        }
    }

    // Start new interactions (working NPCs near each other, not KO'd)
    for i in 0..n {
        if world.npcs[i].state != NpcState::Working { continue; }
        if world.npcs[i].interacting_with.is_some() { continue; }
        if world.npcs[i].state == NpcState::KnockedOut { continue; }
        // Cooldown: interaction_timer > 0 means recently interacted (reused as cooldown)
        if world.npcs[i].interaction_timer > 0.0 {
            world.npcs[i].interaction_timer -= dt;
            continue;
        }

        for j in (i + 1)..n {
            if world.npcs[j].state != NpcState::Working { continue; }
            if world.npcs[j].interacting_with.is_some() { continue; }
            if world.npcs[j].interaction_timer > 0.0 { continue; }

            let dx = world.npcs[j].x - world.npcs[i].x;
            let dz = world.npcs[j].z - world.npcs[i].z;
            if dx * dx + dz * dz > 2.25 { continue; } // within 1.5m

            // 0.1% chance per frame
            if world.npcs[i].rng.next() % 1000 != 0 { continue; }

            let duration = 3.0 + (world.npcs[i].rng.next() % 50) as f32 * 0.1; // 3-8s

            // Face each other
            let angle_i_to_j = (-dx).atan2(-dz);
            let angle_j_to_i = dx.atan2(dz);
            world.npcs[i].rot_y = angle_i_to_j;
            world.npcs[j].rot_y = angle_j_to_i;

            world.npcs[i].state = NpcState::Interacting;
            world.npcs[i].interacting_with = Some(j);
            world.npcs[i].interaction_timer = duration;
            world.npcs[i].walk_phase = 0.0;

            world.npcs[j].state = NpcState::Interacting;
            world.npcs[j].interacting_with = Some(i);
            world.npcs[j].interaction_timer = duration;
            world.npcs[j].walk_phase = 0.0;

            // Vendor buying
            if world.npcs[i].job == NpcJob::StreetVendor {
                world.npcs[i].money += 1.0;
            }
            if world.npcs[j].job == NpcJob::StreetVendor {
                world.npcs[j].money += 1.0;
            }

            break; // only one interaction per NPC per frame
        }
    }
}
