// NPC life simulation: state machine, physics, pathfinding, item pickup/deposit, night spawning

use crate::state::*;
use crate::world::{check_npc_walk_collision, surface_at, point_to_segment_dist, on_river_not_bridge, on_any_road};
use crate::rng::Rng;

// Home task duration: 4 game-hours = 4 * 60 = 240 real seconds
const HOME_TASK_DURATION: f32 = 240.0;
// Work duration: 12 game-hours = 720 real seconds
const WORK_DURATION: f32 = 720.0;
// Item spawn interval: ~1 item every 1.5 seconds (items are collected efficiently)
const NIGHT_SPAWN_INTERVAL: f32 = 1.0;

pub fn sys_npc(
    world: &mut WorldData, road_network: &mut RoadNetwork, terrain: &Terrain,
    dt: f32, time_of_day: f32, brains: &mut [crate::neat::NeatBrain],
    player_x: f32, player_z: f32,
) {
    let n = world.npcs.len();
    for i in 0..n {
        // Physics: gravity + ground snap
        npc_physics(world, i, terrain, dt);

        // State machine
        let prev_state = world.npcs[i].state;
        match prev_state {
            NpcState::Sleeping => npc_sleeping(world, i, time_of_day),
            NpcState::HomeTask => npc_home_task(world, i, terrain, dt),
            NpcState::GoingToWork => npc_going_to_work(world, i, road_network, terrain, dt),
            NpcState::Working => npc_working(world, i, road_network, terrain, dt, brains, time_of_day, player_x, player_z),
            NpcState::GoingHome => npc_going_home(world, i, road_network, terrain, dt),
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

/// Final pass: ensure no NPC is on the river after all systems have run.
/// Call this AFTER collision, combat, knockback — it's the last line of defense.
pub fn sys_river_escape(world: &mut WorldData, terrain: &Terrain) {
    for i in 0..world.npcs.len() {
        let npc = &world.npcs[i];
        if npc.in_vehicle || npc.state == NpcState::Sleeping { continue; }
        if !on_river_not_bridge(npc.x, npc.z, &world.river_segments, &world.bridges) { continue; }

        // Find nearest bank using 8-direction search at increasing radii
        let cx = world.npcs[i].x;
        let cz = world.npcs[i].z;
        let mut escaped = false;
        for radius in [5.0, 10.0, 20.0, 40.0] {
            for angle_i in 0..8 {
                let a = angle_i as f32 * std::f32::consts::TAU / 8.0;
                let try_x = cx + a.cos() * radius;
                let try_z = cz + a.sin() * radius;
                if !on_river_not_bridge(try_x, try_z, &world.river_segments, &world.bridges) {
                    world.npcs[i].x = try_x;
                    world.npcs[i].z = try_z;
                    world.npcs[i].y = terrain.height_at(try_x, try_z);
                    world.npcs[i].knockback_vx = 0.0;
                    world.npcs[i].knockback_vz = 0.0;
                    escaped = true;
                    break;
                }
            }
            if escaped { break; }
        }
    }
}

fn npc_physics(world: &mut WorldData, i: usize, terrain: &Terrain, dt: f32) {
    let npc = &mut world.npcs[i];
    if npc.in_vehicle || npc.state == NpcState::Sleeping || npc.state == NpcState::Interacting { return; }

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

    // Smooth terrain normal for slope tilting (clamped to 25° max visual tilt)
    let raw_n = terrain.normal_at(npc.x, npc.z);
    let target_n = crate::math::clamp_normal_tilt(raw_n, 25.0);
    let lerp_rate = 8.0 * dt;
    npc.terrain_normal = crate::math::v3_normalize(crate::math::v3_lerp(npc.terrain_normal, target_n, lerp_rate.min(1.0)));

    // Slope sliding: if terrain is steep and NPC is on ground, slide downhill
    // slope = 1 - cos(angle): 30°→0.13, 40°→0.23, 50°→0.36, 60°→0.50
    if npc.on_ground {
        let slope = (1.0 - raw_n[1]).max(0.0);
        if slope > 0.12 { // ~28° threshold
            let slide_force = slope * slope * 40.0 * dt;
            let slide_x = npc.x - raw_n[0] * slide_force;
            let slide_z = npc.z - raw_n[2] * slide_force;
            // Don't slide into rivers
            if !on_river_not_bridge(slide_x, slide_z, &world.river_segments, &world.bridges) {
                npc.x = slide_x;
                npc.z = slide_z;
                npc.y = terrain.height_at(npc.x, npc.z);
            }
        }
    }

    // River escape: push NPCs off river toward nearest bank (but not on bridges)
    if on_river_not_bridge(npc.x, npc.z, &world.river_segments, &world.bridges) {
        let mut best_dist = f32::MAX;
        let mut best_half_width = 5.0f32;
        let mut push_x = 0.0f32;
        let mut push_z = 0.0f32;
        for seg in &world.river_segments {
            let d = crate::world::point_to_segment_dist(npc.x, npc.z, seg.x1, seg.z1, seg.x2, seg.z2);
            if d < best_dist {
                best_dist = d;
                best_half_width = seg.width * 0.5;
                // Perpendicular direction away from river center
                let sdx = seg.x2 - seg.x1;
                let sdz = seg.z2 - seg.z1;
                let len = (sdx * sdx + sdz * sdz).sqrt().max(0.01);
                let px = -sdz / len;
                let pz = sdx / len;
                // Which side is NPC on?
                let dot = (npc.x - seg.x1) * px + (npc.z - seg.z1) * pz;
                let sign = if dot >= 0.0 { 1.0 } else { -1.0 };
                push_x = px * sign;
                push_z = pz * sign;
            }
        }
        // Strong push outward — must exceed walking speed
        npc.x += push_x * 20.0 * dt;
        npc.z += push_z * 20.0 * dt;
        // If still on river after push, snap to bank with escalating distance
        if on_river_not_bridge(npc.x, npc.z, &world.river_segments, &world.bridges) {
            // Try increasing snap distances until we're off the river
            for mult in [1.0, 2.0, 3.0] {
                let snap_dist = (best_half_width + 2.0) * mult;
                let try_x = npc.x + push_x * snap_dist;
                let try_z = npc.z + push_z * snap_dist;
                if !on_river_not_bridge(try_x, try_z, &world.river_segments, &world.bridges) {
                    npc.x = try_x;
                    npc.z = try_z;
                    break;
                }
            }
            // Last resort: if still on river (junction/bend), try 8 directions at increasing radii
            if on_river_not_bridge(npc.x, npc.z, &world.river_segments, &world.bridges) {
                let cx = npc.x;
                let cz = npc.z;
                'escape: for radius in [best_half_width + 3.0, best_half_width + 8.0, best_half_width + 15.0] {
                    for angle_i in 0..8 {
                        let a = angle_i as f32 * std::f32::consts::TAU / 8.0;
                        let try_x = cx + a.cos() * radius;
                        let try_z = cz + a.sin() * radius;
                        if !on_river_not_bridge(try_x, try_z, &world.river_segments, &world.bridges) {
                            npc.x = try_x;
                            npc.z = try_z;
                            break 'escape;
                        }
                    }
                }
            }
        }
    }
}

/// Walk NPC toward (target_x, target_z) with collision avoidance, pathfinding, and surface-aware speed
pub fn npc_walk_toward(world: &mut WorldData, i: usize, tx: f32, tz: f32, net: &RoadNetwork, terrain: &Terrain, dt: f32) -> f32 {
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
        // Drift toward sidewalk edge (but not into river)
        let drift_x = npc.x + best_perp_x * 0.5 * dt;
        let drift_z = npc.z + best_perp_z * 0.5 * dt;
        if !on_river_not_bridge(drift_x, drift_z, &world.river_segments, &world.bridges) {
            npc.x = drift_x;
            npc.z = drift_z;
        }
    }

    // Try to move forward
    let rot = npc.rot_y;
    let new_x = npc.x - rot.sin() * speed * dt;
    let new_z = npc.z - rot.cos() * speed * dt;

    let home_idx = npc.home_idx;

    // Push NPC out of any overlapping placed bin (collision resolution, not rescue)
    for bi in 0..world.trash_bins.len() {
        if world.trash_bins[bi].carried_by.is_some() { continue; }
        let bx = world.trash_bins[bi].x;
        let bz = world.trash_bins[bi].z;
        let nx = world.npcs[i].x;
        let nz = world.npcs[i].z;
        let bdx = nx - bx;
        let bdz = nz - bz;
        let d2 = bdx * bdx + bdz * bdz;
        let r = 0.85; // npc radius(0.4) + bin radius(0.4) + margin
        if d2 < r * r && d2 > 0.001 {
            let d = d2.sqrt();
            let push = (r - d) + 0.05;
            let px = nx + bdx / d * push;
            let pz = nz + bdz / d * push;
            if !check_npc_walk_collision(world, px, pz, 0.4, home_idx)
                && !on_river_not_bridge(px, pz, &world.river_segments, &world.bridges)
            {
                world.npcs[i].x = px;
                world.npcs[i].z = pz;
            }
            break;
        }
    }

    // Check river with margin (2m wider than actual river) so NPCs don't hug the edge
    let river_margin = 2.0;
    let on_river_x = {
        let mut hit = false;
        for seg in &world.river_segments {
            let d = crate::world::point_to_segment_dist(new_x, world.npcs[i].z, seg.x1, seg.z1, seg.x2, seg.z2);
            if d < seg.width * 0.5 + river_margin { hit = true; break; }
        }
        hit
    };
    let on_river_z = {
        let mut hit = false;
        for seg in &world.river_segments {
            let d = crate::world::point_to_segment_dist(world.npcs[i].x, new_z, seg.x1, seg.z1, seg.x2, seg.z2);
            if d < seg.width * 0.5 + river_margin { hit = true; break; }
        }
        hit
    };
    let collides_x = check_npc_walk_collision(world, new_x, world.npcs[i].z, 0.4, home_idx) || on_river_x;
    let collides_z = check_npc_walk_collision(world, world.npcs[i].x, new_z, 0.4, home_idx) || on_river_z;

    // Apply movement
    let old_x = world.npcs[i].x;
    let old_z = world.npcs[i].z;
    if !collides_x { world.npcs[i].x = new_x; }
    if !collides_z { world.npcs[i].z = new_z; }

    // Post-movement diagonal river check — axis-separated checks can miss diagonal steps
    // Only apply if we weren't already on the river (don't trap NPCs at river-edge homes)
    if !on_river_not_bridge(old_x, old_z, &world.river_segments, &world.bridges)
        && on_river_not_bridge(world.npcs[i].x, world.npcs[i].z, &world.river_segments, &world.bridges)
    {
        world.npcs[i].x = old_x;
        world.npcs[i].z = old_z;
    }

    // Animate walk if NPC physically moved (any direction)
    let any_movement = ((world.npcs[i].x - old_x).abs() + (world.npcs[i].z - old_z).abs()) > speed * dt * 0.3;
    if any_movement {
        world.npcs[i].walk_phase += dt * speed * 2.5;
    }

    // Stuck detection: check progress toward current target (detour or real).
    // Wall-sliding (perpendicular movement) shouldn't reset stuck_timer.
    let old_dist = ((actual_tx - old_x) * (actual_tx - old_x) + (actual_tz - old_z) * (actual_tz - old_z)).sqrt();
    let new_dist = ((actual_tx - world.npcs[i].x) * (actual_tx - world.npcs[i].x)
                  + (actual_tz - world.npcs[i].z) * (actual_tz - world.npcs[i].z)).sqrt();
    let progressed = old_dist - new_dist > speed * dt * 0.2;

    if progressed {
        world.npcs[i].stuck_timer = 0.0;
    } else {
        world.npcs[i].stuck_timer += dt;

        if world.npcs[i].stuck_timer > 0.3 && !world.npcs[i].detouring {
            let perp_x = -dz / dist.max(0.01) * 8.0;
            let perp_z = dx / dist.max(0.01) * 8.0;
            let sign = if world.npcs[i].rng.next() % 2 == 0 { 1.0 } else { -1.0 };
            let det_x = world.npcs[i].x + perp_x * sign;
            let det_z = world.npcs[i].z + perp_z * sign;
            // Try other side if first detour hits river
            if on_river_not_bridge(det_x, det_z, &world.river_segments, &world.bridges) {
                let det_x2 = world.npcs[i].x - perp_x * sign;
                let det_z2 = world.npcs[i].z - perp_z * sign;
                if !on_river_not_bridge(det_x2, det_z2, &world.river_segments, &world.bridges) {
                    world.npcs[i].detour_x = det_x2;
                    world.npcs[i].detour_z = det_z2;
                    world.npcs[i].detouring = true;
                }
                // else: don't detour at all — both sides are river
            } else {
                world.npcs[i].detour_x = det_x;
                world.npcs[i].detour_z = det_z;
                world.npcs[i].detouring = true;
            }
        }

        if world.npcs[i].stuck_timer > 0.8 && world.npcs[i].stuck_timer < 1.0 && world.npcs[i].on_ground {
            world.npcs[i].vel_y = NPC_JUMP_VELOCITY;
            world.npcs[i].on_ground = false;
        }

        if world.npcs[i].stuck_timer > 5.0 {
            // Severely stuck — teleport to a clear position
            let cur_x = world.npcs[i].x;
            let cur_z = world.npcs[i].z;

            let toward_x = (tx - cur_x).clamp(-15.0, 15.0);
            let toward_z = (tz - cur_z).clamp(-15.0, 15.0);

            let mut dest_x = cur_x;
            let mut dest_z = cur_z;
            let mut found = false;

            // Try 4 directions, validate no collision + no river
            let candidates = [
                (cur_x + toward_x, cur_z + toward_z),
                (cur_x + toward_z, cur_z - toward_x),
                (cur_x - toward_z, cur_z + toward_x),
                (cur_x - toward_x, cur_z - toward_z),
            ];
            for (cx, cz) in candidates {
                let cx = cx.clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
                let cz = cz.clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
                if !on_river_not_bridge(cx, cz, &world.river_segments, &world.bridges)
                    && !check_npc_walk_collision(world, cx, cz, 0.4, home_idx)
                {
                    dest_x = cx;
                    dest_z = cz;
                    found = true;
                    break;
                }
            }
            // All 4 directions blocked — jump to nearest road node (escape trap)
            if !found && !net.nodes.is_empty() {
                let mut best_d = f32::MAX;
                for node in &net.nodes {
                    let ndx = node[0] - cur_x;
                    let ndz = node[1] - cur_z;
                    let d = ndx * ndx + ndz * ndz;
                    if d < best_d && !on_river_not_bridge(node[0], node[1], &world.river_segments, &world.bridges) {
                        best_d = d;
                        dest_x = node[0];
                        dest_z = node[1];
                        found = true;
                    }
                }
            }
            if !found {
                // Last resort: fallback toward target ignoring collision
                let fx = (cur_x + toward_x).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
                let fz = (cur_z + toward_z).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
                if !on_river_not_bridge(fx, fz, &world.river_segments, &world.bridges) {
                    dest_x = fx;
                    dest_z = fz;
                }
            }

            world.npcs[i].x = dest_x;
            world.npcs[i].z = dest_z;
            world.npcs[i].stuck_timer = 0.0;
            world.npcs[i].detouring = false;

            // Abandon current target — cooldown so NPC picks a different one.
            if let Some(item_idx) = world.npcs[i].target_item {
                if item_idx < world.items.len() {
                    world.items[item_idx].claimed_by = None;
                    world.items[item_idx].skip_until = 30.0;
                }
                world.npcs[i].target_item = None;
            }
            if let Some(bin_idx) = world.npcs[i].target_bin {
                if bin_idx < world.trash_bins.len() {
                    world.trash_bins[bin_idx].carried_by = None;
                }
                world.npcs[i].target_bin = None;
            }
        } else if world.npcs[i].stuck_timer > 3.0 {
            let prev = world.npcs[i].stuck_timer - dt;
            if (world.npcs[i].stuck_timer / 3.0) as u32 != (prev / 3.0) as u32 {
                let angle = world.npcs[i].rng.range(0.0, std::f32::consts::TAU);
                let nx = world.npcs[i].x;
                let nz = world.npcs[i].z;
                world.npcs[i].detour_x = nx + angle.cos() * 10.0;
                world.npcs[i].detour_z = nz + angle.sin() * 10.0;
            }
        }
    }

    // Soft boundary: push NPCs back when they wander too far (>200m from center)
    {
        let npc = &mut world.npcs[i];
        let dist_from_center = (npc.x * npc.x + npc.z * npc.z).sqrt();
        if dist_from_center > 200.0 {
            let push_strength = ((dist_from_center - 200.0) / 50.0).min(1.0) * 2.0 * dt;
            let inv = 1.0 / dist_from_center.max(0.01);
            let push_x = npc.x - npc.x * inv * push_strength * dist_from_center;
            let push_z = npc.z - npc.z * inv * push_strength * dist_from_center;
            if !on_river_not_bridge(push_x, push_z, &world.river_segments, &world.bridges) {
                npc.x = push_x;
                npc.z = push_z;
            }
        }
        npc.x = npc.x.clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
        npc.z = npc.z.clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
        npc.y = terrain.height_at(npc.x, npc.z).max(npc.y);
    }

    let dx2 = tx - world.npcs[i].x;
    let dz2 = tz - world.npcs[i].z;
    (dx2 * dx2 + dz2 * dz2).sqrt()
}

fn find_nearest_vending_machine(world: &WorldData, x: f32, z: f32) -> Option<(f32, f32)> {
    let mut best_dist = f32::MAX;
    let mut best = None;
    for inter in &world.interactibles {
        if inter.kind != InteractibleKind::VendingMachine { continue; }
        let dx = inter.x - x;
        let dz = inter.z - z;
        let d = dx * dx + dz * dz;
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

fn npc_going_to_work(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    world.npcs[i].state_timer += dt;

    // Timeout: if can't reach item in 60s, just start working where you are
    if world.npcs[i].state_timer > 60.0 {
        world.npcs[i].state = NpcState::Working;
        world.npcs[i].state_timer = 0.0;
        world.npcs[i].target_item = None;
        world.npcs[i].detouring = false;
        unclaim_item(world, i);
        return;
    }

    // Find nearest unclaimed item to set as work target area
    let npc = &world.npcs[i];
    if npc.target_item.is_none() {
        let item_idx = find_best_item(world, i);
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
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    if remaining < 3.0 {
        world.npcs[i].state = NpcState::Working;
        world.npcs[i].state_timer = 0.0;
        world.npcs[i].stuck_timer = 0.0;
        world.npcs[i].detouring = false;
    }
}

fn npc_working(
    world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain,
    dt: f32, brains: &mut [crate::neat::NeatBrain], time_of_day: f32,
    player_x: f32, player_z: f32,
) {
    world.npcs[i].state_timer += dt;

    if world.npcs[i].state_timer >= WORK_DURATION {
        // Work day done, go home
        world.npcs[i].state = NpcState::GoingHome;
        world.npcs[i].state_timer = 0.0;
        world.npcs[i].target_item = None;
        world.npcs[i].target_bin = None;
        world.npcs[i].interaction_target = None;
        world.npcs[i].detouring = false;
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
                npc_walk_toward(world, i, vm_x, vm_z, net, terrain, dt);
            }
            return;
        }
    }

    // Job AI handles work behavior and money earning (mut net for vehicle parking)
    crate::jobs::npc_work_dispatch(world, i, net, terrain, dt);

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

fn npc_going_home(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
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
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
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

    // Sync NPC position to vehicle
    world.npcs[i].x = world.vehicles[car_idx].x;
    world.npcs[i].y = world.vehicles[car_idx].y;
    world.npcs[i].z = world.vehicles[car_idx].z;

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
    }

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
        if item.skip_until > 0.0 { continue; }
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
            if check_npc_walk_collision(world, x, z, 0.5, usize::MAX) { continue; }
            if on_river_not_bridge(x, z, &world.river_segments, &world.bridges) { continue; }
            break;
        }
        let y = 40.0 + rng.range(0.0, 20.0);
        let kind = [ItemKind::Health, ItemKind::Money, ItemKind::Stamina][rng.next() as usize % 3];

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
            npc.stuck_count = 0;
            npc.detouring = false;
            npc.wander_cooldown = 0.0;
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
            let dx = px - item.x;
            let dz = pz - item.z;
            let d2 = dx * dx + dz * dz;
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
                ItemKind::Stamina => 0xFF33FF33,
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

    // Priority 5: Near interactible
    {
        let interact_dist_sq = INTERACT_DIST * INTERACT_DIST;
        let mut best_dist = interact_dist_sq;
        let mut best_ii = None;
        for (ii, inter) in world.interactibles.iter().enumerate() {
            if inter.cooldown > 0.0 { continue; }
            let dx = px - inter.x;
            let dz = pz - inter.z;
            let d2 = dx * dx + dz * dz;
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
