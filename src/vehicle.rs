// sys_vehicle: GTA V-style traffic AI — lane discipline, pathfinding, intersection stops,
// following distance, speed variation, smooth steering, parking
// Player enters/exits with Interact key, drives with movement keys

use crate::state::*;
use crate::world::{check_walk_collision, surface_at};
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
                state.world.vehicles[vi].parked = false;
                // Release parking spot
                if let Some(si) = state.world.vehicles[vi].parking_target {
                    if si < state.road_network.parking_spots.len() {
                        state.road_network.parking_spots[si].occupied_by = None;
                    }
                    state.world.vehicles[vi].parking_target = None;
                }
                state.player.in_vehicle = Some(vi);
            }
        }
    }

    // Update driven vehicle
    if let Some(vi) = state.player.in_vehicle {
        drive_vehicle(state, vi, dt);
    }

    // AI for NPC-driven vehicles
    let n = state.world.vehicles.len();
    for i in 0..n {
        if state.world.vehicles[i].occupied { continue; }
        if !state.world.vehicles[i].ai_active { continue; }
        if state.world.vehicles[i].parked { continue; }
        ai_drive(i, &mut state.world, &state.road_network, &state.terrain, dt);
    }

    // Snap newly-parked vehicles that ended up off-road to nearest parking/road position
    for i in 0..n {
        if !state.world.vehicles[i].parked { continue; }
        if state.world.vehicles[i].parking_target.is_some() { continue; } // already at a spot
        // Check if vehicle is on a road
        let vx = state.world.vehicles[i].x;
        let vz = state.world.vehicles[i].z;
        let surf = surface_at(vx, vz, &state.road_network);
        if surf == Surface::CarRoad { continue; } // already on road, fine
        // Off-road — snap to nearest parking spot or road edge
        snap_to_parking(i, &mut state.world, &mut state.road_network, &state.terrain);
    }

    // Vehicle-vehicle collision separation — push overlapping vehicles apart
    separate_vehicles(&mut state.world, &state.terrain);
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

    if !check_walk_collision(&state.world, new_x, cur_z, 1.5, None) {
        state.world.vehicles[vi].x = new_x;
    } else {
        state.world.vehicles[vi].speed *= -0.3;
    }
    let cur_x = state.world.vehicles[vi].x;
    if !check_walk_collision(&state.world, cur_x, new_z, 1.5, None) {
        state.world.vehicles[vi].z = new_z;
    } else {
        state.world.vehicles[vi].speed *= -0.3;
    }

    state.world.vehicles[vi].x = state.world.vehicles[vi].x.clamp(-WORLD_HALF, WORLD_HALF);
    state.world.vehicles[vi].z = state.world.vehicles[vi].z.clamp(-WORLD_HALF, WORLD_HALF);
    let vx = state.world.vehicles[vi].x;
    let vz = state.world.vehicles[vi].z;
    state.world.vehicles[vi].y = state.terrain.height_at(vx, vz) + VEHICLE_GROUND_OFFSET;

    // Smooth terrain normal for slope tilting (clamped to 30° max visual tilt)
    let target_n = crate::math::clamp_normal_tilt(state.terrain.normal_at(vx, vz), 30.0);
    let lerp_rate = 6.0 * dt;
    let v = &mut state.world.vehicles[vi];
    v.terrain_normal = crate::math::v3_normalize(crate::math::v3_lerp(v.terrain_normal, target_n, lerp_rate.min(1.0)));

    // Slope speed effect: uphill slows, downhill accelerates
    let (sr, cr) = state.world.vehicles[vi].rot_y.sin_cos();
    let fwd_dir = [-sr, 0.0, -cr];
    let slope_dot = target_n[0] * fwd_dir[0] + target_n[2] * fwd_dir[2]; // positive = facing uphill
    state.world.vehicles[vi].speed -= slope_dot * 15.0 * dt;

    state.player.x = state.world.vehicles[vi].x;
    state.player.y = state.world.vehicles[vi].y;
    state.player.z = state.world.vehicles[vi].z;
    state.player.rot_y = state.world.vehicles[vi].rot_y;
    state.player.terrain_normal = state.world.vehicles[vi].terrain_normal;

    // Speed limit check: player speeding on car road
    if state.world.vehicles[vi].speed.abs() > SPEED_LIMIT {
        let surf = surface_at(state.world.vehicles[vi].x, state.world.vehicles[vi].z, &state.road_network);
        if surf == Surface::CarRoad {
            state.player.wanted_vehicle_hit = true;
            state.player.bounty += 5.0 * dt;
        }
    }
}

// --- Dijkstra pathfinding on road graph ---

fn find_nearest_node(x: f32, z: f32, nodes: &[[f32; 2]]) -> usize {
    let mut best = 0;
    let mut best_d = f32::MAX;
    for (i, n) in nodes.iter().enumerate() {
        let dx = x - n[0];
        let dz = z - n[1];
        let d = dx * dx + dz * dz;
        if d < best_d { best_d = d; best = i; }
    }
    best
}

fn dijkstra(graph: &RoadGraph, start: usize, end: usize, node_count: usize) -> Vec<PathWaypoint> {
    if start == end || node_count == 0 { return Vec::new(); }

    let mut dist = vec![f32::MAX; node_count];
    let mut prev: Vec<Option<(usize, usize)>> = vec![None; node_count]; // (prev_node, seg_idx)
    let mut visited = vec![false; node_count];
    dist[start] = 0.0;

    for _ in 0..node_count {
        // Find unvisited node with minimum distance (linear scan — fine for ~15 nodes)
        let mut u = usize::MAX;
        let mut u_dist = f32::MAX;
        for n in 0..node_count {
            if !visited[n] && dist[n] < u_dist {
                u_dist = dist[n];
                u = n;
            }
        }
        if u == usize::MAX || u == end { break; }
        visited[u] = true;

        if u >= graph.adjacency.len() { break; }
        for &(neighbor, seg_idx, edge_dist) in &graph.adjacency[u] {
            let alt = dist[u] + edge_dist;
            if alt < dist[neighbor] {
                dist[neighbor] = alt;
                prev[neighbor] = Some((u, seg_idx));
            }
        }
    }

    // Reconstruct path
    if dist[end] == f32::MAX { return Vec::new(); }
    let mut path = Vec::new();
    let mut cur = end;
    while let Some((p, seg)) = prev[cur] {
        path.push(PathWaypoint { node_idx: cur, segment_idx: seg });
        cur = p;
    }
    path.reverse();
    path
}

fn plan_route(v: &mut Vehicle, net: &RoadNetwork) {
    if net.nodes.is_empty() || net.graph.adjacency.is_empty() { return; }

    let start = find_nearest_node(v.x, v.z, &net.nodes);
    let end = v.rng.next() as usize % net.nodes.len();
    if start == end {
        let end2 = (end + 1) % net.nodes.len();
        v.path = dijkstra(&net.graph, start, end2, net.nodes.len());
    } else {
        v.path = dijkstra(&net.graph, start, end, net.nodes.len());
    }
    v.path_idx = 0;
    v.intersection_state = IntersectionState::Cruising;

    if !v.path.is_empty() {
        v.current_segment = Some(v.path[0].segment_idx);
        // Set ai_target to first waypoint node for NPC driving compatibility
        let ni = v.path[0].node_idx;
        v.ai_target_x = net.nodes[ni][0];
        v.ai_target_z = net.nodes[ni][1];
    }
}

/// Compute lane center position at parameter t along a segment, offset to the right
fn lane_center(seg: &RoadSegment, t: f32, dir: LaneDirection) -> (f32, f32) {
    let dx = seg.x1 - seg.x0;
    let dz = seg.z1 - seg.z0;
    let len = (dx * dx + dz * dz).sqrt().max(0.01);
    let dir_x = dx / len;
    let dir_z = dz / len;

    // Perpendicular: always offset to right of travel direction
    let (perp_x, perp_z) = match dir {
        LaneDirection::Forward => (dir_z, -dir_x),   // right of A→B
        LaneDirection::Reverse => (-dir_z, dir_x),    // right of B→A (left of A→B)
    };

    let cx = seg.x0 + dx * t + perp_x * LANE_OFFSET;
    let cz = seg.z0 + dz * t + perp_z * LANE_OFFSET;
    (cx, cz)
}

/// Determine lane direction for a vehicle on a segment
fn compute_lane_dir(vx: f32, vz: f32, seg: &RoadSegment, target_x: f32, target_z: f32) -> LaneDirection {
    let dx = seg.x1 - seg.x0;
    let dz = seg.z1 - seg.z0;
    // Direction we're heading
    let tx = target_x - vx;
    let tz = target_z - vz;
    // Dot with segment direction
    let dot = tx * dx + tz * dz;
    if dot >= 0.0 { LaneDirection::Forward } else { LaneDirection::Reverse }
}

/// Wrap angle difference to [-PI, PI]
fn angle_diff(a: f32, b: f32) -> f32 {
    let mut d = a - b;
    while d > std::f32::consts::PI { d -= 2.0 * std::f32::consts::PI; }
    while d < -std::f32::consts::PI { d += 2.0 * std::f32::consts::PI; }
    d
}

fn ai_drive(vi: usize, world: &mut WorldData, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // 1. If no path → plan_route()
    if world.vehicles[vi].path.is_empty() {
        plan_route(&mut world.vehicles[vi], net);
        if world.vehicles[vi].path.is_empty() {
            // Fallback: no graph, just idle
            world.vehicles[vi].speed = 0.0;
            return;
        }
    }

    let path_idx = world.vehicles[vi].path_idx;
    if path_idx >= world.vehicles[vi].path.len() {
        // Arrived at end of path — deactivate (NPC handles next drive)
        world.vehicles[vi].ai_active = false;
        world.vehicles[vi].speed = 0.0;
        world.vehicles[vi].path.clear();
        return;
    }

    let path_idx = world.vehicles[vi].path_idx;
    let waypoint = world.vehicles[vi].path[path_idx];
    let target_node = net.nodes[waypoint.node_idx];
    let seg_idx = waypoint.segment_idx;

    // 2. Distance to current waypoint node
    let vx = world.vehicles[vi].x;
    let vz = world.vehicles[vi].z;
    let dx_node = target_node[0] - vx;
    let dz_node = target_node[1] - vz;
    let dist_to_node = (dx_node * dx_node + dz_node * dz_node).sqrt();

    // 3. Advance path when within 3m of current waypoint
    if dist_to_node < 3.0 {
        world.vehicles[vi].path_idx += 1;
        world.vehicles[vi].intersection_state = IntersectionState::Cruising;
        world.vehicles[vi].intersection_wait_timer = 0.0;

        if world.vehicles[vi].path_idx >= world.vehicles[vi].path.len() {
            // End of path — deactivate (NPC handles next drive)
            world.vehicles[vi].ai_active = false;
            world.vehicles[vi].speed = 0.0;
            world.vehicles[vi].path.clear();
            return;
        }

        // Update target from new waypoint
        let new_idx = world.vehicles[vi].path_idx;
        if new_idx < world.vehicles[vi].path.len() {
            let wp = world.vehicles[vi].path[new_idx];
            if wp.node_idx < net.nodes.len() {
                world.vehicles[vi].ai_target_x = net.nodes[wp.node_idx][0];
                world.vehicles[vi].ai_target_z = net.nodes[wp.node_idx][1];
            }
            world.vehicles[vi].current_segment = Some(wp.segment_idx);
        }
        return; // process new waypoint next frame
    }

    // 4. Update intersection state based on distance + cross-traffic
    let cruise_speed = world.vehicles[vi].cruise_speed;
    let mut target_speed = cruise_speed;

    match world.vehicles[vi].intersection_state {
        IntersectionState::Cruising => {
            if dist_to_node < INTERSECTION_APPROACH_DIST {
                world.vehicles[vi].intersection_state = IntersectionState::Approaching;
            }
        }
        IntersectionState::Approaching => {
            // Decelerate proportionally
            let factor = (dist_to_node / INTERSECTION_APPROACH_DIST).clamp(0.2, 1.0);
            target_speed = cruise_speed * factor;

            if dist_to_node < INTERSECTION_STOP_DIST {
                // Check for cross-traffic
                let has_cross_traffic = check_cross_traffic(vi, world, net, waypoint.node_idx);
                if has_cross_traffic {
                    world.vehicles[vi].intersection_state = IntersectionState::Waiting;
                    world.vehicles[vi].intersection_wait_timer = 0.0;
                } else {
                    world.vehicles[vi].intersection_state = IntersectionState::Turning;
                }
            }
        }
        IntersectionState::Waiting => {
            target_speed = 0.0;
            world.vehicles[vi].intersection_wait_timer += dt;

            // Re-check yield each frame, force through after max wait
            let has_cross_traffic = check_cross_traffic(vi, world, net, waypoint.node_idx);
            if !has_cross_traffic || world.vehicles[vi].intersection_wait_timer > INTERSECTION_WAIT_MAX {
                world.vehicles[vi].intersection_state = IntersectionState::Turning;
            }
        }
        IntersectionState::Turning => {
            target_speed = cruise_speed * 0.4;
        }
    }

    // 5. Compute turn angle for speed adjustment
    let desired = (-dx_node).atan2(-dz_node);
    let turn_angle = angle_diff(desired, world.vehicles[vi].rot_y).abs();
    let turn_factor = if turn_angle < 0.3 { 1.0 }
                      else if turn_angle < 1.0 { 0.6 }
                      else { 0.35 };
    target_speed *= turn_factor;

    // 6. Vehicle-ahead awareness + close-range omnidirectional avoidance
    {
        let vx = world.vehicles[vi].x;
        let vz = world.vehicles[vi].z;
        let rot = world.vehicles[vi].rot_y;
        let (sin_r, cos_r) = rot.sin_cos();
        let fwd_x = -sin_r;
        let fwd_z = -cos_r;

        let n_veh = world.vehicles.len();
        for j in 0..n_veh {
            if j == vi { continue; }
            let ojx = world.vehicles[j].x - vx;
            let ojz = world.vehicles[j].z - vz;
            let dist_sq = ojx * ojx + ojz * ojz;

            // Close-range omnidirectional brake: any vehicle within collision radius
            // regardless of heading — prevents intersection pileups
            let close_dist = VEHICLE_COLLISION_RADIUS * 2.0;
            if dist_sq < close_dist * close_dist && dist_sq > 0.01 {
                let dist = dist_sq.sqrt();
                // Emergency brake when very close, proportional slowdown otherwise
                if dist < VEHICLE_COLLISION_RADIUS * 1.2 {
                    target_speed = 0.0;
                } else {
                    let factor = ((dist - VEHICLE_COLLISION_RADIUS * 1.2) / (close_dist - VEHICLE_COLLISION_RADIUS * 1.2)).clamp(0.0, 1.0);
                    let close_speed = cruise_speed * factor;
                    if close_speed < target_speed {
                        target_speed = close_speed;
                    }
                }
            }

            // Forward-lane following distance (longer range, directional)
            if world.vehicles[j].parked { continue; }

            // Project onto forward vector
            let proj = ojx * fwd_x + ojz * fwd_z;
            if proj < 0.0 || proj > FOLLOW_DISTANCE { continue; }

            // Lateral offset
            let lateral = (ojx * fwd_z - ojz * fwd_x).abs();
            if lateral > 2.5 { continue; }

            // Same lane ahead — reduce speed
            if proj < MIN_FOLLOW_DISTANCE {
                target_speed = 0.0; // emergency brake
            } else {
                let factor = ((proj - MIN_FOLLOW_DISTANCE) / (FOLLOW_DISTANCE - MIN_FOLLOW_DISTANCE)).clamp(0.0, 1.0);
                let follow_speed = cruise_speed * factor;
                if follow_speed < target_speed {
                    target_speed = follow_speed;
                }
            }
        }
    }

    world.vehicles[vi].target_speed = target_speed;

    // 7. Lane discipline — steer toward lane center point ahead on segment
    let steer_target_x;
    let steer_target_z;

    if seg_idx < net.segments.len() {
        let seg = &net.segments[seg_idx];
        let vx = world.vehicles[vi].x;
        let vz = world.vehicles[vi].z;
        let target_x = world.vehicles[vi].ai_target_x;
        let target_z = world.vehicles[vi].ai_target_z;

        let lane_dir = compute_lane_dir(vx, vz, seg, target_x, target_z);
        world.vehicles[vi].lane_dir = lane_dir;

        // Find parameter t along segment closest to vehicle
        let sdx = seg.x1 - seg.x0;
        let sdz = seg.z1 - seg.z0;
        let len_sq = sdx * sdx + sdz * sdz;
        let t = if len_sq > 1e-8 {
            ((vx - seg.x0) * sdx + (vz - seg.z0) * sdz) / len_sq
        } else { 0.5 };

        // Look 5m ahead along segment
        let seg_len = len_sq.sqrt().max(0.01);
        let t_ahead = match lane_dir {
            LaneDirection::Forward => (t + 5.0 / seg_len).min(1.0),
            LaneDirection::Reverse => (t - 5.0 / seg_len).max(0.0),
        };

        let (lx, lz) = lane_center(seg, t_ahead, lane_dir);

        // Blend between lane center and waypoint node when close
        if dist_to_node < 8.0 {
            let blend = dist_to_node / 8.0;
            steer_target_x = lx * blend + target_node[0] * (1.0 - blend);
            steer_target_z = lz * blend + target_node[1] * (1.0 - blend);
        } else {
            steer_target_x = lx;
            steer_target_z = lz;
        }
    } else {
        steer_target_x = target_node[0];
        steer_target_z = target_node[1];
    }

    // 8. Steer toward target
    {
        let vx = world.vehicles[vi].x;
        let vz = world.vehicles[vi].z;
        let dx = steer_target_x - vx;
        let dz = steer_target_z - vz;
        let desired = (-dx).atan2(-dz);
        let diff = angle_diff(desired, world.vehicles[vi].rot_y);
        world.vehicles[vi].rot_y += diff.clamp(-VEHICLE_TURN_SPEED * dt, VEHICLE_TURN_SPEED * dt);
    }

    // 9. Accelerate/brake toward target_speed
    {
        let v = &mut world.vehicles[vi];
        let ts = v.target_speed;
        if v.speed < ts {
            v.speed = (v.speed + VEHICLE_ACCEL * 0.5 * dt).min(ts);
        } else if v.speed > ts {
            v.speed = (v.speed - VEHICLE_BRAKE * 0.5 * dt).max(ts).max(0.0);
        }
    }

    // 10. Move + building/river collision check
    {
        let rot = world.vehicles[vi].rot_y;
        let spd = world.vehicles[vi].speed;
        let cur_x = world.vehicles[vi].x;
        let cur_z = world.vehicles[vi].z;
        let new_x = cur_x - rot.sin() * spd * dt;
        let new_z = cur_z - rot.cos() * spd * dt;

        let river_x = crate::world::on_river_not_bridge(new_x, cur_z, &world.river_segments, &world.bridges);
        if !check_walk_collision(world, new_x, cur_z, 1.5, None) && !river_x {
            world.vehicles[vi].x = new_x;
        } else {
            world.vehicles[vi].speed *= -0.3;
            // Try to recover by adjusting rotation
            world.vehicles[vi].rot_y += 0.3;
        }
        let cur_x = world.vehicles[vi].x;
        let river_z = crate::world::on_river_not_bridge(cur_x, new_z, &world.river_segments, &world.bridges);
        if !check_walk_collision(world, cur_x, new_z, 1.5, None) && !river_z {
            world.vehicles[vi].z = new_z;
        } else {
            world.vehicles[vi].speed *= -0.3;
        }
    }

    // 11. Clamp + terrain follow
    world.vehicles[vi].x = world.vehicles[vi].x.clamp(-WORLD_HALF, WORLD_HALF);
    world.vehicles[vi].z = world.vehicles[vi].z.clamp(-WORLD_HALF, WORLD_HALF);
    world.vehicles[vi].y = terrain.height_at(world.vehicles[vi].x, world.vehicles[vi].z) + VEHICLE_GROUND_OFFSET;

    // Smooth terrain normal for slope tilting (clamped to 30° max visual tilt)
    let target_n = crate::math::clamp_normal_tilt(terrain.normal_at(world.vehicles[vi].x, world.vehicles[vi].z), 30.0);
    let lerp_rate = 6.0 * dt;
    world.vehicles[vi].terrain_normal = crate::math::v3_normalize(crate::math::v3_lerp(world.vehicles[vi].terrain_normal, target_n, lerp_rate.min(1.0)));

    // Slope speed effect: uphill slows, downhill accelerates
    let (sr, cr) = world.vehicles[vi].rot_y.sin_cos();
    let slope_dot = target_n[0] * (-sr) + target_n[2] * (-cr);
    world.vehicles[vi].speed -= slope_dot * 15.0 * dt;

    // 12. Gridlock recovery — auto-park vehicles stuck at low speed for too long
    if world.vehicles[vi].speed.abs() < 0.5 {
        world.vehicles[vi].idle_timer += dt;
        if world.vehicles[vi].idle_timer > 8.0 {
            world.vehicles[vi].ai_active = false;
            world.vehicles[vi].speed = 0.0;
            world.vehicles[vi].parked = true;
            world.vehicles[vi].path.clear();
            world.vehicles[vi].idle_timer = 0.0;
            return;
        }
    } else {
        world.vehicles[vi].idle_timer = 0.0;
    }

    // Speed limit check for NPC drivers
    if let Some(owner) = world.vehicles[vi].owner_npc {
        if world.vehicles[vi].speed.abs() > SPEED_LIMIT && owner < world.npcs.len() {
            let surf = surface_at(world.vehicles[vi].x, world.vehicles[vi].z, net);
            if surf == Surface::CarRoad && world.npcs[owner].violation_timer <= 0.0 {
                world.npcs[owner].wanted = true;
                world.npcs[owner].bounty += 5.0;
                world.npcs[owner].violation_timer = 10.0;
            }
        }
    }
}

/// Check if there's cross-traffic at intersection node that we should yield to
fn check_cross_traffic(vi: usize, world: &WorldData, net: &RoadNetwork, node_idx: usize) -> bool {
    if node_idx >= net.nodes.len() { return false; }
    let node = net.nodes[node_idx];
    let our_rot = world.vehicles[vi].rot_y;
    let (our_sin, our_cos) = our_rot.sin_cos();
    let our_fwd_x = -our_sin;
    let our_fwd_z = -our_cos;
    let our_dist_to_node = {
        let dx = node[0] - world.vehicles[vi].x;
        let dz = node[1] - world.vehicles[vi].z;
        (dx * dx + dz * dz).sqrt()
    };

    for (j, other) in world.vehicles.iter().enumerate() {
        if j == vi { continue; }
        if other.parked || other.speed < 0.5 { continue; }

        let dx = other.x - node[0];
        let dz = other.z - node[1];
        let dist = (dx * dx + dz * dz).sqrt();
        if dist > 12.0 { continue; }

        // Check if perpendicular heading (|dot product| < 0.5)
        let (o_sin, o_cos) = other.rot_y.sin_cos();
        let o_fwd_x = -o_sin;
        let o_fwd_z = -o_cos;
        let dot = our_fwd_x * o_fwd_x + our_fwd_z * o_fwd_z;

        if dot.abs() < 0.5 {
            // Perpendicular traffic — yield if they're closer or already turning
            if dist < our_dist_to_node || other.intersection_state == IntersectionState::Turning {
                return true;
            }
        }
    }
    false
}

/// Find nearest free parking spot within max_dist of (x, z)
pub fn find_nearest_parking_spot(net: &RoadNetwork, x: f32, z: f32, max_dist: f32) -> Option<usize> {
    let max_d2 = max_dist * max_dist;
    let mut best_d2 = max_d2;
    let mut best = None;
    for (i, spot) in net.parking_spots.iter().enumerate() {
        if spot.occupied_by.is_some() { continue; }
        let dx = spot.x - x;
        let dz = spot.z - z;
        let d2 = dx * dx + dz * dz;
        if d2 < best_d2 { best_d2 = d2; best = Some(i); }
    }
    best
}

/// Find the nearest point on any road segment. Returns (x, z, rot_y) of the
/// closest point on a road, clamped to the road edge (lane position).
pub fn nearest_road_position(net: &RoadNetwork, x: f32, z: f32) -> Option<(f32, f32, f32)> {
    let mut best_d2 = f32::MAX;
    let mut best_pos: Option<(f32, f32, f32)> = None;
    for seg in &net.segments {
        if seg.tier != RoadTier::CarRoad { continue; }
        let dx = seg.x1 - seg.x0;
        let dz = seg.z1 - seg.z0;
        let len_sq = dx * dx + dz * dz;
        if len_sq < 0.01 { continue; }
        // Project (x,z) onto segment
        let t = ((x - seg.x0) * dx + (z - seg.z0) * dz) / len_sq;
        let t = t.clamp(0.05, 0.95); // stay away from intersections
        let px = seg.x0 + dx * t;
        let pz = seg.z0 + dz * t;
        // Offset to road edge (right lane curb)
        let len = len_sq.sqrt();
        let perp_x = -dz / len;
        let perp_z = dx / len;
        let curb_offset = CAR_ROAD_WIDTH * 0.5 - 1.0; // park near road edge
        let rx = px + perp_x * curb_offset;
        let rz = pz + perp_z * curb_offset;
        let ddx = rx - x;
        let ddz = rz - z;
        let d2 = ddx * ddx + ddz * ddz;
        if d2 < best_d2 {
            best_d2 = d2;
            let rot = (-dx).atan2(-dz);
            best_pos = Some((rx, rz, rot));
        }
        // Try opposite side too
        let lx = px - perp_x * curb_offset;
        let lz = pz - perp_z * curb_offset;
        let ddx = lx - x;
        let ddz = lz - z;
        let d2 = ddx * ddx + ddz * ddz;
        if d2 < best_d2 {
            best_d2 = d2;
            let rot = (-dx).atan2(-dz) + std::f32::consts::PI;
            best_pos = Some((lx, lz, rot));
        }
    }
    best_pos
}

/// Snap a vehicle to the nearest parking spot or road edge when it needs to park.
/// Returns true if the vehicle was snapped to a valid position.
pub fn snap_to_parking(vi: usize, world: &mut WorldData, net: &mut RoadNetwork, terrain: &Terrain) -> bool {
    let vx = world.vehicles[vi].x;
    let vz = world.vehicles[vi].z;

    // First try: find a free parking spot within 30m
    if let Some(si) = find_nearest_parking_spot(net, vx, vz, 30.0) {
        let spot = &net.parking_spots[si];
        world.vehicles[vi].x = spot.x;
        world.vehicles[vi].z = spot.z;
        world.vehicles[vi].y = terrain.height_at(spot.x, spot.z);
        world.vehicles[vi].rot_y = spot.rot_y;
        world.vehicles[vi].parking_target = Some(si);
        net.parking_spots[si].occupied_by = Some(vi);
        return true;
    }

    // Second try: snap to nearest road edge
    if let Some((rx, rz, rot)) = nearest_road_position(net, vx, vz) {
        world.vehicles[vi].x = rx;
        world.vehicles[vi].z = rz;
        world.vehicles[vi].y = terrain.height_at(rx, rz);
        world.vehicles[vi].rot_y = rot;
        return true;
    }

    false
}

/// Push overlapping vehicles apart so they don't interpenetrate at intersections.
/// Uses a simple radial separation with oriented-box awareness for elongated vehicles.
fn separate_vehicles(world: &mut WorldData, terrain: &Terrain) {
    let n = world.vehicles.len();
    let sep_radius = VEHICLE_COLLISION_RADIUS * 2.0; // diameter for pair check
    let sep_radius_sq = sep_radius * sep_radius;

    for i in 0..n {
        // Skip parked vehicles that aren't being driven — they don't need separation
        // (keeps parked cars in their spots undisturbed)
        let i_moving = world.vehicles[i].speed.abs() > 0.1
            || world.vehicles[i].ai_active
            || world.vehicles[i].occupied;
        if !i_moving && world.vehicles[i].parked { continue; }

        for j in (i + 1)..n {
            let j_moving = world.vehicles[j].speed.abs() > 0.1
                || world.vehicles[j].ai_active
                || world.vehicles[j].occupied;
            // At least one vehicle must be moving/active for separation
            if !i_moving && !j_moving { continue; }

            let dx = world.vehicles[j].x - world.vehicles[i].x;
            let dz = world.vehicles[j].z - world.vehicles[i].z;
            let d2 = dx * dx + dz * dz;
            if d2 >= sep_radius_sq || d2 < 0.001 { continue; }

            let dist = d2.sqrt();
            let overlap = sep_radius - dist;
            if overlap <= 0.0 { continue; }

            // Separation direction (from i toward j)
            let inv_d = 1.0 / dist;
            let nx = dx * inv_d;
            let nz = dz * inv_d;

            // Push each vehicle by half the overlap (symmetric)
            let push = overlap * 0.5 + 0.05; // small extra margin

            // Determine push weights — faster vehicle yields less, slower yields more
            let si = world.vehicles[i].speed.abs();
            let sj = world.vehicles[j].speed.abs();
            let total_speed = si + sj + 0.1; // avoid div by zero
            // Higher speed = less push (you have right-of-way if faster)
            let wi = sj / total_speed; // push weight for vehicle i (pushed more if j is faster)
            let wj = si / total_speed;

            // Don't push parked vehicles
            let (pi, pj) = if world.vehicles[i].parked {
                (0.0, 1.0)
            } else if world.vehicles[j].parked {
                (1.0, 0.0)
            } else {
                (wi, wj)
            };

            world.vehicles[i].x -= nx * push * pi;
            world.vehicles[i].z -= nz * push * pi;
            world.vehicles[j].x += nx * push * pj;
            world.vehicles[j].z += nz * push * pj;

            // Slow down colliding vehicles
            if si > 1.0 { world.vehicles[i].speed *= 0.85; }
            if sj > 1.0 { world.vehicles[j].speed *= 0.85; }
        }
    }

    // Re-snap Y to terrain after separation pushes
    for i in 0..n {
        let vx = world.vehicles[i].x;
        let vz = world.vehicles[i].z;
        world.vehicles[i].y = terrain.height_at(vx, vz) + VEHICLE_GROUND_OFFSET;
    }
}
