// OBB collision system: 2D oriented bounding boxes in XZ plane, SAT intersection
// Handles vehicle-NPC, vehicle-vehicle, vehicle-player, NPC-NPC, player-NPC collisions

use crate::state::*;

pub struct Obb2d {
    pub cx: f32, pub cz: f32,         // center
    pub half_w: f32, pub half_d: f32,  // half-extents (local X, local Z)
    pub sin_r: f32, pub cos_r: f32,    // precomputed rotation
}

impl Obb2d {
    pub fn new(cx: f32, cz: f32, w: f32, d: f32, rot: f32) -> Self {
        let (sin_r, cos_r) = rot.sin_cos();
        Obb2d { cx, cz, half_w: w * 0.5, half_d: d * 0.5, sin_r, cos_r }
    }

    pub fn from_vehicle(v: &Vehicle) -> Self {
        // Match visual mesh: 1.86m wide x 4.6m long, scaled per vehicle
        Self::new(v.x, v.z, 1.86 * v.scale, 4.6 * v.scale, v.rot_y)
    }

    pub fn from_npc(n: &Npc) -> Self {
        Self::new(n.x, n.z, 0.6, 0.35, n.rot_y)
    }

    pub fn from_player(p: &Player) -> Self {
        Self::new(p.x, p.z, 0.6, 0.35, p.rot_y)
    }

    // Returns the 4 corners in world space (XZ)
    fn corners(&self) -> [[f32; 2]; 4] {
        let dx_w = self.cos_r * self.half_w;
        let dz_w = -self.sin_r * self.half_w;
        let dx_d = self.sin_r * self.half_d;
        let dz_d = self.cos_r * self.half_d;
        [
            [self.cx + dx_w + dx_d, self.cz + dz_w + dz_d],
            [self.cx - dx_w + dx_d, self.cz - dz_w + dz_d],
            [self.cx - dx_w - dx_d, self.cz - dz_w - dz_d],
            [self.cx + dx_w - dx_d, self.cz + dz_w - dz_d],
        ]
    }
}

/// SAT intersection test. Returns Some((normal_x, normal_z, depth)) or None.
pub fn obb_intersect(a: &Obb2d, b: &Obb2d) -> Option<(f32, f32, f32)> {
    let ca = a.corners();
    let cb = b.corners();

    let axes = [
        // A's local axes
        [a.cos_r, -a.sin_r],
        [a.sin_r, a.cos_r],
        // B's local axes
        [b.cos_r, -b.sin_r],
        [b.sin_r, b.cos_r],
    ];

    let mut min_depth = f32::MAX;
    let mut min_axis = [0.0f32; 2];

    for axis in &axes {
        let (a_min, a_max) = project_corners(&ca, axis);
        let (b_min, b_max) = project_corners(&cb, axis);

        // Check for separation
        if a_max < b_min || b_max < a_min {
            return None; // separating axis found
        }

        let overlap = a_max.min(b_max) - a_min.max(b_min);
        if overlap < min_depth {
            min_depth = overlap;
            // Normal points from A to B
            let center_proj_a = (a_min + a_max) * 0.5;
            let center_proj_b = (b_min + b_max) * 0.5;
            if center_proj_b < center_proj_a {
                min_axis = [-axis[0], -axis[1]];
            } else {
                min_axis = *axis;
            }
        }
    }

    Some((min_axis[0], min_axis[1], min_depth))
}

fn project_corners(corners: &[[f32; 2]; 4], axis: &[f32; 2]) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for c in corners {
        let proj = c[0] * axis[0] + c[1] * axis[1];
        if proj < min { min = proj; }
        if proj > max { max = proj; }
    }
    (min, max)
}

/// Initialize ragdoll from NPC's current position using the articulated skeleton.
/// Falls back to legacy 7-point initialization and also activates the skeleton ragdoll.
fn init_ragdoll(npc: &mut Npc, impulse_x: f32, impulse_y: f32, impulse_z: f32) {
    let x = npc.x;
    let y = npc.y;
    let z = npc.z;

    // Activate skeleton ragdoll (new system)
    let impulse = [impulse_x, impulse_y, impulse_z];
    npc.skeleton.activate_ragdoll([x, y, z], npc.rot_y, impulse);
    npc.skeleton.ragdoll_timer = RAGDOLL_DURATION;

    // Init legacy ragdoll points for rendering compatibility
    npc.ragdoll_points = npc.skeleton.to_ragdoll_points();
    npc.ragdoll_active = true;
    npc.ragdoll_timer = RAGDOLL_DURATION;
}

fn push_apart_npcs_safe(world: &mut WorldData, i: usize, j: usize, d: f32, dx: f32, dz: f32, min_dist: f32) {
    let overlap = min_dist - d;
    let nx = dx / d;
    let nz = dz / d;

    // Mass-based push: lighter NPC moves more
    let mass_i = world.npcs[i].body.mass;
    let mass_j = world.npcs[j].body.mass;
    let inv_sum = 1.0 / mass_i + 1.0 / mass_j;
    let push_i = overlap * (1.0 / mass_i) / inv_sum;
    let push_j = overlap * (1.0 / mass_j) / inv_sum;

    // Impulse for separation
    let sep_impulse = overlap * 50.0; // gentle push force
    world.npcs[i].body.apply_impulse([-nx * sep_impulse, 0.0, -nz * sep_impulse]);
    world.npcs[j].body.apply_impulse([nx * sep_impulse, 0.0, nz * sep_impulse]);

    // Positional correction
    let new_ix = world.npcs[i].body.pos[0] - nx * push_i;
    let new_iz = world.npcs[i].body.pos[2] - nz * push_i;
    let new_jx = world.npcs[j].body.pos[0] + nx * push_j;
    let new_jz = world.npcs[j].body.pos[2] + nz * push_j;
    if !crate::world::on_river_not_bridge(new_ix, new_iz, &world.river_segments, &world.bridges) {
        world.npcs[i].body.pos[0] = new_ix;
        world.npcs[i].body.pos[2] = new_iz;
        world.npcs[i].x = new_ix;
        world.npcs[i].z = new_iz;
    }
    if !crate::world::on_river_not_bridge(new_jx, new_jz, &world.river_segments, &world.bridges) {
        world.npcs[j].body.pos[0] = new_jx;
        world.npcs[j].body.pos[2] = new_jz;
        world.npcs[j].x = new_jx;
        world.npcs[j].z = new_jz;
    }
}

fn decay_violation_timers(world: &mut WorldData, dt: f32) {
    for npc in &mut world.npcs {
        if npc.violation_timer > 0.0 {
            npc.violation_timer -= dt;
        }
    }
}

/// Apply a radial explosion at a world position, affecting all nearby entities.
/// Vehicles, NPCs, and player receive impulses proportional to distance.
/// NPCs above the force threshold are ragdolled.
pub fn apply_world_explosion(
    world: &mut WorldData, player: &mut Player,
    origin: crate::math::Vec3, radius: f32, force: f32,
) {
    // Vehicles
    for v in &mut world.vehicles {
        crate::physics::apply_explosion(&mut v.body, origin, radius, force);
        // Sync legacy fields
        v.x = v.body.pos[0];
        v.z = v.body.pos[2];
        let fwd = crate::math::quat_forward(v.body.quat);
        v.speed = crate::math::v3_dot(v.body.vel, fwd);
    }

    // NPCs
    for ni in 0..world.npcs.len() {
        if world.npcs[ni].ragdoll_active { continue; }
        let npc_pos = [world.npcs[ni].x, world.npcs[ni].y + 0.9, world.npcs[ni].z];
        let d = crate::math::v3_sub(npc_pos, origin);
        let dist = crate::math::v3_len(d);
        if dist >= radius || dist < 0.01 { continue; }

        let falloff = 1.0 - dist / radius;
        let impulse_mag = force * falloff;

        // Apply impulse to NPC body
        let dir = crate::math::v3_scale(d, 1.0 / dist);
        let impulse = [
            dir[0] * impulse_mag,
            (dir[1] + 0.3) * impulse_mag, // upward bias
            dir[2] * impulse_mag,
        ];
        world.npcs[ni].body.apply_impulse(impulse);

        // Ragdoll if impulse exceeds threshold (force > mass * 5 m/s → ragdoll)
        let speed_change = impulse_mag / world.npcs[ni].body.mass;
        if speed_change > 5.0 {
            init_ragdoll(&mut world.npcs[ni], impulse[0], impulse[1], impulse[2]);
            world.npcs[ni].health -= impulse_mag * 0.1;
        }
    }

    // Player
    let player_pos = [player.x, player.y + 0.9, player.z];
    let mag = crate::physics::apply_explosion(&mut player.body, player_pos, radius, force);
    if mag > 0.0 {
        player.x = player.body.pos[0];
        player.z = player.body.pos[2];
        let damage = mag * 0.1;
        player.health = (player.health - damage).max(0.0);
        player.damage_shake = (mag / force * 2.0).min(1.0);
    }
}

/// Full collision pass each frame
pub fn sys_collisions(world: &mut WorldData, player: &mut Player, _terrain: &Terrain, dt: f32) {
    let nv = world.vehicles.len();
    let nn = world.npcs.len();

    // --- Vehicle → NPC collisions ---
    for vi in 0..nv {
        let vspeed = world.vehicles[vi].speed.abs();
        if vspeed < 0.5 { continue; } // skip slow/parked vehicles

        let vobb = Obb2d::from_vehicle(&world.vehicles[vi]);

        for ni in 0..nn {
            if world.npcs[ni].in_vehicle { continue; }
            if world.npcs[ni].state == NpcState::Sleeping { continue; }
            if world.npcs[ni].ragdoll_active { continue; }

            // Broad phase
            let dx = world.npcs[ni].x - world.vehicles[vi].x;
            let dz = world.npcs[ni].z - world.vehicles[vi].z;
            if dx.abs() > 10.0 || dz.abs() > 10.0 { continue; }

            let nobb = Obb2d::from_npc(&world.npcs[ni]);
            if let Some((nx, nz, _depth)) = obb_intersect(&vobb, &nobb) {
                // Vehicle hits NPC
                let damage = vspeed * VEHICLE_HIT_DAMAGE_MULT;
                world.npcs[ni].health -= damage;
                world.npcs[ni].hit_flash = HIT_FLASH_DURATION;

                // Mass-based momentum transfer for launch
                let v_mass = world.vehicles[vi].body.mass;
                let n_mass = world.npcs[ni].body.mass;
                let inv_mass_sum = 1.0 / v_mass + 1.0 / n_mass;
                let restitution = 0.5;
                let j_mag = (1.0 + restitution) * vspeed / inv_mass_sum;

                // Launch NPC with physics impulse
                let vrot = world.vehicles[vi].rot_y;
                let fwd_x = -vrot.sin();
                let fwd_z = -vrot.cos();
                let launch_x = fwd_x * j_mag / n_mass + nx * 3.0;
                let launch_z = fwd_z * j_mag / n_mass + nz * 3.0;

                // Apply impulse to NPC body always
                world.npcs[ni].body.apply_impulse([launch_x, VEHICLE_HIT_LAUNCH_UP, launch_z]);

                // Ragdoll only above force threshold: velocity change > 3 m/s
                // Low-speed bumps push the NPC but don't ragdoll them
                let speed_change = j_mag / n_mass;
                if speed_change > 3.0 {
                    init_ragdoll(&mut world.npcs[ni], launch_x, VEHICLE_HIT_LAUNCH_UP, launch_z);
                }

                // Vehicle slows proportional to mass ratio (barely for car vs person)
                world.vehicles[vi].body.apply_impulse([-fwd_x * j_mag, 0.0, -fwd_z * j_mag]);
                let fwd_v = crate::math::quat_forward(world.vehicles[vi].body.quat);
                world.vehicles[vi].speed = crate::math::v3_dot(world.vehicles[vi].body.vel, fwd_v);

                // Mark driver as wanted (hit-and-run)
                if let Some(owner) = world.vehicles[vi].owner_npc {
                    if owner < nn {
                        world.npcs[owner].wanted = true;
                        world.npcs[owner].bounty += damage * 0.5;
                    }
                }
                // If player is driving this vehicle
                if world.vehicles[vi].occupied {
                    player.wanted_vehicle_hit = true;
                }

                // KO check
                if world.npcs[ni].health <= 0.0 {
                    crate::combat::knockout_npc(&mut world.npcs[ni]);
                }
            }
        }
    }

    // --- Vehicle → Player collision ---
    if player.in_vehicle.is_none() {
        let pobb = Obb2d::from_player(player);
        for vi in 0..nv {
            let vspeed = world.vehicles[vi].speed.abs();
            if vspeed < 0.5 { continue; }
            if world.vehicles[vi].occupied { continue; } // can't hit yourself

            let dx = player.x - world.vehicles[vi].x;
            let dz = player.z - world.vehicles[vi].z;
            if dx.abs() > 10.0 || dz.abs() > 10.0 { continue; }

            let vobb = Obb2d::from_vehicle(&world.vehicles[vi]);
            if let Some((nx, nz, depth)) = obb_intersect(&vobb, &pobb) {
                let damage = vspeed * VEHICLE_HIT_DAMAGE_MULT;
                player.health = (player.health - damage).max(0.0);
                player.hit_flash = HIT_FLASH_DURATION;
                player.damage_shake = CAMERA_SHAKE_INTENSITY;

                // Mass-based push: player body gets impulse
                let v_mass = world.vehicles[vi].body.mass;
                let p_mass = player.body.mass;
                let inv_mass_sum = 1.0 / v_mass + 1.0 / p_mass;
                let restitution = 0.4;
                let j_mag = (1.0 + restitution) * vspeed / inv_mass_sum;
                player.body.apply_impulse([nx * j_mag, j_mag * 0.2, nz * j_mag]);
                player.body.pos[0] += nx * (depth + 0.3);
                player.body.pos[2] += nz * (depth + 0.3);
                player.x = player.body.pos[0];
                player.z = player.body.pos[2];

                // Vehicle slows proportional to mass ratio
                world.vehicles[vi].body.apply_impulse([-nx * j_mag, 0.0, -nz * j_mag]);
                let fwd_v = crate::math::quat_forward(world.vehicles[vi].body.quat);
                world.vehicles[vi].speed = crate::math::v3_dot(world.vehicles[vi].body.vel, fwd_v);

                // Mark driver as wanted
                if let Some(owner) = world.vehicles[vi].owner_npc {
                    if owner < nn {
                        world.npcs[owner].wanted = true;
                        world.npcs[owner].bounty += damage * 0.5;
                    }
                }
            }
        }
    }

    // --- Vehicle → Wall collision ---
    for vi in 0..nv {
        if world.vehicles[vi].speed.abs() < 0.1 { continue; }
        let vobb = Obb2d::from_vehicle(&world.vehicles[vi]);
        for w in &world.walls {
            // Broad phase
            let dx = world.vehicles[vi].x - w.x;
            let dz = world.vehicles[vi].z - w.z;
            if dx.abs() > w.hw + 3.0 || dz.abs() > w.hd + 3.0 { continue; }
            // AABB collision with push-out
            let overlap_x = (w.hw + vobb.half_w) - dx.abs();
            let overlap_z = (w.hd + vobb.half_d) - dz.abs();
            if overlap_x > 0.0 && overlap_z > 0.0 {
                if overlap_x < overlap_z {
                    let sign = if dx > 0.0 { 1.0 } else { -1.0 };
                    world.vehicles[vi].body.pos[0] += sign * overlap_x;
                } else {
                    let sign = if dz > 0.0 { 1.0 } else { -1.0 };
                    world.vehicles[vi].body.pos[2] += sign * overlap_z;
                }
                let restitution = 0.2;
                if overlap_x < overlap_z {
                    world.vehicles[vi].body.vel[0] *= -restitution;
                } else {
                    world.vehicles[vi].body.vel[2] *= -restitution;
                }
                world.vehicles[vi].x = world.vehicles[vi].body.pos[0];
                world.vehicles[vi].z = world.vehicles[vi].body.pos[2];
                let fwd = crate::math::quat_forward(world.vehicles[vi].body.quat);
                world.vehicles[vi].speed = crate::math::v3_dot(world.vehicles[vi].body.vel, fwd);
            }
        }
        // Vehicle → street light collision
        for sl in &world.street_lights {
            let dx = world.vehicles[vi].x - sl.x;
            let dz = world.vehicles[vi].z - sl.z;
            let d2 = dx * dx + dz * dz;
            let min_r = 0.3 + vobb.half_w.max(vobb.half_d);
            if d2 < min_r * min_r && d2 > 0.001 {
                let d = d2.sqrt();
                let push = min_r - d;
                let norm_x = dx / d;
                let norm_z = dz / d;
                world.vehicles[vi].body.pos[0] += norm_x * push;
                world.vehicles[vi].body.pos[2] += norm_z * push;
                // Reflect velocity component along collision normal
                let vn = world.vehicles[vi].body.vel[0] * norm_x + world.vehicles[vi].body.vel[2] * norm_z;
                if vn < 0.0 {
                    world.vehicles[vi].body.vel[0] -= norm_x * vn * 1.3;
                    world.vehicles[vi].body.vel[2] -= norm_z * vn * 1.3;
                }
                world.vehicles[vi].x = world.vehicles[vi].body.pos[0];
                world.vehicles[vi].z = world.vehicles[vi].body.pos[2];
                let fwd = crate::math::quat_forward(world.vehicles[vi].body.quat);
                world.vehicles[vi].speed = crate::math::v3_dot(world.vehicles[vi].body.vel, fwd);
            }
        }
    }

    // --- Vehicle → Vehicle collision ---
    for i in 0..nv {
        if world.vehicles[i].speed.abs() < 0.1 { continue; }
        for j in (i + 1)..nv {
            let dx = world.vehicles[j].x - world.vehicles[i].x;
            let dz = world.vehicles[j].z - world.vehicles[i].z;
            if dx.abs() > 10.0 || dz.abs() > 10.0 { continue; }

            let obb_a = Obb2d::from_vehicle(&world.vehicles[i]);
            let obb_b = Obb2d::from_vehicle(&world.vehicles[j]);

            if let Some((nx, nz, depth)) = obb_intersect(&obb_a, &obb_b) {
                // Mass-based collision response
                let mass_i = world.vehicles[i].body.mass;
                let mass_j = world.vehicles[j].body.mass;
                let inv_mass_sum = 1.0 / mass_i + 1.0 / mass_j;

                // Positional correction (mass-weighted: lighter vehicle moves more)
                let push = depth + 0.05;
                let wi = (1.0 / mass_i) / inv_mass_sum;
                let wj = (1.0 / mass_j) / inv_mass_sum;
                world.vehicles[i].body.pos[0] -= nx * push * wi;
                world.vehicles[i].body.pos[2] -= nz * push * wi;
                world.vehicles[j].body.pos[0] += nx * push * wj;
                world.vehicles[j].body.pos[2] += nz * push * wj;

                // Impulse-based velocity exchange
                let rel_vn = (world.vehicles[i].body.vel[0] - world.vehicles[j].body.vel[0]) * nx
                           + (world.vehicles[i].body.vel[2] - world.vehicles[j].body.vel[2]) * nz;
                let relative_speed = rel_vn.abs();
                if rel_vn > 0.0 {
                    let restitution = 0.3;
                    let j_mag = (1.0 + restitution) * rel_vn / inv_mass_sum;
                    world.vehicles[i].body.apply_impulse([-nx * j_mag, 0.0, -nz * j_mag]);
                    world.vehicles[j].body.apply_impulse([nx * j_mag, 0.0, nz * j_mag]);
                }

                // Sync legacy fields from body
                world.vehicles[i].x = world.vehicles[i].body.pos[0];
                world.vehicles[i].z = world.vehicles[i].body.pos[2];
                let fwd_i = crate::math::quat_forward(world.vehicles[i].body.quat);
                world.vehicles[i].speed = crate::math::v3_dot(world.vehicles[i].body.vel, fwd_i);
                world.vehicles[j].x = world.vehicles[j].body.pos[0];
                world.vehicles[j].z = world.vehicles[j].body.pos[2];
                let fwd_j = crate::math::quat_forward(world.vehicles[j].body.quat);
                world.vehicles[j].speed = crate::math::v3_dot(world.vehicles[j].body.vel, fwd_j);

                // Panel deformation from crash energy
                if relative_speed > 2.0 {
                    let mass_i = world.vehicles[i].body.mass;
                    let mass_j = world.vehicles[j].body.mass;
                    let energy = 0.5 * (mass_i * mass_j / (mass_i + mass_j)) * relative_speed * relative_speed;
                    // Impact point in each vehicle's local space
                    let impact_i = [nx * obb_a.half_d, 0.0, nz * obb_a.half_d];
                    let impact_j = [-nx * obb_b.half_d, 0.0, -nz * obb_b.half_d];
                    world.vehicles[i].deformation.apply_impact(impact_i, energy, obb_a.half_w, obb_a.half_d);
                    world.vehicles[j].deformation.apply_impact(impact_j, energy, obb_b.half_w, obb_b.half_d);

                    // Explosion when a vehicle becomes totaled from this impact
                    if world.vehicles[i].deformation.is_totaled() {
                        let origin = world.vehicles[i].body.pos;
                        let blast_force = energy.sqrt().min(500.0);
                        // Apply explosion impulse to all nearby NPCs
                        for ni in 0..nn {
                            if world.npcs[ni].ragdoll_active { continue; }
                            let mag = crate::physics::apply_explosion(&mut world.npcs[ni].body, origin, 10.0, blast_force);
                            if mag / world.npcs[ni].body.mass > 5.0 {
                                let imp = crate::math::v3_scale(
                                    crate::math::v3_sub(world.npcs[ni].body.pos, origin),
                                    mag / crate::math::v3_len(crate::math::v3_sub(world.npcs[ni].body.pos, origin)).max(0.1),
                                );
                                init_ragdoll(&mut world.npcs[ni], imp[0], imp[1] + mag * 0.3, imp[2]);
                            }
                        }
                        // Explosion impulse to player
                        crate::physics::apply_explosion(&mut player.body, origin, 10.0, blast_force);
                        // Impulse to the other vehicle
                        crate::physics::apply_explosion(&mut world.vehicles[j].body, origin, 10.0, blast_force);
                    }
                    if world.vehicles[j].deformation.is_totaled() {
                        let origin = world.vehicles[j].body.pos;
                        let blast_force = energy.sqrt().min(500.0);
                        for ni in 0..nn {
                            if world.npcs[ni].ragdoll_active { continue; }
                            let mag = crate::physics::apply_explosion(&mut world.npcs[ni].body, origin, 10.0, blast_force);
                            if mag / world.npcs[ni].body.mass > 5.0 {
                                let imp = crate::math::v3_scale(
                                    crate::math::v3_sub(world.npcs[ni].body.pos, origin),
                                    mag / crate::math::v3_len(crate::math::v3_sub(world.npcs[ni].body.pos, origin)).max(0.1),
                                );
                                init_ragdoll(&mut world.npcs[ni], imp[0], imp[1] + mag * 0.3, imp[2]);
                            }
                        }
                        crate::physics::apply_explosion(&mut player.body, origin, 10.0, blast_force);
                        crate::physics::apply_explosion(&mut world.vehicles[i].body, origin, 10.0, blast_force);
                    }
                }

                // Occupant damage
                let occupant_damage = relative_speed * VEHICLE_CRASH_SELF_DAMAGE;
                if world.vehicles[i].occupied {
                    player.health = (player.health - occupant_damage).max(0.0);
                    player.damage_shake = CAMERA_SHAKE_INTENSITY;
                }
                if world.vehicles[j].occupied {
                    player.health = (player.health - occupant_damage).max(0.0);
                    player.damage_shake = CAMERA_SHAKE_INTENSITY;
                }
                // NPC driver damage
                if let Some(owner) = world.vehicles[i].owner_npc {
                    if owner < nn && world.npcs[owner].in_vehicle {
                        world.npcs[owner].health -= occupant_damage;
                    }
                }
                if let Some(owner) = world.vehicles[j].owner_npc {
                    if owner < nn && world.npcs[owner].in_vehicle {
                        world.npcs[owner].health -= occupant_damage;
                    }
                }
            }
        }
    }

    // --- NPC → NPC soft push-apart ---
    for i in 0..nn {
        if world.npcs[i].in_vehicle || world.npcs[i].state == NpcState::Sleeping { continue; }
        if world.npcs[i].ragdoll_active { continue; }
        for j in (i + 1)..nn {
            if world.npcs[j].in_vehicle || world.npcs[j].state == NpcState::Sleeping { continue; }
            if world.npcs[j].ragdoll_active { continue; }

            let dx = world.npcs[j].x - world.npcs[i].x;
            let dz = world.npcs[j].z - world.npcs[i].z;
            let d2 = dx * dx + dz * dz;
            let min_dist = 0.6; // NPC body width
            if d2 < min_dist * min_dist && d2 > 0.001 {
                let d = d2.sqrt();
                push_apart_npcs_safe(world, i, j, d, dx, dz, min_dist);
            }
        }
    }

    // --- Player → NPC soft push-apart ---
    if player.in_vehicle.is_none() {
        for ni in 0..nn {
            if world.npcs[ni].in_vehicle || world.npcs[ni].state == NpcState::Sleeping { continue; }
            if world.npcs[ni].ragdoll_active { continue; }

            let dx = world.npcs[ni].x - player.x;
            let dz = world.npcs[ni].z - player.z;
            let d2 = dx * dx + dz * dz;
            let min_dist = 0.6;
            if d2 < min_dist * min_dist && d2 > 0.001 {
                let d = d2.sqrt();
                let overlap = min_dist - d;
                let nx = dx / d;
                let nz = dz / d;
                // Mass-based push
                let p_mass = player.body.mass;
                let n_mass = world.npcs[ni].body.mass;
                let inv_sum = 1.0 / p_mass + 1.0 / n_mass;
                let push_p = overlap * (1.0 / p_mass) / inv_sum;
                let push_n = overlap * (1.0 / n_mass) / inv_sum;
                world.npcs[ni].body.pos[0] += nx * push_n;
                world.npcs[ni].body.pos[2] += nz * push_n;
                world.npcs[ni].x = world.npcs[ni].body.pos[0];
                world.npcs[ni].z = world.npcs[ni].body.pos[2];
                player.body.pos[0] -= nx * push_p;
                player.body.pos[2] -= nz * push_p;
                player.x = player.body.pos[0];
                player.z = player.body.pos[2];
            }
        }
    }

    // Tick violation timers
    decay_violation_timers(world, dt);
}

/// Headless collision pass for training (no player, no rendering)
pub fn sys_collisions_headless(world: &mut WorldData, _terrain: &Terrain, dt: f32) {
    let nv = world.vehicles.len();
    let nn = world.npcs.len();

    // Vehicle → NPC
    for vi in 0..nv {
        let vspeed = world.vehicles[vi].speed.abs();
        if vspeed < 0.5 { continue; }
        let vobb = Obb2d::from_vehicle(&world.vehicles[vi]);

        for ni in 0..nn {
            if world.npcs[ni].in_vehicle || world.npcs[ni].state == NpcState::Sleeping { continue; }
            if world.npcs[ni].ragdoll_active { continue; }

            let dx = world.npcs[ni].x - world.vehicles[vi].x;
            let dz = world.npcs[ni].z - world.vehicles[vi].z;
            if dx.abs() > 10.0 || dz.abs() > 10.0 { continue; }

            let nobb = Obb2d::from_npc(&world.npcs[ni]);
            if let Some((nx, nz, _depth)) = obb_intersect(&vobb, &nobb) {
                let damage = vspeed * VEHICLE_HIT_DAMAGE_MULT;
                world.npcs[ni].health -= damage;

                // Mass-based launch
                let v_mass = world.vehicles[vi].body.mass;
                let n_mass = world.npcs[ni].body.mass;
                let inv_mass_sum = 1.0 / v_mass + 1.0 / n_mass;
                let j_mag = 1.5 * vspeed / inv_mass_sum;
                let vrot = world.vehicles[vi].rot_y;
                let fwd_x = -vrot.sin();
                let fwd_z = -vrot.cos();
                let launch_x = fwd_x * j_mag / n_mass + nx * 3.0;
                let launch_z = fwd_z * j_mag / n_mass + nz * 3.0;

                // Apply impulse always, ragdoll only above force threshold
                world.npcs[ni].body.apply_impulse([launch_x, VEHICLE_HIT_LAUNCH_UP, launch_z]);
                let speed_change = j_mag / n_mass;
                if speed_change > 3.0 {
                    init_ragdoll(&mut world.npcs[ni], launch_x, VEHICLE_HIT_LAUNCH_UP, launch_z);
                }

                world.vehicles[vi].body.apply_impulse([-fwd_x * j_mag, 0.0, -fwd_z * j_mag]);
                let fwd_v = crate::math::quat_forward(world.vehicles[vi].body.quat);
                world.vehicles[vi].speed = crate::math::v3_dot(world.vehicles[vi].body.vel, fwd_v);

                if let Some(owner) = world.vehicles[vi].owner_npc {
                    if owner < nn {
                        world.npcs[owner].wanted = true;
                        world.npcs[owner].bounty += damage * 0.5;
                    }
                }

                if world.npcs[ni].health <= 0.0 {
                    crate::combat::knockout_npc(&mut world.npcs[ni]);
                }
            }
        }
    }

    // Vehicle → Wall (headless)
    for vi in 0..nv {
        if world.vehicles[vi].speed.abs() < 0.1 { continue; }
        let vobb = Obb2d::from_vehicle(&world.vehicles[vi]);
        for w in &world.walls {
            let dx = world.vehicles[vi].x - w.x;
            let dz = world.vehicles[vi].z - w.z;
            if dx.abs() > w.hw + 3.0 || dz.abs() > w.hd + 3.0 { continue; }
            let overlap_x = (w.hw + vobb.half_w) - dx.abs();
            let overlap_z = (w.hd + vobb.half_d) - dz.abs();
            if overlap_x > 0.0 && overlap_z > 0.0 {
                if overlap_x < overlap_z {
                    let sign = if dx > 0.0 { 1.0 } else { -1.0 };
                    world.vehicles[vi].body.pos[0] += sign * overlap_x;
                } else {
                    let sign = if dz > 0.0 { 1.0 } else { -1.0 };
                    world.vehicles[vi].body.pos[2] += sign * overlap_z;
                }
                let restitution = 0.2;
                if overlap_x < overlap_z {
                    world.vehicles[vi].body.vel[0] *= -restitution;
                } else {
                    world.vehicles[vi].body.vel[2] *= -restitution;
                }
                world.vehicles[vi].x = world.vehicles[vi].body.pos[0];
                world.vehicles[vi].z = world.vehicles[vi].body.pos[2];
                let fwd = crate::math::quat_forward(world.vehicles[vi].body.quat);
                world.vehicles[vi].speed = crate::math::v3_dot(world.vehicles[vi].body.vel, fwd);
            }
        }
        for sl in &world.street_lights {
            let dx = world.vehicles[vi].x - sl.x;
            let dz = world.vehicles[vi].z - sl.z;
            let d2 = dx * dx + dz * dz;
            let min_r = 0.3 + vobb.half_w.max(vobb.half_d);
            if d2 < min_r * min_r && d2 > 0.001 {
                let d = d2.sqrt();
                let push = min_r - d;
                let norm_x = dx / d;
                let norm_z = dz / d;
                world.vehicles[vi].body.pos[0] += norm_x * push;
                world.vehicles[vi].body.pos[2] += norm_z * push;
                // Reflect velocity component along collision normal
                let vn = world.vehicles[vi].body.vel[0] * norm_x + world.vehicles[vi].body.vel[2] * norm_z;
                if vn < 0.0 {
                    world.vehicles[vi].body.vel[0] -= norm_x * vn * 1.3;
                    world.vehicles[vi].body.vel[2] -= norm_z * vn * 1.3;
                }
                world.vehicles[vi].x = world.vehicles[vi].body.pos[0];
                world.vehicles[vi].z = world.vehicles[vi].body.pos[2];
                let fwd = crate::math::quat_forward(world.vehicles[vi].body.quat);
                world.vehicles[vi].speed = crate::math::v3_dot(world.vehicles[vi].body.vel, fwd);
            }
        }
    }

    // Vehicle → Vehicle
    for i in 0..nv {
        if world.vehicles[i].speed.abs() < 0.1 { continue; }
        for j in (i + 1)..nv {
            let dx = world.vehicles[j].x - world.vehicles[i].x;
            let dz = world.vehicles[j].z - world.vehicles[i].z;
            if dx.abs() > 10.0 || dz.abs() > 10.0 { continue; }

            let obb_a = Obb2d::from_vehicle(&world.vehicles[i]);
            let obb_b = Obb2d::from_vehicle(&world.vehicles[j]);
            if let Some((nx, nz, depth)) = obb_intersect(&obb_a, &obb_b) {
                // Mass-based collision response
                let mass_i = world.vehicles[i].body.mass;
                let mass_j = world.vehicles[j].body.mass;
                let inv_mass_sum = 1.0 / mass_i + 1.0 / mass_j;

                let push = depth + 0.05;
                let wi = (1.0 / mass_i) / inv_mass_sum;
                let wj = (1.0 / mass_j) / inv_mass_sum;
                world.vehicles[i].body.pos[0] -= nx * push * wi;
                world.vehicles[i].body.pos[2] -= nz * push * wi;
                world.vehicles[j].body.pos[0] += nx * push * wj;
                world.vehicles[j].body.pos[2] += nz * push * wj;

                let rel_vn = (world.vehicles[i].body.vel[0] - world.vehicles[j].body.vel[0]) * nx
                           + (world.vehicles[i].body.vel[2] - world.vehicles[j].body.vel[2]) * nz;
                let relative_speed = rel_vn.abs();
                if rel_vn > 0.0 {
                    let restitution = 0.3;
                    let j_mag = (1.0 + restitution) * rel_vn / inv_mass_sum;
                    world.vehicles[i].body.apply_impulse([-nx * j_mag, 0.0, -nz * j_mag]);
                    world.vehicles[j].body.apply_impulse([nx * j_mag, 0.0, nz * j_mag]);
                }
                world.vehicles[i].x = world.vehicles[i].body.pos[0];
                world.vehicles[i].z = world.vehicles[i].body.pos[2];
                let fwd_i = crate::math::quat_forward(world.vehicles[i].body.quat);
                world.vehicles[i].speed = crate::math::v3_dot(world.vehicles[i].body.vel, fwd_i);
                world.vehicles[j].x = world.vehicles[j].body.pos[0];
                world.vehicles[j].z = world.vehicles[j].body.pos[2];
                let fwd_j = crate::math::quat_forward(world.vehicles[j].body.quat);
                world.vehicles[j].speed = crate::math::v3_dot(world.vehicles[j].body.vel, fwd_j);

                // Panel deformation
                if relative_speed > 2.0 {
                    let mass_i = world.vehicles[i].body.mass;
                    let mass_j = world.vehicles[j].body.mass;
                    let energy = 0.5 * (mass_i * mass_j / (mass_i + mass_j)) * relative_speed * relative_speed;
                    let impact_i = [nx * obb_a.half_d, 0.0, nz * obb_a.half_d];
                    let impact_j = [-nx * obb_b.half_d, 0.0, -nz * obb_b.half_d];
                    world.vehicles[i].deformation.apply_impact(impact_i, energy, obb_a.half_w, obb_a.half_d);
                    world.vehicles[j].deformation.apply_impact(impact_j, energy, obb_b.half_w, obb_b.half_d);
                }

                let occupant_damage = relative_speed * VEHICLE_CRASH_SELF_DAMAGE;
                if let Some(owner) = world.vehicles[i].owner_npc {
                    if owner < nn && world.npcs[owner].in_vehicle {
                        world.npcs[owner].health -= occupant_damage;
                    }
                }
                if let Some(owner) = world.vehicles[j].owner_npc {
                    if owner < nn && world.npcs[owner].in_vehicle {
                        world.npcs[owner].health -= occupant_damage;
                    }
                }
            }
        }
    }

    // NPC → NPC push-apart (avoid pushing into river)
    for i in 0..nn {
        if world.npcs[i].in_vehicle || world.npcs[i].state == NpcState::Sleeping { continue; }
        if world.npcs[i].ragdoll_active { continue; }
        for j in (i + 1)..nn {
            if world.npcs[j].in_vehicle || world.npcs[j].state == NpcState::Sleeping { continue; }
            if world.npcs[j].ragdoll_active { continue; }

            let dx = world.npcs[j].x - world.npcs[i].x;
            let dz = world.npcs[j].z - world.npcs[i].z;
            let d2 = dx * dx + dz * dz;
            if d2 < 0.36 && d2 > 0.001 {
                let d = d2.sqrt();
                push_apart_npcs_safe(world, i, j, d, dx, dz, 0.6);
            }
        }
    }

    decay_violation_timers(world, dt);
}
