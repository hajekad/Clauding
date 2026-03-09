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
        Self::new(v.x, v.z, 1.8, 3.6, v.rot_y)
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

/// Initialize ragdoll points from NPC's current position
fn init_ragdoll(npc: &mut Npc, impulse_x: f32, impulse_y: f32, impulse_z: f32) {
    let x = npc.x;
    let y = npc.y;
    let z = npc.z;

    // hips=0, chest=1, head=2, l_hand=3, r_hand=4, l_foot=5, r_foot=6
    npc.ragdoll_points = [
        [x, y + 0.7, z],           // hips
        [x, y + 1.4, z],           // chest
        [x, y + 1.85, z],          // head
        [x - 0.45, y + 1.1, z],    // l_hand
        [x + 0.45, y + 1.1, z],    // r_hand
        [x - 0.15, y + 0.0, z],    // l_foot
        [x + 0.15, y + 0.0, z],    // r_foot
    ];

    // Set previous positions offset by impulse (Verlet: velocity = pos - prev)
    for i in 0..RAGDOLL_POINT_COUNT {
        npc.ragdoll_prev[i] = [
            npc.ragdoll_points[i][0] - impulse_x * 0.016,
            npc.ragdoll_points[i][1] - impulse_y * 0.016,
            npc.ragdoll_points[i][2] - impulse_z * 0.016,
        ];
    }

    npc.ragdoll_active = true;
    npc.ragdoll_timer = RAGDOLL_DURATION;
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

                // Launch direction: vehicle's forward + up
                let vrot = world.vehicles[vi].rot_y;
                let fwd_x = -vrot.sin();
                let fwd_z = -vrot.cos();
                let launch_x = fwd_x * vspeed * 0.5 + nx * 3.0;
                let launch_z = fwd_z * vspeed * 0.5 + nz * 3.0;

                init_ragdoll(&mut world.npcs[ni], launch_x, VEHICLE_HIT_LAUNCH_UP, launch_z);

                // Slow vehicle on impact
                world.vehicles[vi].speed *= 0.7;

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
                    world.npcs[ni].health = 0.0;
                    world.npcs[ni].state = NpcState::KnockedOut;
                    world.npcs[ni].knockout_timer = KNOCKOUT_TIME;
                    world.npcs[ni].carrying_item = false;
                    world.npcs[ni].carrying_bin = None;
                    world.npcs[ni].fitness_knockouts += 1;
                    world.npcs[ni].sound = [0.0; 3];
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

                // Push player out
                player.x += nx * (depth + 0.5);
                player.z += nz * (depth + 0.5);

                // Slow vehicle
                world.vehicles[vi].speed *= 0.7;

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
                    world.vehicles[vi].x += sign * overlap_x;
                } else {
                    let sign = if dz > 0.0 { 1.0 } else { -1.0 };
                    world.vehicles[vi].z += sign * overlap_z;
                }
                world.vehicles[vi].speed *= 0.3;
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
                world.vehicles[vi].x += (dx / d) * push;
                world.vehicles[vi].z += (dz / d) * push;
                world.vehicles[vi].speed *= 0.5;
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
                let relative_speed = (world.vehicles[i].speed - world.vehicles[j].speed).abs();

                // Push apart
                let push = depth * 0.5 + 0.1;
                world.vehicles[i].x -= nx * push;
                world.vehicles[i].z -= nz * push;
                world.vehicles[j].x += nx * push;
                world.vehicles[j].z += nz * push;

                // Bounce speeds
                let avg = (world.vehicles[i].speed + world.vehicles[j].speed) * 0.5;
                world.vehicles[i].speed = avg - nx * relative_speed * 0.3;
                world.vehicles[j].speed = avg + nx * relative_speed * 0.3;

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
                let overlap = min_dist - d;
                let nx = dx / d;
                let nz = dz / d;
                let push = overlap * 0.5;
                let new_ix = world.npcs[i].x - nx * push;
                let new_iz = world.npcs[i].z - nz * push;
                let new_jx = world.npcs[j].x + nx * push;
                let new_jz = world.npcs[j].z + nz * push;
                if !crate::world::on_river_not_bridge(new_ix, new_iz, &world.river_segments, &world.bridges) {
                    world.npcs[i].x = new_ix;
                    world.npcs[i].z = new_iz;
                }
                if !crate::world::on_river_not_bridge(new_jx, new_jz, &world.river_segments, &world.bridges) {
                    world.npcs[j].x = new_jx;
                    world.npcs[j].z = new_jz;
                }
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
                // Push NPC more than player (70/30)
                world.npcs[ni].x += nx * overlap * 0.7;
                world.npcs[ni].z += nz * overlap * 0.7;
                player.x -= nx * overlap * 0.3;
                player.z -= nz * overlap * 0.3;
            }
        }
    }

    // Tick violation timers
    for npc in &mut world.npcs {
        if npc.violation_timer > 0.0 {
            npc.violation_timer -= dt;
        }
    }
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

                let vrot = world.vehicles[vi].rot_y;
                let fwd_x = -vrot.sin();
                let fwd_z = -vrot.cos();
                let launch_x = fwd_x * vspeed * 0.5 + nx * 3.0;
                let launch_z = fwd_z * vspeed * 0.5 + nz * 3.0;
                init_ragdoll(&mut world.npcs[ni], launch_x, VEHICLE_HIT_LAUNCH_UP, launch_z);

                world.vehicles[vi].speed *= 0.7;

                if let Some(owner) = world.vehicles[vi].owner_npc {
                    if owner < nn {
                        world.npcs[owner].wanted = true;
                        world.npcs[owner].bounty += damage * 0.5;
                    }
                }

                if world.npcs[ni].health <= 0.0 {
                    world.npcs[ni].health = 0.0;
                    world.npcs[ni].state = NpcState::KnockedOut;
                    world.npcs[ni].knockout_timer = KNOCKOUT_TIME;
                    world.npcs[ni].carrying_item = false;
                    world.npcs[ni].carrying_bin = None;
                    world.npcs[ni].fitness_knockouts += 1;
                    world.npcs[ni].sound = [0.0; 3];
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
                    world.vehicles[vi].x += sign * overlap_x;
                } else {
                    let sign = if dz > 0.0 { 1.0 } else { -1.0 };
                    world.vehicles[vi].z += sign * overlap_z;
                }
                world.vehicles[vi].speed *= 0.3;
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
                world.vehicles[vi].x += (dx / d) * push;
                world.vehicles[vi].z += (dz / d) * push;
                world.vehicles[vi].speed *= 0.5;
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
                let relative_speed = (world.vehicles[i].speed - world.vehicles[j].speed).abs();
                let push = depth * 0.5 + 0.1;
                world.vehicles[i].x -= nx * push;
                world.vehicles[i].z -= nz * push;
                world.vehicles[j].x += nx * push;
                world.vehicles[j].z += nz * push;

                let avg = (world.vehicles[i].speed + world.vehicles[j].speed) * 0.5;
                world.vehicles[i].speed = avg - nx * relative_speed * 0.3;
                world.vehicles[j].speed = avg + nx * relative_speed * 0.3;

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
                let overlap = 0.6 - d;
                let nx = dx / d;
                let nz = dz / d;
                let push = overlap * 0.5;
                let new_ix = world.npcs[i].x - nx * push;
                let new_iz = world.npcs[i].z - nz * push;
                let new_jx = world.npcs[j].x + nx * push;
                let new_jz = world.npcs[j].z + nz * push;
                if !crate::world::on_river_not_bridge(new_ix, new_iz, &world.river_segments, &world.bridges) {
                    world.npcs[i].x = new_ix;
                    world.npcs[i].z = new_iz;
                }
                if !crate::world::on_river_not_bridge(new_jx, new_jz, &world.river_segments, &world.bridges) {
                    world.npcs[j].x = new_jx;
                    world.npcs[j].z = new_jz;
                }
            }
        }
    }

    for npc in &mut world.npcs {
        if npc.violation_timer > 0.0 {
            npc.violation_timer -= dt;
        }
    }
}
