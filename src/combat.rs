// Combat system: melee attacks for player and NPCs, knockback, knockout/recovery

use crate::state::*;
use crate::input::{Action, KeyBinds};
use crate::particle::ParticleSystem;

pub fn sys_combat(
    world: &mut WorldData, player: &mut Player, particles: &mut ParticleSystem,
    terrain: &Terrain, keys: &[bool; 256], prev_keys: &[bool; 256],
    keybinds: &KeyBinds, dt: f32,
) {
    // Tick cooldowns
    player.attack_cooldown = (player.attack_cooldown - dt).max(0.0);
    player.attack_phase = (player.attack_phase - dt).max(0.0);
    player.hit_flash = (player.hit_flash - dt).max(0.0);
    player.damage_shake = (player.damage_shake - dt * CAMERA_SHAKE_DECAY).max(0.0);

    for npc in &mut world.npcs {
        npc.attack_cooldown = (npc.attack_cooldown - dt).max(0.0);
        npc.attack_phase = (npc.attack_phase - dt).max(0.0);
        npc.hit_flash = (npc.hit_flash - dt).max(0.0);
    }

    // Player attack (edge-triggered)
    let attack_now = keybinds.is_pressed(Action::Attack, keys);
    let attack_prev = keybinds.is_pressed(Action::Attack, prev_keys);
    if attack_now && !attack_prev && player.attack_cooldown <= 0.0 && player.in_vehicle.is_none() {
        player.attack_cooldown = ATTACK_COOLDOWN;
        player.attack_phase = ATTACK_ANIM_DURATION;
        player_attack_npcs(world, player, particles);
    }

    // NPC attack intents (set by NEAT, processed here)
    let n = world.npcs.len();
    for i in 0..n {
        let intent = world.npcs[i].attack_intent;
        world.npcs[i].attack_intent = 0;

        if intent == 0 { continue; }
        if world.npcs[i].attack_cooldown > 0.0 { continue; }
        if world.npcs[i].state == NpcState::KnockedOut { continue; }
        if world.npcs[i].state == NpcState::Sleeping { continue; }

        if intent == 1 {
            // Attack player
            let npc = &world.npcs[i];
            if npc_attack_player(npc.x, npc.z, npc.rot_y, player, particles) {
                world.npcs[i].attack_cooldown = ATTACK_COOLDOWN;
                world.npcs[i].attack_phase = ATTACK_ANIM_DURATION;
                world.npcs[i].fitness_hits_landed += 1;
            }
        } else if intent == 2 {
            // Attack nearest NPC
            let ax = world.npcs[i].x;
            let az = world.npcs[i].z;
            let arot = world.npcs[i].rot_y;
            let mut best_dist = ATTACK_RANGE * ATTACK_RANGE;
            let mut best_j = None;
            for j in 0..n {
                if j == i { continue; }
                if world.npcs[j].state == NpcState::KnockedOut { continue; }
                let dx = world.npcs[j].x - ax;
                let dz = world.npcs[j].z - az;
                let d2 = dx * dx + dz * dz;
                if d2 < best_dist {
                    best_dist = d2;
                    best_j = Some(j);
                }
            }
            if let Some(j) = best_j {
                if npc_attack_npc_check(ax, az, arot, world.npcs[j].x, world.npcs[j].z) {
                    world.npcs[i].attack_cooldown = ATTACK_COOLDOWN;
                    world.npcs[i].attack_phase = ATTACK_ANIM_DURATION;
                    world.npcs[i].fitness_hits_landed += 1;

                    let tx = world.npcs[j].x;
                    let tz = world.npcs[j].z;
                    let ty = world.npcs[j].y;
                    emit_hit_particles(particles, tx, ty + 1.0, tz);

                    world.npcs[j].health -= NPC_ATTACK_DAMAGE;
                    world.npcs[j].hit_flash = HIT_FLASH_DURATION;

                    // Knockback (70% force for NPC-vs-NPC)
                    let dx = tx - ax;
                    let dz = tz - az;
                    let dist = (dx * dx + dz * dz).sqrt().max(0.01);
                    world.npcs[j].knockback_vx += dx / dist * KNOCKBACK_FORCE * 0.7;
                    world.npcs[j].knockback_vz += dz / dist * KNOCKBACK_FORCE * 0.7;
                    world.npcs[j].vel_y = KNOCKBACK_UP * 0.7;

                    // KO check
                    if world.npcs[j].health <= 0.0 {
                        world.npcs[j].health = 0.0;
                        world.npcs[j].state = NpcState::KnockedOut;
                        world.npcs[j].knockout_timer = KNOCKOUT_TIME;
                        world.npcs[j].carrying_item = false;
                        world.npcs[j].carrying_bin = None;
                        world.npcs[j].fitness_knockouts += 1;
                        world.npcs[j].sound = [0.0; 3];
                    }
                }
            }
        }
    }

    // Apply knockback friction + knockout recovery + passive health regen
    for i in 0..n {
        let npc = &mut world.npcs[i];

        // Knockback movement
        if npc.knockback_vx.abs() > 0.01 || npc.knockback_vz.abs() > 0.01 {
            npc.x += npc.knockback_vx * dt;
            npc.z += npc.knockback_vz * dt;
            npc.x = npc.x.clamp(-WORLD_HALF, WORLD_HALF);
            npc.z = npc.z.clamp(-WORLD_HALF, WORLD_HALF);
            // Friction
            let friction = (-KNOCKBACK_FRICTION * dt).exp();
            npc.knockback_vx *= friction;
            npc.knockback_vz *= friction;
        }

        // Knockout recovery (skip for starving_dead — they stay down until midnight)
        if npc.state == NpcState::KnockedOut && !npc.starving_dead {
            npc.knockout_timer -= dt;
            if npc.knockout_timer <= 0.0 {
                npc.health = KNOCKOUT_REGEN_HP;
                npc.state = NpcState::Working;
                npc.knockout_timer = 0.0;
                npc.state_timer = 0.0;
            }
        }

        // Passive health regen (not while KO'd)
        if npc.state != NpcState::KnockedOut && npc.health < NPC_HEALTH_MAX {
            npc.health = (npc.health + HEALTH_REGEN_RATE * dt).min(NPC_HEALTH_MAX);
        }

        // Snap to terrain after knockback
        let ground = terrain.height_at(npc.x, npc.z);
        if npc.y < ground {
            npc.y = ground;
        }
    }
}

fn player_attack_npcs(world: &mut WorldData, player: &mut Player, particles: &mut ParticleSystem) {
    let (sin_r, cos_r) = player.rot_y.sin_cos();
    let fwd_x = -sin_r;
    let fwd_z = -cos_r;

    for npc in &mut world.npcs {
        if npc.state == NpcState::KnockedOut || npc.state == NpcState::Sleeping { continue; }

        let dx = npc.x - player.x;
        let dz = npc.z - player.z;
        let dist_sq = dx * dx + dz * dz;
        if dist_sq > ATTACK_RANGE * ATTACK_RANGE { continue; }

        let dist = dist_sq.sqrt();
        if dist < 0.01 { continue; }

        // Facing cone check
        let dot = (dx / dist) * fwd_x + (dz / dist) * fwd_z;
        if dot < ATTACK_CONE_COS { continue; }

        // Hit!
        npc.health -= ATTACK_DAMAGE;
        npc.hit_flash = HIT_FLASH_DURATION;

        // Knockback
        npc.knockback_vx += dx / dist * KNOCKBACK_FORCE;
        npc.knockback_vz += dz / dist * KNOCKBACK_FORCE;
        npc.vel_y = KNOCKBACK_UP;

        emit_hit_particles(particles, npc.x, npc.y + 1.0, npc.z);

        // KO check
        if npc.health <= 0.0 {
            npc.health = 0.0;
            npc.state = NpcState::KnockedOut;
            npc.knockout_timer = KNOCKOUT_TIME;
            npc.carrying_item = false;
            npc.carrying_bin = None;
            npc.fitness_knockouts += 1;
            npc.sound = [0.0; 3];
        }
    }
}

fn npc_attack_player(nx: f32, nz: f32, nrot: f32, player: &mut Player, particles: &mut ParticleSystem) -> bool {
    let dx = player.x - nx;
    let dz = player.z - nz;
    let dist_sq = dx * dx + dz * dz;
    if dist_sq > ATTACK_RANGE * ATTACK_RANGE { return false; }

    let dist = dist_sq.sqrt();
    if dist < 0.01 { return false; }

    // Facing cone
    let (sin_r, cos_r) = nrot.sin_cos();
    let fwd_x = -sin_r;
    let fwd_z = -cos_r;
    let dot = (dx / dist) * fwd_x + (dz / dist) * fwd_z;
    if dot < ATTACK_CONE_COS { return false; }

    player.health = (player.health - NPC_ATTACK_DAMAGE).max(0.0);
    player.hit_flash = HIT_FLASH_DURATION;
    player.damage_shake = CAMERA_SHAKE_INTENSITY;

    emit_hit_particles(particles, player.x, player.y + 1.0, player.z);
    true
}

fn npc_attack_npc_check(ax: f32, az: f32, arot: f32, tx: f32, tz: f32) -> bool {
    let dx = tx - ax;
    let dz = tz - az;
    let dist_sq = dx * dx + dz * dz;
    if dist_sq > ATTACK_RANGE * ATTACK_RANGE { return false; }

    let dist = dist_sq.sqrt();
    if dist < 0.01 { return false; }

    let (sin_r, cos_r) = arot.sin_cos();
    let fwd_x = -sin_r;
    let fwd_z = -cos_r;
    let dot = (dx / dist) * fwd_x + (dz / dist) * fwd_z;
    dot >= ATTACK_CONE_COS
}

/// Verlet ragdoll physics update — called each frame after sys_collisions
pub fn sys_ragdoll_update(world: &mut WorldData, terrain: &Terrain, dt: f32) {
    let gravity_dt2 = GRAVITY * dt * dt;

    for npc in &mut world.npcs {
        if !npc.ragdoll_active { continue; }

        // Verlet integration for each point
        for i in 0..RAGDOLL_POINT_COUNT {
            let cur = npc.ragdoll_points[i];
            let prev = npc.ragdoll_prev[i];

            let new_x = cur[0] + (cur[0] - prev[0]) * 0.98; // 0.98 = damping
            let new_y = cur[1] + (cur[1] - prev[1]) * 0.98 - gravity_dt2;
            let new_z = cur[2] + (cur[2] - prev[2]) * 0.98;

            npc.ragdoll_prev[i] = cur;
            npc.ragdoll_points[i] = [new_x, new_y, new_z];
        }

        // Ground collision per point
        for i in 0..RAGDOLL_POINT_COUNT {
            let ground = terrain.height_at(npc.ragdoll_points[i][0], npc.ragdoll_points[i][2]);
            if npc.ragdoll_points[i][1] < ground {
                npc.ragdoll_points[i][1] = ground;
                // Friction: reduce horizontal velocity
                let dx = npc.ragdoll_points[i][0] - npc.ragdoll_prev[i][0];
                let dz = npc.ragdoll_points[i][2] - npc.ragdoll_prev[i][2];
                npc.ragdoll_prev[i][0] = npc.ragdoll_points[i][0] - dx * 0.3;
                npc.ragdoll_prev[i][2] = npc.ragdoll_points[i][2] - dz * 0.3;
                // Bounce: reverse vertical
                npc.ragdoll_prev[i][1] = npc.ragdoll_points[i][1] + (npc.ragdoll_points[i][1] - npc.ragdoll_prev[i][1]) * 0.2;
            }
        }

        // Distance constraint solving: 4 iterations
        for _ in 0..4 {
            for &(a, b, rest) in &RAGDOLL_CONSTRAINTS {
                let ax = npc.ragdoll_points[a][0];
                let ay = npc.ragdoll_points[a][1];
                let az = npc.ragdoll_points[a][2];
                let bx = npc.ragdoll_points[b][0];
                let by = npc.ragdoll_points[b][1];
                let bz = npc.ragdoll_points[b][2];

                let dx = bx - ax;
                let dy = by - ay;
                let dz = bz - az;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);
                let diff = (dist - rest) / dist * 0.5;

                npc.ragdoll_points[a][0] += dx * diff;
                npc.ragdoll_points[a][1] += dy * diff;
                npc.ragdoll_points[a][2] += dz * diff;
                npc.ragdoll_points[b][0] -= dx * diff;
                npc.ragdoll_points[b][1] -= dy * diff;
                npc.ragdoll_points[b][2] -= dz * diff;
            }
        }

        // World bounds
        for i in 0..RAGDOLL_POINT_COUNT {
            npc.ragdoll_points[i][0] = npc.ragdoll_points[i][0].clamp(-WORLD_HALF, WORLD_HALF);
            npc.ragdoll_points[i][2] = npc.ragdoll_points[i][2].clamp(-WORLD_HALF, WORLD_HALF);
        }

        // Timer
        npc.ragdoll_timer -= dt;
        if npc.ragdoll_timer <= 0.0 {
            // Snap NPC to hips position
            npc.x = npc.ragdoll_points[0][0];
            npc.y = terrain.height_at(npc.ragdoll_points[0][0], npc.ragdoll_points[0][2]);
            npc.z = npc.ragdoll_points[0][2];
            npc.ragdoll_active = false;
            npc.knockback_vx = 0.0;
            npc.knockback_vz = 0.0;

            // Enter KnockedOut if not already
            if npc.state != NpcState::KnockedOut {
                npc.state = NpcState::KnockedOut;
                npc.knockout_timer = KNOCKOUT_TIME;
                npc.carrying_item = false;
                npc.carrying_bin = None;
            }
        }

        // Keep NPC position synced to hips during ragdoll
        npc.x = npc.ragdoll_points[0][0];
        npc.y = npc.ragdoll_points[0][1];
        npc.z = npc.ragdoll_points[0][2];
    }
}

fn emit_hit_particles(ps: &mut ParticleSystem, x: f32, y: f32, z: f32) {
    for i in 0..6 {
        let angle = i as f32 * std::f32::consts::TAU / 6.0;
        let speed = 2.5;
        let vx = angle.cos() * speed;
        let vz = angle.sin() * speed;
        ps.emit(x, y, z, vx, 2.0, vz, 0.4, 0xFFFF3333);
    }
}
