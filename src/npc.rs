// sys_npc: NPC pedestrian AI - wander near roads, avoid obstacles

use crate::state::*;

pub fn sys_npc(world: &mut WorldData, dt: f32) {
    let n = world.npcs.len();
    for i in 0..n {
        let npc = &world.npcs[i];
        let dx = npc.target_x - npc.x;
        let dz = npc.target_z - npc.z;
        let dist = (dx * dx + dz * dz).sqrt();

        // Pick new target when close
        if dist < 2.0 {
            pick_npc_target(&mut world.npcs[i]);
            continue;
        }

        // Turn toward target
        let desired = (-dx).atan2(-dz);
        let mut diff = desired - world.npcs[i].rot_y;
        while diff > std::f32::consts::PI { diff -= 2.0 * std::f32::consts::PI; }
        while diff < -std::f32::consts::PI { diff += 2.0 * std::f32::consts::PI; }
        world.npcs[i].rot_y += diff.clamp(-3.0 * dt, 3.0 * dt);

        // Walk forward
        let rot = world.npcs[i].rot_y;
        let new_x = world.npcs[i].x - rot.sin() * NPC_SPEED * dt;
        let new_z = world.npcs[i].z - rot.cos() * NPC_SPEED * dt;

        // Collision check against buildings/rocks (borrow buildings/rocks separately)
        let collides_building = world.buildings.iter().any(|b| {
            new_x + 0.4 > b.x - b.w * 0.5 && new_x - 0.4 < b.x + b.w * 0.5
            && new_z + 0.4 > b.z - b.d * 0.5 && new_z - 0.4 < b.z + b.d * 0.5
        });
        let collides_rock = world.rocks.iter().any(|r| {
            let rdx = new_x - r.x;
            let rdz = new_z - r.z;
            rdx * rdx + rdz * rdz < (r.size + 0.4) * (r.size + 0.4)
        });

        if !collides_building && !collides_rock {
            world.npcs[i].x = new_x;
            world.npcs[i].z = new_z;
            world.npcs[i].walk_phase += dt * NPC_SPEED * 2.5;
        } else {
            pick_npc_target(&mut world.npcs[i]);
        }

        world.npcs[i].x = world.npcs[i].x.clamp(-WORLD_HALF, WORLD_HALF);
        world.npcs[i].z = world.npcs[i].z.clamp(-WORLD_HALF, WORLD_HALF);
    }
}

fn pick_npc_target(npc: &mut Npc) {
    // Wander near current position, biased toward roads
    let hash = ((npc.x * 17.3 + npc.z * 43.7 + npc.walk_phase * 7.1) * 1000.0) as i32;

    // Find nearest road and walk along it
    let mut best_road = 0.0f32;
    let mut best_dist = f32::MAX;
    let mut is_x_road = true;
    for &r in &ROAD_POSITIONS {
        let dz = (npc.z - r).abs();
        if dz < best_dist { best_dist = dz; best_road = r; is_x_road = true; }
        let dx = (npc.x - r).abs();
        if dx < best_dist { best_dist = dx; best_road = r; is_x_road = false; }
    }

    // Walk along sidewalk (offset from road center)
    let sidewalk_offset = ROAD_WIDTH * 0.5 + 1.5;
    let side = if hash % 2 == 0 { sidewalk_offset } else { -sidewalk_offset };
    let along = npc.x + ((hash % 40) as f32 - 20.0);
    let along_z = npc.z + (((hash / 40) % 40) as f32 - 20.0);

    if is_x_road {
        npc.target_x = along.clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
        npc.target_z = best_road + side;
    } else {
        npc.target_x = best_road + side;
        npc.target_z = along_z.clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
    }
}

pub struct PickupEvent {
    pub x: f32, pub z: f32, pub color: u32,
}

pub fn sys_items(world: &mut WorldData, player: &mut Player, dt: f32) -> Vec<PickupEvent> {
    let mut pickups = Vec::new();
    for item in &mut world.items {
        if item.active {
            item.spin_phase += dt * 3.0;
            let dx = player.x - item.x;
            let dz = player.z - item.z;
            if dx * dx + dz * dz < ITEM_PICKUP_DIST * ITEM_PICKUP_DIST {
                let color = match item.kind {
                    ItemKind::Health => { player.health = (player.health + 25.0).min(100.0); 0xFFFF3333 }
                    ItemKind::Money => { player.money += 20.0; 0xFFFFDD33 }
                    ItemKind::Stamina => { player.stamina = (player.stamina + 50.0).min(100.0); 0xFF33FF33 }
                };
                player.score += 10;
                item.active = false;
                item.respawn_timer = ITEM_RESPAWN_TIME;
                pickups.push(PickupEvent { x: item.x, z: item.z, color });
            }
        } else {
            item.respawn_timer -= dt;
            if item.respawn_timer <= 0.0 {
                item.active = true;
            }
        }
    }
    pickups
}
