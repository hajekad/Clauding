// NPC job-specific AI behaviors
// Each job function is called during the Working state based on npc.job

use crate::state::*;
use crate::npc::npc_walk_toward;

/// Dispatch NPC work behavior by job type
pub fn npc_work_dispatch(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    match world.npcs[i].job {
        NpcJob::Collector => npc_work_collector(world, i, net, terrain, dt),
        NpcJob::GarbageCollector => npc_work_garbage(world, i, net, terrain, dt),
        NpcJob::TaxiDriver => npc_work_taxi(world, i, net, terrain, dt),
        NpcJob::DeliveryCourier => npc_work_delivery(world, i, net, terrain, dt),
        NpcJob::MailCarrier => npc_work_mail(world, i, net, terrain, dt),
        NpcJob::Paramedic => npc_work_paramedic(world, i, net, terrain, dt),
        NpcJob::Firefighter => npc_work_firefighter(world, i, net, terrain, dt),
        NpcJob::PolicePatrol => npc_work_police(world, i, net, terrain, dt),
        NpcJob::StreetVendor => npc_work_vendor(world, i, net, terrain, dt),
        NpcJob::Mechanic => npc_work_mechanic(world, i, net, terrain, dt),
        NpcJob::ConstructionWorker => npc_work_construction(world, i, net, terrain, dt),
        NpcJob::Fisherman => npc_work_fisherman(world, i, net, terrain, dt),
        NpcJob::Farmer => npc_work_farmer(world, i, net, terrain, dt),
        NpcJob::Lumberjack => npc_work_lumberjack(world, i, net, terrain, dt),
        NpcJob::Scavenger => npc_work_scavenger(world, i, net, terrain, dt),
    }
}

// ---- Collector (existing behavior, extracted from npc.rs) ----

fn npc_work_collector(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    let npc = &world.npcs[i];

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
                world.npcs[i].carrying_item = false;
                world.npcs[i].target_bin = None;
                world.npcs[i].items_deposited_today += 1;
                world.npcs[i].money += 1.0;
                world.trash_bins[bi].items_held += 1;
            }
        } else {
            pick_wander(world, i);
            let tx = world.npcs[i].target_x;
            let tz = world.npcs[i].target_z;
            npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        }
    } else if world.npcs[i].carrying_bin.is_some() {
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        if remaining < 2.0 {
            if let Some(bi) = world.npcs[i].carrying_bin {
                world.trash_bins[bi].x = world.npcs[i].x;
                world.trash_bins[bi].z = world.npcs[i].z;
                world.trash_bins[bi].y = terrain.height_at(world.npcs[i].x, world.npcs[i].z);
                world.trash_bins[bi].carried_by = None;
                world.npcs[i].carrying_bin = None;
            }
        }
    } else {
        // Find item to pick up
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
                    if bin_dist > 20.0 && world.trash_bins[bi].carried_by.is_none() {
                        world.npcs[i].carrying_bin = Some(bi);
                        world.trash_bins[bi].carried_by = Some(i);
                        world.npcs[i].target_x = world.items[idx].x;
                        world.npcs[i].target_z = world.items[idx].z;
                        world.items[idx].claimed_by = None;
                        world.npcs[i].target_item = None;
                        world.npcs[i].target_x = world.trash_bins[bi].x;
                        world.npcs[i].target_z = world.trash_bins[bi].z;
                        return;
                    }
                }
            } else {
                pick_wander(world, i);
            }
        }

        if let Some(item_idx) = world.npcs[i].target_item {
            if item_idx < world.items.len() && world.items[item_idx].active && !world.items[item_idx].falling {
                let ix = world.items[item_idx].x;
                let iz = world.items[item_idx].z;
                let remaining = npc_walk_toward(world, i, ix, iz, net, terrain, dt);
                if remaining < NPC_PICKUP_DIST {
                    world.items[item_idx].active = false;
                    world.items[item_idx].claimed_by = None;
                    world.npcs[i].carrying_item = true;
                    world.npcs[i].target_item = None;
                }
            } else {
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

// ---- Service Jobs ----

fn npc_work_garbage(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Similar to collector but targets dumpster interactibles instead of bins
    let npc = &world.npcs[i];

    if npc.carrying_item {
        // Find nearest dumpster
        if world.npcs[i].interaction_target.is_none() {
            let best = find_nearest_interactible(world, npc.x, npc.z, InteractibleKind::Dumpster);
            world.npcs[i].interaction_target = best;
        }
        if let Some(di) = world.npcs[i].interaction_target {
            let dx = world.interactibles[di].x;
            let dz = world.interactibles[di].z;
            let remaining = npc_walk_toward(world, i, dx, dz, net, terrain, dt);
            if remaining < NPC_BIN_DIST {
                world.npcs[i].carrying_item = false;
                world.npcs[i].interaction_target = None;
                world.npcs[i].items_deposited_today += 1;
                world.npcs[i].money += 1.5;
            }
        }
    } else {
        // Find items near roads
        if world.npcs[i].target_item.is_none() {
            let item_idx = find_best_item(world, i);
            if let Some(idx) = item_idx {
                world.npcs[i].target_item = Some(idx);
                world.items[idx].claimed_by = Some(i);
                world.npcs[i].target_x = world.items[idx].x;
                world.npcs[i].target_z = world.items[idx].z;
            } else {
                pick_wander(world, i);
            }
        }

        if let Some(item_idx) = world.npcs[i].target_item {
            if item_idx < world.items.len() && world.items[item_idx].active && !world.items[item_idx].falling {
                let ix = world.items[item_idx].x;
                let iz = world.items[item_idx].z;
                let remaining = npc_walk_toward(world, i, ix, iz, net, terrain, dt);
                if remaining < NPC_PICKUP_DIST {
                    world.items[item_idx].active = false;
                    world.items[item_idx].claimed_by = None;
                    world.npcs[i].carrying_item = true;
                    world.npcs[i].target_item = None;
                }
            } else {
                if item_idx < world.items.len() { world.items[item_idx].claimed_by = None; }
                world.npcs[i].target_item = None;
            }
        } else {
            let tx = world.npcs[i].target_x;
            let tz = world.npcs[i].target_z;
            npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        }
    }
}

fn npc_work_taxi(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Drive around road nodes
    world.npcs[i].job_timer += dt;
    let car_idx = world.npcs[i].car_idx;
    if car_idx < world.vehicles.len() && !world.vehicles[car_idx].occupied {
        // If not in vehicle, try to enter
        if !world.npcs[i].in_vehicle {
            let cdx = world.vehicles[car_idx].x - world.npcs[i].x;
            let cdz = world.vehicles[car_idx].z - world.npcs[i].z;
            if cdx * cdx + cdz * cdz < 25.0 {
                world.npcs[i].in_vehicle = true;
                world.npcs[i].state = NpcState::Driving;
                world.vehicles[car_idx].ai_active = true;
                // Pick random road node as destination
                if !net.nodes.is_empty() {
                    let ni = world.npcs[i].rng.next() as usize % net.nodes.len();
                    world.npcs[i].target_x = net.nodes[ni][0];
                    world.npcs[i].target_z = net.nodes[ni][1];
                }
            } else {
                npc_walk_toward(world, i, world.vehicles[car_idx].x, world.vehicles[car_idx].z, net, terrain, dt);
            }
        }
    } else {
        // Wander on foot
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        if remaining < 3.0 { pick_wander(world, i); }
    }
}

fn npc_work_delivery(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    let npc = &world.npcs[i];

    if npc.carrying_item {
        // Walk to target building
        let tx = world.npcs[i].job_target_x;
        let tz = world.npcs[i].job_target_z;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        if remaining < 3.0 {
            world.npcs[i].carrying_item = false;
            world.npcs[i].items_deposited_today += 1;
            world.npcs[i].money += 2.0;
            // Pick new source
            pick_random_building_target(world, i);
        }
    } else {
        // Walk to source building to pick up
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        if remaining < 3.0 {
            world.npcs[i].carrying_item = true;
            // Pick delivery destination
            let dest = world.npcs[i].rng.next() as usize % world.buildings.len();
            world.npcs[i].job_target_x = world.buildings[dest].x;
            world.npcs[i].job_target_z = world.buildings[dest].z;
        }
    }
}

fn npc_work_mail(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Visit mailbox interactibles in sequence
    world.npcs[i].job_timer += dt;

    if world.npcs[i].interaction_target.is_none() {
        // Find nearest unvisited mailbox
        let best = find_nearest_interactible(world, world.npcs[i].x, world.npcs[i].z, InteractibleKind::Mailbox);
        world.npcs[i].interaction_target = best;
    }

    if let Some(mi) = world.npcs[i].interaction_target {
        let mx = world.interactibles[mi].x;
        let mz = world.interactibles[mi].z;
        let remaining = npc_walk_toward(world, i, mx, mz, net, terrain, dt);
        if remaining < 2.0 {
            // "Deliver" to this mailbox
            world.npcs[i].job_timer = 0.0;
            world.npcs[i].money += 1.0;
            world.npcs[i].items_deposited_today += 1;
            world.npcs[i].interaction_target = None;
        }
    } else {
        pick_wander(world, i);
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    }
}

// ---- Emergency Jobs ----

fn npc_work_paramedic(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Find NPCs that are stuck (stuck_timer > 5)
    world.npcs[i].job_timer += dt;

    let mut target_x = world.npcs[i].target_x;
    let mut target_z = world.npcs[i].target_z;
    let mut found_patient = false;

    if world.npcs[i].job_timer > 3.0 {
        // Scan for stuck NPCs
        let n = world.npcs.len();
        let mut best_dist = f32::MAX;
        for j in 0..n {
            if j == i { continue; }
            if world.npcs[j].stuck_timer < 5.0 { continue; }
            if world.npcs[j].state == NpcState::Sleeping { continue; }
            let dx = world.npcs[j].x - world.npcs[i].x;
            let dz = world.npcs[j].z - world.npcs[i].z;
            let d = dx * dx + dz * dz;
            if d < best_dist {
                best_dist = d;
                target_x = world.npcs[j].x;
                target_z = world.npcs[j].z;
                found_patient = true;
            }
        }
        world.npcs[i].job_timer = 0.0;
    }

    if found_patient {
        world.npcs[i].target_x = target_x;
        world.npcs[i].target_z = target_z;
    }

    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    if remaining < 2.0 {
        // "Treat" nearby stuck NPCs
        let n = world.npcs.len();
        for j in 0..n {
            if j == i { continue; }
            let dx = world.npcs[j].x - world.npcs[i].x;
            let dz = world.npcs[j].z - world.npcs[i].z;
            if dx * dx + dz * dz < 9.0 && world.npcs[j].stuck_timer > 3.0 {
                world.npcs[j].stuck_timer = 0.0;
                world.npcs[j].detouring = false;
                world.npcs[i].money += 2.0;
            }
        }
        pick_wander(world, i);
    }
}

fn npc_work_firefighter(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Patrol between fire hydrants
    if world.npcs[i].interaction_target.is_none() {
        let best = find_nearest_interactible(world, world.npcs[i].x, world.npcs[i].z, InteractibleKind::FireHydrant);
        world.npcs[i].interaction_target = best;
    }

    if let Some(hi) = world.npcs[i].interaction_target {
        let hx = world.interactibles[hi].x;
        let hz = world.interactibles[hi].z;
        let remaining = npc_walk_toward(world, i, hx, hz, net, terrain, dt);
        if remaining < 2.0 {
            world.npcs[i].job_timer += dt;
            if world.npcs[i].job_timer > 5.0 {
                // Occasionally activate hydrant
                if world.npcs[i].rng.next() % 4 == 0 {
                    world.interactibles[hi].state_val = 5.0;
                }
                world.npcs[i].money += 1.0;
                world.npcs[i].interaction_target = None;
                world.npcs[i].job_timer = 0.0;
            }
        }
    } else {
        pick_wander(world, i);
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    }
}

fn npc_work_police(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Drive patrol route between road nodes
    world.npcs[i].job_timer += dt;
    let car_idx = world.npcs[i].car_idx;

    if car_idx < world.vehicles.len() && !world.vehicles[car_idx].occupied && !world.npcs[i].in_vehicle {
        let cdx = world.vehicles[car_idx].x - world.npcs[i].x;
        let cdz = world.vehicles[car_idx].z - world.npcs[i].z;
        if cdx * cdx + cdz * cdz < 25.0 {
            world.npcs[i].in_vehicle = true;
            world.npcs[i].state = NpcState::Driving;
            world.vehicles[car_idx].ai_active = true;
            if !net.nodes.is_empty() {
                let ni = world.npcs[i].rng.next() as usize % net.nodes.len();
                world.npcs[i].target_x = net.nodes[ni][0];
                world.npcs[i].target_z = net.nodes[ni][1];
            }
        } else {
            npc_walk_toward(world, i, world.vehicles[car_idx].x, world.vehicles[car_idx].z, net, terrain, dt);
        }
    } else if !world.npcs[i].in_vehicle {
        // Walk patrol
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        if remaining < 3.0 { pick_wander(world, i); }
    }
}

// ---- Commerce Jobs ----

fn npc_work_vendor(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Stand at intersection, earn money when NPCs come near
    world.npcs[i].job_timer += dt;

    // Walk to a road node if not already near one
    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    let dx = tx - world.npcs[i].x;
    let dz = tz - world.npcs[i].z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist > 3.0 {
        npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    } else {
        // Stand still, check for nearby NPCs every few seconds
        world.npcs[i].walk_phase = 0.0;
        if world.npcs[i].job_timer > 30.0 {
            // Check for nearby NPCs (customers)
            let n = world.npcs.len();
            for j in 0..n {
                if j == i { continue; }
                if world.npcs[j].state != NpcState::Working { continue; }
                if world.npcs[j].job == NpcJob::StreetVendor { continue; }
                let cdx = world.npcs[j].x - world.npcs[i].x;
                let cdz = world.npcs[j].z - world.npcs[i].z;
                if cdx * cdx + cdz * cdz < 9.0 {
                    world.npcs[i].money += 1.0;
                    world.npcs[i].job_timer = 0.0;
                    break;
                }
            }
        }
    }
}

fn npc_work_mechanic(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Walk to parked vehicles, stand near them
    world.npcs[i].job_timer += dt;

    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);

    if remaining < 2.0 {
        // "Repair" — stand still for 10 seconds
        world.npcs[i].walk_phase = 0.0;
        if world.npcs[i].job_timer > 10.0 {
            world.npcs[i].money += 2.0;
            world.npcs[i].job_timer = 0.0;
            // Find next vehicle
            let mut best_dist = f32::MAX;
            let mut best_x = world.npcs[i].x;
            let mut best_z = world.npcs[i].z;
            for v in &world.vehicles {
                if v.speed.abs() > 0.5 { continue; } // skip moving vehicles
                let dx = v.x - world.npcs[i].x;
                let dz = v.z - world.npcs[i].z;
                let d = dx * dx + dz * dz;
                if d > 4.0 && d < best_dist {
                    best_dist = d;
                    best_x = v.x;
                    best_z = v.z;
                }
            }
            world.npcs[i].target_x = best_x;
            world.npcs[i].target_z = best_z;
        }
    }
}

fn npc_work_construction(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Walk to dockyard area, do hammering animation
    let tx = world.npcs[i].job_target_x;
    let tz = world.npcs[i].job_target_z;

    // Set target to dockyard if not set
    if tx == world.npcs[i].x && tz == world.npcs[i].z {
        world.npcs[i].job_target_x = world.npcs[i].rng.range(-30.0, 30.0);
        world.npcs[i].job_target_z = DOCK_Z_START + world.npcs[i].rng.range(5.0, 15.0);
    }

    let jtx = world.npcs[i].job_target_x;
    let jtz = world.npcs[i].job_target_z;
    let remaining = npc_walk_toward(world, i, jtx, jtz, net, terrain, dt);
    if remaining < 3.0 {
        // Hammering animation (rapid walk phase)
        world.npcs[i].walk_phase += dt * 15.0;
        world.npcs[i].job_timer += dt;
        if world.npcs[i].job_timer > 20.0 {
            world.npcs[i].money += 3.0;
            world.npcs[i].job_timer = 0.0;
            // Move to new spot
            world.npcs[i].job_target_x = world.npcs[i].rng.range(-30.0, 30.0);
            world.npcs[i].job_target_z = DOCK_Z_START + world.npcs[i].rng.range(5.0, 15.0);
        }
    }
}

// ---- Outdoor Jobs ----

fn npc_work_fisherman(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Walk to fishing pier, stand at edge, earn money periodically
    let pier_x = match world.npcs[i].rng.clone().next() % 3 {
        0 => -30.0, 1 => 0.0, _ => 30.0,
    };
    let pier_z = DOCK_Z_START + 35.0;

    if world.npcs[i].job_target_x == world.npcs[i].x && world.npcs[i].job_target_z == world.npcs[i].z {
        world.npcs[i].job_target_x = pier_x;
        world.npcs[i].job_target_z = pier_z;
    }

    let tx = world.npcs[i].job_target_x;
    let tz = world.npcs[i].job_target_z;
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    if remaining < 4.0 {
        // Fishing — stand still
        world.npcs[i].walk_phase = 0.0;
        world.npcs[i].job_timer += dt;
        if world.npcs[i].job_timer > 30.0 + world.npcs[i].rng.range(0.0, 30.0) {
            world.npcs[i].money += 2.0;
            world.npcs[i].job_timer = 0.0;
        }
    }
}

fn npc_work_farmer(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Walk back and forth in rows on terrain
    world.npcs[i].job_timer += dt;

    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    if remaining < 2.0 {
        // Pick next row point
        let row_offset = if world.npcs[i].job_timer as u32 % 2 == 0 { 8.0 } else { -8.0 };
        world.npcs[i].target_x = world.npcs[i].x + row_offset;
        world.npcs[i].target_z = world.npcs[i].z + world.npcs[i].rng.range(-1.0, 1.0);
        world.npcs[i].target_x = world.npcs[i].target_x.clamp(-WORLD_HALF + 10.0, WORLD_HALF - 10.0);
        world.npcs[i].target_z = world.npcs[i].target_z.clamp(-WORLD_HALF + 10.0, DOCK_Z_START - 5.0);

        if world.npcs[i].job_timer > 60.0 {
            world.npcs[i].money += 1.0;
            world.npcs[i].job_timer = 0.0;
        }
    }
}

fn npc_work_lumberjack(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Walk to nearest tree, stand near it for 15s, earn money
    world.npcs[i].job_timer += dt;

    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);

    if remaining < 2.0 {
        // Chopping animation
        world.npcs[i].walk_phase += dt * 12.0;
        if world.npcs[i].job_timer > 15.0 {
            world.npcs[i].money += 3.0;
            world.npcs[i].job_timer = 0.0;
            // Find next tree
            let mut best_dist = f32::MAX;
            let mut best_x = world.npcs[i].x;
            let mut best_z = world.npcs[i].z;
            for t in &world.trees {
                let dx = t.x - world.npcs[i].x;
                let dz = t.z - world.npcs[i].z;
                let d = dx * dx + dz * dz;
                if d > 4.0 && d < best_dist {
                    best_dist = d;
                    best_x = t.x + 1.5;
                    best_z = t.z;
                }
            }
            world.npcs[i].target_x = best_x;
            world.npcs[i].target_z = best_z;
        }
    }
}

fn npc_work_scavenger(world: &mut WorldData, i: usize, net: &RoadNetwork, terrain: &Terrain, dt: f32) {
    // Visit dumpster interactibles, search them for loot
    world.npcs[i].job_timer += dt;

    if world.npcs[i].interaction_target.is_none() {
        let best = find_nearest_interactible(world, world.npcs[i].x, world.npcs[i].z, InteractibleKind::Dumpster);
        world.npcs[i].interaction_target = best;
    }

    if let Some(di) = world.npcs[i].interaction_target {
        let dx = world.interactibles[di].x;
        let dz = world.interactibles[di].z;
        let remaining = npc_walk_toward(world, i, dx, dz, net, terrain, dt);
        if remaining < 2.0 {
            // Searching...
            world.npcs[i].walk_phase = 0.0;
            if world.npcs[i].job_timer > 5.0 {
                let loot = 1.0 + (world.npcs[i].rng.next() % 3) as f32;
                world.npcs[i].money += loot;
                world.npcs[i].job_timer = 0.0;
                world.npcs[i].interaction_target = None;
            }
        }
    } else {
        pick_wander(world, i);
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    }
}

// ---- Helpers ----

fn find_best_item(world: &WorldData, npc_idx: usize) -> Option<usize> {
    let npc = &world.npcs[npc_idx];
    let mut best_dist = f32::MAX;
    let mut best_idx = None;
    for (idx, item) in world.items.iter().enumerate() {
        if !item.active || item.falling { continue; }
        if let Some(claimer) = item.claimed_by {
            if claimer != npc_idx { continue; }
        }
        let dx = item.x - npc.x;
        let dz = item.z - npc.z;
        let dist = dx * dx + dz * dz;
        if dist < best_dist { best_dist = dist; best_idx = Some(idx); }
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
        if dist < best_dist { best_dist = dist; best_idx = Some(idx); }
    }
    best_idx
}

fn find_nearest_interactible(world: &WorldData, x: f32, z: f32, kind: InteractibleKind) -> Option<usize> {
    let mut best_dist = f32::MAX;
    let mut best_idx = None;
    for (idx, inter) in world.interactibles.iter().enumerate() {
        if inter.kind != kind { continue; }
        let dx = inter.x - x;
        let dz = inter.z - z;
        let dist = dx * dx + dz * dz;
        if dist < best_dist { best_dist = dist; best_idx = Some(idx); }
    }
    best_idx
}

fn pick_wander(world: &mut WorldData, i: usize) {
    world.npcs[i].target_x = world.npcs[i].x + world.npcs[i].rng.range(-15.0, 15.0);
    world.npcs[i].target_z = world.npcs[i].z + world.npcs[i].rng.range(-15.0, 15.0);
    world.npcs[i].target_x = world.npcs[i].target_x.clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
    world.npcs[i].target_z = world.npcs[i].target_z.clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
}

fn pick_random_building_target(world: &mut WorldData, i: usize) {
    let bi = world.npcs[i].rng.next() as usize % world.buildings.len();
    world.npcs[i].target_x = world.buildings[bi].x;
    world.npcs[i].target_z = world.buildings[bi].z;
}
