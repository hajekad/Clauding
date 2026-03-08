// NPC job-specific AI behaviors — traditional AI handles work, money earning
// Each job function is called during the Working state based on npc.job

use crate::state::*;
use crate::npc::{npc_walk_toward, npc_enter_car};
use crate::world::on_river_not_bridge;

/// Handle stuck recovery with escalation — called when NPC gives up on a target
fn stuck_recovery(world: &mut WorldData, i: usize, net: &RoadNetwork) {
    world.npcs[i].stuck_timer = 0.0;
    world.npcs[i].detouring = false;
    world.npcs[i].stuck_count += 1;

    // Escalation level 2: teleport to car or road node
    if world.npcs[i].stuck_count >= 3 && world.npcs[i].stuck_count < 6 {
        // Teleport to NPC's car and try to drive away
        let car_idx = world.npcs[i].car_idx;
        if car_idx < world.vehicles.len() && !world.vehicles[car_idx].occupied {
            let cx = world.vehicles[car_idx].x;
            let cz = world.vehicles[car_idx].z;
            if !on_river_not_bridge(cx, cz, &world.river_segments, &world.bridges) {
                world.npcs[i].x = cx;
                world.npcs[i].z = cz;
                world.npcs[i].stuck_count = 0;
                if !net.nodes.is_empty() {
                    let ni = world.npcs[i].rng.next() as usize % net.nodes.len();
                    world.npcs[i].target_x = net.nodes[ni][0];
                    world.npcs[i].target_z = net.nodes[ni][1];
                }
                world.npcs[i].wander_cooldown = 5.0;
                return;
            }
        }
        // Fallback: teleport to a distant road node
        if !net.nodes.is_empty() {
            let nx = world.npcs[i].x;
            let nz = world.npcs[i].z;
            for _ in 0..10 {
                let ni = world.npcs[i].rng.next() as usize % net.nodes.len();
                let node = &net.nodes[ni];
                let dx = node[0] - nx;
                let dz = node[1] - nz;
                let d = dx * dx + dz * dz;
                if d > 2500.0 && d < 40000.0
                    && !on_river_not_bridge(node[0], node[1], &world.river_segments, &world.bridges)
                {
                    world.npcs[i].x = node[0];
                    world.npcs[i].z = node[1];
                    world.npcs[i].stuck_count = 0;
                    world.npcs[i].wander_cooldown = 5.0;
                    return;
                }
            }
        }
        // Don't reset stuck_count on failure — let it escalate to level 3
    }

    // Escalation level 3: teleport to a road node near center (guaranteed items nearby)
    if world.npcs[i].stuck_count >= 6 {
        if !net.nodes.is_empty() {
            let mut best_d = f32::MAX;
            let mut best_node = None;
            for (ni, node) in net.nodes.iter().enumerate() {
                let d = node[0] * node[0] + node[1] * node[1]; // distance from center
                if d < best_d
                    && !on_river_not_bridge(node[0], node[1], &world.river_segments, &world.bridges)
                {
                    best_d = d;
                    best_node = Some(ni);
                }
            }
            if let Some(ni) = best_node {
                world.npcs[i].x = net.nodes[ni][0];
                world.npcs[i].z = net.nodes[ni][1];
            }
        }
        world.npcs[i].stuck_count = 0;
        world.npcs[i].wander_cooldown = 8.0;
        pick_wander(world, i, net);
        return;
    }

    world.npcs[i].wander_cooldown = 3.0;
    pick_wander(world, i, net);
}

/// Try to drive to a work target instead of walking — returns true if NPC entered vehicle
fn try_drive_to_target(world: &mut WorldData, i: usize, tx: f32, tz: f32, terrain: &Terrain, net: &mut RoadNetwork) -> bool {
    let npc = &world.npcs[i];
    if npc.carrying_item || npc.carrying_bin.is_some() { return false; }

    let dx = tx - npc.x;
    let dz = tz - npc.z;
    let target_dist = (dx * dx + dz * dz).sqrt();
    if target_dist <= NPC_DRIVE_THRESHOLD { return false; }

    let car_idx = npc.car_idx;
    if car_idx >= world.vehicles.len() { return false; }
    if world.vehicles[car_idx].occupied { return false; }

    let cdx = world.vehicles[car_idx].x - npc.x;
    let cdz = world.vehicles[car_idx].z - npc.z;
    let car_dist = (cdx * cdx + cdz * cdz).sqrt();
    if car_dist > 5.0 { return false; } // car must be right here

    // Set target destination, then enter car
    world.npcs[i].target_x = tx;
    world.npcs[i].target_z = tz;
    npc_enter_car(world, i, terrain, net)
}

/// Dispatch NPC work behavior by job type
pub fn npc_work_dispatch(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    // Tick down wander cooldown — prevents re-targeting same unreachable item
    if world.npcs[i].wander_cooldown > 0.0 {
        world.npcs[i].wander_cooldown -= dt;
    }
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

fn npc_work_collector(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    let npc = &world.npcs[i];

    if npc.carrying_item {
        // Find nearest bin to deposit
        if npc.target_bin.is_none() {
            let bin_idx = find_nearest_bin(world, npc.x, npc.z, npc.home_idx);
            world.npcs[i].target_bin = bin_idx;
            if bin_idx.is_some() {
                world.npcs[i].stuck_timer = 0.0;
                world.npcs[i].detouring = false;
            }
        }
        if let Some(bi) = world.npcs[i].target_bin {
            let was_stuck = world.npcs[i].stuck_timer > 4.0;
            let bx = world.trash_bins[bi].x;
            let bz = world.trash_bins[bi].z;
            let remaining = npc_walk_toward(world, i, bx, bz, net, terrain, dt);
            if remaining < NPC_BIN_DIST {
                world.npcs[i].carrying_item = false;
                world.npcs[i].target_bin = None;
                world.npcs[i].items_deposited_today += 1;
                world.npcs[i].money += 1.0;
                world.npcs[i].fitness_money_earned += 1.0;
                world.npcs[i].stuck_count = 0;
                world.trash_bins[bi].items_held += 1;
            } else if was_stuck {
                // Can't reach bin — drop item and pick new bin
                world.npcs[i].target_bin = None;
                stuck_recovery(world, i, net);
            }
        } else {
            pick_wander(world, i, net);
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
        // Find item to pick up (skip if on cooldown from stuck recovery)
        if world.npcs[i].target_item.is_none() && world.npcs[i].wander_cooldown <= 0.0 {
            let item_idx = find_best_item(world, i);
            if let Some(idx) = item_idx {
                world.npcs[i].target_item = Some(idx);
                world.items[idx].claimed_by = Some(i);
                world.npcs[i].target_x = world.items[idx].x;
                world.npcs[i].target_z = world.items[idx].z;
                world.npcs[i].stuck_timer = 0.0; // fresh start for new target
                world.npcs[i].detouring = false;

                // Check if we should relocate a bin closer
                let nearest_bin = find_nearest_bin(world, world.items[idx].x, world.items[idx].z, world.npcs[i].home_idx);
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
            } else if world.npcs[i].wander_cooldown <= 0.0 {
                pick_wander(world, i, net);
            }
        }

        if let Some(item_idx) = world.npcs[i].target_item {
            if item_idx < world.items.len() && world.items[item_idx].active && !world.items[item_idx].falling {
                let ix = world.items[item_idx].x;
                let iz = world.items[item_idx].z;
                // Try driving if item is far and car is nearby
                if try_drive_to_target(world, i, ix, iz, terrain, net) { return; }
                // Check stuck BEFORE walking (walk resets timer on teleport)
                let was_stuck = world.npcs[i].stuck_timer > 4.0;
                let remaining = npc_walk_toward(world, i, ix, iz, net, terrain, dt);
                if remaining < NPC_PICKUP_DIST {
                    world.items[item_idx].active = false;
                    world.items[item_idx].claimed_by = None;
                    world.npcs[i].carrying_item = true;
                    world.npcs[i].target_item = None;
                    world.npcs[i].fitness_items_picked += 1;
                    world.npcs[i].stuck_count = 0;
                } else if was_stuck {
                    // Can't reach item — give up, mark item suspect
                    world.items[item_idx].claimed_by = None;
                    world.items[item_idx].skip_until = 10.0;
                    world.npcs[i].target_item = None;
                    stuck_recovery(world, i, net);
                }
            } else {
                if item_idx < world.items.len() {
                    world.items[item_idx].claimed_by = None;
                }
                world.npcs[i].target_item = None;
            }
        } else {
            // Wandering (during cooldown or no target available)
            let tx = world.npcs[i].target_x;
            let tz = world.npcs[i].target_z;
            let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
            if remaining < 2.0 {
                pick_wander(world, i, net);
            }
        }
    }
}

// ---- Service Jobs ----

fn npc_work_garbage(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    // Similar to collector but targets dumpster interactibles instead of bins
    let npc = &world.npcs[i];

    if npc.carrying_item {
        // Find nearest dumpster
        if world.npcs[i].interaction_target.is_none() {
            let best = find_nearest_interactible(world, npc.x, npc.z, InteractibleKind::Dumpster, npc.home_idx);
            world.npcs[i].interaction_target = best;
            if best.is_some() {
                world.npcs[i].stuck_timer = 0.0;
                world.npcs[i].detouring = false;
            }
        }
        if let Some(di) = world.npcs[i].interaction_target {
            let was_stuck = world.npcs[i].stuck_timer > 4.0;
            let dx = world.interactibles[di].x;
            let dz = world.interactibles[di].z;
            let remaining = npc_walk_toward(world, i, dx, dz, net, terrain, dt);
            if remaining < NPC_BIN_DIST {
                world.npcs[i].carrying_item = false;
                world.npcs[i].interaction_target = None;
                world.npcs[i].items_deposited_today += 1;
                world.npcs[i].money += 1.5;
                world.npcs[i].fitness_money_earned += 1.5;
                world.npcs[i].stuck_count = 0;
            } else if was_stuck {
                // Can't reach dumpster — drop item and try another
                world.npcs[i].interaction_target = None;
                stuck_recovery(world, i, net);
            }
        }
    } else {
        // Find items near roads (skip if on cooldown from stuck recovery)
        if world.npcs[i].target_item.is_none() && world.npcs[i].wander_cooldown <= 0.0 {
            let item_idx = find_best_item(world, i);
            if let Some(idx) = item_idx {
                world.npcs[i].target_item = Some(idx);
                world.items[idx].claimed_by = Some(i);
                world.npcs[i].target_x = world.items[idx].x;
                world.npcs[i].target_z = world.items[idx].z;
                world.npcs[i].stuck_timer = 0.0; // fresh start for new target
                world.npcs[i].detouring = false;
            } else if world.npcs[i].wander_cooldown <= 0.0 {
                pick_wander(world, i, net);
            }
        }

        if let Some(item_idx) = world.npcs[i].target_item {
            if item_idx < world.items.len() && world.items[item_idx].active && !world.items[item_idx].falling {
                let ix = world.items[item_idx].x;
                let iz = world.items[item_idx].z;
                // Try driving if item is far and car is nearby
                if try_drive_to_target(world, i, ix, iz, terrain, net) { return; }
                let was_stuck = world.npcs[i].stuck_timer > 4.0;
                let remaining = npc_walk_toward(world, i, ix, iz, net, terrain, dt);
                if remaining < NPC_PICKUP_DIST {
                    world.items[item_idx].active = false;
                    world.items[item_idx].claimed_by = None;
                    world.npcs[i].carrying_item = true;
                    world.npcs[i].target_item = None;
                    world.npcs[i].fitness_items_picked += 1;
                    world.npcs[i].stuck_count = 0;
                } else if was_stuck {
                    world.items[item_idx].claimed_by = None;
                    world.items[item_idx].skip_until = 10.0;
                    world.npcs[i].target_item = None;
                    stuck_recovery(world, i, net);
                }
            } else {
                if item_idx < world.items.len() { world.items[item_idx].claimed_by = None; }
                world.npcs[i].target_item = None;
            }
        } else {
            // Wandering (during cooldown or no target available)
            let tx = world.npcs[i].target_x;
            let tz = world.npcs[i].target_z;
            let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
            if remaining < 2.0 {
                pick_wander(world, i, net);
            }
        }
    }
}

fn npc_work_taxi(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    // Drive around road nodes
    world.npcs[i].job_timer += dt;
    let car_idx = world.npcs[i].car_idx;
    if car_idx < world.vehicles.len() && !world.vehicles[car_idx].occupied {
        // If not in vehicle, try to enter
        if !world.npcs[i].in_vehicle {
            let cdx = world.vehicles[car_idx].x - world.npcs[i].x;
            let cdz = world.vehicles[car_idx].z - world.npcs[i].z;
            if cdx * cdx + cdz * cdz < 25.0 {
                // Pick random road node as destination
                if !net.nodes.is_empty() {
                    let ni = world.npcs[i].rng.next() as usize % net.nodes.len();
                    world.npcs[i].target_x = net.nodes[ni][0];
                    world.npcs[i].target_z = net.nodes[ni][1];
                }
                world.npcs[i].in_vehicle = true;
                world.npcs[i].state = NpcState::Driving;
                world.vehicles[car_idx].ai_active = true;
                world.vehicles[car_idx].parked = false;
                world.vehicles[car_idx].path.clear();
                world.vehicles[car_idx].ai_target_x = world.npcs[i].target_x;
                world.vehicles[car_idx].ai_target_z = world.npcs[i].target_z;
            } else {
                npc_walk_toward(world, i, world.vehicles[car_idx].x, world.vehicles[car_idx].z, net, terrain, dt);
            }
        }
    } else {
        // Wander on foot
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        if remaining < 3.0 { pick_wander(world, i, net); }
    }
}

fn npc_work_delivery(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    let npc = &world.npcs[i];

    if npc.carrying_item {
        // Drive/walk to target building
        let tx = world.npcs[i].job_target_x;
        let tz = world.npcs[i].job_target_z;
        let was_stuck = world.npcs[i].stuck_timer > 4.0;
        // Delivery carrying → can't drive (carrying_item check in try_drive), just walk
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        if remaining < 3.0 {
            world.npcs[i].carrying_item = false;
            world.npcs[i].items_deposited_today += 1;
            world.npcs[i].money += 2.0;
            world.npcs[i].fitness_money_earned += 2.0;
            // Pick new source
            pick_random_building_target(world, i);
        } else if was_stuck {
            // Can't reach destination — pick new one
            world.npcs[i].carrying_item = false;
            pick_random_building_target(world, i);
            stuck_recovery(world, i, net);
        }
    } else {
        // Drive/walk to source building to pick up
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        if try_drive_to_target(world, i, tx, tz, terrain, net) { return; }
        let was_stuck = world.npcs[i].stuck_timer > 4.0;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        if remaining < 3.0 {
            world.npcs[i].carrying_item = true;
            world.npcs[i].fitness_items_picked += 1;
            // Pick delivery destination
            let dest = world.npcs[i].rng.next() as usize % world.buildings.len();
            world.npcs[i].job_target_x = world.buildings[dest].x;
            world.npcs[i].job_target_z = world.buildings[dest].z;
        } else if was_stuck {
            pick_random_building_target(world, i);
            stuck_recovery(world, i, net);
        }
    }
}

fn npc_work_mail(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    // Visit mailbox interactibles in sequence
    world.npcs[i].job_timer += dt;

    if world.npcs[i].interaction_target.is_none() {
        // Wait before finding next mailbox so we walk away from the last one
        if world.npcs[i].job_timer < 5.0 {
            let tx = world.npcs[i].target_x;
            let tz = world.npcs[i].target_z;
            npc_walk_toward(world, i, tx, tz, net, terrain, dt);
            return;
        }
        let best = find_nearest_interactible(world, world.npcs[i].x, world.npcs[i].z, InteractibleKind::Mailbox, world.npcs[i].home_idx);
        world.npcs[i].interaction_target = best;
    }

    if let Some(mi) = world.npcs[i].interaction_target {
        let mx = world.interactibles[mi].x;
        let mz = world.interactibles[mi].z;
        if try_drive_to_target(world, i, mx, mz, terrain, net) { return; }
        let was_stuck = world.npcs[i].stuck_timer > 4.0;
        let remaining = npc_walk_toward(world, i, mx, mz, net, terrain, dt);
        if remaining < 2.0 {
            // "Deliver" to this mailbox
            world.npcs[i].job_timer = 0.0;
            world.npcs[i].money += 1.0;
            world.npcs[i].fitness_money_earned += 1.0;
            world.npcs[i].items_deposited_today += 1;
            world.npcs[i].interaction_target = None;
            pick_wander(world, i, net); // walk away before finding next
        } else if was_stuck {
            world.npcs[i].interaction_target = None;
            stuck_recovery(world, i, net);
        }
    } else {
        pick_wander(world, i, net);
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    }
}

// ---- Emergency Jobs ----

fn npc_work_paramedic(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
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
    if try_drive_to_target(world, i, tx, tz, terrain, net) { return; }
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
                world.npcs[i].fitness_money_earned += 2.0;
            }
        }
        pick_wander(world, i, net);
    }
}

fn npc_work_firefighter(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    // Patrol between fire hydrants
    if world.npcs[i].interaction_target.is_none() {
        let best = find_nearest_interactible(world, world.npcs[i].x, world.npcs[i].z, InteractibleKind::FireHydrant, world.npcs[i].home_idx);
        world.npcs[i].interaction_target = best;
    }

    if let Some(hi) = world.npcs[i].interaction_target {
        let hx = world.interactibles[hi].x;
        let hz = world.interactibles[hi].z;
        if try_drive_to_target(world, i, hx, hz, terrain, net) { return; }
        let was_stuck = world.npcs[i].stuck_timer > 4.0;
        let remaining = npc_walk_toward(world, i, hx, hz, net, terrain, dt);
        if was_stuck && remaining > 2.0 {
            world.npcs[i].interaction_target = None;
            world.npcs[i].job_timer = 0.0;
            stuck_recovery(world, i, net);
        } else if remaining < 2.0 {
            world.npcs[i].job_timer += dt;
            if world.npcs[i].job_timer > 5.0 {
                // Occasionally activate hydrant
                if world.npcs[i].rng.next() % 4 == 0 {
                    world.interactibles[hi].state_val = 5.0;
                }
                world.npcs[i].money += 1.0;
                world.npcs[i].fitness_money_earned += 1.0;
                world.npcs[i].interaction_target = None;
                world.npcs[i].job_timer = 0.0;
            }
        }
    } else {
        pick_wander(world, i, net);
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    }
}

fn npc_work_police(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    world.npcs[i].job_timer += dt;

    // Scan for wanted NPCs within 50m (rescan every 2s or when no target)
    if world.npcs[i].police_target.is_none() || world.npcs[i].job_timer > 2.0 {
        let mut best_d = 50.0 * 50.0;
        let mut best_j = None;
        let n = world.npcs.len();
        for j in 0..n {
            if j == i { continue; }
            if !world.npcs[j].wanted { continue; }
            if world.npcs[j].state == NpcState::Sleeping { continue; }
            let dx = world.npcs[j].x - world.npcs[i].x;
            let dz = world.npcs[j].z - world.npcs[i].z;
            let d2 = dx * dx + dz * dz;
            if d2 < best_d { best_d = d2; best_j = Some(j); }
        }
        if best_j.is_some() {
            world.npcs[i].police_target = best_j;
            world.npcs[i].job_timer = 0.0;
        }
    }

    if let Some(target) = world.npcs[i].police_target {
        // Validate target still exists and is wanted
        if target >= world.npcs.len() || !world.npcs[target].wanted
            || world.npcs[target].state == NpcState::Sleeping {
            world.npcs[i].police_target = None;
            return;
        }

        let tx = world.npcs[target].x;
        let tz = world.npcs[target].z;
        if try_drive_to_target(world, i, tx, tz, terrain, net) { return; }
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);

        if remaining < 2.0 {
            // Catch! Take 50% of money
            let fine = world.npcs[target].money * 0.5;
            world.npcs[target].money -= fine;
            world.npcs[i].money += fine;
            world.npcs[i].fitness_money_earned += fine;
            world.npcs[target].wanted = false;
            world.npcs[target].bounty = 0.0;
            world.npcs[i].police_target = None;
        }
    } else {
        // Normal patrol: walk between road nodes
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
        if remaining < 3.0 { pick_wander(world, i, net); }
    }
}

// ---- Commerce Jobs ----

fn npc_work_vendor(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
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
                    world.npcs[i].fitness_money_earned += 1.0;
                    world.npcs[i].job_timer = 0.0;
                    break;
                }
            }
        }
    }
}

fn npc_work_mechanic(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    // Walk to parked vehicles, stand near them
    world.npcs[i].job_timer += dt;

    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    if try_drive_to_target(world, i, tx, tz, terrain, net) { return; }
    let was_stuck = world.npcs[i].stuck_timer > 4.0;
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);

    if was_stuck && remaining > 2.0 {
        stuck_recovery(world, i, net);
    } else if remaining < 2.0 {
        // "Repair" — stand still for 10 seconds
        world.npcs[i].walk_phase = 0.0;
        if world.npcs[i].job_timer > 10.0 {
            world.npcs[i].money += 2.0;
            world.npcs[i].fitness_money_earned += 2.0;
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

fn npc_work_construction(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    // Walk to dockyard area, do hammering animation
    let tx = world.npcs[i].job_target_x;
    let tz = world.npcs[i].job_target_z;

    // Set target to dockyard if not set or reached
    let dx = tx - world.npcs[i].x;
    let dz = tz - world.npcs[i].z;
    if (tx == 0.0 && tz == 0.0) || dx * dx + dz * dz < 9.0 {
        world.npcs[i].job_target_x = world.npcs[i].rng.range(-30.0, 30.0);
        world.npcs[i].job_target_z = DOCK_Z_START + world.npcs[i].rng.range(5.0, 15.0);
    }

    let jtx = world.npcs[i].job_target_x;
    let jtz = world.npcs[i].job_target_z;
    if try_drive_to_target(world, i, jtx, jtz, terrain, net) { return; }
    let remaining = npc_walk_toward(world, i, jtx, jtz, net, terrain, dt);
    if remaining < 3.0 {
        // Hammering animation (rapid walk phase)
        world.npcs[i].walk_phase += dt * 15.0;
        world.npcs[i].job_timer += dt;
        if world.npcs[i].job_timer > 20.0 {
            world.npcs[i].money += 3.0;
            world.npcs[i].fitness_money_earned += 3.0;
            world.npcs[i].job_timer = 0.0;
            // Move to new spot
            world.npcs[i].job_target_x = world.npcs[i].rng.range(-30.0, 30.0);
            world.npcs[i].job_target_z = DOCK_Z_START + world.npcs[i].rng.range(5.0, 15.0);
        }
    }
}

// ---- Outdoor Jobs ----

fn npc_work_fisherman(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    // Walk to fishing pier, stand at edge, earn money periodically
    let pier_x = match world.npcs[i].rng.clone().next() % 3 {
        0 => -30.0, 1 => 0.0, _ => 30.0,
    };
    let pier_z = DOCK_Z_START + 5.0; // stay within world bounds

    let ftx = world.npcs[i].job_target_x;
    let ftz = world.npcs[i].job_target_z;
    let fdx = ftx - world.npcs[i].x;
    let fdz = ftz - world.npcs[i].z;
    if (ftx == 0.0 && ftz == 0.0) || fdx * fdx + fdz * fdz < 16.0 {
        world.npcs[i].job_target_x = pier_x;
        world.npcs[i].job_target_z = pier_z;
    }

    let tx = world.npcs[i].job_target_x;
    let tz = world.npcs[i].job_target_z;
    if try_drive_to_target(world, i, tx, tz, terrain, net) { return; }
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    if remaining < 4.0 {
        // Fishing — stand still
        world.npcs[i].walk_phase = 0.0;
        world.npcs[i].job_timer += dt;
        if world.npcs[i].job_timer > 30.0 + world.npcs[i].rng.range(0.0, 30.0) {
            world.npcs[i].money += 2.0;
            world.npcs[i].fitness_money_earned += 2.0;
            world.npcs[i].job_timer = 0.0;
        }
    }
}

fn npc_work_farmer(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
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
            world.npcs[i].fitness_money_earned += 1.0;
            world.npcs[i].job_timer = 0.0;
        }
    }
}

fn npc_work_lumberjack(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    // Walk to nearest tree, stand near it for 15s, earn money
    world.npcs[i].job_timer += dt;

    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    if try_drive_to_target(world, i, tx, tz, terrain, net) { return; }
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt);

    if remaining < 2.0 {
        // Chopping animation
        world.npcs[i].walk_phase += dt * 12.0;
        if world.npcs[i].job_timer > 15.0 {
            world.npcs[i].money += 3.0;
            world.npcs[i].fitness_money_earned += 3.0;
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

fn npc_work_scavenger(world: &mut WorldData, i: usize, net: &mut RoadNetwork, terrain: &Terrain, dt: f32) {
    // Visit dumpster interactibles, search them for loot
    world.npcs[i].job_timer += dt;

    if world.npcs[i].interaction_target.is_none() {
        let best = find_nearest_interactible(world, world.npcs[i].x, world.npcs[i].z, InteractibleKind::Dumpster, world.npcs[i].home_idx);
        world.npcs[i].interaction_target = best;
    }

    if let Some(di) = world.npcs[i].interaction_target {
        let dx = world.interactibles[di].x;
        let dz = world.interactibles[di].z;
        if try_drive_to_target(world, i, dx, dz, terrain, net) { return; }
        let remaining = npc_walk_toward(world, i, dx, dz, net, terrain, dt);
        if remaining < 2.0 {
            // Searching...
            world.npcs[i].walk_phase = 0.0;
            if world.npcs[i].job_timer > 5.0 {
                let loot = 1.0 + (world.npcs[i].rng.next() % 3) as f32;
                world.npcs[i].money += loot;
                world.npcs[i].fitness_money_earned += loot;
                world.npcs[i].job_timer = 0.0;
                world.npcs[i].interaction_target = None;
            }
        }
    } else {
        pick_wander(world, i, net);
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        npc_walk_toward(world, i, tx, tz, net, terrain, dt);
    }
}

// ---- Helpers ----

/// Check if the straight-line path from (ax,az) to (bx,bz) crosses a river.
/// Only checks river — NPCs can walk around buildings via detour/stuck recovery,
/// but rivers are truly impassable. Samples every ~8m along the path.
fn path_clear(world: &WorldData, ax: f32, az: f32, bx: f32, bz: f32, _home_idx: usize) -> bool {
    let dx = bx - ax;
    let dz = bz - az;
    let dist = (dx * dx + dz * dz).sqrt();
    let steps = ((dist / 8.0) as usize).clamp(2, 8);
    for step in 1..=steps {
        let t = step as f32 / (steps + 1) as f32;
        let sx = ax + dx * t;
        let sz = az + dz * t;
        if on_river_not_bridge(sx, sz, &world.river_segments, &world.bridges) {
            return false;
        }
    }
    true
}

fn find_best_item(world: &WorldData, npc_idx: usize) -> Option<usize> {
    let npc = &world.npcs[npc_idx];
    let home = npc.home_idx;
    let mut best_dist = f32::MAX;
    let mut best_idx = None;
    let max_dist_sq = 120.0 * 120.0;
    for (idx, item) in world.items.iter().enumerate() {
        if !item.active || item.falling { continue; }
        if item.skip_until > 0.0 { continue; }
        if let Some(claimer) = item.claimed_by {
            if claimer != npc_idx { continue; }
        }
        let dx = item.x - npc.x;
        let dz = item.z - npc.z;
        let dist = dx * dx + dz * dz;
        if dist > max_dist_sq { continue; }
        if dist < best_dist && path_clear(world, npc.x, npc.z, item.x, item.z, home) {
            best_dist = dist;
            best_idx = Some(idx);
        }
    }
    best_idx
}

fn find_nearest_bin(world: &WorldData, x: f32, z: f32, home_idx: usize) -> Option<usize> {
    let mut best_dist = f32::MAX;
    let mut best_idx = None;
    for (idx, bin) in world.trash_bins.iter().enumerate() {
        if bin.carried_by.is_some() { continue; }
        let dx = bin.x - x;
        let dz = bin.z - z;
        let dist = dx * dx + dz * dz;
        if dist < best_dist && path_clear(world, x, z, bin.x, bin.z, home_idx) {
            best_dist = dist;
            best_idx = Some(idx);
        }
    }
    best_idx
}

fn find_nearest_interactible(world: &WorldData, x: f32, z: f32, kind: InteractibleKind, home_idx: usize) -> Option<usize> {
    let mut best_dist = f32::MAX;
    let mut best_idx = None;
    for (idx, inter) in world.interactibles.iter().enumerate() {
        if inter.kind != kind { continue; }
        let dx = inter.x - x;
        let dz = inter.z - z;
        let dist = dx * dx + dz * dz;
        if dist < best_dist && path_clear(world, x, z, inter.x, inter.z, home_idx) {
            best_dist = dist;
            best_idx = Some(idx);
        }
    }
    best_idx
}

fn pick_wander(world: &mut WorldData, i: usize, net: &RoadNetwork) {
    let nx = world.npcs[i].x;
    let nz = world.npcs[i].z;
    let dist_from_center = (nx * nx + nz * nz).sqrt();

    // NPCs far from center: bias toward center to return to populated areas with items
    if dist_from_center > 60.0 {
        let toward_center_angle = (-nz).atan2(-nx);
        let spread = 0.5; // ±30° cone toward center
        for _ in 0..3 {
            let angle = toward_center_angle + world.npcs[i].rng.range(-spread, spread);
            let dist = world.npcs[i].rng.range(20.0, 50.0);
            let tx = (nx + angle.cos() * dist).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            let tz = (nz + angle.sin() * dist).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
            if !on_river_not_bridge(tx, tz, &world.river_segments, &world.bridges) {
                world.npcs[i].target_x = tx;
                world.npcs[i].target_z = tz;
                return;
            }
        }
    }

    // Try a few random road nodes — fast O(1), gets NPCs onto walkable paths
    let n_nodes = net.nodes.len();
    if n_nodes > 0 {
        for _ in 0..5 {
            let ni = world.npcs[i].rng.next() as usize % n_nodes;
            let node = &net.nodes[ni];
            let dx = node[0] - nx;
            let dz = node[1] - nz;
            let d = dx * dx + dz * dz;
            // Must be 10-80m away and not on river
            if d > 100.0 && d < 6400.0 && !on_river_not_bridge(node[0], node[1], &world.river_segments, &world.bridges) {
                world.npcs[i].target_x = node[0];
                world.npcs[i].target_z = node[1];
                return;
            }
        }
    }
    // Fallback: random directions, avoid river
    for _ in 0..5 {
        let angle = world.npcs[i].rng.range(0.0, std::f32::consts::TAU);
        let dist = world.npcs[i].rng.range(10.0, 25.0);
        let tx = (nx + angle.cos() * dist).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
        let tz = (nz + angle.sin() * dist).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
        if !on_river_not_bridge(tx, tz, &world.river_segments, &world.bridges) {
            world.npcs[i].target_x = tx;
            world.npcs[i].target_z = tz;
            return;
        }
    }
    // Last resort: random direction ignoring river
    let angle = world.npcs[i].rng.range(0.0, std::f32::consts::TAU);
    let dist = world.npcs[i].rng.range(10.0, 15.0);
    world.npcs[i].target_x = (nx + angle.cos() * dist).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
    world.npcs[i].target_z = (nz + angle.sin() * dist).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
}

fn pick_random_building_target(world: &mut WorldData, i: usize) {
    let bi = world.npcs[i].rng.next() as usize % world.buildings.len();
    world.npcs[i].target_x = world.buildings[bi].x;
    world.npcs[i].target_z = world.buildings[bi].z;
}
