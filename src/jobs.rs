//! NPC job-specific AI behaviors — traditional (non-NEAT) AI for working hours.
//! Each job function is called during the Working state based on `npc.job`.

use crate::math::dist_sq_2d;
use crate::npc::{npc_enter_car, npc_walk_toward};
use crate::state::*;
use crate::world::on_river_not_bridge;

/// Handle path failure — NPC gives up on target, picks a new wander destination
fn path_failure(world: &mut WorldData, i: usize, net: &RoadNetwork) {
    world.npcs[i].stuck_recoveries += 1;
    world.npcs[i].stuck_timer = 0.0;
    world.npcs[i].job_timer = 0.0;
    world.npcs[i].target_item = None;
    world.npcs[i].nav_path.clear();
    world.npcs[i].nav_path_idx = 0;
    pick_wander(world, i, net);
}

/// Try to drive to a work target instead of walking — returns true if NPC entered vehicle
fn try_drive_to_target(
    world: &mut WorldData,
    i: usize,
    tx: f32,
    tz: f32,
    terrain: &Terrain,
    net: &mut RoadNetwork,
) -> bool {
    let npc = &world.npcs[i];
    if npc.carrying_item || npc.carrying_bin.is_some() {
        return false;
    }

    let dx = tx - npc.x;
    let dz = tz - npc.z;
    let target_dist = (dx * dx + dz * dz).sqrt();
    if target_dist <= NPC_DRIVE_THRESHOLD {
        return false;
    }

    let car_idx = npc.car_idx;
    if car_idx >= world.vehicles.len() {
        return false;
    }
    if world.vehicles[car_idx].occupied {
        return false;
    }

    let cdx = world.vehicles[car_idx].x - npc.x;
    let cdz = world.vehicles[car_idx].z - npc.z;
    let car_dist = (cdx * cdx + cdz * cdz).sqrt();
    if car_dist > 5.0 {
        return false;
    } // car must be right here

    // Set target destination, then enter car
    world.npcs[i].target_x = tx;
    world.npcs[i].target_z = tz;
    npc_enter_car(world, i, terrain, net)
}

/// Dispatch NPC work behavior by job type
pub fn npc_work_dispatch(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    match world.npcs[i].job {
        NpcJob::Collector => npc_work_collector(world, i, net, terrain, dt, walk_grid),
        NpcJob::GarbageCollector => npc_work_garbage(world, i, net, terrain, dt, walk_grid),
        NpcJob::TaxiDriver => npc_work_taxi(world, i, net, terrain, dt, walk_grid),
        NpcJob::DeliveryCourier => npc_work_delivery(world, i, net, terrain, dt, walk_grid),
        NpcJob::MailCarrier => npc_work_mail(world, i, net, terrain, dt, walk_grid),
        NpcJob::Paramedic => npc_work_paramedic(world, i, net, terrain, dt, walk_grid),
        NpcJob::Firefighter => npc_work_firefighter(world, i, net, terrain, dt, walk_grid),
        NpcJob::PolicePatrol => npc_work_police(world, i, net, terrain, dt, walk_grid),
        NpcJob::StreetVendor => npc_work_vendor(world, i, net, terrain, dt, walk_grid),
        NpcJob::Mechanic => npc_work_mechanic(world, i, net, terrain, dt, walk_grid),
        NpcJob::ConstructionWorker => npc_work_construction(world, i, net, terrain, dt, walk_grid),
        NpcJob::Fisherman => npc_work_fisherman(world, i, net, terrain, dt, walk_grid),
        NpcJob::Farmer => npc_work_farmer(world, i, net, terrain, dt, walk_grid),
        NpcJob::Lumberjack => npc_work_lumberjack(world, i, net, terrain, dt, walk_grid),
        NpcJob::Scavenger => npc_work_scavenger(world, i, net, terrain, dt, walk_grid),
    }
}

// ---- Collector (existing behavior, extracted from npc.rs) ----

fn npc_work_collector(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    let npc = &world.npcs[i];

    if npc.carrying_item {
        // Find nearest bin to deposit
        if npc.target_bin.is_none() {
            let bin_idx = find_nearest_bin(world, npc.x, npc.z, npc.home_idx);
            world.npcs[i].target_bin = bin_idx;
            if bin_idx.is_some() {
                world.npcs[i].stuck_timer = 0.0;
                world.npcs[i].job_timer = 0.0;
            } else {
                world.npcs[i].find_bin_failures += 1;
            }
        }
        if let Some(bi) = world.npcs[i].target_bin {
            world.npcs[i].job_timer += dt;
            let no_progress = world.npcs[i].job_timer > 15.0;
            let bx = world.trash_bins[bi].x;
            let bz = world.trash_bins[bi].z;
            let remaining = npc_walk_toward(world, i, bx, bz, net, terrain, dt, walk_grid);
            if remaining < NPC_BIN_DIST {
                world.npcs[i].carrying_item = false;
                world.npcs[i].target_bin = None;
                world.npcs[i].items_deposited_today += 1;
                world.npcs[i].money += 1.0;
                world.npcs[i].fitness_money_earned += 1.0;
                world.trash_bins[bi].items_held += 1;
            } else if no_progress {
                // Can't reach bin — drop item and pick new bin
                world.npcs[i].target_bin = None;
                path_failure(world, i, net);
            }
        } else {
            // No bin found — wander toward center to find one (pick target ONCE)
            world.npcs[i].job_timer += dt;
            if world.npcs[i].job_timer > 8.0 || world.npcs[i].target_x == 0.0 {
                world.npcs[i].job_timer = 0.0;
                pick_wander(world, i, net);
                // Also retry bin search after moving
                let bin_idx = find_nearest_bin(
                    world,
                    world.npcs[i].x,
                    world.npcs[i].z,
                    world.npcs[i].home_idx,
                );
                world.npcs[i].target_bin = bin_idx;
            }
            let tx = world.npcs[i].target_x;
            let tz = world.npcs[i].target_z;
            npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
        }
    } else if world.npcs[i].carrying_bin.is_some() {
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        world.npcs[i].job_timer += dt;
        let no_progress = world.npcs[i].job_timer > 15.0;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
        if remaining < 2.0 {
            if let Some(bi) = world.npcs[i].carrying_bin {
                world.trash_bins[bi].x = world.npcs[i].x;
                world.trash_bins[bi].z = world.npcs[i].z;
                world.trash_bins[bi].y = terrain.height_at(world.npcs[i].x, world.npcs[i].z);
                world.trash_bins[bi].terrain_normal =
                    terrain.normal_at(world.npcs[i].x, world.npcs[i].z);
                world.trash_bins[bi].carried_by = None;
                world.npcs[i].carrying_bin = None;
            }
        } else if no_progress {
            // Can't reach target — drop bin here and give up
            if let Some(bi) = world.npcs[i].carrying_bin {
                world.trash_bins[bi].x = world.npcs[i].x;
                world.trash_bins[bi].z = world.npcs[i].z;
                world.trash_bins[bi].y = terrain.height_at(world.npcs[i].x, world.npcs[i].z);
                world.trash_bins[bi].terrain_normal =
                    terrain.normal_at(world.npcs[i].x, world.npcs[i].z);
                world.trash_bins[bi].carried_by = None;
                world.npcs[i].carrying_bin = None;
            }
            path_failure(world, i, net);
        }
    } else {
        // Find item to pick up (skip if on cooldown from stuck recovery)
        if world.npcs[i].target_item.is_none() {
            let item_idx = find_best_item(world, i);
            if let Some(idx) = item_idx {
                world.npcs[i].target_item = Some(idx);
                world.items[idx].claimed_by = Some(i);
                world.npcs[i].target_x = world.items[idx].x;
                world.npcs[i].target_z = world.items[idx].z;
                world.npcs[i].stuck_timer = 0.0; // fresh start for new target
                world.npcs[i].job_timer = 0.0; // progress timer

                // Check if we should relocate a bin closer
                let nearest_bin = find_nearest_bin(
                    world,
                    world.items[idx].x,
                    world.items[idx].z,
                    world.npcs[i].home_idx,
                );
                if let Some(bi) = nearest_bin {
                    let bdx = world.trash_bins[bi].x - world.items[idx].x;
                    let bdz = world.trash_bins[bi].z - world.items[idx].z;
                    let bin_dist = (bdx * bdx + bdz * bdz).sqrt();
                    if bin_dist > 20.0 && world.trash_bins[bi].carried_by.is_none() {
                        world.npcs[i].carrying_bin = Some(bi);
                        world.trash_bins[bi].carried_by = Some(i);
                        world.items[idx].claimed_by = None;
                        world.npcs[i].target_item = None;
                        // Walk to item area to deposit bin there (closer to items)
                        world.npcs[i].target_x = world.items[idx].x;
                        world.npcs[i].target_z = world.items[idx].z;
                        return;
                    }
                }
            } else {
                world.npcs[i].find_item_failures += 1;
                pick_wander(world, i, net);
            }
        }

        if let Some(item_idx) = world.npcs[i].target_item {
            if item_idx < world.items.len()
                && world.items[item_idx].active
                && !world.items[item_idx].falling
            {
                let ix = world.items[item_idx].x;
                let iz = world.items[item_idx].z;
                // Try driving if item is far and car is nearby
                if try_drive_to_target(world, i, ix, iz, terrain, net) {
                    return;
                }
                // Check stuck BEFORE walking (walk resets timer on teleport)
                world.npcs[i].job_timer += dt;
                let no_progress = world.npcs[i].job_timer > 15.0;
                let remaining = npc_walk_toward(world, i, ix, iz, net, terrain, dt, walk_grid);
                if remaining < NPC_PICKUP_DIST {
                    world.items[item_idx].active = false;
                    world.items[item_idx].claimed_by = None;
                    world.npcs[i].carrying_item = true;
                    world.npcs[i].target_item = None;
                    world.npcs[i].fitness_items_picked += 1;
                } else if no_progress {
                    // Can't reach item — give up, mark item suspect
                    world.items[item_idx].claimed_by = None;
                    world.npcs[i].target_item = None;
                    path_failure(world, i, net);
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
            world.npcs[i].job_timer += dt;
            let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
            if remaining < 2.0 {
                world.npcs[i].job_timer = 0.0;
                pick_wander(world, i, net);
            } else if world.npcs[i].job_timer > 15.0 {
                // Stuck wandering — just pick a new wander target instead of escalating
                world.npcs[i].job_timer = 0.0;
                pick_wander(world, i, net);
            }
        }
    }
}

// ---- Service Jobs ----

fn npc_work_garbage(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    // Similar to collector but targets dumpster interactibles instead of bins
    let npc = &world.npcs[i];

    if npc.carrying_item {
        // Find nearest dumpster
        if world.npcs[i].interaction_target.is_none() {
            let best = find_nearest_interactible(
                world,
                npc.x,
                npc.z,
                InteractibleKind::Dumpster,
                npc.home_idx,
            );
            world.npcs[i].interaction_target = best;
            if best.is_some() {
                world.npcs[i].stuck_timer = 0.0;
                world.npcs[i].job_timer = 0.0;
            }
        }
        if let Some(di) = world.npcs[i].interaction_target {
            world.npcs[i].job_timer += dt;
            let no_progress = world.npcs[i].job_timer > 15.0;
            let dx = world.interactibles[di].x;
            let dz = world.interactibles[di].z;
            let remaining = npc_walk_toward(world, i, dx, dz, net, terrain, dt, walk_grid);
            if remaining < NPC_BIN_DIST {
                world.npcs[i].carrying_item = false;
                world.npcs[i].interaction_target = None;
                world.npcs[i].items_deposited_today += 1;
                world.npcs[i].money += 1.5;
                world.npcs[i].fitness_money_earned += 1.5;
            } else if no_progress {
                // Can't reach dumpster — drop item and try another
                world.npcs[i].interaction_target = None;
                path_failure(world, i, net);
            }
        } else {
            // No dumpster found — wander to find one (pick target periodically, not every frame)
            world.npcs[i].job_timer += dt;
            if world.npcs[i].job_timer > 8.0 || world.npcs[i].target_x == 0.0 {
                world.npcs[i].job_timer = 0.0;
                pick_wander(world, i, net);
                // Retry dumpster search after moving
                let best = find_nearest_interactible(
                    world,
                    world.npcs[i].x,
                    world.npcs[i].z,
                    InteractibleKind::Dumpster,
                    world.npcs[i].home_idx,
                );
                world.npcs[i].interaction_target = best;
            }
            let tx = world.npcs[i].target_x;
            let tz = world.npcs[i].target_z;
            npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
        }
    } else {
        // Find items near roads (skip if on cooldown from stuck recovery)
        if world.npcs[i].target_item.is_none() {
            let item_idx = find_best_item(world, i);
            if let Some(idx) = item_idx {
                world.npcs[i].target_item = Some(idx);
                world.items[idx].claimed_by = Some(i);
                world.npcs[i].target_x = world.items[idx].x;
                world.npcs[i].target_z = world.items[idx].z;
                world.npcs[i].stuck_timer = 0.0; // fresh start for new target
                world.npcs[i].job_timer = 0.0;
            } else {
                world.npcs[i].find_item_failures += 1;
                pick_wander(world, i, net);
            }
        }

        if let Some(item_idx) = world.npcs[i].target_item {
            if item_idx < world.items.len()
                && world.items[item_idx].active
                && !world.items[item_idx].falling
            {
                let ix = world.items[item_idx].x;
                let iz = world.items[item_idx].z;
                // Try driving if item is far and car is nearby
                if try_drive_to_target(world, i, ix, iz, terrain, net) {
                    return;
                }
                world.npcs[i].job_timer += dt;
                let no_progress = world.npcs[i].job_timer > 15.0;
                let remaining = npc_walk_toward(world, i, ix, iz, net, terrain, dt, walk_grid);
                if remaining < NPC_PICKUP_DIST {
                    world.items[item_idx].active = false;
                    world.items[item_idx].claimed_by = None;
                    world.npcs[i].carrying_item = true;
                    world.npcs[i].target_item = None;
                    world.npcs[i].fitness_items_picked += 1;
                } else if no_progress {
                    world.items[item_idx].claimed_by = None;
                    world.npcs[i].target_item = None;
                    path_failure(world, i, net);
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
            world.npcs[i].job_timer += dt;
            let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
            if remaining < 2.0 {
                world.npcs[i].job_timer = 0.0;
                pick_wander(world, i, net);
            } else if world.npcs[i].job_timer > 15.0 {
                // Stuck wandering — just pick a new wander target instead of escalating
                world.npcs[i].job_timer = 0.0;
                pick_wander(world, i, net);
            }
        }
    }
}

fn npc_work_taxi(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
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
                npc_walk_toward(
                    world,
                    i,
                    world.vehicles[car_idx].x,
                    world.vehicles[car_idx].z,
                    net,
                    terrain,
                    dt,
                    walk_grid,
                );
            }
        }
    } else {
        // Wander on foot
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
        if remaining < 3.0 {
            pick_wander(world, i, net);
        }
    }
}

fn npc_work_delivery(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    let npc = &world.npcs[i];

    if npc.carrying_item {
        // Drive/walk to target building
        let tx = world.npcs[i].job_target_x;
        let tz = world.npcs[i].job_target_z;
        world.npcs[i].job_timer += dt;
        let no_progress = world.npcs[i].job_timer > 15.0;
        // Delivery carrying → can't drive (carrying_item check in try_drive), just walk
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
        if remaining < 3.0 {
            world.npcs[i].carrying_item = false;
            world.npcs[i].items_deposited_today += 1;
            world.npcs[i].money += 2.0;
            world.npcs[i].fitness_money_earned += 2.0;
            world.npcs[i].job_timer = 0.0;
            // Pick new source
            pick_random_building_target(world, i);
        } else if no_progress {
            // Can't reach destination — pick new one
            world.npcs[i].carrying_item = false;
            pick_random_building_target(world, i);
            path_failure(world, i, net);
        }
    } else {
        // Drive/walk to source building to pick up
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        if try_drive_to_target(world, i, tx, tz, terrain, net) {
            return;
        }
        world.npcs[i].job_timer += dt;
        let no_progress = world.npcs[i].job_timer > 15.0;
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
        if remaining < 3.0 {
            world.npcs[i].carrying_item = true;
            world.npcs[i].fitness_items_picked += 1;
            world.npcs[i].job_timer = 0.0;
            // Pick delivery destination
            if !world.buildings.is_empty() {
                let dest = world.npcs[i].rng.next() as usize % world.buildings.len();
                world.npcs[i].job_target_x = world.buildings[dest].x;
                world.npcs[i].job_target_z = world.buildings[dest].z;
            }
        } else if no_progress {
            pick_random_building_target(world, i);
            path_failure(world, i, net);
        }
    }
}

fn npc_work_mail(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    // Visit mailbox interactibles in sequence
    world.npcs[i].job_timer += dt;

    if world.npcs[i].interaction_target.is_none() {
        // Wait before finding next mailbox so we walk away from the last one
        if world.npcs[i].job_timer < 5.0 {
            let tx = world.npcs[i].target_x;
            let tz = world.npcs[i].target_z;
            npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
            return;
        }
        let best = find_nearest_interactible(
            world,
            world.npcs[i].x,
            world.npcs[i].z,
            InteractibleKind::Mailbox,
            world.npcs[i].home_idx,
        );
        world.npcs[i].interaction_target = best;
    }

    if let Some(mi) = world.npcs[i].interaction_target {
        let mx = world.interactibles[mi].x;
        let mz = world.interactibles[mi].z;
        if try_drive_to_target(world, i, mx, mz, terrain, net) {
            return;
        }
        let no_progress = world.npcs[i].job_timer > 25.0;
        let remaining = npc_walk_toward(world, i, mx, mz, net, terrain, dt, walk_grid);
        if remaining < 2.0 {
            // "Deliver" to this mailbox
            world.npcs[i].job_timer = 0.0;
            world.npcs[i].money += 1.0;
            world.npcs[i].fitness_money_earned += 1.0;
            world.npcs[i].items_deposited_today += 1;
            world.npcs[i].interaction_target = None;
            pick_wander(world, i, net); // walk away before finding next
        } else if no_progress {
            world.npcs[i].interaction_target = None;
            path_failure(world, i, net);
        }
    } else {
        pick_wander(world, i, net);
        let tx = world.npcs[i].target_x;
        let tz = world.npcs[i].target_z;
        npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
    }
}

// ---- Emergency Jobs ----

fn npc_work_paramedic(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
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
            if j == i {
                continue;
            }
            if world.npcs[j].stuck_timer < 5.0 {
                continue;
            }
            if world.npcs[j].state == NpcState::Sleeping {
                continue;
            }
            let d = dist_sq_2d(
                world.npcs[j].x,
                world.npcs[j].z,
                world.npcs[i].x,
                world.npcs[i].z,
            );
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
    if try_drive_to_target(world, i, tx, tz, terrain, net) {
        return;
    }
    world.npcs[i].job_timer += dt;
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
    if world.npcs[i].job_timer > 20.0 && remaining > 2.0 {
        path_failure(world, i, net);
    } else if remaining < 2.0 {
        // "Treat" nearby NPCs with high stuck_timer
        let n = world.npcs.len();
        for j in 0..n {
            if j == i {
                continue;
            }
            if dist_sq_2d(
                world.npcs[j].x,
                world.npcs[j].z,
                world.npcs[i].x,
                world.npcs[i].z,
            ) < 9.0
                && world.npcs[j].stuck_timer > 3.0
            {
                world.npcs[j].stuck_timer = 0.0;
                world.npcs[i].money += 2.0;
                world.npcs[i].fitness_money_earned += 2.0;
            }
        }
        pick_wander(world, i, net);
    }
}

fn npc_work_firefighter(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    // Patrol between fire hydrants
    if world.npcs[i].interaction_target.is_none() {
        let best = find_nearest_interactible(
            world,
            world.npcs[i].x,
            world.npcs[i].z,
            InteractibleKind::FireHydrant,
            world.npcs[i].home_idx,
        );
        world.npcs[i].interaction_target = best;
    }

    if let Some(hi) = world.npcs[i].interaction_target {
        let hx = world.interactibles[hi].x;
        let hz = world.interactibles[hi].z;
        if try_drive_to_target(world, i, hx, hz, terrain, net) {
            return;
        }
        world.npcs[i].job_timer += dt;
        let remaining = npc_walk_toward(world, i, hx, hz, net, terrain, dt, walk_grid);
        if world.npcs[i].job_timer > 20.0 && remaining > 2.0 {
            world.npcs[i].interaction_target = None;
            world.npcs[i].job_timer = 0.0;
            path_failure(world, i, net);
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
        npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
    }
}

fn npc_work_police(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    world.npcs[i].job_timer += dt;

    // Scan for wanted NPCs within 50m (rescan every 2s or when no target)
    if world.npcs[i].police_target.is_none() || world.npcs[i].job_timer > 2.0 {
        let mut best_d = 50.0 * 50.0;
        let mut best_j = None;
        let n = world.npcs.len();
        for j in 0..n {
            if j == i {
                continue;
            }
            if !world.npcs[j].wanted {
                continue;
            }
            if world.npcs[j].state == NpcState::Sleeping {
                continue;
            }
            let d2 = dist_sq_2d(
                world.npcs[j].x,
                world.npcs[j].z,
                world.npcs[i].x,
                world.npcs[i].z,
            );
            if d2 < best_d {
                best_d = d2;
                best_j = Some(j);
            }
        }
        if best_j.is_some() {
            world.npcs[i].police_target = best_j;
            world.npcs[i].job_timer = 0.0;
        }
    }

    if let Some(target) = world.npcs[i].police_target {
        // Validate target still exists and is wanted
        if target >= world.npcs.len()
            || !world.npcs[target].wanted
            || world.npcs[target].state == NpcState::Sleeping
        {
            world.npcs[i].police_target = None;
            return;
        }

        let tx = world.npcs[target].x;
        let tz = world.npcs[target].z;
        if try_drive_to_target(world, i, tx, tz, terrain, net) {
            return;
        }
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);

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
        let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
        if remaining < 3.0 {
            pick_wander(world, i, net);
        } else if world.npcs[i].job_timer > 15.0 {
            world.npcs[i].job_timer = 0.0;
            path_failure(world, i, net);
        }
    }
}

// ---- Commerce Jobs ----

fn npc_work_vendor(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    // Stand at intersection, earn money when NPCs come near
    world.npcs[i].job_timer += dt;

    // Walk to a road node if not already near one
    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    let dx = tx - world.npcs[i].x;
    let dz = tz - world.npcs[i].z;
    let dist = (dx * dx + dz * dz).sqrt();

    if dist > 3.0 {
        npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
    } else {
        // Stand still, check for nearby NPCs every few seconds
        world.npcs[i].walk_phase = 0.0;
        if world.npcs[i].job_timer > 30.0 {
            // Check for nearby NPCs (customers)
            let n = world.npcs.len();
            for j in 0..n {
                if j == i {
                    continue;
                }
                if world.npcs[j].state != NpcState::Working {
                    continue;
                }
                if world.npcs[j].job == NpcJob::StreetVendor {
                    continue;
                }
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

fn npc_work_mechanic(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    // Walk to parked vehicles, stand near them
    world.npcs[i].job_timer += dt;

    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    if try_drive_to_target(world, i, tx, tz, terrain, net) {
        return;
    }
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);

    if world.npcs[i].job_timer > 20.0 && remaining > 2.0 {
        path_failure(world, i, net);
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
                if v.speed.abs() > 0.5 {
                    continue;
                } // skip moving vehicles
                let d = dist_sq_2d(v.x, v.z, world.npcs[i].x, world.npcs[i].z);
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

fn npc_work_construction(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    // Walk to dockyard area, do hammering animation
    let tx = world.npcs[i].job_target_x;
    let tz = world.npcs[i].job_target_z;

    // Set target to dockyard if not set or reached
    if (tx == 0.0 && tz == 0.0) || dist_sq_2d(tx, tz, world.npcs[i].x, world.npcs[i].z) < 9.0 {
        world.npcs[i].job_target_x = world.npcs[i].rng.range(-30.0, 30.0);
        world.npcs[i].job_target_z = DOCK_Z_START + world.npcs[i].rng.range(5.0, 15.0);
    }

    let jtx = world.npcs[i].job_target_x;
    let jtz = world.npcs[i].job_target_z;
    if try_drive_to_target(world, i, jtx, jtz, terrain, net) {
        return;
    }
    let remaining = npc_walk_toward(world, i, jtx, jtz, net, terrain, dt, walk_grid);
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

fn npc_work_fisherman(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    // Walk to fishing pier, stand at edge, earn money periodically
    let pier_x = match world.npcs[i].rng.clone().next() % 3 {
        0 => -30.0,
        1 => 0.0,
        _ => 30.0,
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
    if try_drive_to_target(world, i, tx, tz, terrain, net) {
        return;
    }
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
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

fn npc_work_farmer(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    // Walk back and forth in rows on terrain
    world.npcs[i].job_timer += dt;

    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
    if remaining < 2.0 {
        // Pick next row point
        let row_offset = if world.npcs[i].job_timer as u32 % 2 == 0 {
            8.0
        } else {
            -8.0
        };
        world.npcs[i].target_x = world.npcs[i].x + row_offset;
        world.npcs[i].target_z = world.npcs[i].z + world.npcs[i].rng.range(-1.0, 1.0);
        world.npcs[i].target_x = world.npcs[i]
            .target_x
            .clamp(-WORLD_HALF + 10.0, WORLD_HALF - 10.0);
        world.npcs[i].target_z = world.npcs[i]
            .target_z
            .clamp(-WORLD_HALF + 10.0, DOCK_Z_START - 5.0);

        if world.npcs[i].job_timer > 60.0 {
            world.npcs[i].money += 1.0;
            world.npcs[i].fitness_money_earned += 1.0;
            world.npcs[i].job_timer = 0.0;
        }
    }
}

fn npc_work_lumberjack(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    // Walk to nearest tree, stand near it for 15s, earn money
    world.npcs[i].job_timer += dt;

    let tx = world.npcs[i].target_x;
    let tz = world.npcs[i].target_z;
    if try_drive_to_target(world, i, tx, tz, terrain, net) {
        return;
    }
    let remaining = npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);

    if remaining < 2.0 {
        // Chopping animation
        world.npcs[i].walk_phase += dt * 12.0;
        if world.npcs[i].job_timer > 10.0 {
            world.npcs[i].money += 3.0;
            world.npcs[i].fitness_money_earned += 3.0;
            world.npcs[i].job_timer = 0.0;
            // Find next tree
            let mut best_dist = f32::MAX;
            let mut best_x = world.npcs[i].x;
            let mut best_z = world.npcs[i].z;
            for t in &world.trees {
                let d = dist_sq_2d(t.x, t.z, world.npcs[i].x, world.npcs[i].z);
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

fn npc_work_scavenger(
    world: &mut WorldData,
    i: usize,
    net: &mut RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    // Visit dumpster interactibles, search them for loot
    world.npcs[i].job_timer += dt;

    if world.npcs[i].interaction_target.is_none() {
        let best = find_nearest_interactible(
            world,
            world.npcs[i].x,
            world.npcs[i].z,
            InteractibleKind::Dumpster,
            world.npcs[i].home_idx,
        );
        world.npcs[i].interaction_target = best;
    }

    if let Some(di) = world.npcs[i].interaction_target {
        let dx = world.interactibles[di].x;
        let dz = world.interactibles[di].z;
        if try_drive_to_target(world, i, dx, dz, terrain, net) {
            return;
        }
        let remaining = npc_walk_toward(world, i, dx, dz, net, terrain, dt, walk_grid);
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
        npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
    }
}

// ---- Helpers ----

/// Check if the straight-line path from (ax,az) to (bx,bz) crosses a river.
/// Only checks river — NPCs can walk around buildings via detour/stuck recovery,
/// but rivers are truly impassable. Samples every ~8m along the path.
fn path_clear(world: &WorldData, ax: f32, az: f32, bx: f32, bz: f32, home_idx: usize) -> bool {
    let dx = bx - ax;
    let dz = bz - az;
    let dist = (dx * dx + dz * dz).sqrt();
    let steps = ((dist / 6.0) as usize).clamp(2, 10);
    for step in 1..=steps {
        let t = step as f32 / (steps + 1) as f32;
        let sx = ax + dx * t;
        let sz = az + dz * t;
        if on_river_not_bridge(sx, sz, &world.river_segments, &world.bridges) {
            return false;
        }
        // Reject paths through large buildings
        for (bi, b) in world.buildings.iter().enumerate() {
            if bi == home_idx {
                continue;
            }
            if b.w + b.d < 6.0 {
                continue;
            }
            if sx > b.x - b.w * 0.5
                && sx < b.x + b.w * 0.5
                && sz > b.z - b.d * 0.5
                && sz < b.z + b.d * 0.5
            {
                return false;
            }
        }
        // Reject paths through walls (linear barriers)
        for w in &world.walls {
            if sx > w.x - w.hw - 0.5
                && sx < w.x + w.hw + 0.5
                && sz > w.z - w.hd - 0.5
                && sz < w.z + w.hd + 0.5
            {
                return false;
            }
        }
        // Reject paths through large rocks
        for r in &world.rocks {
            if r.size < 1.0 {
                continue;
            }
            let rdx = sx - r.x;
            let rdz = sz - r.z;
            if rdx * rdx + rdz * rdz < (r.size + 0.5) * (r.size + 0.5) {
                return false;
            }
        }
    }
    true
}

fn find_best_item(world: &WorldData, npc_idx: usize) -> Option<usize> {
    let npc = &world.npcs[npc_idx];
    let home = npc.home_idx;
    let mut best_dist = f32::MAX;
    let mut best_idx = None;
    let mut fallback_dist = f32::MAX;
    let mut fallback_idx = None;
    let max_range = 120.0;
    let max_dist_sq = max_range * max_range;
    for (idx, item) in world.items.iter().enumerate() {
        if !item.active || item.falling {
            continue;
        }
        if let Some(claimer) = item.claimed_by {
            if claimer != npc_idx {
                continue;
            }
        }
        let dist = dist_sq_2d(item.x, item.z, npc.x, npc.z);
        if dist > max_dist_sq {
            continue;
        }
        if path_clear(world, npc.x, npc.z, item.x, item.z, home) {
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(idx);
            }
        } else if best_idx.is_none() && dist < fallback_dist {
            // Track closest item without clear path — NPC can detour around obstacles
            fallback_dist = dist;
            fallback_idx = Some(idx);
        }
    }
    best_idx.or(fallback_idx)
}

fn find_nearest_with_fallback<T, F, G>(
    world: &WorldData,
    x: f32,
    z: f32,
    home_idx: usize,
    items: &[T],
    skip: F,
    pos: G,
) -> Option<usize>
where
    F: Fn(&T) -> bool,
    G: Fn(&T) -> (f32, f32),
{
    let mut best_dist = f32::MAX;
    let mut best_idx = None;
    let mut fallback_dist = f32::MAX;
    let mut fallback_idx = None;
    for (idx, item) in items.iter().enumerate() {
        if skip(item) {
            continue;
        }
        let (ix, iz) = pos(item);
        let dist = dist_sq_2d(ix, iz, x, z);
        if path_clear(world, x, z, ix, iz, home_idx) {
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(idx);
            }
        } else if best_idx.is_none() && dist < fallback_dist {
            fallback_dist = dist;
            fallback_idx = Some(idx);
        }
    }
    best_idx.or(fallback_idx)
}

fn find_nearest_bin(world: &WorldData, x: f32, z: f32, home_idx: usize) -> Option<usize> {
    find_nearest_with_fallback(
        world,
        x,
        z,
        home_idx,
        &world.trash_bins,
        |bin| bin.carried_by.is_some(),
        |bin| (bin.x, bin.z),
    )
}

fn find_nearest_interactible(
    world: &WorldData,
    x: f32,
    z: f32,
    kind: InteractibleKind,
    home_idx: usize,
) -> Option<usize> {
    find_nearest_with_fallback(
        world,
        x,
        z,
        home_idx,
        &world.interactibles,
        |inter| inter.kind != kind,
        |inter| (inter.x, inter.z),
    )
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
            let d = dist_sq_2d(node[0], node[1], nx, nz);
            // Must be 5-200m away, not on river
            if d > 25.0
                && d < 40000.0
                && !on_river_not_bridge(node[0], node[1], &world.river_segments, &world.bridges)
            {
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
    if world.buildings.is_empty() {
        return;
    }
    let bi = world.npcs[i].rng.next() as usize % world.buildings.len();
    world.npcs[i].target_x = world.buildings[bi].x;
    world.npcs[i].target_z = world.buildings[bi].z;
}
