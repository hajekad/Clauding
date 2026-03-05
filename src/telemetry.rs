// Telemetry: periodic game state dump to /tmp/clauding_state.txt

use crate::state::*;
use std::fmt::Write;

const DUMP_INTERVAL: u64 = 300; // ~5 seconds at 60 ticks/sec
const MAP_SIZE: usize = 50;

pub fn sys_telemetry(game: &GameState) {
    if game.frame_counter % DUMP_INTERVAL != 0 { return; }

    let mut s = String::with_capacity(8192);

    // Header
    let hour = game.time_of_day as u32;
    let minute = ((game.time_of_day - hour as f32) * 60.0) as u32;
    let _ = writeln!(s, "=== CLAUDING STATE ===");
    let _ = writeln!(s, "Day {} | {:02}:{:02} | frame {} | tick {}", game.day_count, hour, minute, game.frame_counter, game.frame_counter);
    let _ = writeln!(s);

    // Player
    let p = &game.player;
    let _ = writeln!(s, "--- PLAYER ---");
    let _ = writeln!(s, "pos=({:.1}, {:.1}, {:.1}) rot={:.2}", p.x, p.y, p.z, p.rot_y);
    let _ = writeln!(s, "health={:.0} stamina={:.0} money=${:.0} bank=${:.0}", p.health, p.stamina, p.money, p.bank_balance);
    let _ = write!(s, "flags:");
    if p.in_vehicle.is_some() { let _ = write!(s, " in_vehicle({})", p.in_vehicle.unwrap()); }
    if p.carrying_item { let _ = write!(s, " carrying_item"); }
    if p.carrying_bin.is_some() { let _ = write!(s, " carrying_bin({})", p.carrying_bin.unwrap()); }
    if p.sitting { let _ = write!(s, " sitting"); }
    if p.job_menu_open { let _ = write!(s, " job_menu_open"); }
    if p.sprinting { let _ = write!(s, " sprinting"); }
    let _ = writeln!(s);
    if p.active_job.job_type != PlayerJobType::None {
        let _ = writeln!(s, "job: {} items={}/{} time={:.0}s",
            player_job_name(p.active_job.job_type),
            p.active_job.items_done, p.active_job.items_needed,
            p.active_job.time_remaining);
    }
    let _ = writeln!(s);

    // ASCII minimap
    let _ = writeln!(s, "--- MAP (50x50, [-100,100]) ---");
    let w = &game.world;

    // Build grid
    let mut grid = [[b'.'; MAP_SIZE]; MAP_SIZE];

    // Buildings
    for b in &w.buildings {
        let (gx, gz) = world_to_grid(b.x, b.z);
        if gx < MAP_SIZE && gz < MAP_SIZE { grid[gz][gx] = b'#'; }
    }

    // Water (dockyard area z > 70)
    for row in 0..MAP_SIZE {
        for col in 0..MAP_SIZE {
            let (_, wz) = grid_to_world(col, row);
            if wz > DOCK_Z_START { grid[row][col] = b'~'; }
        }
    }

    // Vehicles
    for v in &w.vehicles {
        let (gx, gz) = world_to_grid(v.x, v.z);
        if gx < MAP_SIZE && gz < MAP_SIZE { grid[gz][gx] = b'v'; }
    }

    // Items (active)
    for item in &w.items {
        if !item.active { continue; }
        let (gx, gz) = world_to_grid(item.x, item.z);
        if gx < MAP_SIZE && gz < MAP_SIZE { grid[gz][gx] = b'*'; }
    }

    // NPCs
    for npc in &w.npcs {
        let (gx, gz) = world_to_grid(npc.x, npc.z);
        if gx >= MAP_SIZE || gz >= MAP_SIZE { continue; }
        grid[gz][gx] = match npc.state {
            NpcState::Working => b'W',
            NpcState::Sleeping => b'S',
            NpcState::Driving => b'D',
            NpcState::Interacting => b'I',
            NpcState::HomeTask => b'H',
            NpcState::GoingToWork | NpcState::GoingHome => b'G',
        };
    }

    // Player on top
    let (px, pz) = world_to_grid(game.player.x, game.player.z);
    if px < MAP_SIZE && pz < MAP_SIZE { grid[pz][px] = b'@'; }

    for row in &grid {
        let _ = s.push_str(std::str::from_utf8(row).unwrap_or(""));
        let _ = s.push('\n');
    }
    let _ = writeln!(s, "Legend: @=player W=working S=sleeping D=driving I=interacting H=home G=going v=vehicle *=item #=building ~=water");
    let _ = writeln!(s);

    // NPC state summary
    let _ = writeln!(s, "--- NPC SUMMARY ({} total) ---", w.npcs.len());
    let mut state_counts = [0u32; 7];
    for npc in &w.npcs {
        let idx = match npc.state {
            NpcState::Sleeping => 0,
            NpcState::HomeTask => 1,
            NpcState::GoingToWork => 2,
            NpcState::Working => 3,
            NpcState::GoingHome => 4,
            NpcState::Driving => 5,
            NpcState::Interacting => 6,
        };
        state_counts[idx] += 1;
    }
    let _ = writeln!(s, "Sleeping={} HomeTask={} GoingToWork={} Working={} GoingHome={} Driving={} Interacting={}",
        state_counts[0], state_counts[1], state_counts[2], state_counts[3],
        state_counts[4], state_counts[5], state_counts[6]);

    // Job counts
    let mut job_counts = [0u32; NPC_JOB_COUNT];
    for npc in &w.npcs {
        let idx = npc_job_index(npc.job);
        if idx < NPC_JOB_COUNT { job_counts[idx] += 1; }
    }
    let _ = write!(s, "Jobs:");
    let job_names = ["Collector", "Garbage", "Taxi", "Delivery", "Mail", "Paramedic",
        "Firefighter", "Police", "Vendor", "Mechanic", "Construction", "Fisherman",
        "Farmer", "Lumberjack", "Scavenger"];
    for (i, name) in job_names.iter().enumerate() {
        if job_counts[i] > 0 {
            let _ = write!(s, " {}={}", name, job_counts[i]);
        }
    }
    let _ = writeln!(s);
    let _ = writeln!(s);

    // Per-NPC detail
    let _ = writeln!(s, "--- NPC DETAIL ---");
    for (i, npc) in w.npcs.iter().enumerate() {
        let _ = writeln!(s, "[{:2}] {:12} {:10} pos=({:6.1},{:6.1}) money=${:<5.0} carry={} bin={} dep={}",
            i,
            npc_job_name(npc.job),
            npc_state_name(npc.state),
            npc.x, npc.z,
            npc.money,
            npc.carrying_item,
            npc.carrying_bin.is_some(),
            npc.items_deposited_today);
    }
    let _ = writeln!(s);

    // Vehicle summary
    let _ = writeln!(s, "--- VEHICLES ({}) ---", w.vehicles.len());
    for (i, v) in w.vehicles.iter().enumerate() {
        if v.speed.abs() > 0.1 || v.occupied || v.ai_active {
            let _ = writeln!(s, "[{:2}] pos=({:6.1},{:6.1}) spd={:5.1} occupied={} ai={} owner={:?}",
                i, v.x, v.z, v.speed, v.occupied, v.ai_active, v.owner_npc);
        }
    }
    let _ = writeln!(s);

    // Interactibles (only active ones)
    let active_inter: Vec<_> = w.interactibles.iter().enumerate()
        .filter(|(_, inter)| inter.cooldown > 0.0 || inter.state_val > 0.0)
        .collect();
    if !active_inter.is_empty() {
        let _ = writeln!(s, "--- ACTIVE INTERACTIBLES ---");
        for (i, inter) in &active_inter {
            let _ = writeln!(s, "[{:2}] {:16} cd={:5.1} state={:5.1} pos=({:.1},{:.1})",
                i, interactible_name(inter.kind), inter.cooldown, inter.state_val, inter.x, inter.z);
        }
        let _ = writeln!(s);
    }

    // World stats
    let active_items = w.items.iter().filter(|it| it.active).count();
    let falling_items = w.items.iter().filter(|it| it.falling).count();
    let total_npc_money: f32 = w.npcs.iter().map(|n| n.money).sum();
    let total_deposited: u32 = w.npcs.iter().map(|n| n.items_deposited_today).sum();
    let _ = writeln!(s, "--- WORLD STATS ---");
    let _ = writeln!(s, "active_items={} falling={} total_npc_money=${:.0} deposited_today={}",
        active_items, falling_items, total_npc_money, total_deposited);

    let _ = std::fs::write("/tmp/clauding_state.txt", s);
}

fn world_to_grid(x: f32, z: f32) -> (usize, usize) {
    let gx = ((x + WORLD_HALF) / WORLD_SIZE * MAP_SIZE as f32) as usize;
    let gz = ((z + WORLD_HALF) / WORLD_SIZE * MAP_SIZE as f32) as usize;
    (gx.min(MAP_SIZE - 1), gz.min(MAP_SIZE - 1))
}

fn grid_to_world(_col: usize, row: usize) -> (f32, f32) {
    let wz = (row as f32 / MAP_SIZE as f32) * WORLD_SIZE - WORLD_HALF;
    (0.0, wz)
}

fn npc_state_name(s: NpcState) -> &'static str {
    match s {
        NpcState::Sleeping => "Sleeping",
        NpcState::HomeTask => "HomeTask",
        NpcState::GoingToWork => "GoToWork",
        NpcState::Working => "Working",
        NpcState::GoingHome => "GoHome",
        NpcState::Driving => "Driving",
        NpcState::Interacting => "Interacting",
    }
}

fn npc_job_name(j: NpcJob) -> &'static str {
    match j {
        NpcJob::Collector => "Collector",
        NpcJob::GarbageCollector => "Garbage",
        NpcJob::TaxiDriver => "Taxi",
        NpcJob::DeliveryCourier => "Delivery",
        NpcJob::MailCarrier => "Mail",
        NpcJob::Paramedic => "Paramedic",
        NpcJob::Firefighter => "Firefighter",
        NpcJob::PolicePatrol => "Police",
        NpcJob::StreetVendor => "Vendor",
        NpcJob::Mechanic => "Mechanic",
        NpcJob::ConstructionWorker => "Construction",
        NpcJob::Fisherman => "Fisherman",
        NpcJob::Farmer => "Farmer",
        NpcJob::Lumberjack => "Lumberjack",
        NpcJob::Scavenger => "Scavenger",
    }
}

fn npc_job_index(j: NpcJob) -> usize {
    match j {
        NpcJob::Collector => 0,
        NpcJob::GarbageCollector => 1,
        NpcJob::TaxiDriver => 2,
        NpcJob::DeliveryCourier => 3,
        NpcJob::MailCarrier => 4,
        NpcJob::Paramedic => 5,
        NpcJob::Firefighter => 6,
        NpcJob::PolicePatrol => 7,
        NpcJob::StreetVendor => 8,
        NpcJob::Mechanic => 9,
        NpcJob::ConstructionWorker => 10,
        NpcJob::Fisherman => 11,
        NpcJob::Farmer => 12,
        NpcJob::Lumberjack => 13,
        NpcJob::Scavenger => 14,
    }
}

fn player_job_name(j: PlayerJobType) -> &'static str {
    match j {
        PlayerJobType::None => "None",
        PlayerJobType::GarbageCollector => "Garbage",
        PlayerJobType::TaxiDriver => "Taxi",
        PlayerJobType::DeliveryCourier => "Delivery",
        PlayerJobType::MailCarrier => "Mail",
        PlayerJobType::Paramedic => "Paramedic",
        PlayerJobType::Firefighter => "Firefighter",
        PlayerJobType::PolicePatrol => "Police",
        PlayerJobType::StreetVendor => "Vendor",
        PlayerJobType::Mechanic => "Mechanic",
        PlayerJobType::ConstructionWorker => "Construction",
        PlayerJobType::Fisherman => "Fisherman",
        PlayerJobType::Farmer => "Farmer",
        PlayerJobType::Lumberjack => "Lumberjack",
        PlayerJobType::Scavenger => "Scavenger",
    }
}

fn interactible_name(k: InteractibleKind) -> &'static str {
    match k {
        InteractibleKind::VendingMachine => "VendingMachine",
        InteractibleKind::ParkBench => "ParkBench",
        InteractibleKind::Dumpster => "Dumpster",
        InteractibleKind::Atm => "ATM",
        InteractibleKind::PhoneBooth => "PhoneBooth",
        InteractibleKind::FireHydrant => "FireHydrant",
        InteractibleKind::NewspaperStand => "NewspaperStand",
        InteractibleKind::Mailbox => "Mailbox",
        InteractibleKind::Payphone => "Payphone",
    }
}
