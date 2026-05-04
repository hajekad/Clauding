//! Player-side job acceptance, progress tracking, and completion payout.

use crate::state::*;

/// Update player job progress (called each fixed tick)
pub fn sys_player_job(game: &mut GameState, _dt: f32) {
    if game.player.active_job.job_type == PlayerJobType::None {
        return;
    }

    // Decrement timer
    game.player.active_job.time_remaining -= _dt;
    if game.player.active_job.time_remaining <= 0.0 {
        // Job expired — fail
        game.player.active_job = PlayerJob::none();
        return;
    }

    // Check completion
    if game.player.active_job.items_done >= game.player.active_job.items_needed {
        // Pay bonus
        let bonus = match game.player.active_job.job_type {
            PlayerJobType::GarbageCollector => 10.0,
            PlayerJobType::DeliveryCourier => 15.0,
            PlayerJobType::MailCarrier => 12.0,
            PlayerJobType::Fisherman => 20.0,
            PlayerJobType::Lumberjack => 18.0,
            _ => 10.0,
        };
        game.player.money += bonus;
        game.player.bank_balance += bonus;
        game.player.active_job = PlayerJob::none();
    }
}

/// Accept a job from the phone booth menu
pub fn accept_job(game: &mut GameState) {
    let cursor = game.player.job_menu_cursor;
    let (job_type, items_needed, time) = match cursor {
        0 => (PlayerJobType::GarbageCollector, 5, 300.0),
        1 => (PlayerJobType::TaxiDriver, 3, 300.0),
        2 => (PlayerJobType::DeliveryCourier, 4, 300.0),
        3 => (PlayerJobType::MailCarrier, 6, 300.0),
        4 => (PlayerJobType::Paramedic, 3, 300.0),
        5 => (PlayerJobType::Firefighter, 3, 300.0),
        6 => (PlayerJobType::PolicePatrol, 3, 300.0),
        7 => (PlayerJobType::StreetVendor, 5, 300.0),
        8 => (PlayerJobType::Mechanic, 3, 300.0),
        9 => (PlayerJobType::ConstructionWorker, 3, 300.0),
        10 => (PlayerJobType::Fisherman, 4, 300.0),
        11 => (PlayerJobType::Farmer, 5, 300.0),
        12 => (PlayerJobType::Lumberjack, 3, 300.0),
        13 => (PlayerJobType::Scavenger, 5, 300.0),
        _ => return,
    };
    game.player.active_job = PlayerJob {
        job_type,
        time_remaining: time,
        items_done: 0,
        items_needed,
    };
}

/// Update interactible cooldowns
pub fn sys_interactibles_update(world: &mut WorldData, dt: f32) {
    for inter in &mut world.interactibles {
        if inter.cooldown > 0.0 {
            inter.cooldown -= dt;
        }
        if inter.state_val > 0.0 {
            inter.state_val -= dt;
        }
    }
}
