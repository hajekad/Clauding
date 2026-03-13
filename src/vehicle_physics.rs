// Vehicle physics step: suspension → tire forces → rigid body integration
//
// Bridges the gap between the AI/player driving inputs and the rigid body solver.
// Each frame: compute suspension forces → tire forces → integrate body → sync back to Vehicle fields.

use crate::math::*;
use crate::state::{Vehicle, Terrain, RoadNetwork};
use crate::physics;
use crate::tire;
use crate::suspension;
use crate::material;

/// Step physics for a single vehicle. Called once per fixed timestep.
/// This replaces the old direct position/speed manipulation with force-based movement.
pub fn step_vehicle_physics(v: &mut Vehicle, terrain: &Terrain, road_network: &RoadNetwork, dt: f32) {
    // Skip physics for parked vehicles with no driver
    if v.parked && !v.occupied && !v.ai_active {
        // Just sync body position from legacy fields
        v.body.pos = [v.x, v.y, v.z];
        v.body.quat = quat_from_rot_y(v.rot_y);
        v.body.vel = [0.0; 3];
        v.body.ang_vel = [0.0; 3];
        // Reset suspension so cold-start fires correctly on re-activation
        for s in &mut v.suspension {
            s.compression = 0.0;
            s.prev_compression = 0.0;
        }
        return;
    }

    // Cold-start: if suspension was fully unloaded (parked → active), initialize body Y
    // to suspension equilibrium height so springs don't launch the vehicle
    if v.suspension.iter().all(|s| s.compression == 0.0 && s.prev_compression == 0.0) {
        let ground_y = terrain.height_at(v.body.pos[0], v.body.pos[2]);
        let wr = v.wheels[0].radius;
        let local_y = v.wheels[0].local_pos[1];
        let equil_comp = (v.body.mass * 9.81 * 0.25) / v.suspension[0].params.spring_rate;
        let equil_length = v.suspension[0].params.rest_length - equil_comp;
        v.body.pos[1] = ground_y + wr + equil_length - local_y;
        v.body.vel[1] = 0.0;
        for s in &mut v.suspension {
            s.compression = equil_comp;
            s.prev_compression = equil_comp;
        }
    }

    // Deformation damage factors
    let dmg_engine = v.deformation.engine_factor();
    let dmg_steer = v.deformation.steering_factor();
    let dmg_grip = v.deformation.grip_factor();

    // Distribute drivetrain inputs to wheels (steering reduced by front-end damage)
    let steer = v.drivetrain.steer_input * v.drivetrain.max_steer * dmg_steer;
    v.wheels[0].steer_angle = steer; // FL
    v.wheels[1].steer_angle = steer; // FR
    v.wheels[2].steer_angle = 0.0;   // RL
    v.wheels[3].steer_angle = 0.0;   // RR

    // Electronic speed governor: forward + reverse
    let max_reverse_speed = 8.33; // 30 km/h reverse
    let throttle_input = v.drivetrain.throttle; // can be negative for reverse
    let governed_throttle = if throttle_input < 0.0 {
        // Reverse governor
        let rev_speed = (-v.speed).max(0.0);
        if rev_speed >= max_reverse_speed {
            0.0
        } else if rev_speed > max_reverse_speed * 0.9 {
            let t = (max_reverse_speed - rev_speed) / (max_reverse_speed * 0.1);
            throttle_input * t
        } else {
            throttle_input
        }
    } else if v.drivetrain.max_speed > 0.0 {
        let speed = v.speed.abs();
        if speed >= v.drivetrain.max_speed {
            0.0
        } else if speed > v.drivetrain.max_speed * 0.95 {
            // Smooth ramp-down in final 5% to avoid jerkiness
            let t = (v.drivetrain.max_speed - speed) / (v.drivetrain.max_speed * 0.05);
            throttle_input * t
        } else {
            throttle_input
        }
    } else {
        throttle_input
    };

    // Engine torque to driven wheels (reduced by engine bay damage)
    // Power-limited: at high wheel speed, cap torque so power doesn't exceed max_power
    let is_reverse = governed_throttle < 0.0;
    let effective_gear = if is_reverse { v.drivetrain.gear_ratio * 0.6 } else { v.drivetrain.gear_ratio };
    let base_torque = v.drivetrain.engine_torque * governed_throttle * effective_gear * dmg_engine;
    let avg_wheel_speed = {
        let sum: f32 = v.wheels.iter().map(|w| w.ang_vel.abs() * w.radius).sum();
        (sum * 0.25).max(0.1) // average ground speed from wheels, floor at 0.1 m/s
    };
    let power_at_wheels = (base_torque * avg_wheel_speed / v.wheels[0].radius).abs();
    let power_scale = if !is_reverse && power_at_wheels > v.drivetrain.max_power && v.drivetrain.max_power > 0.0 {
        v.drivetrain.max_power / power_at_wheels
    } else {
        1.0
    };
    let engine_torque = base_torque * power_scale;
    let per_wheel = engine_torque * 0.25; // 4WD split
    for w in &mut v.wheels {
        w.drive_torque = per_wheel;
    }

    // Brake torque with front/rear bias (S5: 350mm front discs, 330mm rear)
    // 60/40 front/rear split — front brakes do more work, matches weight transfer
    let brake_input = v.drivetrain.brake;
    let front_brake = 4200.0 * brake_input; // per front wheel (6-piston calipers)
    let rear_brake = 2800.0 * brake_input;  // per rear wheel (1-piston floating)
    v.wheels[0].brake_torque = front_brake; // FL
    v.wheels[1].brake_torque = front_brake; // FR
    v.wheels[2].brake_torque = rear_brake;  // RL
    v.wheels[3].brake_torque = rear_brake;  // RR

    // ABS enabled on all wheels by default
    for w in &mut v.wheels {
        w.abs = true;
    }

    // Handbrake: lock rear wheels only, bypass ABS (causes oversteer/drift)
    if v.drivetrain.handbrake {
        v.wheels[2].brake_torque = 6000.0;
        v.wheels[3].brake_torque = 6000.0;
        v.wheels[2].abs = false;
        v.wheels[3].abs = false;
    }

    // Engine braking: when off-throttle, engine creates drag through drivetrain
    if governed_throttle.abs() < 0.01 && v.speed.abs() > 1.0 {
        let engine_drag = 80.0; // N·m per wheel — compression braking
        for w in &mut v.wheels {
            if w.brake_torque < 10.0 {
                w.brake_torque = engine_drag;
            }
        }
    }

    // Get body orientation vectors
    let body_up = quat_up(v.body.quat);
    let body_fwd = quat_forward(v.body.quat);
    let body_right = quat_right(v.body.quat);

    // Base surface at vehicle center (used as fallback / reference)
    let center_surface = v.surface_override
        .unwrap_or_else(|| crate::world::surface_at(v.body.pos[0], v.body.pos[2], road_network));

    let mut total_force = [0.0f32; 3];
    let mut total_torque = [0.0f32; 3];

    // Process each wheel: suspension → tire forces (with deformation damage)
    for i in 0..4 {
        // Wheel attachment point in world space
        let local_attach = v.wheels[i].local_pos;
        let attach_world = v3_add(v.body.pos, quat_rotate(v.body.quat, local_attach));

        // Suspension force (spring rate scaled by structural damage at this corner)
        let dmg_susp = v.deformation.suspension_factor(i);
        let susp_force = suspension::compute_suspension(
            &mut v.suspension[i],
            &mut v.wheels[i],
            attach_world,
            body_up,
            terrain,
            dt,
        ) * dmg_susp;

        // Apply suspension force (upward on body at attachment point)
        let susp_force_vec = v3_scale(body_up, susp_force);
        total_force = v3_add(total_force, susp_force_vec);
        let r = v3_sub(attach_world, v.body.pos);
        total_torque = v3_add(total_torque, v3_cross(r, susp_force_vec));

        // Tire forces (only if wheel is on ground, grip reduced by damage)
        if v.wheels[i].on_ground && susp_force > 1.0 {
            // Per-wheel surface: sample at tire contact XZ, blend with center surface
            // At road edges, one side of the car may be on asphalt while the other is on grass
            let wheel_surface = v.surface_override
                .unwrap_or_else(|| crate::world::surface_at(attach_world[0], attach_world[2], road_network));
            let wheel_mat = if wheel_surface == center_surface {
                *material::material_for_surface(wheel_surface)
            } else {
                // Wheel straddles a surface boundary — blend materials for smooth transition
                material::combine_materials(
                    material::material_for_surface(center_surface),
                    material::material_for_surface(wheel_surface),
                )
            };
            let contact_vel = v.body.velocity_at(attach_world);
            let tire_force = tire::compute_tire_forces(
                &mut v.wheels[i],
                contact_vel,
                body_up,
                body_fwd,
                body_right,
                susp_force * dmg_grip, // reduced normal load → less grip
                &wheel_mat,
                dt,
            );
            total_force = v3_add(total_force, tire_force);
            total_torque = v3_add(total_torque, v3_cross(r, tire_force));
        }
    }

    // Aerodynamic drag: F = -0.5 × Cd × A × ρ × v² × v_hat
    // Cd=0.33, A=2.2m², ρ=1.225 → 0.5 × 0.33 × 2.2 × 1.225 ≈ 0.445
    let speed_sq = v3_dot(v.body.vel, v.body.vel);
    if speed_sq > 0.01 {
        let speed = speed_sq.sqrt();
        let drag_mag = 0.445 * speed_sq; // CdA_half_rho × v²
        let drag_force = v3_scale(v.body.vel, -drag_mag / speed);
        total_force = v3_add(total_force, drag_force);
    }

    // Apply accumulated forces to rigid body
    v.body.apply_force(total_force);
    v.body.torque_accum = v3_add(v.body.torque_accum, total_torque);

    // Integrate
    physics::integrate(&mut v.body, dt);

    // Ground contact enforcement — prevent falling through terrain
    let ground_y = terrain.height_at(v.body.pos[0], v.body.pos[2]);
    let min_y = ground_y + crate::state::VEHICLE_GROUND_OFFSET;
    if v.body.pos[1] < min_y {
        v.body.pos[1] = min_y;
        if v.body.vel[1] < 0.0 {
            v.body.vel[1] = 0.0;
        }
    }

    // Sync back to legacy Vehicle fields
    v.x = v.body.pos[0];
    v.y = v.body.pos[1];
    v.z = v.body.pos[2];

    // Extract rot_y from quaternion (yaw angle)
    let fwd_new = quat_forward(v.body.quat);
    v.rot_y = (-fwd_new[0]).atan2(-fwd_new[2]);

    // Speed = projection of velocity onto forward direction
    v.speed = v3_dot(v.body.vel, fwd_new);

    // Terrain normal for rendering
    let target_n = clamp_normal_tilt(terrain.normal_at(v.x, v.z), 30.0);
    let lerp_rate = (6.0 * dt).min(1.0);
    v.terrain_normal = v3_normalize(v3_lerp(v.terrain_normal, target_n, lerp_rate));

    // World bounds
    v.x = v.x.clamp(-crate::state::WORLD_HALF, crate::state::WORLD_HALF);
    v.z = v.z.clamp(-crate::state::WORLD_HALF, crate::state::WORLD_HALF);
    v.body.pos[0] = v.x;
    v.body.pos[2] = v.z;
}

/// Update drivetrain from player input (called before step_vehicle_physics)
/// throttle: -1..1 (negative = reverse)
pub fn player_drive_input(v: &mut Vehicle, throttle: f32, brake: f32, steer: f32) {
    v.drivetrain.throttle = throttle.clamp(-1.0, 1.0);
    v.drivetrain.brake = brake.clamp(0.0, 1.0);
    v.drivetrain.steer_input = steer.clamp(-1.0, 1.0);
}

/// Update drivetrain from AI (converts legacy speed/steering to physics inputs)
pub fn ai_drive_input(v: &mut Vehicle) {
    // Convert target_speed to throttle/brake
    let speed_diff = v.target_speed - v.speed;
    if speed_diff > 0.5 {
        v.drivetrain.throttle = (speed_diff / 5.0).clamp(0.1, 1.0);
        v.drivetrain.brake = 0.0;
    } else if speed_diff < -0.5 {
        v.drivetrain.throttle = 0.0;
        v.drivetrain.brake = (-speed_diff / 10.0).clamp(0.1, 1.0);
    } else {
        v.drivetrain.throttle = 0.05; // idle
        v.drivetrain.brake = 0.0;
    }

    // steer_input is set by AI steering logic in ai_drive(), not here
}
