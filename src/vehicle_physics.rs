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
        return;
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

    // Engine torque to driven wheels (reduced by engine bay damage)
    let engine_torque = v.drivetrain.engine_torque * v.drivetrain.throttle * v.drivetrain.gear_ratio * dmg_engine;
    let per_wheel = engine_torque * 0.25; // 4WD split
    for w in &mut v.wheels {
        w.drive_torque = per_wheel;
    }

    // Brake torque
    let brake_torque = 3000.0 * v.drivetrain.brake;
    for w in &mut v.wheels {
        w.brake_torque = brake_torque;
    }

    // Get body orientation vectors
    let body_up = quat_up(v.body.quat);
    let body_fwd = quat_forward(v.body.quat);
    let body_right = quat_right(v.body.quat);

    // Query actual surface material at vehicle position
    let surface = crate::world::surface_at(v.body.pos[0], v.body.pos[2], road_network);
    let surface_mat = *material::material_for_surface(surface);

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
            let contact_vel = v.body.velocity_at(attach_world);
            let tire_force = tire::compute_tire_forces(
                &mut v.wheels[i],
                contact_vel,
                body_up,
                body_fwd,
                body_right,
                susp_force * dmg_grip, // reduced normal load → less grip
                &surface_mat,
                dt,
            );
            total_force = v3_add(total_force, tire_force);
            total_torque = v3_add(total_torque, v3_cross(r, tire_force));
        }
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
pub fn player_drive_input(v: &mut Vehicle, throttle: f32, brake: f32, steer: f32) {
    v.drivetrain.throttle = throttle.clamp(0.0, 1.0);
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
