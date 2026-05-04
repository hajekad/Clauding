//! Pacejka "Magic Formula" tire model — wheel state and drivetrain.
//!
//! Computes longitudinal/lateral forces from slip ratios and slip angles.
//! Each wheel tracks angular velocity and steering angle, producing forces
//! that feed back into the rigid body solver.

use crate::material::SurfaceMaterial;
use crate::math::*;

// ── Pacejka Magic Formula ────────────────────────────────────────────────

/// Pacejka coefficients for a single axis (longitudinal or lateral)
#[derive(Clone, Copy)]
pub struct PacejkaCoeffs {
    pub b: f32, // stiffness
    pub c: f32, // shape
    pub d: f32, // peak (force = d * Fz)
    pub e: f32, // curvature
}

/// Default coefficients for performance tires (d > 1.0 = tire compound exceeds surface μ)
pub const PACEJKA_LONG: PacejkaCoeffs = PacejkaCoeffs {
    b: 10.0,
    c: 1.9,
    d: 1.15,
    e: 0.97,
};
pub const PACEJKA_LAT: PacejkaCoeffs = PacejkaCoeffs {
    b: 12.0,
    c: 2.3,
    d: 1.05,
    e: 0.97,
};

/// Evaluate the Magic Formula: F = D * sin(C * atan(B*x - E*(B*x - atan(B*x))))
fn pacejka(p: &PacejkaCoeffs, slip: f32) -> f32 {
    let bx = p.b * slip;
    let inner = bx - p.e * (bx - bx.atan());
    p.d * (p.c * inner.atan()).sin()
}

// ── Wheel state ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct WheelState {
    pub ang_vel: f32,      // wheel angular velocity (rad/s)
    pub steer_angle: f32,  // steering angle (rad, 0 for rear wheels)
    pub radius: f32,       // tire radius (m)
    pub inertia: f32,      // wheel rotational inertia (kg*m^2)
    pub brake_torque: f32, // applied brake torque (N*m)
    pub drive_torque: f32, // applied drive torque from engine (N*m)
    pub punctured: bool,   // punctured tire: reduced friction, no grip recovery
    pub abs: bool,         // ABS active on this wheel (false for handbrake)
    // Suspension attachment point in local vehicle space
    pub local_pos: Vec3,
    // Output (computed each frame)
    pub contact_force: Vec3, // world-space force from this tire
    pub on_ground: bool,
    pub compression: f32, // current suspension compression (0..1)
    pub ground_y: f32,    // terrain height at contact
    pub slip_ratio: f32,  // last computed longitudinal slip ratio
}

impl WheelState {
    pub fn new(local_pos: Vec3, radius: f32) -> Self {
        WheelState {
            ang_vel: 0.0,
            steer_angle: 0.0,
            radius,
            inertia: 1.2, // typical car wheel ~1.2 kg*m^2
            brake_torque: 0.0,
            drive_torque: 0.0,
            punctured: false,
            abs: true,
            local_pos,
            contact_force: [0.0; 3],
            on_ground: false,
            compression: 0.0,
            ground_y: 0.0,
            slip_ratio: 0.0,
        }
    }
}

// ── Drivetrain ───────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Drivetrain {
    pub engine_torque: f32, // max engine torque (N*m)
    pub gear_ratio: f32,    // current gear ratio * final drive
    pub max_power: f32,     // max engine power (W) — limits force at high speed
    pub max_speed: f32,     // electronic speed governor (m/s) — 0 = no limit
    pub throttle: f32,      // 0..1
    pub brake: f32,         // 0..1
    pub handbrake: bool,
    pub max_steer: f32,   // max steering angle (rad)
    pub steer_input: f32, // -1..1 steering input
}

impl Drivetrain {
    pub fn new(engine_torque: f32, max_steer_deg: f32) -> Self {
        Drivetrain {
            engine_torque,
            gear_ratio: 5.0,
            max_power: 265_000.0, // 265 kW ≈ 360 HP — 0-100 in ~4.7s, 0-200 in ~17s
            max_speed: 69.44,     // 250 km/h electronic speed governor
            throttle: 0.0,
            brake: 0.0,
            handbrake: false,
            max_steer: max_steer_deg.to_radians(),
            steer_input: 0.0,
        }
    }
}

// ── Tire force computation ───────────────────────────────────────────────

/// Compute tire forces for one wheel given the vehicle body state.
/// Returns (world_force, tire_torque_feedback) applied to the rigid body.
pub fn compute_tire_forces(
    wheel: &mut WheelState,
    body_vel_at_contact: Vec3, // velocity of body at wheel contact point (world space)
    _body_up: Vec3,            // vehicle up direction (from quat)
    body_fwd: Vec3,            // vehicle forward direction (from quat)
    body_right: Vec3,          // vehicle right direction (from quat)
    normal_load: f32,          // suspension force pushing tire into ground (N)
    surface: &SurfaceMaterial,
    dt: f32,
) -> Vec3 {
    if normal_load < 1.0 {
        wheel.on_ground = false;
        wheel.contact_force = [0.0; 3];
        // Free-spinning wheel: apply drive/brake torque to spin
        let net_torque = wheel.drive_torque - wheel.brake_torque * wheel.ang_vel.signum();
        wheel.ang_vel += net_torque / wheel.inertia * dt;
        // Decay free-spinning wheel
        wheel.ang_vel *= 0.99;
        return [0.0; 3];
    }
    wheel.on_ground = true;

    // Local tire coordinate system (respecting steering)
    let (sin_s, cos_s) = wheel.steer_angle.sin_cos();
    let tire_fwd = v3_add(v3_scale(body_fwd, cos_s), v3_scale(body_right, sin_s));
    let tire_right = v3_sub(v3_scale(body_right, cos_s), v3_scale(body_fwd, sin_s));

    // Project contact velocity onto tire axes (ground plane)
    let vx_tire = v3_dot(body_vel_at_contact, tire_fwd); // longitudinal
    let vy_tire = v3_dot(body_vel_at_contact, tire_right); // lateral

    // Longitudinal slip ratio: (wheel_speed - ground_speed) / max(|ground_speed|, |wheel_speed|, small)
    let wheel_speed = wheel.ang_vel * wheel.radius;
    let denom = vx_tire.abs().max(wheel_speed.abs()).max(0.5);
    let slip_ratio = (wheel_speed - vx_tire) / denom;

    // Lateral slip angle: atan(vy / |vx|) with smooth low-speed blending
    // Use total contact speed for damping (not just vx) so drifts still generate grip
    let contact_speed = (vx_tire * vx_tire + vy_tire * vy_tire).sqrt();
    let min_speed = 0.5f32;
    let raw_slip_angle = (vy_tire / vx_tire.abs().max(min_speed)).atan();
    let low_speed_factor = (contact_speed / min_speed).min(1.0); // 0 at standstill, 1 at 0.5+ m/s
    let slip_angle = raw_slip_angle * low_speed_factor;

    wheel.slip_ratio = slip_ratio;

    // ABS: modulate brake torque to keep slip in the Pacejka plateau (~0.08-0.25)
    // The Pacejka curve stays near peak force until slip ~0.25, then drops
    let effective_brake = if wheel.abs && wheel.brake_torque > 50.0 {
        let slip_mag = slip_ratio.abs();
        if slip_mag > 0.50 {
            wheel.brake_torque * 0.05
        } else if slip_mag > 0.25 {
            let t = (slip_mag - 0.25) / 0.25;
            wheel.brake_torque * (1.0 - t * 0.95)
        } else {
            wheel.brake_torque
        }
    } else {
        wheel.brake_torque // no ABS: handbrake or low brake torque
    };

    // Traction control (TC): gentle intervention at extreme wheelspin
    // Low-speed exemption: TC doesn't activate below 2 m/s to allow smooth launch
    let effective_drive = if wheel.abs && wheel.drive_torque.abs() > 10.0 && contact_speed > 2.0 {
        let slip_mag = slip_ratio.abs();
        if slip_mag > 0.80 {
            wheel.drive_torque * 0.15
        } else if slip_mag > 0.40 {
            let t = (slip_mag - 0.40) / 0.40;
            wheel.drive_torque * (1.0 - t * 0.85)
        } else {
            wheel.drive_torque
        }
    } else {
        wheel.drive_torque
    };

    // Punctured tire: drastically reduced friction (riding on rim)
    let puncture_mult = if wheel.punctured { 0.3 } else { 1.0 };
    let effective_friction = surface.dynamic_friction * puncture_mult;

    // Pacejka forces (scaled by normal load and surface friction)
    let fx = pacejka(&PACEJKA_LONG, slip_ratio) * normal_load * effective_friction;
    let fy = -pacejka(&PACEJKA_LAT, slip_angle) * normal_load * effective_friction;

    // Rolling resistance
    let f_roll = -vx_tire.signum() * normal_load * surface.rolling_resistance;

    // World-space force from this tire
    let force = v3_add(
        v3_add(v3_scale(tire_fwd, fx + f_roll), v3_scale(tire_right, fy)),
        [0.0; 3], // no vertical tire force (that's suspension)
    );

    // Tire angular velocity update: torque balance on wheel
    // Use implicit integration to prevent oscillation: solve for ang_vel that balances
    // drive torque against the tire's grip reaction (linearized Pacejka around current slip)
    let _ground_speed_ang = vx_tire / wheel.radius;
    let tire_reaction_torque = -fx * wheel.radius;
    let net_torque =
        effective_drive + tire_reaction_torque - effective_brake * wheel.ang_vel.signum();
    wheel.ang_vel += net_torque / wheel.inertia * dt;

    // Stabilization: strongly blend wheel speed toward no-slip speed
    // This creates reliable traction without oscillation
    let target_ang_vel = vx_tire / wheel.radius + effective_drive / (wheel.inertia * 60.0);
    wheel.ang_vel = wheel.ang_vel * 0.8 + target_ang_vel * 0.2;

    // Brake can stop wheel completely
    if effective_brake > 0.0 && wheel.ang_vel.abs() < 0.5 && wheel.drive_torque.abs() < 1.0 {
        wheel.ang_vel *= 0.9;
    }

    wheel.contact_force = force;
    force
}
