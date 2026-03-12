// Vehicle suspension: spring-damper per wheel
//
// Each wheel has an independent spring-damper that:
// 1. Raycasts down from the wheel attachment point
// 2. Compresses/extends based on terrain height
// 3. Produces a normal force that both supports the vehicle and feeds into tire model

use crate::math::*;
use crate::tire::WheelState;

// ── Suspension parameters ────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct SuspensionParams {
    pub rest_length: f32,      // natural spring length (m)
    pub spring_rate: f32,      // spring constant k (N/m)
    pub damper_rate: f32,      // damping constant c (N*s/m)
    pub max_compression: f32,  // maximum compression travel (m)
    pub max_extension: f32,    // maximum droop travel (m)
}

impl SuspensionParams {
    /// Typical SUV/sedan suspension
    pub fn default_car() -> Self {
        SuspensionParams {
            rest_length: 0.35,
            spring_rate: 35000.0,   // 35 kN/m per wheel (total ~140 kN for 1500kg car)
            damper_rate: 3500.0,    // critical damping ≈ 2*sqrt(k*m) ≈ 2*sqrt(35000*375) ≈ 7245
            max_compression: 0.15,
            max_extension: 0.20,
        }
    }
}

// ── Suspension state per wheel ───────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct SuspensionState {
    pub params: SuspensionParams,
    pub compression: f32,       // current compression distance (positive = compressed)
    pub prev_compression: f32,  // previous frame compression (for velocity)
}

impl SuspensionState {
    pub fn new(params: SuspensionParams) -> Self {
        SuspensionState { params, compression: 0.0, prev_compression: 0.0 }
    }
}

// ── Suspension force computation ─────────────────────────────────────────

/// Compute suspension force for one wheel.
///
/// Takes the wheel's world-space attachment point on the body, raycasts down
/// to find terrain, and returns:
/// - The normal force magnitude (N) pushing up on the body (and down on the ground)
/// - Updates wheel.ground_y and wheel.compression
///
/// `attach_world`: world-space position of the suspension top mount
/// `body_up`: vehicle's up direction (from quaternion)
/// `terrain`: for ground height queries
/// `dt`: timestep
pub fn compute_suspension(
    susp: &mut SuspensionState,
    wheel: &mut WheelState,
    attach_world: Vec3,
    _body_up: Vec3,
    terrain: &crate::state::Terrain,
    dt: f32,
) -> f32 {
    // Raycast straight down from attachment point (in vehicle's up direction)
    let ray_length = susp.params.rest_length + susp.params.max_extension + wheel.radius;

    // Ground height at wheel XZ position
    let ground_y = terrain.height_at(attach_world[0], attach_world[2]);
    wheel.ground_y = ground_y;

    // Distance from attach point to ground (along up axis, approximated as Y)
    let dist_to_ground = attach_world[1] - ground_y;

    // Wheel center wants to be at: ground_y + wheel.radius
    // Suspension length = attach_y - wheel_center_y = attach_y - ground_y - wheel.radius
    let current_length = dist_to_ground - wheel.radius;

    if current_length > ray_length {
        // Wheel is fully unloaded (in the air)
        wheel.on_ground = false;
        wheel.compression = 0.0;
        susp.prev_compression = susp.compression;
        susp.compression = 0.0;
        return 0.0;
    }

    // Compression = how much shorter than rest length
    let compression = susp.params.rest_length - current_length;
    let compression = compression.clamp(-susp.params.max_extension, susp.params.max_compression);

    // Compression velocity (for damping)
    susp.prev_compression = susp.compression;
    susp.compression = compression;
    let comp_velocity = if dt > 1e-6 {
        (compression - susp.prev_compression) / dt
    } else {
        0.0
    };

    // Spring + damper force: F = k*x + c*v
    let spring_force = susp.params.spring_rate * compression;
    let damper_force = susp.params.damper_rate * comp_velocity;
    let total_force = (spring_force + damper_force).max(0.0); // suspension can only push, not pull

    wheel.compression = (compression / susp.params.max_compression).clamp(0.0, 1.0);
    wheel.on_ground = true;

    total_force
}
