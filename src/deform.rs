// Vehicle panel deformation: visible crash damage
//
// Each vehicle has deformation state per panel region.
// On collision, impact energy deforms panels proportional to relative velocity.
// Deformation offsets are applied during mesh generation in render.rs.

use crate::math::*;

/// Number of deformation sample points around the vehicle shell
pub const DEFORM_POINTS: usize = 8;

/// Deformation state for a vehicle
#[derive(Clone, Copy)]
pub struct VehicleDeformation {
    /// Per-point deformation offset (inward, in meters). 0 = pristine, positive = dented in.
    pub offsets: [f32; DEFORM_POINTS],
    /// Maximum deformation before panel is "destroyed" (structural limit)
    pub max_deform: f32,
}

impl VehicleDeformation {
    pub fn new() -> Self {
        VehicleDeformation {
            offsets: [0.0; DEFORM_POINTS],
            max_deform: 0.3, // 30cm max dent
        }
    }

    /// Apply impact deformation from a collision.
    /// `local_impact`: impact point in vehicle local space
    /// `energy`: collision energy (mass * relative_speed^2 * 0.5)
    /// `half_w`, `half_d`: vehicle half-extents for mapping impact to panel
    pub fn apply_impact(&mut self, local_impact: Vec3, energy: f32, _half_w: f32, _half_d: f32) {
        // Map impact point to panel index (8 panels around perimeter)
        let angle = local_impact[2].atan2(local_impact[0]); // -PI..PI
        let idx = ((angle + std::f32::consts::PI) / std::f32::consts::TAU * DEFORM_POINTS as f32) as usize;
        let idx = idx.min(DEFORM_POINTS - 1);

        // Deformation amount: sqrt(energy) scaled to reasonable range
        let deform = (energy * 0.0001).sqrt().min(self.max_deform);

        // Apply to primary panel and neighbors (spread damage)
        self.offsets[idx] = (self.offsets[idx] + deform).min(self.max_deform);
        let prev = if idx == 0 { DEFORM_POINTS - 1 } else { idx - 1 };
        let next = (idx + 1) % DEFORM_POINTS;
        self.offsets[prev] = (self.offsets[prev] + deform * 0.5).min(self.max_deform);
        self.offsets[next] = (self.offsets[next] + deform * 0.5).min(self.max_deform);
    }

    /// Get deformation offset at an arbitrary angle around the vehicle perimeter
    pub fn sample_at_angle(&self, angle: f32) -> f32 {
        let t = (angle + std::f32::consts::PI) / std::f32::consts::TAU * DEFORM_POINTS as f32;
        let idx = t as usize;
        let frac = t - idx as f32;
        let a = self.offsets[idx % DEFORM_POINTS];
        let b = self.offsets[(idx + 1) % DEFORM_POINTS];
        a + (b - a) * frac
    }

    /// Total damage as a fraction 0..1 (average deformation / max)
    pub fn damage_fraction(&self) -> f32 {
        let total: f32 = self.offsets.iter().sum();
        total / (DEFORM_POINTS as f32 * self.max_deform)
    }

    /// Is the vehicle totaled? (>80% max deformation on average)
    pub fn is_totaled(&self) -> bool {
        self.damage_fraction() > 0.8
    }

    // ── Handling effects from structural damage ──

    /// Engine torque multiplier (front panels damaged → less power).
    /// Panel indices 1-3 correspond to the front arc of the vehicle.
    pub fn engine_factor(&self) -> f32 {
        let front_avg = (self.offsets[1] + self.offsets[2] + self.offsets[3]) / (3.0 * self.max_deform);
        (1.0 - front_avg * 0.8).max(0.05) // up to 80% torque loss, never fully zero
    }

    /// Suspension spring rate multiplier for a given wheel.
    /// Each wheel maps to the nearest panel pair.
    pub fn suspension_factor(&self, wheel_idx: usize) -> f32 {
        let panel_avg = match wheel_idx {
            0 => (self.offsets[1] + self.offsets[2]) * 0.5, // FL — front-left panels
            1 => (self.offsets[3] + self.offsets[4]) * 0.5, // FR — front-right panels
            2 => (self.offsets[6] + self.offsets[7]) * 0.5, // RL — rear-left panels
            3 => (self.offsets[5] + self.offsets[6]) * 0.5, // RR — rear-right panels
            _ => 0.0,
        };
        let damage = panel_avg / self.max_deform;
        (1.0 - damage * 0.5).max(0.3) // up to 50% spring loss
    }

    /// Steering angle multiplier (front axle damage → reduced lock).
    pub fn steering_factor(&self) -> f32 {
        let front_avg = (self.offsets[1] + self.offsets[2] + self.offsets[3]) / (3.0 * self.max_deform);
        (1.0 - front_avg * 0.6).max(0.2) // up to 60% lock reduction
    }

    /// Tire grip multiplier (overall damage → reduced contact patch).
    pub fn grip_factor(&self) -> f32 {
        (1.0 - self.damage_fraction() * 0.3).max(0.5) // up to 30% grip loss
    }
}
