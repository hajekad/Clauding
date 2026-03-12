// Articulated skeleton: 15-bone hierarchy with joint constraints
//
// Replaces the old Verlet 7-point ragdoll with proper bones that have:
// - Position + orientation (quaternion)
// - Joint angle constraints (cone + twist limits)
// - Ability to blend between ragdoll and animation poses

use crate::math::*;

// ── Bone IDs ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum BoneId {
    Hips = 0,
    Spine,
    Chest,
    Neck,
    Head,
    LeftUpperArm,
    LeftForearm,
    RightUpperArm,
    RightForearm,
    LeftUpperLeg,
    LeftLowerLeg,
    LeftFoot,
    RightUpperLeg,
    RightLowerLeg,
    RightFoot,
}

pub const BONE_COUNT: usize = 15;

// ── Joint constraint ─────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct JointConstraint {
    pub cone_angle: f32,    // max angle from parent's axis (radians)
    pub twist_min: f32,     // min twist around bone axis (radians)
    pub twist_max: f32,     // max twist around bone axis (radians)
}

impl JointConstraint {
    pub const fn new(cone_deg: f32, twist_min_deg: f32, twist_max_deg: f32) -> Self {
        JointConstraint {
            cone_angle: cone_deg * (std::f32::consts::PI / 180.0),
            twist_min: twist_min_deg * (std::f32::consts::PI / 180.0),
            twist_max: twist_max_deg * (std::f32::consts::PI / 180.0),
        }
    }

    pub const fn free() -> Self {
        JointConstraint { cone_angle: std::f32::consts::PI, twist_min: -std::f32::consts::PI, twist_max: std::f32::consts::PI }
    }
}

// ── Bone ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Bone {
    pub local_pos: Vec3,        // offset from parent in parent's local space
    pub local_rot: Quat,        // orientation relative to parent
    pub length: f32,            // bone length (distance to child attachment)
    pub parent: Option<u8>,     // parent bone index (None for root)
    pub constraint: JointConstraint,
    // Physics state for ragdoll mode
    pub world_pos: Vec3,        // cached world position (computed from hierarchy)
    pub world_rot: Quat,        // cached world rotation
    pub vel: Vec3,              // linear velocity (ragdoll mode)
    pub ang_vel: Vec3,          // angular velocity (ragdoll mode)
    pub mass: f32,              // bone mass for physics
}

impl Bone {
    pub fn new(parent: Option<u8>, local_pos: Vec3, length: f32, mass: f32, constraint: JointConstraint) -> Self {
        Bone {
            local_pos,
            local_rot: QUAT_IDENTITY,
            length,
            parent,
            constraint,
            world_pos: [0.0; 3],
            world_rot: QUAT_IDENTITY,
            vel: [0.0; 3],
            ang_vel: [0.0; 3],
            mass,
        }
    }
}

// ── Skeleton ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Skeleton {
    pub bones: [Bone; BONE_COUNT],
    pub ragdoll_active: bool,
    pub ragdoll_timer: f32,
    pub ragdoll_blend: f32,     // 0 = full ragdoll, 1 = full animation (for blending back)
}

impl Skeleton {
    /// Create default humanoid skeleton with anatomical proportions.
    /// Height ~1.85m, proportions matching the existing character mesh.
    pub fn new_humanoid() -> Self {
        use BoneId::*;

        let bones = [
            // Hips (root) — at ~0.95m
            Bone::new(None,                         [0.0, 0.95, 0.0],  0.15, 8.0,  JointConstraint::free()),
            // Spine — above hips
            Bone::new(Some(Hips as u8),             [0.0, 0.15, 0.0],  0.20, 6.0,  JointConstraint::new(30.0, -20.0, 20.0)),
            // Chest — above spine
            Bone::new(Some(Spine as u8),            [0.0, 0.20, 0.0],  0.25, 8.0,  JointConstraint::new(25.0, -15.0, 15.0)),
            // Neck — above chest
            Bone::new(Some(Chest as u8),            [0.0, 0.25, 0.0],  0.10, 2.0,  JointConstraint::new(40.0, -30.0, 30.0)),
            // Head — above neck
            Bone::new(Some(Neck as u8),             [0.0, 0.10, 0.0],  0.18, 4.5,  JointConstraint::new(35.0, -20.0, 20.0)),
            // Left upper arm
            Bone::new(Some(Chest as u8),            [-0.22, 0.20, 0.0], 0.28, 2.5, JointConstraint::new(90.0, -90.0, 90.0)),
            // Left forearm
            Bone::new(Some(LeftUpperArm as u8),     [0.0, -0.28, 0.0], 0.25, 1.5,  JointConstraint::new(5.0, -140.0, 0.0)),
            // Right upper arm
            Bone::new(Some(Chest as u8),            [0.22, 0.20, 0.0], 0.28, 2.5,  JointConstraint::new(90.0, -90.0, 90.0)),
            // Right forearm
            Bone::new(Some(RightUpperArm as u8),    [0.0, -0.28, 0.0], 0.25, 1.5,  JointConstraint::new(5.0, 0.0, 140.0)),
            // Left upper leg
            Bone::new(Some(Hips as u8),             [-0.10, -0.0, 0.0], 0.42, 5.0, JointConstraint::new(80.0, -30.0, 30.0)),
            // Left lower leg
            Bone::new(Some(LeftUpperLeg as u8),     [0.0, -0.42, 0.0], 0.40, 3.0,  JointConstraint::new(5.0, 0.0, 140.0)),
            // Left foot
            Bone::new(Some(LeftLowerLeg as u8),     [0.0, -0.40, 0.0], 0.12, 1.0,  JointConstraint::new(30.0, -20.0, 20.0)),
            // Right upper leg
            Bone::new(Some(Hips as u8),             [0.10, -0.0, 0.0], 0.42, 5.0,  JointConstraint::new(80.0, -30.0, 30.0)),
            // Right lower leg
            Bone::new(Some(RightUpperLeg as u8),    [0.0, -0.42, 0.0], 0.40, 3.0,  JointConstraint::new(5.0, 0.0, 140.0)),
            // Right foot
            Bone::new(Some(RightLowerLeg as u8),    [0.0, -0.40, 0.0], 0.12, 1.0,  JointConstraint::new(30.0, -20.0, 20.0)),
        ];

        Skeleton {
            bones,
            ragdoll_active: false,
            ragdoll_timer: 0.0,
            ragdoll_blend: 1.0,
        }
    }

    /// Compute world transforms by walking the bone hierarchy from root.
    /// Call after modifying local_rot values (e.g., from animation or IK).
    pub fn compute_world_transforms(&mut self, root_pos: Vec3, root_rot: Quat) {
        // Root bone
        self.bones[0].world_pos = v3_add(root_pos, self.bones[0].local_pos);
        self.bones[0].world_rot = quat_mul(root_rot, self.bones[0].local_rot);

        // Children
        for i in 1..BONE_COUNT {
            let parent_idx = self.bones[i].parent.unwrap_or(0) as usize;
            let parent_pos = self.bones[parent_idx].world_pos;
            let parent_rot = self.bones[parent_idx].world_rot;

            // World position = parent_pos + parent_rot * local_pos
            let offset = quat_rotate(parent_rot, self.bones[i].local_pos);
            self.bones[i].world_pos = v3_add(parent_pos, offset);
            self.bones[i].world_rot = quat_mul(parent_rot, self.bones[i].local_rot);
        }
    }

    /// Get world-space bone endpoint (tip) for rendering
    pub fn bone_tip(&self, bone_id: BoneId) -> Vec3 {
        let b = &self.bones[bone_id as usize];
        let dir = quat_rotate(b.world_rot, [0.0, -b.length, 0.0]); // bones point downward by default
        v3_add(b.world_pos, dir)
    }

    /// Get positions for the old 7-point ragdoll format (backward compatibility with rendering).
    /// Returns [hips, chest, head, l_hand, r_hand, l_foot, r_foot]
    pub fn to_ragdoll_points(&self) -> [[f32; 3]; 7] {
        [
            self.bones[BoneId::Hips as usize].world_pos,
            self.bones[BoneId::Chest as usize].world_pos,
            self.bones[BoneId::Head as usize].world_pos,
            self.bone_tip(BoneId::LeftForearm),
            self.bone_tip(BoneId::RightForearm),
            self.bone_tip(BoneId::LeftFoot),
            self.bone_tip(BoneId::RightFoot),
        ]
    }

    /// Initialize skeleton from NPC position for ragdoll activation.
    /// Sets bone positions to default T-pose at the character's location.
    pub fn activate_ragdoll(&mut self, pos: Vec3, rot_y: f32, impulse: Vec3) {
        let root_rot = quat_from_rot_y(rot_y);
        self.compute_world_transforms(pos, root_rot);

        // Set initial velocities from impulse (distributed across bones, more to extremities)
        let base_vel = v3_scale(impulse, 0.5);
        for i in 0..BONE_COUNT {
            self.bones[i].vel = base_vel;
        }
        // Extra impulse to extremities
        let extra = v3_scale(impulse, 0.3);
        self.bones[BoneId::Head as usize].vel = v3_add(self.bones[BoneId::Head as usize].vel, extra);
        self.bones[BoneId::LeftForearm as usize].vel = v3_add(self.bones[BoneId::LeftForearm as usize].vel, extra);
        self.bones[BoneId::RightForearm as usize].vel = v3_add(self.bones[BoneId::RightForearm as usize].vel, extra);

        self.ragdoll_active = true;
        self.ragdoll_blend = 0.0;
    }

    /// Step ragdoll physics: gravity, bone velocities, constraint enforcement.
    pub fn step_ragdoll(&mut self, terrain: &crate::state::Terrain, dt: f32) {
        if !self.ragdoll_active { return; }

        let gravity = [0.0f32, -9.81, 0.0];

        // Integrate each bone independently
        for i in 0..BONE_COUNT {
            let b = &mut self.bones[i];
            // Apply gravity
            b.vel = v3_add(b.vel, v3_scale(gravity, dt));
            // Damping
            b.vel = v3_scale(b.vel, 0.98);
            // Integrate position
            b.world_pos = v3_add(b.world_pos, v3_scale(b.vel, dt));

            // Ground collision
            let ground_y = terrain.height_at(b.world_pos[0], b.world_pos[2]);
            if b.world_pos[1] < ground_y {
                b.world_pos[1] = ground_y;
                // Bounce + friction
                if b.vel[1] < 0.0 {
                    b.vel[1] *= -0.2; // low bounce
                }
                b.vel[0] *= 0.7; // ground friction
                b.vel[2] *= 0.7;
            }

            // World bounds
            b.world_pos[0] = b.world_pos[0].clamp(-crate::state::WORLD_HALF, crate::state::WORLD_HALF);
            b.world_pos[2] = b.world_pos[2].clamp(-crate::state::WORLD_HALF, crate::state::WORLD_HALF);
        }

        // Enforce distance constraints (parent-child bone lengths)
        for _ in 0..4 {
            for i in 1..BONE_COUNT {
                let parent_idx = self.bones[i].parent.unwrap_or(0) as usize;
                let target_dist = v3_len(self.bones[i].local_pos);
                if target_dist < 0.001 { continue; }

                let delta = v3_sub(self.bones[i].world_pos, self.bones[parent_idx].world_pos);
                let dist = v3_len(delta);
                if dist < 0.001 { continue; }

                let diff = (dist - target_dist) / dist;
                let correction = v3_scale(delta, diff * 0.5);

                // Mass-weighted correction
                let mi = self.bones[i].mass;
                let mp = self.bones[parent_idx].mass;
                let total = mi + mp;
                let wi = mp / total;
                let wp = mi / total;

                self.bones[i].world_pos = v3_sub(self.bones[i].world_pos, v3_scale(correction, wi));
                self.bones[parent_idx].world_pos = v3_add(self.bones[parent_idx].world_pos, v3_scale(correction, wp));
            }
        }

        // Timer
        self.ragdoll_timer -= dt;
    }
}

// ── Two-bone IK solver ───────────────────────────────────────────────────

/// Analytic two-bone IK: given bone lengths a and b, and target distance,
/// compute the mid-joint angle. Returns the bend angle in radians.
pub fn two_bone_ik_angle(len_a: f32, len_b: f32, target_dist: f32) -> f32 {
    let d = target_dist.clamp(0.001, len_a + len_b - 0.001);
    // Law of cosines: c^2 = a^2 + b^2 - 2ab*cos(C)
    let cos_angle = (len_a * len_a + len_b * len_b - d * d) / (2.0 * len_a * len_b);
    cos_angle.clamp(-1.0, 1.0).acos()
}

/// Solve two-bone IK chain. Given upper bone (shoulder/hip), lower bone (elbow/knee),
/// and a target world position, compute the required joint rotations.
/// `pole_dir` is the preferred bend direction (e.g., forward for knees, backward for elbows).
pub fn solve_two_bone_ik(
    root_pos: Vec3,
    target_pos: Vec3,
    len_upper: f32,
    len_lower: f32,
    pole_dir: Vec3,
) -> (Quat, Quat) {
    let to_target = v3_sub(target_pos, root_pos);
    let target_dist = v3_len(to_target);

    if target_dist < 0.001 {
        return (QUAT_IDENTITY, QUAT_IDENTITY);
    }

    let target_dir = v3_scale(to_target, 1.0 / target_dist);

    // Mid-joint angle
    let mid_angle = two_bone_ik_angle(len_upper, len_lower, target_dist);

    // Upper bone: rotates toward target, offset by the mid-joint bend
    let upper_angle = if target_dist >= len_upper + len_lower - 0.001 {
        0.0 // fully extended
    } else {
        let cos_upper = (len_upper * len_upper + target_dist * target_dist - len_lower * len_lower)
            / (2.0 * len_upper * target_dist);
        cos_upper.clamp(-1.0, 1.0).acos()
    };

    // Build rotation for upper bone (simplified — rotate toward target with offset)
    let bend_axis = v3_normalize(v3_cross(target_dir, pole_dir));
    let bend_axis = if v3_len(bend_axis) < 0.01 {
        [1.0, 0.0, 0.0] // fallback axis
    } else {
        bend_axis
    };

    let upper_rot = quat_from_axis_angle(bend_axis, upper_angle);
    let lower_rot = quat_from_axis_angle(bend_axis, -(std::f32::consts::PI - mid_angle));

    (upper_rot, lower_rot)
}
