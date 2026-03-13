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

// ── Per-foot ground contact ──────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct FootContact {
    pub grounded: bool,         // foot is planted on surface
    pub ground_y: f32,          // terrain height under this foot
    pub surface_normal: Vec3,   // terrain normal under this foot (for slope adaptation)
    pub target_pos: Vec3,       // IK target in world space
    pub plant_pos: Vec3,        // position where foot was planted (stays fixed while grounded)
    pub lift_height: f32,       // current lift above ground during swing phase
    pub push_force: Vec3,       // ground reaction force this foot exerts (for locomotion)
}

impl FootContact {
    pub fn new() -> Self {
        FootContact {
            grounded: true,
            ground_y: 0.0,
            surface_normal: [0.0, 1.0, 0.0],
            target_pos: [0.0; 3],
            plant_pos: [0.0; 3],
            lift_height: 0.0,
            push_force: [0.0; 3],
        }
    }
}

// ── Locomotion gait states ───────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
pub enum Gait {
    Idle,
    Walk,
    Run,
    Sprint,
}

impl Gait {
    /// Stride frequency (steps/sec) at this gait's natural speed
    pub fn stride_freq(self) -> f32 {
        match self {
            Gait::Idle => 0.0,
            Gait::Walk => 2.5,    // 8 km/h brisk walk (NPC default)
            Gait::Run => 3.0,     // 11 km/h jog (player default)
            Gait::Sprint => 4.5,  // 28 km/h sprint
        }
    }

    /// Maximum stride length (meters forward per step)
    pub fn stride_len(self) -> f32 {
        match self {
            Gait::Idle => 0.0,
            Gait::Walk => 0.89,   // 2.5 × 0.89 = 2.225 m/s ≈ 8 km/h
            Gait::Run => 1.02,    // 3.0 × 1.02 = 3.06 m/s ≈ 11 km/h
            Gait::Sprint => 1.73, // 4.5 × 1.73 = 7.785 m/s ≈ 28 km/h
        }
    }

    /// Natural speed for this gait: emerges from stride frequency × stride length.
    /// This is the speed the character achieves when legs are pumping at full cadence.
    pub fn natural_speed(self) -> f32 {
        self.stride_freq() * self.stride_len()
    }

    /// Foot lift height during swing phase (meters)
    pub fn foot_lift(self) -> f32 {
        match self {
            Gait::Idle => 0.0,
            Gait::Walk => 0.08,   // brisk walk
            Gait::Run => 0.12,
            Gait::Sprint => 0.22, // high knee lift at full sprint
        }
    }

    /// Arm swing amplitude (radians)
    pub fn arm_swing(self) -> f32 {
        match self {
            Gait::Idle => 0.0,
            Gait::Walk => 0.35,
            Gait::Run => 0.6,
            Gait::Sprint => 1.0,
        }
    }

    /// Forward spine lean (radians)
    pub fn spine_lean(self) -> f32 {
        match self {
            Gait::Idle => 0.0,
            Gait::Walk => 0.04,
            Gait::Run => 0.08,
            Gait::Sprint => 0.20, // aggressive forward lean at sprint
        }
    }

    /// CoM lean threshold before stumble triggers (meters).
    /// Faster gaits have wider stance → higher tolerance before losing balance.
    pub fn balance_threshold(self) -> f32 {
        match self {
            Gait::Idle => 0.20,   // standing still — tips easily
            Gait::Walk => 0.30,
            Gait::Run => 0.40,    // wider dynamic stance
            Gait::Sprint => 0.60, // momentum carries through mild lean
        }
    }

    /// Speed thresholds with hysteresis for animation gait transition
    /// Returns (enter_speed, exit_speed) — enter > exit prevents flickering
    pub fn speed_range(self) -> (f32, f32) {
        match self {
            Gait::Idle => (0.0, 0.0),
            Gait::Walk => (0.5, 0.3),    // start walk animation at 0.5 m/s
            Gait::Run => (2.5, 1.8),     // transition to run above ~2.5 m/s
            Gait::Sprint => (5.5, 4.5),  // transition to sprint above ~5.5 m/s
        }
    }

    /// All animation parameters as a tuple for interpolation during gait transitions
    fn params(self) -> GaitParams {
        GaitParams {
            stride_freq: self.stride_freq(),
            stride_len: self.stride_len(),
            foot_lift: self.foot_lift(),
            arm_swing: self.arm_swing(),
            spine_lean: self.spine_lean(),
        }
    }
}

/// Snapshot of gait animation parameters for smooth blending between gaits
#[derive(Clone, Copy)]
struct GaitParams {
    stride_freq: f32,
    stride_len: f32,
    foot_lift: f32,
    arm_swing: f32,
    spine_lean: f32,
}

impl GaitParams {
    fn lerp(a: &GaitParams, b: &GaitParams, t: f32) -> GaitParams {
        GaitParams {
            stride_freq: a.stride_freq + (b.stride_freq - a.stride_freq) * t,
            stride_len: a.stride_len + (b.stride_len - a.stride_len) * t,
            foot_lift: a.foot_lift + (b.foot_lift - a.foot_lift) * t,
            arm_swing: a.arm_swing + (b.arm_swing - a.arm_swing) * t,
            spine_lean: a.spine_lean + (b.spine_lean - a.spine_lean) * t,
        }
    }
}

/// Select gait from current speed with hysteresis
fn select_gait(speed: f32, current: Gait) -> Gait {
    // Try to stay in current gait (hysteresis prevents flickering)
    let (_, exit) = current.speed_range();
    if speed >= exit {
        // Check if we should upgrade to a faster gait
        let next = match current {
            Gait::Idle => Gait::Walk,
            Gait::Walk => Gait::Run,
            Gait::Run => Gait::Sprint,
            Gait::Sprint => Gait::Sprint,
        };
        let (enter, _) = next.speed_range();
        if speed >= enter { return next; }
        return current;
    }
    // Downgrade
    match current {
        Gait::Sprint => Gait::Run,
        Gait::Run => Gait::Walk,
        Gait::Walk => Gait::Idle,
        Gait::Idle => Gait::Idle,
    }
}

// ── Skeleton ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Skeleton {
    pub bones: [Bone; BONE_COUNT],
    pub ragdoll_active: bool,
    pub ragdoll_timer: f32,
    pub ragdoll_blend: f32,     // 0 = full ragdoll, 1 = full animation (for blending back)
    // Procedural animation state
    pub walk_phase: f32,        // 0..TAU, drives left/right foot alternation
    pub feet: [FootContact; 2], // [left, right]
    pub com_offset: Vec3,       // center of mass offset from hips (body sway)
    pub com_world: Vec3,        // actual center of mass position (mass-weighted bone average)
    pub com_lean: Vec3,         // lean direction relative to support base (for balance)
    pub landing_speed: f32,     // vertical speed at last ground contact (for stumble)
    pub stumble_timer: f32,     // >0 = stumbling, decrements to 0
    pub stumble_dir: Vec3,      // stumble lean direction (world space)
    pub stumble_brake: f32,     // 0..1 locomotion force multiplier during stumble (0 = full brake)
    pub total_push_force: Vec3, // accumulated ground reaction force from both feet
    pub gait: Gait,             // current locomotion state (idle/walk/run/sprint)
    pub gait_blend: f32,        // 0..1 blend progress during gait transition
    prev_gait_params: GaitParams, // snapshot of previous gait's parameters for blending
    // Jump compression phase
    pub jump_phase: f32,        // 0 = none, 0→1 = compressing, 1→2 = extending (launch)
    pub jump_crouch: f32,       // current crouch depth (meters, for visual + force timing)
    // Euphoria-style active ragdoll
    pub bone_active: [bool; BONE_COUNT],      // per-bone: true = spring motor active, false = passive ragdoll
    pub bone_target_pos: [Vec3; BONE_COUNT],  // target world-space positions for active motors
    pub vehicle_contact: Option<VehicleContact>, // continuous body-on-vehicle contact
    pub anticipation_timer: f32,             // >0 = NPC is bracing/dodging before impact
    pub anticipation_dir: Vec3,              // direction of incoming threat
}

/// Continuous contact state: body resting on/sliding along a vehicle surface.
/// Contact position is stored in vehicle local space so it tracks the vehicle's movement.
#[derive(Clone, Copy)]
pub struct VehicleContact {
    pub vehicle_idx: usize,         // index of vehicle in world.vehicles
    pub bone_contacts: [bool; BONE_COUNT], // per-bone: true if in contact with vehicle
    pub surface_normal: Vec3,       // vehicle surface normal at contact (world space, updated each frame)
    pub local_offset: Vec3,         // contact offset in vehicle local space (tracks with vehicle)
    pub friction: f32,              // contact friction (0.3 for painted metal)
    pub time: f32,                  // how long contact has been active
}

// Position-based spring motor gains for active ragdoll
const RAGDOLL_KP: f32 = 200.0;     // spring stiffness (N/m)
const RAGDOLL_KD: f32 = 20.0;      // velocity damping (N·s/m)

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
            walk_phase: 0.0,
            feet: [FootContact::new(); 2],
            com_offset: [0.0; 3],
            com_world: [0.0; 3],
            com_lean: [0.0; 3],
            landing_speed: 0.0,
            stumble_timer: 0.0,
            stumble_dir: [0.0; 3],
            stumble_brake: 1.0,
            total_push_force: [0.0; 3],
            gait: Gait::Idle,
            gait_blend: 1.0,
            prev_gait_params: Gait::Idle.params(),
            jump_phase: 0.0,
            jump_crouch: 0.0,
            bone_active: [true; BONE_COUNT],
            bone_target_pos: [[0.0; 3]; BONE_COUNT],
            vehicle_contact: None,
            anticipation_timer: 0.0,
            anticipation_dir: [0.0; 3],
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

    // ── Procedural animation ──────────────────────────────────────────

    /// Step procedural animation: walk cycle from velocity, foot IK, CoM sway, stumble.
    /// Call once per frame for walking/standing characters. Not called during ragdoll.
    /// `vel` = character velocity from rigid body, `pos` = character world position,
    /// `rot_y` = facing direction, `terrain` = for foot ground/normal queries,
    /// `mass` = character mass for ground reaction force computation.
    pub fn step_animation(&mut self, vel: Vec3, pos: Vec3, rot_y: f32, terrain: &crate::state::Terrain, on_ground: bool, _mass: f32, dt: f32) {
        if self.ragdoll_active { return; }

        let horiz_speed = (vel[0] * vel[0] + vel[2] * vel[2]).sqrt();
        let pi = std::f32::consts::PI;
        let tau = std::f32::consts::TAU;

        // ── Gait selection with hysteresis (momentum-governed transitions) ──
        let new_gait = select_gait(horiz_speed, self.gait);
        if new_gait != self.gait {
            // Snapshot current (blended) params before switching
            let cur = self.gait.params();
            self.prev_gait_params = if self.gait_blend < 1.0 {
                GaitParams::lerp(&self.prev_gait_params, &cur, self.gait_blend)
            } else {
                cur
            };
            self.gait = new_gait;
            self.gait_blend = 0.0; // start blending from old params to new gait
        }
        // Blend rate governed by momentum: heavier/faster = slower transitions
        let blend_rate = 3.0 / (1.0 + horiz_speed * 0.3); // ~3/s idle, ~1.5/s at sprint
        self.gait_blend = (self.gait_blend + blend_rate * dt).min(1.0);

        // Interpolate between previous gait params and current gait params
        let blended = GaitParams::lerp(&self.prev_gait_params, &self.gait.params(), self.gait_blend);
        let stride_freq = blended.stride_freq;
        let stride_len = blended.stride_len;
        let foot_lift_h = blended.foot_lift;
        let arm_swing_amp = blended.arm_swing;
        let spine_lean_amt = blended.spine_lean;

        // ── Walk phase: leg cadence from gait stride frequency ──
        let freq = if self.gait == Gait::Idle { 0.0 } else { stride_freq };
        self.walk_phase = (self.walk_phase + freq * dt) % tau;

        let fwd = [-rot_y.sin(), 0.0, -rot_y.cos()];
        let right = [rot_y.cos(), 0.0, -rot_y.sin()];

        let hip_y = pos[1] + 0.95;

        self.total_push_force = [0.0; 3];

        // ── Per-foot: ground contact, surface normal, IK, ground reaction force ──
        for side in 0..2 {
            let lateral_sign: f32 = if side == 0 { -1.0 } else { 1.0 };
            let phase = self.walk_phase + if side == 1 { pi } else { 0.0 };
            let sin_phase = phase.sin();
            let cos_phase = phase.cos();

            let hip_pos = [
                pos[0] + right[0] * 0.10 * lateral_sign,
                hip_y,
                pos[2] + right[2] * 0.10 * lateral_sign,
            ];

            if self.gait != Gait::Idle {
                let foot_fwd_offset = sin_phase * stride_len;
                let foot_lift = (1.0 - cos_phase.abs()) * foot_lift_h;

                let foot_x = pos[0] + right[0] * 0.10 * lateral_sign + fwd[0] * foot_fwd_offset;
                let foot_z = pos[2] + right[2] * 0.10 * lateral_sign + fwd[2] * foot_fwd_offset;
                let ground_y = terrain.height_at(foot_x, foot_z);
                let foot_normal = terrain.normal_at(foot_x, foot_z);

                // Foot grounded during push phase (foot behind body, pressing against ground)
                let is_contact = cos_phase > 0.0;

                self.feet[side].grounded = is_contact && on_ground;
                self.feet[side].ground_y = ground_y;
                self.feet[side].surface_normal = foot_normal;
                self.feet[side].lift_height = if is_contact { 0.0 } else { foot_lift };
                self.feet[side].target_pos = [foot_x, ground_y + self.feet[side].lift_height, foot_z];

                if is_contact {
                    self.feet[side].plant_pos = self.feet[side].target_pos;
                }

                // Ground reaction force: leg pushes backward+down against surface,
                // normal reaction propels body forward. Force angle adjusted by surface normal.
                // Steeper slopes → push angle shifts, reducing effective forward force.
                if self.feet[side].grounded {
                    // Foot pushes along surface tangent (forward direction projected onto surface)
                    let n = foot_normal;
                    // Remove surface normal component from forward direction
                    let dot_fn = fwd[0] * n[0] + fwd[1] * n[1] + fwd[2] * n[2];
                    let tangent_fwd = v3_normalize([
                        fwd[0] - n[0] * dot_fn,
                        fwd[1] - n[1] * dot_fn,
                        fwd[2] - n[2] * dot_fn,
                    ]);
                    // Grip: how much of the foot's push actually propels (surface normal Y component)
                    let grip = n[1].max(0.0); // flat=1.0, vertical=0.0
                    // Push magnitude: proportional to phase (strongest at mid-push)
                    let push_phase = cos_phase.max(0.0); // 0..1 during contact
                    let push_mag = push_phase * grip;
                    self.feet[side].push_force = v3_scale(tangent_fwd, push_mag);
                    self.total_push_force = v3_add(self.total_push_force, self.feet[side].push_force);
                } else {
                    self.feet[side].push_force = [0.0; 3];
                }
            } else {
                // Idle: feet planted at rest
                let foot_x = pos[0] + right[0] * 0.10 * lateral_sign;
                let foot_z = pos[2] + right[2] * 0.10 * lateral_sign;
                let ground_y = terrain.height_at(foot_x, foot_z);
                let foot_normal = terrain.normal_at(foot_x, foot_z);
                self.feet[side].grounded = on_ground;
                self.feet[side].ground_y = ground_y;
                self.feet[side].surface_normal = foot_normal;
                self.feet[side].lift_height = 0.0;
                self.feet[side].target_pos = [foot_x, ground_y, foot_z];
                self.feet[side].plant_pos = self.feet[side].target_pos;
                self.feet[side].push_force = [0.0; 3];
            }

            // Two-bone IK: solve leg chain from hip to foot target
            let upper_len = 0.42;
            let lower_len = 0.40;
            let pole_dir = fwd; // knees bend forward
            let (upper_rot, lower_rot) = solve_two_bone_ik(
                hip_pos,
                self.feet[side].target_pos,
                upper_len,
                lower_len,
                pole_dir,
            );

            let upper_bone = if side == 0 { BoneId::LeftUpperLeg } else { BoneId::RightUpperLeg };
            let lower_bone = if side == 0 { BoneId::LeftLowerLeg } else { BoneId::RightLowerLeg };
            self.bones[upper_bone as usize].local_rot = upper_rot;
            self.bones[lower_bone as usize].local_rot = lower_rot;

            // Foot bone: align to surface normal when grounded
            if self.feet[side].grounded {
                let n = self.feet[side].surface_normal;
                // Foot tilts to match surface: rotation from default (pointing down) to surface plane
                let tilt_x = n[2].atan2(n[1]); // pitch from normal
                let tilt_z = -n[0].atan2(n[1]); // roll from normal
                let foot_bone = if side == 0 { BoneId::LeftFoot } else { BoneId::RightFoot };
                self.bones[foot_bone as usize].local_rot = quat_mul(
                    quat_from_axis_angle([1.0, 0.0, 0.0], tilt_x * 0.5),
                    quat_from_axis_angle([0.0, 0.0, 1.0], tilt_z * 0.5),
                );
            }
        }

        // ── Center of mass tracking ──
        // Compute actual CoM from mass-weighted bone positions
        let mut com_sum = [0.0f32; 3];
        let mut mass_sum = 0.0f32;
        for b in &self.bones {
            com_sum = v3_add(com_sum, v3_scale(b.world_pos, b.mass));
            mass_sum += b.mass;
        }
        if mass_sum > 0.0 {
            self.com_world = v3_scale(com_sum, 1.0 / mass_sum);
        }

        // Support base: midpoint between grounded feet (or single foot, or hips if airborne)
        let support_base = if self.feet[0].grounded && self.feet[1].grounded {
            v3_scale(v3_add(self.feet[0].plant_pos, self.feet[1].plant_pos), 0.5)
        } else if self.feet[0].grounded {
            self.feet[0].plant_pos
        } else if self.feet[1].grounded {
            self.feet[1].plant_pos
        } else {
            pos // airborne: use character position
        };

        // Lean = horizontal offset of CoM from support base
        self.com_lean = [
            self.com_world[0] - support_base[0],
            0.0,
            self.com_world[2] - support_base[2],
        ];
        let lean_mag = (self.com_lean[0] * self.com_lean[0] + self.com_lean[2] * self.com_lean[2]).sqrt();

        // Lateral sway: shift hips toward planted foot during walk
        let sway_amount = if self.gait != Gait::Idle {
            self.walk_phase.cos() * 0.015 * (horiz_speed / 1.4).min(1.0)
        } else {
            0.0
        };
        self.com_offset = [right[0] * sway_amount, 0.0, right[2] * sway_amount];

        // ── Arm IK (two-bone solver, same as legs) ──
        // Arms swing forward/backward during locomotion using IK targets.
        // At idle, arms hang at rest alongside the body.
        // Derive bone lengths from skeleton definition (not hardcoded)
        let upper_arm_len = self.bones[BoneId::LeftUpperArm as usize].length;  // 0.28
        let forearm_len = self.bones[BoneId::LeftForearm as usize].length;     // 0.25
        let elbow_pole = v3_scale(fwd, -1.0); // elbows bend backward

        // Shoulder height: sum up the chain hips→spine→chest + chest attachment Y
        // Uses actual bone local_pos values so it stays correct if proportions change
        let shoulder_y = pos[1]
            + self.bones[BoneId::Hips as usize].local_pos[1]      // hips offset from root
            + self.bones[BoneId::Spine as usize].local_pos[1]     // spine above hips
            + self.bones[BoneId::Chest as usize].local_pos[1]     // chest above spine
            + self.bones[BoneId::LeftUpperArm as usize].local_pos[1]; // arm attachment on chest

        for arm_side in 0..2 {
            let upper_bone = if arm_side == 0 { BoneId::LeftUpperArm } else { BoneId::RightUpperArm };
            let lower_bone = if arm_side == 0 { BoneId::LeftForearm } else { BoneId::RightForearm };

            // Shoulder lateral offset from skeleton definition
            let arm_local_x = self.bones[upper_bone as usize].local_pos[0]; // ±0.22
            let shoulder_pos = v3_add(
                [pos[0], shoulder_y, pos[2]],
                v3_scale(right, arm_local_x),
            );

            // Arm swing: left arm swings opposite to left leg (counter-rotation)
            let phase_offset = if arm_side == 0 { std::f32::consts::PI } else { 0.0 };
            let swing_angle = (self.walk_phase + phase_offset).sin() * arm_swing_amp;

            // IK target: hand position at end of swing arc
            let arm_reach = (upper_arm_len + forearm_len) * 0.6;
            let hand_fwd_offset = [fwd[0] * swing_angle * arm_reach, 0.0, fwd[2] * swing_angle * arm_reach];
            // At rest (idle), hands hang below shoulder
            let rest_drop = upper_arm_len + forearm_len - 0.02;
            let hand_target = [
                shoulder_pos[0] + hand_fwd_offset[0],
                shoulder_pos[1] - rest_drop + swing_angle.abs() * 0.05,
                shoulder_pos[2] + hand_fwd_offset[2],
            ];

            let (upper_rot, lower_rot) = solve_two_bone_ik(
                shoulder_pos,
                hand_target,
                upper_arm_len,
                forearm_len,
                elbow_pole,
            );

            self.bones[upper_bone as usize].local_rot = upper_rot;
            self.bones[lower_bone as usize].local_rot = lower_rot;
        }

        // ── Spine lean (gait-specific, forward lean increases with speed) ──
        if self.gait != Gait::Idle {
            self.bones[BoneId::Spine as usize].local_rot =
                quat_from_axis_angle([1.0, 0.0, 0.0], spine_lean_amt);
        } else {
            self.bones[BoneId::Spine as usize].local_rot = QUAT_IDENTITY;
        }

        // ── Anticipation: flinch/brace before incoming vehicle impact ──
        if self.anticipation_timer > 0.0 {
            self.anticipation_timer -= dt;
            let brace_t = (1.0 - self.anticipation_timer).clamp(0.0, 1.0);

            // Turn head toward threat (proper angle wrapping)
            let head_yaw = self.anticipation_dir[0].atan2(self.anticipation_dir[2]);
            let mut rel_yaw = head_yaw - rot_y;
            // Wrap to [-π, π]
            while rel_yaw > std::f32::consts::PI { rel_yaw -= std::f32::consts::TAU; }
            while rel_yaw < -std::f32::consts::PI { rel_yaw += std::f32::consts::TAU; }
            self.bones[BoneId::Head as usize].local_rot = quat_mul(
                self.bones[BoneId::Head as usize].local_rot,
                quat_from_axis_angle([0.0, 1.0, 0.0], rel_yaw.clamp(-0.5, 0.5) * brace_t),
            );

            // Raise arms defensively (flinch: arms cross in front of face/chest)
            let flinch_angle = brace_t * 1.5;
            self.bones[BoneId::LeftUpperArm as usize].local_rot =
                quat_from_axis_angle([1.0, 0.0, 0.3], flinch_angle);
            self.bones[BoneId::RightUpperArm as usize].local_rot =
                quat_from_axis_angle([1.0, 0.0, -0.3], flinch_angle);
            // Forearms fold inward
            self.bones[BoneId::LeftForearm as usize].local_rot =
                quat_from_axis_angle([1.0, 0.0, 0.0], -flinch_angle * 0.8);
            self.bones[BoneId::RightForearm as usize].local_rot =
                quat_from_axis_angle([1.0, 0.0, 0.0], -flinch_angle * 0.8);

            // Slight crouch (spine bend)
            self.bones[BoneId::Spine as usize].local_rot =
                quat_from_axis_angle([1.0, 0.0, 0.0], brace_t * 0.15);
        }

        // ── Stumble triggers (affect BOTH animation AND rigid body) ──
        // 1. Hard landing (vertical impact velocity)
        if on_ground && self.landing_speed < -4.0 {
            let severity = (-self.landing_speed - 4.0) / 8.0;
            self.stumble_timer = severity.clamp(0.2, 1.5);
            // Harder landing = more braking (0.0 = full stop, 0.5 = half speed)
            self.stumble_brake = (1.0 - severity).clamp(0.0, 0.5);
            if horiz_speed > 0.5 {
                self.stumble_dir = v3_normalize([vel[0], 0.0, vel[2]]);
            } else {
                self.stumble_dir = fwd;
            }
            self.landing_speed = 0.0;
        }
        // 2. CoM lean exceeding per-gait balance threshold (off-balance stumble)
        let threshold = self.gait.balance_threshold();
        if on_ground && self.stumble_timer <= 0.0 && lean_mag > threshold {
            self.stumble_timer = (lean_mag - threshold).clamp(0.2, 1.0);
            self.stumble_brake = 0.3; // significant braking when off-balance
            self.stumble_dir = if lean_mag > 0.01 {
                v3_normalize(self.com_lean)
            } else {
                fwd
            };
        }

        // Record vertical velocity for next-frame landing detection
        if !on_ground {
            self.landing_speed = vel[1];
        } else {
            self.landing_speed = 0.0;
        }

        // ── Stumble animation + physics effect ──
        if self.stumble_timer > 0.0 {
            self.stumble_timer -= dt;
            // Gradually recover locomotion control as stumble fades
            if self.stumble_timer <= 0.0 {
                self.stumble_brake = 1.0; // fully recovered
            } else {
                // Ramp brake back toward 1.0 as timer decreases
                let recovery = 1.0 - (self.stumble_timer / 1.5).min(1.0);
                self.stumble_brake = self.stumble_brake + (1.0 - self.stumble_brake) * recovery * 0.5;
            }
            let t = self.stumble_timer.max(0.0);
            let lean_fwd = t * 0.5;
            let wobble = (t * 12.0).sin() * t * 0.15;
            self.bones[BoneId::Spine as usize].local_rot =
                quat_from_axis_angle([1.0, 0.0, 0.0], lean_fwd);
            self.bones[BoneId::Chest as usize].local_rot =
                quat_from_axis_angle([0.0, 0.0, 1.0], wobble);
            let flail = t * 0.8;
            self.bones[BoneId::LeftUpperArm as usize].local_rot =
                quat_from_axis_angle([0.0, 0.0, 1.0], flail);
            self.bones[BoneId::RightUpperArm as usize].local_rot =
                quat_from_axis_angle([0.0, 0.0, -1.0], flail);
        }

        // ── Update world transforms ──
        let root_rot = quat_from_rot_y(rot_y);
        let root_pos = v3_add(pos, self.com_offset);
        self.compute_world_transforms(root_pos, root_rot);
    }

    /// Compute ground reaction force for locomotion.
    /// Returns force vector to apply to the character's rigid body.
    /// This is the actual "legs pushing on ground" force.
    ///
    /// Speed is EMERGENT from the gait's stride parameters (frequency × length),
    /// Animate upper body while driving based on suspension state.
    /// Produces lateral sway, pitch from braking/acceleration, steering lean, and head stabilization.
    /// `suspension_comp`: per-wheel compression [FL, FR, RL, RR]
    /// `speed`: vehicle forward speed (m/s)
    /// `steer`: steering input (-1..1, negative=left, positive=right)
    /// Optionally call with vehicle world pos/rot for steering wheel IK (Phase 4).
    pub fn step_driving_animation(&mut self, suspension_comp: &[f32; 4], speed: f32, steer: f32, _dt: f32) {
        use BoneId::*;

        // Average suspension compression → vertical bounce via Hips offset
        let avg = (suspension_comp[0] + suspension_comp[1] + suspension_comp[2] + suspension_comp[3]) * 0.25;
        let rest = 0.35; // SuspensionParams default rest_length
        let vert_offset = (avg - rest * 0.5) * 0.15;
        self.bones[Hips as usize].local_pos[1] = 0.95 + vert_offset;

        // Lateral sway: left vs right suspension diff → Spine/Chest roll
        let left_comp = (suspension_comp[0] + suspension_comp[2]) * 0.5;
        let right_comp = (suspension_comp[1] + suspension_comp[3]) * 0.5;
        let roll = (left_comp - right_comp) * 0.08; // subtle roll angle

        // Pitch: front vs rear diff → Spine forward/backward lean
        let front_comp = (suspension_comp[0] + suspension_comp[1]) * 0.5;
        let rear_comp = (suspension_comp[2] + suspension_comp[3]) * 0.5;
        let pitch = (front_comp - rear_comp) * 0.06; // braking pitches forward

        // Steering lean: lean into turns at speed
        let steer_lean = steer * speed.abs().min(20.0) * 0.003;

        // Apply to Spine: roll + pitch
        self.bones[Spine as usize].local_rot = quat_mul(
            quat_from_axis_angle([0.0, 0.0, 1.0], roll + steer_lean),
            quat_from_axis_angle([1.0, 0.0, 0.0], pitch),
        );

        // Chest: amplify the sway slightly
        self.bones[Chest as usize].local_rot = quat_mul(
            quat_from_axis_angle([0.0, 0.0, 1.0], (roll + steer_lean) * 0.5),
            quat_from_axis_angle([1.0, 0.0, 0.0], pitch * 0.5),
        );

        // Head stabilization: counter-rotate to stay upright (vestibulo-ocular reflex)
        // Head compensates ~60% of body roll and ~40% of pitch
        let head_roll = -(roll + steer_lean) * 0.9;
        let head_pitch = -pitch * 0.6;
        self.bones[Neck as usize].local_rot = QUAT_IDENTITY;
        self.bones[Head as usize].local_rot = quat_mul(
            quat_from_axis_angle([0.0, 0.0, 1.0], head_roll),
            quat_from_axis_angle([1.0, 0.0, 0.0], head_pitch),
        );

        // Arms: relax into a neutral driving position (slightly forward, hands at ~10-2 o'clock)
        // Upper arms angle forward-down slightly
        let arm_pitch = -0.5; // ~30° forward
        self.bones[LeftUpperArm as usize].local_rot = quat_from_axis_angle([1.0, 0.0, 0.0], arm_pitch);
        self.bones[RightUpperArm as usize].local_rot = quat_from_axis_angle([1.0, 0.0, 0.0], arm_pitch);
        // Forearms bend at elbow ~90°
        let elbow_bend = -1.2; // ~70° bend
        self.bones[LeftForearm as usize].local_rot = quat_from_axis_angle([1.0, 0.0, 0.0], elbow_bend);
        self.bones[RightForearm as usize].local_rot = quat_from_axis_angle([1.0, 0.0, 0.0], elbow_bend);

        // Legs: seated position (thighs forward, knees bent)
        let thigh_angle = 1.4; // ~80° forward
        let knee_bend = -1.8;  // ~100° bend
        self.bones[LeftUpperLeg as usize].local_rot = quat_from_axis_angle([1.0, 0.0, 0.0], thigh_angle);
        self.bones[LeftLowerLeg as usize].local_rot = quat_from_axis_angle([1.0, 0.0, 0.0], knee_bend);
        self.bones[RightUpperLeg as usize].local_rot = quat_from_axis_angle([1.0, 0.0, 0.0], thigh_angle);
        self.bones[RightLowerLeg as usize].local_rot = quat_from_axis_angle([1.0, 0.0, 0.0], knee_bend);
        self.bones[LeftFoot as usize].local_rot = QUAT_IDENTITY;
        self.bones[RightFoot as usize].local_rot = QUAT_IDENTITY;
    }

    /// Apply steering wheel arm IK after step_driving_animation + compute_world_transforms.
    /// `wheel_pos`: world-space steering wheel center
    /// `steer_angle`: current steering wheel rotation (radians, from steer_input * max_steer)
    pub fn apply_steering_wheel_ik(&mut self, wheel_pos: Vec3, steer_angle: f32) {
        use BoneId::*;
        let upper_len = self.bones[LeftUpperArm as usize].length;
        let lower_len = self.bones[LeftForearm as usize].length;

        // Steering wheel grip points rotate with the wheel
        let (sin_s, cos_s) = steer_angle.sin_cos();
        // Left hand at 10 o'clock, right at 2 o'clock positions on wheel (radius ~0.17m)
        let grip_r = 0.17;
        let left_offset = [-grip_r * cos_s, grip_r * sin_s, 0.0];
        let right_offset = [grip_r * cos_s, -grip_r * sin_s, 0.0];

        let left_target = v3_add(wheel_pos, left_offset);
        let right_target = v3_add(wheel_pos, right_offset);

        // Pole direction: forward-down (elbows point forward and slightly down)
        let pole = [0.0, -0.3, -1.0];

        // Left arm IK
        let left_shoulder = self.bones[LeftUpperArm as usize].world_pos;
        let (l_upper, l_lower) = solve_two_bone_ik(left_shoulder, left_target, upper_len, lower_len, pole);
        self.bones[LeftUpperArm as usize].world_rot = l_upper;
        self.bones[LeftForearm as usize].world_rot = l_lower;

        // Right arm IK
        let right_shoulder = self.bones[RightUpperArm as usize].world_pos;
        let (r_upper, r_lower) = solve_two_bone_ik(right_shoulder, right_target, upper_len, lower_len, pole);
        self.bones[RightUpperArm as usize].world_rot = r_upper;
        self.bones[RightForearm as usize].world_rot = r_lower;
    }

    /// NOT from an externally imposed target. The gait determines how fast legs pump;
    /// the surface friction caps how much of that force translates to movement.
    ///
    /// `desired_dir`: normalized direction the character wants to move (zero = braking)
    /// `desired_gait`: the gait the character wants to achieve (Walk/Run/Sprint/Idle)
    /// `current_vel`: current body velocity
    /// `mass`: character mass
    /// `surface_friction`: dynamic friction coefficient of the surface (0..1, from material system)
    pub fn compute_locomotion_force(&self, desired_dir: Vec3, desired_gait: Gait, current_vel: Vec3, mass: f32, surface_friction: f32) -> Vec3 {
        // No force if no foot is grounded
        if !self.feet[0].grounded && !self.feet[1].grounded {
            return [0.0; 3];
        }

        // Speed emerges from gait: stride_freq × stride_len
        // Walk: 2.5 × 0.89 = 2.22 m/s (8 km/h), Run: 3.0 × 1.02 = 3.06 m/s (11 km/h), Sprint: 4.5 × 1.73 = 7.78 m/s (28 km/h)
        let desired_speed = desired_gait.natural_speed();

        // Average slope grip from grounded feet (Y component of surface normal)
        let mut slope_grip = 0.0f32;
        let mut foot_count = 0.0f32;
        for foot in &self.feet {
            if foot.grounded {
                slope_grip += foot.surface_normal[1].max(0.0);
                foot_count += 1.0;
            }
        }
        if foot_count > 0.0 { slope_grip /= foot_count; }

        // Total grip = slope factor × surface friction coefficient
        // Flat asphalt: 1.0 * 0.7 = 0.7 (good grip)
        // Flat ice: 1.0 * 0.04 = 0.04 (nearly no grip)
        // Steep grass: 0.6 * 0.4 = 0.24 (poor grip)
        let total_grip = slope_grip * surface_friction;

        // Push direction: project desired direction onto average surface tangent
        let avg_normal = if self.feet[0].grounded && self.feet[1].grounded {
            v3_normalize(v3_scale(
                v3_add(self.feet[0].surface_normal, self.feet[1].surface_normal), 0.5
            ))
        } else if self.feet[0].grounded {
            self.feet[0].surface_normal
        } else {
            self.feet[1].surface_normal
        };

        // Remove normal component from desired direction (project onto surface)
        let dir_len = v3_len(desired_dir);
        let surface_dir = if dir_len > 0.01 {
            let dot_dn = v3_dot(desired_dir, avg_normal);
            v3_normalize([
                desired_dir[0] - avg_normal[0] * dot_dn,
                desired_dir[1] - avg_normal[1] * dot_dn,
                desired_dir[2] - avg_normal[2] * dot_dn,
            ])
        } else {
            // No desired direction: brake (deceleration)
            let horiz_vel = [current_vel[0], 0.0, current_vel[2]];
            let hv_len = v3_len(horiz_vel);
            if hv_len > 0.1 {
                v3_scale(horiz_vel, -1.0 / hv_len) // push against current motion
            } else {
                return [0.0; 3]; // nearly stopped, no force needed
            }
        };

        // Current speed along surface direction
        let current_along = v3_dot(current_vel, surface_dir);
        let speed_error = if dir_len > 0.01 {
            desired_speed - current_along
        } else {
            // Braking: reduce speed to zero
            -current_along
        };

        // Ground reaction force: legs push against surface, limited by friction
        // max force = friction_coefficient × normal_force (mass × g)
        let max_push = total_grip * mass * 9.81;

        // Force = gait-driven push magnitude, capped by friction
        // At the desired gait, legs produce enough force to sustain natural_speed against damping.
        // The push force per stride is: mass × desired_speed / response_time
        // On low friction, this is capped → character can't reach full gait speed (emergent)
        let response_time = 0.15;
        let desired_force = mass * speed_error / response_time;
        let clamped_force = desired_force.clamp(-max_push, max_push);

        // Scale by walk phase: force peaks when a foot is in push phase
        // This makes force pulsed per footstep, not continuous — legs actually drive the motion
        let push_scale = if self.total_push_force[0].abs() + self.total_push_force[2].abs() > 0.01 {
            let push_len = v3_len(self.total_push_force);
            // Foot in push phase contributes more; between pushes, force drops
            push_len.min(1.0) * 0.7 + 0.3
        } else if desired_speed < 0.1 {
            1.0 // braking: full force always
        } else {
            0.3 // no foot pushing: reduced force (between stride pushes)
        };

        // Stumble reduces locomotion force — character loses traction while off-balance
        let stumble_mult = self.stumble_brake;
        v3_scale(surface_dir, clamped_force * push_scale * stumble_mult)
    }

    /// Return walk swing angle for the renderer (gait-appropriate amplitude).
    pub fn walk_swing(&self) -> f32 {
        self.walk_phase.sin() * self.gait.arm_swing().max(0.4)
    }

    /// Should this character enter ragdoll from fall damage?
    /// Call BEFORE step_animation consumes landing_speed.
    pub fn should_ragdoll_from_fall(&self) -> bool {
        self.landing_speed < -10.0
    }

    /// Blend from ragdoll back to animation over time.
    /// Call each frame while ragdoll_blend < 1.0 after ragdoll timer expires.
    pub fn blend_from_ragdoll(&mut self, pos: Vec3, rot_y: f32, dt: f32) {
        if !self.ragdoll_active { return; }

        // Start blending when timer expires
        if self.ragdoll_timer > 0.0 { return; }

        // Increase blend toward animation
        self.ragdoll_blend += dt * 1.5; // ~0.67s full recovery
        if self.ragdoll_blend >= 1.0 {
            self.ragdoll_blend = 1.0;
            self.ragdoll_active = false;
            // Reset bone velocities
            for b in &mut self.bones {
                b.vel = [0.0; 3];
                b.ang_vel = [0.0; 3];
            }
            return;
        }

        // Compute animation pose target
        let root_rot = quat_from_rot_y(rot_y);
        let anim_bones = {
            let mut temp = self.clone();
            temp.ragdoll_active = false;
            temp.compute_world_transforms(pos, root_rot);
            temp.bones
        };

        // Blend each bone: lerp world_pos, slerp world_rot
        let t = self.ragdoll_blend;
        for i in 0..BONE_COUNT {
            self.bones[i].world_pos = v3_lerp(self.bones[i].world_pos, anim_bones[i].world_pos, t);
            self.bones[i].world_rot = quat_slerp(self.bones[i].world_rot, anim_bones[i].world_rot, t);
            // Dampen velocities as blend increases
            self.bones[i].vel = v3_scale(self.bones[i].vel, 1.0 - t);
        }

        // Ground-bracing IK: during early recovery (blend < 0.5), arms reach toward ground
        if t < 0.5 {
            let brace_strength = 1.0 - t * 2.0; // 1.0 at blend=0, 0 at blend=0.5
            let hips_pos = self.bones[BoneId::Hips as usize].world_pos;
            let ground_y = hips_pos[1] - 0.9; // approximate ground below hips
            let upper_len = self.bones[BoneId::LeftUpperArm as usize].length;
            let lower_len = self.bones[BoneId::LeftForearm as usize].length;

            for side in 0..2 {
                let upper_id = if side == 0 { BoneId::LeftUpperArm } else { BoneId::RightUpperArm };
                let lower_id = if side == 0 { BoneId::LeftForearm } else { BoneId::RightForearm };
                let shoulder_pos = self.bones[upper_id as usize].world_pos;
                // Target: ground below shoulder, slightly forward
                let target = [shoulder_pos[0], ground_y, shoulder_pos[2] - 0.1];
                let pole = [0.0, 1.0, 0.0]; // elbows up
                let (upper_rot, lower_rot) = solve_two_bone_ik(
                    shoulder_pos, target, upper_len, lower_len, pole,
                );
                // Blend IK rotations with current ragdoll rotations
                self.bones[upper_id as usize].world_rot = quat_slerp(
                    self.bones[upper_id as usize].world_rot, upper_rot, brace_strength * 0.6,
                );
                self.bones[lower_id as usize].world_rot = quat_slerp(
                    self.bones[lower_id as usize].world_rot, lower_rot, brace_strength * 0.6,
                );
            }
        }
    }

    /// Initialize full ragdoll from NPC position.
    /// All bones go passive (no motors) — full limp ragdoll.
    pub fn activate_ragdoll(&mut self, pos: Vec3, rot_y: f32, impulse: Vec3) {
        let root_rot = quat_from_rot_y(rot_y);
        self.compute_world_transforms(pos, root_rot);

        // Full ragdoll: ALL bones passive (no PD motors fighting physics)
        self.bone_active = [false; BONE_COUNT];

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
        self.vehicle_contact = None;
        self.anticipation_timer = 0.0;
    }

    /// Step ragdoll physics: gravity, active motors, per-bone collision, constraints.
    /// Euphoria-style: bones with `bone_active[i] == true` have spring motors driving them
    /// toward `bone_target_pos[i]`, giving "keep head up" / "brace arms" behaviors.
    pub fn step_ragdoll(&mut self, terrain: &crate::state::Terrain, dt: f32) {
        if !self.ragdoll_active { return; }

        let gravity = [0.0f32, -9.81, 0.0];

        // ── Active ragdoll motors (PD controllers on active bones) ──
        // Compute animation pose targets for active bones
        // Head tries to stay upright, arms brace toward ground on falling
        self.update_active_targets();

        // Integrate each bone independently
        for i in 0..BONE_COUNT {
            // Active ragdoll motor: position-based spring-damper.
            // Drives active bones toward target world-space positions.
            // F = Kp * (target - current) - Kd * velocity
            let motor_vel_change = if self.bone_active[i] && i > 0 {
                let error = v3_sub(self.bone_target_pos[i], self.bones[i].world_pos);
                let spring = v3_scale(error, RAGDOLL_KP);
                let damp = v3_scale(self.bones[i].vel, RAGDOLL_KD);
                let force = v3_sub(spring, damp);
                // F = ma → Δv = (F/m) * dt
                v3_scale(force, dt / self.bones[i].mass.max(0.1))
            } else {
                [0.0; 3]
            };

            let b = &mut self.bones[i];
            b.vel = v3_add(b.vel, v3_scale(gravity, dt));
            b.vel = v3_add(b.vel, motor_vel_change);

            // Damping (slightly higher for active bones to prevent oscillation)
            let damp = if self.bone_active[i] { 0.95 } else { 0.98 };
            b.vel = v3_scale(b.vel, damp);
            // Integrate position
            b.world_pos = v3_add(b.world_pos, v3_scale(b.vel, dt));

            // Ground collision
            let ground_y = terrain.height_at(b.world_pos[0], b.world_pos[2]);
            if b.world_pos[1] < ground_y {
                b.world_pos[1] = ground_y;
                if b.vel[1] < 0.0 {
                    b.vel[1] *= -0.2;
                }
                b.vel[0] *= 0.7;
                b.vel[2] *= 0.7;
            }

            // World bounds
            b.world_pos[0] = b.world_pos[0].clamp(-crate::state::WORLD_HALF, crate::state::WORLD_HALF);
            b.world_pos[2] = b.world_pos[2].clamp(-crate::state::WORLD_HALF, crate::state::WORLD_HALF);
        }

        // ── Continuous vehicle contact ──
        // Process all bones in contact with a vehicle surface.
        // Contact normal and position are updated each frame by the collision system.
        if let Some(vc) = self.vehicle_contact {
            let normal = vc.surface_normal;
            let friction = vc.friction;
            let mut any_contact = false;

            for bi in 0..BONE_COUNT {
                if !vc.bone_contacts[bi] { continue; }

                // Apply surface constraint: prevent penetration, apply friction
                let vn = v3_dot(self.bones[bi].vel, normal);
                if vn < 0.0 {
                    // Remove penetrating velocity
                    self.bones[bi].vel = v3_sub(self.bones[bi].vel, v3_scale(normal, vn));
                    // Apply friction to tangential velocity
                    let v_tangent = v3_sub(self.bones[bi].vel, v3_scale(normal, v3_dot(self.bones[bi].vel, normal)));
                    self.bones[bi].vel = v3_add(
                        v3_scale(normal, v3_dot(self.bones[bi].vel, normal)),
                        v3_scale(v_tangent, 1.0 - friction),
                    );
                    any_contact = true;
                }
            }

            if !any_contact {
                // No bones are actively contacting — clear contact state
                self.vehicle_contact = None;
            }
        }

        // Enforce distance + joint angle constraints
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

                let mi = self.bones[i].mass;
                let mp = self.bones[parent_idx].mass;
                let total = mi + mp;
                let wi = mp / total;
                let wp = mi / total;

                self.bones[i].world_pos = v3_sub(self.bones[i].world_pos, v3_scale(correction, wi));
                self.bones[parent_idx].world_pos = v3_add(self.bones[parent_idx].world_pos, v3_scale(correction, wp));

                // Cone constraint: child direction must stay within cone_angle of parent's axis.
                // Parent axis derived from grandparent→parent direction (not stale world_rot).
                let cone_limit = self.bones[i].constraint.cone_angle;
                if cone_limit < std::f32::consts::PI - 0.01 {
                    let parent_axis = if let Some(gp) = self.bones[parent_idx].parent {
                        let d = v3_sub(self.bones[parent_idx].world_pos, self.bones[gp as usize].world_pos);
                        let dl = v3_len(d);
                        if dl > 0.001 { v3_scale(d, 1.0 / dl) } else { [0.0, -1.0, 0.0] }
                    } else {
                        [0.0, -1.0, 0.0] // root: default downward
                    };
                    let child_dir = v3_sub(self.bones[i].world_pos, self.bones[parent_idx].world_pos);
                    let child_dist = v3_len(child_dir);
                    if child_dist > 0.001 {
                        let child_dir_n = v3_scale(child_dir, 1.0 / child_dist);
                        let dot = v3_dot(child_dir_n, parent_axis).clamp(-1.0, 1.0);
                        let angle = dot.acos();

                        if angle > cone_limit {
                            let cross = v3_cross(parent_axis, child_dir_n);
                            let cross_len = v3_len(cross);
                            if cross_len > 0.001 {
                                let axis = v3_scale(cross, 1.0 / cross_len);
                                let clamped_q = quat_from_axis_angle(axis, cone_limit);
                                let clamped_dir = quat_rotate(clamped_q, parent_axis);
                                let new_pos = v3_add(
                                    self.bones[parent_idx].world_pos,
                                    v3_scale(clamped_dir, child_dist),
                                );
                                let move_vec = v3_sub(new_pos, self.bones[i].world_pos);
                                self.bones[i].world_pos = v3_add(self.bones[i].world_pos, v3_scale(move_vec, wi));
                                let vel_along = v3_dot(self.bones[i].vel, v3_normalize(move_vec));
                                if vel_along < 0.0 {
                                    let dampened = v3_scale(v3_normalize(move_vec), vel_along * 0.5);
                                    self.bones[i].vel = v3_sub(self.bones[i].vel, dampened);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Timer
        self.ragdoll_timer -= dt;
    }

    /// Update active ragdoll motor targets: world-space positions that spring motors drive toward.
    /// Head: stay upright above hips. Arms: brace toward ground when falling.
    /// Spine: resist collapse, keep torso erect. Legs: default hanging positions.
    fn update_active_targets(&mut self) {
        let hips = self.bones[BoneId::Hips as usize].world_pos;

        // Hips target = current position (root is physics-driven, no motor on root)
        self.bone_target_pos[BoneId::Hips as usize] = hips;

        // Spine chain: keep torso upright above hips
        self.bone_target_pos[BoneId::Spine as usize] = v3_add(hips, [0.0, 0.15, 0.0]);
        self.bone_target_pos[BoneId::Chest as usize] = v3_add(hips, [0.0, 0.35, 0.0]);
        self.bone_target_pos[BoneId::Neck as usize] = v3_add(hips, [0.0, 0.60, 0.0]);
        self.bone_target_pos[BoneId::Head as usize] = v3_add(hips, [0.0, 0.70, 0.0]);

        // Arms: behavior depends on vertical velocity
        let hips_vel_y = self.bones[BoneId::Hips as usize].vel[1];
        if hips_vel_y < -2.0 {
            // Falling: brace arms forward and down for impact
            let brace = ((-hips_vel_y - 2.0) / 8.0).clamp(0.0, 1.0);
            let fwd = brace * 0.3;
            let drop = brace * 0.2;
            self.bone_target_pos[BoneId::LeftUpperArm as usize] =
                v3_add(hips, [-0.22, 0.55 - drop, -fwd]);
            self.bone_target_pos[BoneId::RightUpperArm as usize] =
                v3_add(hips, [0.22, 0.55 - drop, -fwd]);
            self.bone_target_pos[BoneId::LeftForearm as usize] =
                v3_add(hips, [-0.22, 0.27 - drop, -fwd * 1.5]);
            self.bone_target_pos[BoneId::RightForearm as usize] =
                v3_add(hips, [0.22, 0.27 - drop, -fwd * 1.5]);
        } else {
            // Not falling: arms hang at sides
            self.bone_target_pos[BoneId::LeftUpperArm as usize] = v3_add(hips, [-0.22, 0.55, 0.0]);
            self.bone_target_pos[BoneId::RightUpperArm as usize] = v3_add(hips, [0.22, 0.55, 0.0]);
            self.bone_target_pos[BoneId::LeftForearm as usize] = v3_add(hips, [-0.22, 0.27, 0.0]);
            self.bone_target_pos[BoneId::RightForearm as usize] = v3_add(hips, [0.22, 0.27, 0.0]);
        }

        // Legs: default positions below hips
        self.bone_target_pos[BoneId::LeftUpperLeg as usize] = v3_add(hips, [-0.10, -0.42, 0.0]);
        self.bone_target_pos[BoneId::RightUpperLeg as usize] = v3_add(hips, [0.10, -0.42, 0.0]);
        self.bone_target_pos[BoneId::LeftLowerLeg as usize] = v3_add(hips, [-0.10, -0.82, 0.0]);
        self.bone_target_pos[BoneId::RightLowerLeg as usize] = v3_add(hips, [0.10, -0.82, 0.0]);
        self.bone_target_pos[BoneId::LeftFoot as usize] = v3_add(hips, [-0.10, -0.95, 0.0]);
        self.bone_target_pos[BoneId::RightFoot as usize] = v3_add(hips, [0.10, -0.95, 0.0]);
    }

    /// Activate partial ragdoll: only specified body region goes limp,
    /// rest stays motor-controlled. Used for localized hits (e.g., knee hit → legs limp).
    pub fn activate_partial_ragdoll(&mut self, pos: Vec3, rot_y: f32, impulse: Vec3, impact_bone: BoneId) {
        if self.ragdoll_active { return; }

        let root_rot = quat_from_rot_y(rot_y);
        self.compute_world_transforms(pos, root_rot);

        // All bones start with motor active
        self.bone_active = [true; BONE_COUNT];

        // Disable motors on the impact bone and its descendants
        let impact_idx = impact_bone as usize;
        self.bone_active[impact_idx] = false;
        // Flood-fill: disable all children of the impact bone
        for i in 0..BONE_COUNT {
            if let Some(parent) = self.bones[i].parent {
                if !self.bone_active[parent as usize] {
                    self.bone_active[i] = false;
                }
            }
        }

        // Apply impulse only to passive (limp) bones
        for i in 0..BONE_COUNT {
            if self.bone_active[i] {
                self.bones[i].vel = [0.0; 3];
            } else {
                self.bones[i].vel = v3_scale(impulse, 1.0 / self.bones[i].mass.max(1.0));
            }
        }

        self.ragdoll_active = true;
        self.ragdoll_blend = 0.0;
        self.vehicle_contact = None;
        self.anticipation_timer = 0.0;
    }

    /// Per-bone vehicle collision: test each bone capsule against a vehicle OBB.
    /// Each bone is treated as a capsule (line segment from root to tip with radius).
    /// Returns the bone index and collision normal for the deepest penetrating bone.
    pub fn collide_bones_vs_vehicle(
        &mut self,
        veh_pos: Vec3,
        veh_rot_y: f32,
        veh_half_w: f32,
        veh_half_d: f32,
        veh_height: f32,    // vehicle hood height
        veh_vel: Vec3,
        bone_radius: f32,
    ) -> Option<(BoneId, Vec3, f32)> {
        // Vehicle local frame
        let (sin_r, cos_r) = veh_rot_y.sin_cos();

        let mut best_pen = 0.0f32;
        let mut best_bone = None;
        let mut best_normal = [0.0f32; 3];

        for i in 0..BONE_COUNT {
            // Capsule collision: test 3 points along the bone (root, midpoint, tip).
            // Bone direction derived from parent→bone position (not stale world_rot).
            let bone_root = self.bones[i].world_pos;
            let parent_c = self.bones[i].parent.unwrap_or(0) as usize;
            let to_bone = v3_sub(bone_root, self.bones[parent_c].world_pos);
            let to_bone_len = v3_len(to_bone);
            let bone_dir = if to_bone_len > 0.001 {
                v3_scale(to_bone, self.bones[i].length / to_bone_len)
            } else {
                [0.0, -self.bones[i].length, 0.0]
            };
            let bone_tip = v3_add(bone_root, bone_dir);
            let bone_mid = v3_add(bone_root, v3_scale(bone_dir, 0.5));

            let sample_points = [bone_root, bone_mid, bone_tip];

            for sp in &sample_points {
                let dx = sp[0] - veh_pos[0];
                let dz = sp[2] - veh_pos[2];
                let local_x = dx * cos_r + dz * sin_r;
                let local_z = -dx * sin_r + dz * cos_r;
                let local_y = sp[1] - veh_pos[1];

                let pen_x = (veh_half_w + bone_radius) - local_x.abs();
                let pen_z = (veh_half_d + bone_radius) - local_z.abs();
                let pen_y = (veh_height + bone_radius) - local_y;

                if pen_x > 0.0 && pen_z > 0.0 && pen_y > 0.0 && local_y > -bone_radius {
                    let min_pen = pen_x.min(pen_z).min(pen_y);
                    if min_pen > best_pen {
                        best_pen = min_pen;
                        best_bone = Some(i);

                        let local_normal = if pen_y <= pen_x && pen_y <= pen_z {
                            [0.0, 1.0, 0.0] // pushed up (on hood)
                        } else if pen_x <= pen_z {
                            [local_x.signum(), 0.0, 0.0] // pushed sideways
                        } else {
                            [0.0, 0.0, local_z.signum()] // pushed forward/back
                        };
                        best_normal = [
                            local_normal[0] * cos_r - local_normal[2] * sin_r,
                            local_normal[1],
                            local_normal[0] * sin_r + local_normal[2] * cos_r,
                        ];
                    }
                }
            }
        }

        if let Some(bi) = best_bone {
            // Determine bone ID
            let bone_id = match bi {
                0 => BoneId::Hips, 1 => BoneId::Spine, 2 => BoneId::Chest,
                3 => BoneId::Neck, 4 => BoneId::Head,
                5 => BoneId::LeftUpperArm, 6 => BoneId::LeftForearm,
                7 => BoneId::RightUpperArm, 8 => BoneId::RightForearm,
                9 => BoneId::LeftUpperLeg, 10 => BoneId::LeftLowerLeg,
                11 => BoneId::LeftFoot, 12 => BoneId::RightUpperLeg,
                13 => BoneId::RightLowerLeg, _ => BoneId::RightFoot,
            };

            // Apply collision response to the hit bone (scoped borrow)
            self.bones[bi].world_pos = v3_add(self.bones[bi].world_pos, v3_scale(best_normal, best_pen));
            let rel_vel = v3_sub(self.bones[bi].vel, veh_vel);
            let rel_vn = v3_dot(rel_vel, best_normal);
            if rel_vn < 0.0 {
                self.bones[bi].vel = v3_sub(self.bones[bi].vel, v3_scale(best_normal, rel_vn * 1.3));
                let veh_vn = v3_dot(veh_vel, best_normal);
                if veh_vn.abs() > 0.5 {
                    self.bones[bi].vel = v3_add(self.bones[bi].vel, v3_scale(best_normal, veh_vn * 0.3));
                }
            }

            // Per-bone contact flags: mark all bones inside vehicle OBB
            let mut contacts = [false; BONE_COUNT];
            for ci in 0..BONE_COUNT {
                let cp = self.bones[ci].world_pos;
                let cdx = cp[0] - veh_pos[0];
                let cdz = cp[2] - veh_pos[2];
                let clx = cdx * cos_r + cdz * sin_r;
                let clz = -cdx * sin_r + cdz * cos_r;
                let cly = cp[1] - veh_pos[1];
                if clx.abs() < veh_half_w + bone_radius
                    && clz.abs() < veh_half_d + bone_radius
                    && cly > -bone_radius && cly < veh_height + bone_radius * 2.0
                {
                    contacts[ci] = true;
                }
            }
            // Convert primary bone's position to vehicle local space for tracking
            let bp = self.bones[bi].world_pos;
            let bdx = bp[0] - veh_pos[0];
            let bdz = bp[2] - veh_pos[2];
            let local_off = [
                bdx * cos_r + bdz * sin_r,
                bp[1] - veh_pos[1],
                -bdx * sin_r + bdz * cos_r,
            ];
            self.vehicle_contact = Some(VehicleContact {
                vehicle_idx: 0, // caller sets this
                bone_contacts: contacts,
                surface_normal: best_normal,
                local_offset: local_off,
                friction: 0.3,
                time: 0.0,
            });

            Some((bone_id, best_normal, best_pen))
        } else {
            None
        }
    }

    /// Set up anticipation state: NPC braces/flinches before incoming impact.
    /// Called when a fast-moving vehicle is detected heading toward the NPC.
    pub fn start_anticipation(&mut self, threat_dir: Vec3, time: f32) {
        self.anticipation_timer = time;
        self.anticipation_dir = threat_dir;
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
