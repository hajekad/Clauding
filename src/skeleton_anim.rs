//! Skeletal animation runtime — evaluates FBX animation clips and skins meshes.
//! Designed for the Mixamo skeleton with 65 bones driving the beauty_girl mesh.

use crate::fbx_anim::{FbxSkeleton, AnimationClip, BoneChannel};
use crate::math::*;
use crate::state::WorldTri;

// ── Animation clip indices (must match load order in fbx_anim::load_all_animations) ──

pub const CLIP_WALKING: usize = 0;
pub const CLIP_RUN_FORWARD: usize = 1;
pub const CLIP_ELBOW_PUNCH: usize = 2;
pub const CLIP_HOOK_PUNCH: usize = 3;
pub const CLIP_ROUNDHOUSE_KICK: usize = 4;
pub const CLIP_DROP_KICK: usize = 5;
pub const CLIP_PICKING_UP: usize = 6;
pub const CLIP_SITTING_POSE: usize = 7;

// ── Pose evaluation ─────────────────────────────────────────────────────

/// Compute skeleton normalization parameters: scale factor and Y offset.
/// Used to scale bone bind_translations so the skeleton matches the 1.8m mesh.
/// The FBX skeleton was created by Mixamo from our 1.8m OBJ export.
/// No normalization needed — the skeleton IS at 1.8m scale already.

/// Evaluate animation at time `t`, returning per-bone world-space transforms.
pub fn evaluate_pose(skeleton: &FbxSkeleton, clip: &AnimationClip, t: f32) -> Vec<Mat4> {
    let n = skeleton.bones.len();
    let mut world_transforms: Vec<Mat4> = vec![M4_IDENTITY; n];

    for i in 0..n {
        let bone = &skeleton.bones[i];
        let mut translation = bone.bind_translation; // already normalized
        let mut rotation_deg = bone.bind_rotation;

        if let Some(channel) = clip.bone_channels.iter().find(|c| c.bone_index == i) {
            let anim_t = if clip.looping && clip.duration > 0.0 {
                t % clip.duration
            } else {
                t.min(clip.duration)
            };
            rotation_deg = interpolate_channel_rotation(channel, anim_t);
            if let Some(ref trans_keys) = channel.translations {
                translation = interpolate_channel_translation(channel, trans_keys, anim_t);
            }
        }

        let rx = rotation_deg[0].to_radians();
        let ry = rotation_deg[1].to_radians();
        let rz = rotation_deg[2].to_radians();
        let rot_quat = quat_from_euler_xyz(rx, ry, rz);
        let rot_mat = quat_to_mat3(rot_quat);
        let local = m4_from_rot3_translation(&rot_mat, translation);

        if let Some(parent) = bone.parent {
            world_transforms[i] = m4_mul(&world_transforms[parent], &local);
        } else {
            world_transforms[i] = local;
        }
    }

    world_transforms
}

/// Compute the bind-pose world transforms, normalized to match the mesh
/// coordinate system (Y=0 at feet, 1.8m tall, centered X/Z).
pub fn compute_bind_pose(skeleton: &FbxSkeleton) -> Vec<Mat4> {
    compute_raw_pose(skeleton, None)
}

/// Compute raw (un-normalized) bone world transforms from the skeleton bind pose,
/// optionally applying animation channel overrides.
fn compute_raw_pose(skeleton: &FbxSkeleton, clip_and_time: Option<(&AnimationClip, f32)>) -> Vec<Mat4> {
    let n = skeleton.bones.len();
    let mut world_transforms: Vec<Mat4> = vec![M4_IDENTITY; n];

    for i in 0..n {
        let bone = &skeleton.bones[i];
        let mut translation = bone.bind_translation;
        let mut rotation_deg = bone.bind_rotation;

        // Apply animation overrides if provided
        if let Some((clip, t)) = clip_and_time {
            if let Some(channel) = clip.bone_channels.iter().find(|c| c.bone_index == i) {
                let anim_t = if clip.looping && clip.duration > 0.0 {
                    t % clip.duration
                } else {
                    t.min(clip.duration)
                };
                rotation_deg = interpolate_channel_rotation(channel, anim_t);
                if let Some(ref trans_keys) = channel.translations {
                    translation = interpolate_channel_translation(channel, trans_keys, anim_t);
                }
            }
        }

        let rx = rotation_deg[0].to_radians();
        let ry = rotation_deg[1].to_radians();
        let rz = rotation_deg[2].to_radians();
        let rot_quat = quat_from_euler_xyz(rx, ry, rz);
        let rot_mat = quat_to_mat3(rot_quat);
        let local = m4_from_rot3_translation(&rot_mat, translation);

        if let Some(parent) = bone.parent {
            world_transforms[i] = m4_mul(&world_transforms[parent], &local);
        } else {
            world_transforms[i] = local;
        }
    }

    world_transforms
}

/// Compute inverse bind-pose matrices (used for skinning).
pub fn compute_inverse_bind_matrices(skeleton: &FbxSkeleton) -> Vec<Mat4> {
    let bind_pose = compute_bind_pose(skeleton);
    bind_pose.iter().map(|m| m4_inverse_affine(m)).collect()
}

// ── Keyframe interpolation ──────────────────────────────────────────────

fn interpolate_channel_rotation(channel: &BoneChannel, t: f32) -> [f32; 3] {
    if channel.rotations.is_empty() {
        return [0.0; 3];
    }
    if channel.rotations.len() == 1 || t <= channel.times[0] {
        return channel.rotations[0];
    }
    if t >= *channel.times.last().unwrap() {
        return *channel.rotations.last().unwrap();
    }

    // Find the two keyframes that bracket t
    let mut k0 = 0;
    for i in 1..channel.times.len() {
        if channel.times[i] >= t {
            k0 = i - 1;
            break;
        }
    }
    let k1 = k0 + 1;
    let dt = channel.times[k1] - channel.times[k0];
    let frac = if dt > 0.0 { (t - channel.times[k0]) / dt } else { 0.0 };

    // Lerp the Euler angles. FBX keyframes are dense enough (60fps) that
    // the inter-frame deltas are small, making linear interpolation safe.
    let r0 = channel.rotations[k0];
    let r1 = channel.rotations[k1];
    [
        r0[0] + (r1[0] - r0[0]) * frac,
        r0[1] + (r1[1] - r0[1]) * frac,
        r0[2] + (r1[2] - r0[2]) * frac,
    ]
}

fn interpolate_channel_translation(channel: &BoneChannel, trans_keys: &[[f32; 3]], t: f32) -> [f32; 3] {
    if trans_keys.is_empty() {
        return [0.0; 3];
    }
    if trans_keys.len() == 1 || t <= channel.times[0] {
        return trans_keys[0];
    }
    if t >= *channel.times.last().unwrap() {
        return *trans_keys.last().unwrap();
    }

    let mut k0 = 0;
    for i in 1..channel.times.len() {
        if channel.times[i] >= t {
            k0 = i - 1;
            break;
        }
    }
    let k1 = k0 + 1;
    let dt = channel.times[k1] - channel.times[k0];
    let frac = if dt > 0.0 { (t - channel.times[k0]) / dt } else { 0.0 };

    let t0 = trans_keys[k0];
    let t1 = trans_keys[k1];
    [
        t0[0] + (t1[0] - t0[0]) * frac,
        t0[1] + (t1[1] - t0[1]) * frac,
        t0[2] + (t1[2] - t0[2]) * frac,
    ]
}

// ── Bone assignment ─────────────────────────────────────────────────────

/// Assign each vertex to the nearest bone based on position.
/// Returns a Vec of bone indices, one per vertex (3 per triangle).
/// Uses the bind-pose bone world positions for nearest-bone lookup.
pub fn compute_bone_assignments(
    skeleton: &FbxSkeleton,
    bind_world: &[Mat4],
    tris: &[WorldTri],
) -> Vec<usize> {
    // Extract world-space bone positions from the bind pose
    let bone_positions: Vec<Vec3> = bind_world.iter()
        .map(|m| [m[12], m[13], m[14]]) // translation column
        .collect();

    // Find the Hips bone Y position to know the model's coordinate system
    let hips_idx = skeleton.bones.iter().position(|b| b.name.contains("Hips")).unwrap_or(0);
    let hips_y = bone_positions[hips_idx][1];

    // Build a list of "important" bones (skip finger bones and end bones for efficiency)
    // The Mixamo skeleton has 65 bones, but for skinning we only need ~20 major ones
    let important_bones: Vec<usize> = (0..skeleton.bones.len())
        .filter(|&i| {
            let name = &skeleton.bones[i].name;
            // Skip end bones (HeadTop_End, Toe_End, etc.)
            if name.contains("_End") || name.contains("End") { return false; }
            // Skip individual finger bones — assign them to the Hand bone
            if name.contains("Thumb") || name.contains("Index") ||
               name.contains("Middle") || name.contains("Ring") ||
               name.contains("Pinky") { return false; }
            true
        })
        .collect();

    let mut assignments = Vec::with_capacity(tris.len() * 3);

    for tri in tris {
        for v in &tri.v {
            // Find the nearest important bone to this vertex
            let mut best_bone = hips_idx;
            let mut best_dist = f32::MAX;

            for &bi in &important_bones {
                let bp = bone_positions[bi];
                let dx = v[0] - bp[0];
                let dy = v[1] - bp[1];
                let dz = v[2] - bp[2];
                let d = dx*dx + dy*dy + dz*dz;

                // Weight Y distance more heavily — bones are arranged vertically
                let dy_weight = (v[1] - bp[1]).abs() * 2.0;
                let weighted_d = d + dy_weight * dy_weight;

                if weighted_d < best_dist {
                    best_dist = weighted_d;
                    best_bone = bi;
                }
            }

            // For finger vertices, remap to the Hand bone
            let final_bone = remap_to_parent_if_finger(skeleton, best_bone);
            assignments.push(final_bone);
        }
    }

    // Hips root motion offset: for the walk animation, we want the character
    // to stay in place (the game drives world position). Record the hips Y so
    // we can subtract root translation in skinning.
    let _ = hips_y;

    assignments
}

/// If a bone is a finger bone, remap it to the nearest Hand bone
fn remap_to_parent_if_finger(skeleton: &FbxSkeleton, bone_idx: usize) -> usize {
    let name = &skeleton.bones[bone_idx].name;
    if name.contains("Thumb") || name.contains("Index") ||
       name.contains("Middle") || name.contains("Ring") ||
       name.contains("Pinky") {
        // Walk up to find the Hand bone
        let mut idx = bone_idx;
        for _ in 0..5 {
            if let Some(parent) = skeleton.bones[idx].parent {
                if skeleton.bones[parent].name.contains("Hand") {
                    return parent;
                }
                idx = parent;
            } else {
                break;
            }
        }
    }
    bone_idx
}

// ── Mesh skinning ───────────────────────────────────────────────────────

/// Apply skeletal animation to a mesh. Transforms each vertex by its assigned
/// bone's world transform, relative to the bind pose.
///
/// The skinning formula per vertex:
///   world_pos = bone_world_transform * inverse_bind_transform * bind_pos
///
/// This deforms the mesh from its bind pose (T-pose) into the animated pose.
pub fn skin_mesh(
    bind_tris: &[WorldTri],
    bone_assignments: &[usize],
    animated_transforms: &[Mat4],
    inverse_bind: &[Mat4],
    output: &mut Vec<WorldTri>,
) {
    // Pre-compute skinning matrices: animated * inverse_bind per bone
    let n_bones = animated_transforms.len().min(inverse_bind.len());
    let mut skin_matrices: Vec<Mat4> = Vec::with_capacity(n_bones);
    for i in 0..n_bones {
        skin_matrices.push(m4_mul(&animated_transforms[i], &inverse_bind[i]));
    }

    let base = output.len();
    output.extend_from_slice(bind_tris);

    let mut vi = 0; // vertex index into bone_assignments
    for tri in &mut output[base..] {
        for v in &mut tri.v {
            if vi < bone_assignments.len() {
                let bi = bone_assignments[vi];
                if bi < skin_matrices.len() {
                    *v = m4_transform_point(&skin_matrices[bi], *v);
                }
            }
            vi += 1;
        }
        // Transform normal using the first vertex's bone
        let tri_vi = vi - 3; // index of first vertex of this tri
        if tri_vi < bone_assignments.len() {
            let bi = bone_assignments[tri_vi];
            if bi < skin_matrices.len() {
                tri.normal = m4_transform_normal(&skin_matrices[bi], tri.normal);
            }
        }
    }
}

/// Recenter the skinned mesh so the Hips bone is at origin (0, hips_y, 0).
/// This removes root motion from the animation so the game controls position.
pub fn recenter_skinned_mesh(
    output: &mut [WorldTri],
    animated_transforms: &[Mat4],
    bind_transforms: &[Mat4],
    hips_bone: usize,
) {
    // The animated Hips position in model space
    let anim_hips = [animated_transforms[hips_bone][12],
                     animated_transforms[hips_bone][13],
                     animated_transforms[hips_bone][14]];
    // The bind Hips position
    let bind_hips = [bind_transforms[hips_bone][12],
                     bind_transforms[hips_bone][13],
                     bind_transforms[hips_bone][14]];

    // Offset to subtract: remove horizontal root motion, keep vertical bobbing
    let dx = anim_hips[0] - bind_hips[0];
    let dz = anim_hips[2] - bind_hips[2];

    if dx.abs() < 0.001 && dz.abs() < 0.001 { return; }

    for tri in output.iter_mut() {
        for v in &mut tri.v {
            v[0] -= dx;
            v[2] -= dz;
        }
    }
}

// ── Animation state helper ──────────────────────────────────────────────

/// Cached animation data ready for per-frame skinning.
/// Stored on GameState, computed once at startup.
pub struct AnimationData {
    pub skeleton: FbxSkeleton,
    pub clips: Vec<AnimationClip>,
    pub model_bone_assignments: Vec<Vec<usize>>, // per-model, per-vertex bone index
    pub inverse_bind: Vec<Mat4>,         // inverse bind-pose matrices
    pub bind_transforms: Vec<Mat4>,      // bind-pose world transforms
    pub hips_bone: usize,                // index of the Hips bone
}

impl AnimationData {
    /// Load all FBX animations and compute bone assignments for ALL character models.
    /// Each model gets its own bone assignment set — the Mixamo skeleton is reused.
    pub fn load(anim_dir: &str, all_models: &[Vec<WorldTri>]) -> Self {
        let (mut skeleton, clips) = crate::fbx_anim::load_all_animations(anim_dir);

        if skeleton.bones.is_empty() {
            eprintln!("[skeleton_anim] No bones found — animation disabled");
            return AnimationData {
                skeleton,
                clips,
                model_bone_assignments: Vec::new(),
                inverse_bind: Vec::new(),
                bind_transforms: Vec::new(),
                hips_bone: 0,
            };
        }

        // Normalize skeleton bind translations to match 1.8m mesh coordinate system
        // No skeleton normalization needed — the FBX was created from our 1.8m OBJ
        let bind_transforms = compute_bind_pose(&skeleton);
        let inverse_bind = compute_inverse_bind_matrices(&skeleton);
        let hips_bone = skeleton.bones.iter()
            .position(|b| b.name.contains("Hips") && !b.name.contains("UpLeg"))
            .unwrap_or(0);

        eprintln!("[skeleton_anim] hips Y={:.3}, {} bones",
            bind_transforms[hips_bone][13], skeleton.bones.len());

        // Compute bone assignments for every character model
        let mut model_bone_assignments = Vec::with_capacity(all_models.len());
        for (i, model_tris) in all_models.iter().enumerate() {
            let assignments = compute_bone_assignments(&skeleton, &bind_transforms, model_tris);
            eprintln!("[skeleton_anim] model[{}]: {} verts assigned to {} bones",
                i, assignments.len(), skeleton.bones.len());
            model_bone_assignments.push(assignments);
        }

        AnimationData {
            skeleton,
            clips,
            model_bone_assignments,
            inverse_bind,
            bind_transforms,
            hips_bone,
        }
    }

    /// Backward-compatible load for a single mesh (used by body_view)
    pub fn load_single(anim_dir: &str, mesh_tris: &[WorldTri]) -> Self {
        Self::load(anim_dir, &[mesh_tris.to_vec()])
    }

    /// Check if animation data is available
    pub fn is_ready(&self) -> bool {
        !self.skeleton.bones.is_empty() && !self.clips.is_empty() && !self.model_bone_assignments.is_empty()
    }

    /// Determine which clip + time to use based on game state.
    /// All 8 animations are wired:
    ///   - walking.fbx: walk_phase > 0, not sprinting
    ///   - run_forward.fbx: walk_phase > 0, sprinting
    ///   - elbow_punch.fbx, hook_punch.fbx: attack_phase > 0 (alternating by attack count)
    ///   - roundhouse_kick.fbx, drop_kick.fbx: kick attacks (future use, mapped via attack_type)
    ///   - picking_up.fbx: picking up items (future use)
    ///   - sitting_pose.fbx: sitting on benches
    pub fn select_clip(
        &self,
        walk_phase: f32,
        attack_phase: f32,
        sitting: bool,
        sprinting: bool,
        attack_type: u8, // 0=punch, 1=hook, 2=kick, 3=dropkick
    ) -> Option<(usize, f32)> {
        if !self.is_ready() { return None; }

        // Sitting pose
        if sitting && CLIP_SITTING_POSE < self.clips.len() {
            return Some((CLIP_SITTING_POSE, 0.0));
        }

        // Attack animations — select from 4 combat clips based on attack_type
        if attack_phase > 0.0 {
            let clip_idx = match attack_type {
                1 => CLIP_HOOK_PUNCH,
                2 => CLIP_ROUNDHOUSE_KICK,
                3 => CLIP_DROP_KICK,
                _ => CLIP_ELBOW_PUNCH,
            };
            if clip_idx < self.clips.len() {
                let clip = &self.clips[clip_idx];
                let t = (attack_phase / crate::state::ATTACK_ANIM_DURATION).clamp(0.0, 1.0) * clip.duration;
                return Some((clip_idx, t));
            }
        }

        // Locomotion: sprint uses run_forward, walk uses walking
        if walk_phase.abs() > 0.01 {
            let clip_idx = if sprinting && CLIP_RUN_FORWARD < self.clips.len() {
                CLIP_RUN_FORWARD
            } else if CLIP_WALKING < self.clips.len() {
                CLIP_WALKING
            } else {
                return None;
            };
            let clip = &self.clips[clip_idx];
            let t = (walk_phase.abs() % clip.duration).abs();
            return Some((clip_idx, t));
        }

        // Idle: first frame of walking for a natural rest pose
        if CLIP_WALKING < self.clips.len() {
            return Some((CLIP_WALKING, 0.0));
        }

        None
    }

    /// Generate animated mesh triangles into the output buffer.
    /// model_index selects which bone assignment set to use.
    pub fn generate_animated_mesh(
        &self,
        bind_tris: &[WorldTri],
        model_index: usize,
        clip_index: usize,
        time: f32,
        output: &mut Vec<WorldTri>,
    ) {
        let assignments = self.model_bone_assignments.get(model_index)
            .or_else(|| self.model_bone_assignments.first());
        if !self.is_ready() || clip_index >= self.clips.len() || assignments.is_none() {
            output.extend_from_slice(bind_tris);
            return;
        }

        let clip = &self.clips[clip_index];
        let animated = evaluate_pose(&self.skeleton, clip, time);

        let start = output.len();
        skin_mesh(bind_tris, assignments.unwrap(), &animated, &self.inverse_bind, output);
        recenter_skinned_mesh(&mut output[start..], &animated, &self.bind_transforms, self.hips_bone);
    }
}
