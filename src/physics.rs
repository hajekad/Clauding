//! Core rigid body physics — integration, collision shapes, contact solver.
//!
//! Semi-implicit Euler integration, sequential impulse contact resolution.
//! All entities (vehicles, NPCs, player) share this system.

use crate::material::SurfaceMaterial;
use crate::math::*;

// ── Rigid body ───────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct RigidBody {
    // Position + orientation
    pub pos: Vec3,
    pub quat: Quat,
    // Linear
    pub vel: Vec3,
    pub force_accum: Vec3,
    pub mass: f32,
    pub inv_mass: f32,
    // Angular
    pub ang_vel: Vec3,
    pub torque_accum: Vec3,
    pub inv_inertia_local: [f32; 9], // diagonal in local space
    pub inv_inertia_world: [f32; 9], // rotated each frame
    // Flags
    pub is_static: bool,
}

impl RigidBody {
    pub fn new_dynamic(pos: Vec3, mass: f32, inertia_diag: Vec3) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let inv_i = if mass > 0.0 {
            mat3_diagonal(
                if inertia_diag[0] > 0.0 {
                    1.0 / inertia_diag[0]
                } else {
                    0.0
                },
                if inertia_diag[1] > 0.0 {
                    1.0 / inertia_diag[1]
                } else {
                    0.0
                },
                if inertia_diag[2] > 0.0 {
                    1.0 / inertia_diag[2]
                } else {
                    0.0
                },
            )
        } else {
            [0.0; 9]
        };
        RigidBody {
            pos,
            quat: QUAT_IDENTITY,
            vel: [0.0; 3],
            force_accum: [0.0; 3],
            mass,
            inv_mass,
            ang_vel: [0.0; 3],
            torque_accum: [0.0; 3],
            inv_inertia_local: inv_i,
            inv_inertia_world: inv_i,
            is_static: false,
        }
    }

    pub fn new_static(pos: Vec3) -> Self {
        RigidBody {
            pos,
            quat: QUAT_IDENTITY,
            vel: [0.0; 3],
            force_accum: [0.0; 3],
            mass: 0.0,
            inv_mass: 0.0,
            ang_vel: [0.0; 3],
            torque_accum: [0.0; 3],
            inv_inertia_local: [0.0; 9],
            inv_inertia_world: [0.0; 9],
            is_static: true,
        }
    }

    /// Apply a force at the center of mass (no torque)
    pub fn apply_force(&mut self, f: Vec3) {
        self.force_accum = v3_add(self.force_accum, f);
    }

    /// Apply a force at a world-space point (generates torque)
    pub fn apply_force_at(&mut self, f: Vec3, world_point: Vec3) {
        self.force_accum = v3_add(self.force_accum, f);
        let r = v3_sub(world_point, self.pos);
        self.torque_accum = v3_add(self.torque_accum, v3_cross(r, f));
    }

    /// Apply an impulse at the center of mass
    pub fn apply_impulse(&mut self, impulse: Vec3) {
        self.vel = v3_add(self.vel, v3_scale(impulse, self.inv_mass));
    }

    /// Apply an impulse at a world-space point
    pub fn apply_impulse_at(&mut self, impulse: Vec3, world_point: Vec3) {
        self.vel = v3_add(self.vel, v3_scale(impulse, self.inv_mass));
        let r = v3_sub(world_point, self.pos);
        let ang_impulse = v3_cross(r, impulse);
        self.ang_vel = v3_add(
            self.ang_vel,
            mat3_mul_vec(&self.inv_inertia_world, ang_impulse),
        );
    }

    /// Update world-space inertia tensor from current orientation
    pub fn update_inertia(&mut self) {
        let rot = quat_to_mat3(self.quat);
        self.inv_inertia_world = mat3_rotate_inertia(&rot, &self.inv_inertia_local);
    }

    /// Point velocity at a world-space point (linear + angular contribution)
    pub fn velocity_at(&self, world_point: Vec3) -> Vec3 {
        let r = v3_sub(world_point, self.pos);
        v3_add(self.vel, v3_cross(self.ang_vel, r))
    }
}

/// Multiply a 3x3 matrix by a vector
fn mat3_mul_vec(m: &[f32; 9], v: Vec3) -> Vec3 {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}

// ── Collision shapes ─────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub enum CollisionShape {
    /// Capsule for characters: radius, half_height (total height = 2*(half_height+radius))
    Capsule { radius: f32, half_height: f32 },
    /// Axis-aligned box in local space: half extents
    Box { half_extents: Vec3 },
    /// Sphere
    Sphere { radius: f32 },
}

impl CollisionShape {
    /// Compute inertia tensor diagonal for this shape at given mass
    pub fn inertia_diag(&self, mass: f32) -> Vec3 {
        match *self {
            CollisionShape::Box { half_extents: h } => {
                let (w, ht, d) = (h[0] * 2.0, h[1] * 2.0, h[2] * 2.0);
                let f = mass / 12.0;
                [
                    f * (ht * ht + d * d),
                    f * (w * w + d * d),
                    f * (w * w + ht * ht),
                ]
            }
            CollisionShape::Sphere { radius: r } => {
                let i = 0.4 * mass * r * r;
                [i, i, i]
            }
            CollisionShape::Capsule {
                radius: r,
                half_height: hh,
            } => {
                // Approximate as cylinder
                let h = (hh + r) * 2.0;
                let iy = 0.5 * mass * r * r;
                let ix = mass * (3.0 * r * r + h * h) / 12.0;
                [ix, iy, ix]
            }
        }
    }
}

// ── Contact ──────────────────────────────────────────────────────────────

/// A contact point between two bodies (or body and world)
pub struct Contact {
    pub body_a: usize, // index into rigid body array (usize::MAX = world/static)
    pub body_b: usize,
    pub point: Vec3,      // world-space contact point
    pub normal: Vec3,     // from A to B
    pub penetration: f32, // positive = overlapping
    pub material: SurfaceMaterial,
    // Solver cached values (filled during solve)
    pub normal_impulse: f32,
    pub tangent_impulse: [f32; 2],
}

impl Contact {
    pub fn new(
        a: usize,
        b: usize,
        point: Vec3,
        normal: Vec3,
        pen: f32,
        mat: SurfaceMaterial,
    ) -> Self {
        Contact {
            body_a: a,
            body_b: b,
            point,
            normal,
            penetration: pen,
            material: mat,
            normal_impulse: 0.0,
            tangent_impulse: [0.0; 2],
        }
    }
}

// ── Integration ──────────────────────────────────────────────────────────

const GRAVITY: Vec3 = [0.0, -9.81, 0.0];
const LINEAR_DAMPING: f32 = 0.002;
const ANGULAR_DAMPING: f32 = 0.02;

/// Semi-implicit Euler: integrate forces → velocity → position
pub fn integrate(body: &mut RigidBody, dt: f32) {
    if body.is_static {
        return;
    }

    // Apply gravity
    let gravity_force = v3_scale(GRAVITY, body.mass);
    body.force_accum = v3_add(body.force_accum, gravity_force);

    // Linear: v += (F/m) * dt
    let lin_accel = v3_scale(body.force_accum, body.inv_mass);
    body.vel = v3_add(body.vel, v3_scale(lin_accel, dt));
    // Damping
    body.vel = v3_scale(body.vel, (1.0 - LINEAR_DAMPING).max(0.0));
    // Position: x += v * dt
    body.pos = v3_add(body.pos, v3_scale(body.vel, dt));

    // Angular: ω += (I^-1 * τ) * dt
    let ang_accel = mat3_mul_vec(&body.inv_inertia_world, body.torque_accum);
    body.ang_vel = v3_add(body.ang_vel, v3_scale(ang_accel, dt));
    // Damping
    body.ang_vel = v3_scale(body.ang_vel, (1.0 - ANGULAR_DAMPING).max(0.0));
    // Orientation: q += 0.5 * ω * q * dt
    let w = body.ang_vel;
    let dq = quat_mul(
        [w[0] * 0.5 * dt, w[1] * 0.5 * dt, w[2] * 0.5 * dt, 0.0],
        body.quat,
    );
    body.quat = quat_normalize([
        body.quat[0] + dq[0],
        body.quat[1] + dq[1],
        body.quat[2] + dq[2],
        body.quat[3] + dq[3],
    ]);

    // Update world-space inertia
    body.update_inertia();

    // Clear accumulators
    body.force_accum = [0.0; 3];
    body.torque_accum = [0.0; 3];
}

// ── Contact solver (sequential impulse) ──────────────────────────────────

const SOLVER_ITERATIONS: usize = 4;
const BAUMGARTE_FACTOR: f32 = 0.2;
const SLOP: f32 = 0.005; // penetration tolerance before correction

/// Solve all contacts for this frame (sequential impulse with Baumgarte stabilization).
/// `bodies` is the mutable slice of all rigid bodies.
/// Contacts reference bodies by index; usize::MAX means the world (infinite mass).
///
/// Currently: ground contact for characters/vehicles is handled by direct enforcement
/// (simpler, stable). This solver handles inter-body contact resolution
/// (vehicle-vehicle, character-on-vehicle stacking).
pub fn solve_contacts(bodies: &mut [RigidBody], contacts: &mut [Contact], dt: f32) {
    if contacts.is_empty() || dt < 1e-8 {
        return;
    }

    let inv_dt = 1.0 / dt;

    for _ in 0..SOLVER_ITERATIONS {
        for contact in contacts.iter_mut() {
            solve_one(bodies, contact, inv_dt);
        }
    }
}

fn solve_one(bodies: &mut [RigidBody], c: &mut Contact, inv_dt: f32) {
    let n = c.normal;

    // Get effective masses and velocities at contact point
    let (vel_a, inv_mass_a, inv_inertia_a, r_a) = if c.body_a != usize::MAX {
        let b = &bodies[c.body_a];
        let r = v3_sub(c.point, b.pos);
        (b.velocity_at(c.point), b.inv_mass, b.inv_inertia_world, r)
    } else {
        ([0.0; 3], 0.0, [0.0f32; 9], [0.0; 3])
    };
    let (vel_b, inv_mass_b, inv_inertia_b, r_b) = if c.body_b != usize::MAX {
        let b = &bodies[c.body_b];
        let r = v3_sub(c.point, b.pos);
        (b.velocity_at(c.point), b.inv_mass, b.inv_inertia_world, r)
    } else {
        ([0.0; 3], 0.0, [0.0f32; 9], [0.0; 3])
    };

    let rel_vel = v3_sub(vel_b, vel_a);

    // ── Normal impulse ───────────────────────────────────────────────
    let vn = v3_dot(rel_vel, n);

    // Effective mass along normal
    let rn_a = v3_cross(r_a, n);
    let rn_b = v3_cross(r_b, n);
    let k_normal = inv_mass_a
        + inv_mass_b
        + v3_dot(v3_cross(mat3_mul_vec(&inv_inertia_a, rn_a), r_a), n)
        + v3_dot(v3_cross(mat3_mul_vec(&inv_inertia_b, rn_b), r_b), n);
    if k_normal < 1e-10 {
        return;
    }

    // Baumgarte positional correction
    let bias = BAUMGARTE_FACTOR * inv_dt * (c.penetration - SLOP).max(0.0);
    // Restitution
    let restitution_bias = if vn < -1.0 {
        -c.material.restitution * vn
    } else {
        0.0
    };

    let lambda_n = (-vn + bias + restitution_bias) / k_normal;
    let old_impulse = c.normal_impulse;
    c.normal_impulse = (old_impulse + lambda_n).max(0.0); // clamp: normal impulse >= 0
    let lambda_n = c.normal_impulse - old_impulse;

    let impulse_n = v3_scale(n, lambda_n);
    if c.body_a != usize::MAX {
        bodies[c.body_a].apply_impulse_at(v3_scale(impulse_n, -1.0), c.point);
    }
    if c.body_b != usize::MAX {
        bodies[c.body_b].apply_impulse_at(impulse_n, c.point);
    }

    // ── Friction impulse ─────────────────────────────────────────────
    // Re-read velocities after normal impulse
    let vel_a2 = if c.body_a != usize::MAX {
        bodies[c.body_a].velocity_at(c.point)
    } else {
        [0.0; 3]
    };
    let vel_b2 = if c.body_b != usize::MAX {
        bodies[c.body_b].velocity_at(c.point)
    } else {
        [0.0; 3]
    };
    let rel_vel2 = v3_sub(vel_b2, vel_a2);

    // Tangent velocity (remove normal component)
    let vt = v3_sub(rel_vel2, v3_scale(n, v3_dot(rel_vel2, n)));
    let vt_len = v3_len(vt);
    if vt_len < 1e-6 {
        return;
    }

    let tangent = v3_scale(vt, 1.0 / vt_len);

    // Effective mass along tangent
    let rt_a = v3_cross(r_a, tangent);
    let rt_b = v3_cross(r_b, tangent);
    let k_tangent = inv_mass_a
        + inv_mass_b
        + v3_dot(v3_cross(mat3_mul_vec(&inv_inertia_a, rt_a), r_a), tangent)
        + v3_dot(v3_cross(mat3_mul_vec(&inv_inertia_b, rt_b), r_b), tangent);
    if k_tangent < 1e-10 {
        return;
    }

    let lambda_t = -vt_len / k_tangent;

    // Coulomb friction cone: |friction impulse| <= μ * normal impulse
    let max_friction = c.material.dynamic_friction * c.normal_impulse;
    let lambda_t = lambda_t.clamp(-max_friction, max_friction);

    let impulse_t = v3_scale(tangent, lambda_t);
    if c.body_a != usize::MAX {
        bodies[c.body_a].apply_impulse_at(v3_scale(impulse_t, -1.0), c.point);
    }
    if c.body_b != usize::MAX {
        bodies[c.body_b].apply_impulse_at(impulse_t, c.point);
    }
}

// ── Ground plane contact generation ──────────────────────────────────────

/// Generate a contact between a rigid body and the terrain ground plane.
/// Returns Some(Contact) if the body's lowest point is below terrain.
/// Used with solve_contacts() for full contact pipeline (not yet wired for characters).
#[allow(dead_code)]
pub fn ground_contact(
    body_idx: usize,
    body: &RigidBody,
    shape: &CollisionShape,
    terrain: &crate::state::Terrain,
    mat: SurfaceMaterial,
) -> Option<Contact> {
    // Get the lowest point of the shape
    let lowest_y = match *shape {
        CollisionShape::Sphere { radius } => body.pos[1] - radius,
        CollisionShape::Capsule {
            radius,
            half_height,
        } => body.pos[1] - half_height - radius,
        CollisionShape::Box { half_extents } => {
            // Check all 8 corners, find lowest in world space
            let mut min_y = f32::MAX;
            for sx in [-1.0f32, 1.0] {
                for sy in [-1.0f32, 1.0] {
                    for sz in [-1.0f32, 1.0] {
                        let local = [
                            half_extents[0] * sx,
                            half_extents[1] * sy,
                            half_extents[2] * sz,
                        ];
                        let world = v3_add(body.pos, quat_rotate(body.quat, local));
                        if world[1] < min_y {
                            min_y = world[1];
                        }
                    }
                }
            }
            min_y
        }
    };

    let ground_y = terrain.height_at(body.pos[0], body.pos[2]);
    let penetration = ground_y - lowest_y;

    if penetration > 0.0 {
        let normal = terrain.normal_at(body.pos[0], body.pos[2]);
        let contact_point = [body.pos[0], ground_y, body.pos[2]];
        Some(Contact::new(
            body_idx,
            usize::MAX,
            contact_point,
            normal,
            penetration,
            mat,
        ))
    } else {
        None
    }
}

// ── Box vs Box collision (SAT) ───────────────────────────────────────────

/// Test two oriented boxes for overlap. Returns contact if overlapping.
/// 3D contact narrow phase for vehicle-vehicle collisions.
/// Uses 6 face axes (sufficient for box-vs-box when mostly axis-aligned).
pub fn box_vs_box(
    idx_a: usize,
    body_a: &RigidBody,
    half_a: Vec3,
    idx_b: usize,
    body_b: &RigidBody,
    half_b: Vec3,
    mat: SurfaceMaterial,
) -> Option<Contact> {
    let rot_a = quat_to_mat3(body_a.quat);
    let rot_b = quat_to_mat3(body_b.quat);

    // Axes of A (columns of rot_a)
    let axes_a = [
        [rot_a[0], rot_a[1], rot_a[2]],
        [rot_a[3], rot_a[4], rot_a[5]],
        [rot_a[6], rot_a[7], rot_a[8]],
    ];
    let axes_b = [
        [rot_b[0], rot_b[1], rot_b[2]],
        [rot_b[3], rot_b[4], rot_b[5]],
        [rot_b[6], rot_b[7], rot_b[8]],
    ];
    let halfs_a = [half_a[0], half_a[1], half_a[2]];
    let halfs_b = [half_b[0], half_b[1], half_b[2]];

    let d = v3_sub(body_b.pos, body_a.pos);

    let mut min_pen = f32::MAX;
    let mut best_axis = [0.0f32; 3];

    // Test 15 axes: 3 from A, 3 from B, 9 cross products
    // For simplicity, test the 6 face axes (sufficient for most game cases)
    for i in 0..3 {
        let axis = axes_a[i];
        let pen = test_sat_axis(axis, d, &axes_a, &halfs_a, &axes_b, &halfs_b);
        if pen < 0.0 {
            return None;
        } // separating axis found
        if pen < min_pen {
            min_pen = pen;
            best_axis = axis;
        }
    }
    for i in 0..3 {
        let axis = axes_b[i];
        let pen = test_sat_axis(axis, d, &axes_a, &halfs_a, &axes_b, &halfs_b);
        if pen < 0.0 {
            return None;
        }
        if pen < min_pen {
            min_pen = pen;
            best_axis = axis;
        }
    }

    // Ensure normal points from A to B
    if v3_dot(best_axis, d) < 0.0 {
        best_axis = v3_scale(best_axis, -1.0);
    }

    let contact_point = v3_scale(v3_add(body_a.pos, body_b.pos), 0.5);

    Some(Contact::new(
        idx_a,
        idx_b,
        contact_point,
        best_axis,
        min_pen,
        mat,
    ))
}

fn test_sat_axis(
    axis: Vec3,
    d: Vec3,
    axes_a: &[Vec3; 3],
    halfs_a: &[f32; 3],
    axes_b: &[Vec3; 3],
    halfs_b: &[f32; 3],
) -> f32 {
    let len = v3_len(axis);
    if len < 1e-6 {
        return f32::MAX;
    } // degenerate axis, skip

    let proj_d = v3_dot(d, axis).abs();
    let mut proj_a = 0.0f32;
    for i in 0..3 {
        proj_a += halfs_a[i] * v3_dot(axes_a[i], axis).abs();
    }
    let mut proj_b = 0.0f32;
    for i in 0..3 {
        proj_b += halfs_b[i] * v3_dot(axes_b[i], axis).abs();
    }

    (proj_a + proj_b) - proj_d // positive = overlap, negative = separated
}

// ── Sphere vs Sphere ─────────────────────────────────────────────────────

#[allow(dead_code)]
pub fn sphere_vs_sphere(
    idx_a: usize,
    body_a: &RigidBody,
    radius_a: f32,
    idx_b: usize,
    body_b: &RigidBody,
    radius_b: f32,
    mat: SurfaceMaterial,
) -> Option<Contact> {
    let d = v3_sub(body_b.pos, body_a.pos);
    let dist_sq = v3_dot(d, d);
    let sum_r = radius_a + radius_b;
    if dist_sq >= sum_r * sum_r {
        return None;
    }
    let dist = dist_sq.sqrt();
    if dist < 1e-6 {
        return None;
    }
    let normal = v3_scale(d, 1.0 / dist);
    let pen = sum_r - dist;
    let point = v3_add(body_a.pos, v3_scale(normal, radius_a - pen * 0.5));
    Some(Contact::new(idx_a, idx_b, point, normal, pen, mat))
}

// ── Explosion / radial impulse ────────────────────────────────────────────

/// Apply a radial explosion impulse to a rigid body.
/// `origin`: world-space explosion center
/// `radius`: maximum effect radius (impulse falls off linearly to zero at edge)
/// `force`: peak impulse magnitude at the explosion center
/// Returns the actual impulse magnitude applied (0 if out of range).
pub fn apply_explosion(body: &mut RigidBody, origin: Vec3, radius: f32, force: f32) -> f32 {
    if body.is_static || radius < 1e-4 {
        return 0.0;
    }

    let d = v3_sub(body.pos, origin);
    let dist = v3_len(d);
    if dist >= radius || dist < 1e-4 {
        return 0.0;
    }

    // Linear falloff: full impulse at center, zero at edge
    let falloff = 1.0 - dist / radius;
    let magnitude = force * falloff;
    let dir = v3_scale(d, 1.0 / dist);
    // Add slight upward bias (explosions lift things)
    let impulse_dir = v3_normalize([dir[0], dir[1] + 0.3, dir[2]]);
    let impulse = v3_scale(impulse_dir, magnitude);
    body.apply_impulse(impulse);
    magnitude
}

/// Apply explosion to a slice of rigid bodies. Returns indices that were affected.
#[allow(dead_code)]
pub fn apply_explosion_to_all(
    bodies: &mut [RigidBody],
    origin: Vec3,
    radius: f32,
    force: f32,
) -> Vec<(usize, f32)> {
    let mut affected = Vec::new();
    for (i, body) in bodies.iter_mut().enumerate() {
        let mag = apply_explosion(body, origin, radius, force);
        if mag > 0.0 {
            affected.push((i, mag));
        }
    }
    affected
}

// ── Capsule vs ground (character-specific) ───────────────────────────────

/// Character capsule ground contact — returns contact at feet position.
/// Used for NPC/player ground response with terrain normal.
pub fn capsule_ground_contact(
    body_idx: usize,
    body: &RigidBody,
    radius: f32,
    half_height: f32,
    terrain: &crate::state::Terrain,
    mat: SurfaceMaterial,
) -> Option<Contact> {
    let feet_y = body.pos[1] - half_height - radius;
    let ground_y = terrain.height_at(body.pos[0], body.pos[2]);
    let pen = ground_y - feet_y;
    if pen > 0.0 {
        let normal = terrain.normal_at(body.pos[0], body.pos[2]);
        let point = [body.pos[0], ground_y, body.pos[2]];
        Some(Contact::new(body_idx, usize::MAX, point, normal, pen, mat))
    } else {
        None
    }
}

// ── Character-on-vehicle surface test ────────────────────────────────────

/// Test if a point (character feet position) is on top of a vehicle.
/// Returns Some((surface_normal, surface_y_world)) if the point XZ is within
/// the vehicle's OBB footprint. Used for characters standing on moving vehicles.
pub fn point_on_vehicle_surface(
    point: Vec3,
    veh_pos: Vec3,
    veh_rot_y: f32,
    half_w: f32,
    half_d: f32,
    roof_height: f32,
) -> Option<(Vec3, f32)> {
    // Transform point into vehicle local space (XZ only)
    let dx = point[0] - veh_pos[0];
    let dz = point[2] - veh_pos[2];
    let (sin_r, cos_r) = veh_rot_y.sin_cos();
    let local_x = dx * cos_r + dz * sin_r;
    let local_z = -dx * sin_r + dz * cos_r;

    // Check if within OBB footprint (with small margin for roof edge)
    if local_x.abs() > half_w + 0.05 || local_z.abs() > half_d + 0.05 {
        return None;
    }

    // Check vertical: character feet must be near roof height (within 0.5m above)
    let surface_y = veh_pos[1] + roof_height;
    let feet_y = point[1];
    if feet_y < surface_y - 0.1 || feet_y > surface_y + 0.5 {
        return None;
    }

    Some(([0.0, 1.0, 0.0], surface_y))
}

// ── Spatial grid (broad phase) ──────────────────────────────────────────

/// Entity type tag for spatial grid entries
#[derive(Clone, Copy, PartialEq)]
pub enum EntityKind {
    Vehicle(usize),
    Npc(usize),
    Player,
}

/// Uniform grid spatial hash for broad-phase collision.
/// 500m world centered at origin → covers -260 to +260 with 10m cells.
pub struct SpatialGrid {
    cell_size: f32,
    grid_w: usize,
    world_min: f32,
    cells: Vec<Vec<EntityKind>>,
}

const GRID_CELL_SIZE: f32 = 10.0;
const GRID_W: usize = 54; // covers 540m → -270 to +270
const GRID_WORLD_MIN: f32 = -270.0;

impl SpatialGrid {
    pub fn new() -> Self {
        let total = GRID_W * GRID_W;
        let mut cells = Vec::with_capacity(total);
        for _ in 0..total {
            cells.push(Vec::new());
        }
        SpatialGrid {
            cell_size: GRID_CELL_SIZE,
            grid_w: GRID_W,
            world_min: GRID_WORLD_MIN,
            cells,
        }
    }

    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    fn cell_coords(&self, x: f32, z: f32) -> (usize, usize) {
        let cx = ((x - self.world_min) / self.cell_size) as isize;
        let cz = ((z - self.world_min) / self.cell_size) as isize;
        let cx = cx.clamp(0, self.grid_w as isize - 1) as usize;
        let cz = cz.clamp(0, self.grid_w as isize - 1) as usize;
        (cx, cz)
    }

    pub fn insert(&mut self, x: f32, z: f32, entity: EntityKind) {
        let (cx, cz) = self.cell_coords(x, z);
        self.cells[cz * self.grid_w + cx].push(entity);
    }

    /// Collect all entities in the 3x3 cell neighborhood around (x, z)
    pub fn query(&self, x: f32, z: f32, buf: &mut Vec<EntityKind>) {
        buf.clear();
        let (cx, cz) = self.cell_coords(x, z);
        let gw = self.grid_w;
        let min_x = if cx > 0 { cx - 1 } else { 0 };
        let max_x = if cx + 1 < gw { cx + 1 } else { gw - 1 };
        let min_z = if cz > 0 { cz - 1 } else { 0 };
        let max_z = if cz + 1 < gw { cz + 1 } else { gw - 1 };
        for nz in min_z..=max_z {
            let row = nz * gw;
            for nx in min_x..=max_x {
                buf.extend_from_slice(&self.cells[row + nx]);
            }
        }
    }
}

/// Contact accumulation buffer for the sequential impulse solver
pub struct ContactBuffer {
    pub contacts: Vec<Contact>,
}

impl ContactBuffer {
    pub fn new() -> Self {
        ContactBuffer {
            contacts: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.contacts.clear();
    }

    pub fn push(&mut self, c: Contact) {
        self.contacts.push(c);
    }
}
