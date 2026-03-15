// 3D math: vec3 as [f32;3], mat4 as [f32;16] column-major, free functions only

pub type Vec3 = [f32; 3];
pub type Mat4 = [f32; 16];

pub fn v3(x: f32, y: f32, z: f32) -> Vec3 { [x, y, z] }

pub fn v3_add(a: Vec3, b: Vec3) -> Vec3 { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
pub fn v3_sub(a: Vec3, b: Vec3) -> Vec3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
pub fn v3_scale(a: Vec3, s: f32) -> Vec3 { [a[0]*s, a[1]*s, a[2]*s] }
pub fn v3_dot(a: Vec3, b: Vec3) -> f32 { a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
pub fn v3_cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}
pub fn v3_len(a: Vec3) -> f32 { v3_dot(a, a).sqrt() }
pub fn v3_normalize(a: Vec3) -> Vec3 {
    let l = v3_len(a);
    if l < 1e-10 { [0.0, 0.0, 0.0] } else { v3_scale(a, 1.0 / l) }
}
// Mat4: column-major [c0r0, c0r1, c0r2, c0r3, c1r0, c1r1, c1r2, c1r3, ...]
// Index: m[col*4 + row]

pub fn m4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut r = [0.0f32; 16];
    for c in 0..4 {
        for row in 0..4 {
            r[c*4 + row] = a[0*4+row]*b[c*4+0] + a[1*4+row]*b[c*4+1]
                         + a[2*4+row]*b[c*4+2] + a[3*4+row]*b[c*4+3];
        }
    }
    r
}

pub fn m4_perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fovy * 0.5).tan();
    let nf = 1.0 / (near - far);
    [f/aspect, 0.0, 0.0,                0.0,
     0.0,      f,   0.0,                0.0,
     0.0,      0.0, (far+near)*nf,     -1.0,
     0.0,      0.0, 2.0*far*near*nf,    0.0]
}

pub fn m4_look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = v3_normalize(v3_sub(target, eye));
    let s = v3_normalize(v3_cross(f, up));
    let u = v3_cross(s, f);
    [s[0],  u[0],  -f[0], 0.0,
     s[1],  u[1],  -f[1], 0.0,
     s[2],  u[2],  -f[2], 0.0,
     -v3_dot(s, eye), -v3_dot(u, eye), v3_dot(f, eye), 1.0]
}

/// Perspective matrix for Vulkan: depth [0,1], Y-flip
pub fn m4_perspective_vk(fovy: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fovy * 0.5).tan();
    let nf = 1.0 / (near - far);
    [f/aspect, 0.0, 0.0,          0.0,
     0.0,     -f,   0.0,          0.0,
     0.0,      0.0, far*nf,      -1.0,
     0.0,      0.0, far*near*nf,  0.0]
}

pub fn m4_transform_no_div(m: &Mat4, p: Vec3) -> [f32; 4] {
    [
        m[0]*p[0] + m[4]*p[1] + m[8]*p[2]  + m[12],
        m[1]*p[0] + m[5]*p[1] + m[9]*p[2]  + m[13],
        m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14],
        m[3]*p[0] + m[7]*p[1] + m[11]*p[2] + m[15],
    ]
}

/// Build a 3x3 rotation matrix that aligns local Y-up with terrain normal,
/// while preserving heading (rot_y). Returns column-major [c0r0..c2r2].
/// This combines heading rotation with slope tilt in a single matrix.
pub fn terrain_rot3x3(normal: Vec3, rot_y: f32) -> [f32; 9] {
    // Up axis = terrain normal (where local Y should point)
    let up = normal;
    // Forward direction from heading (in world XZ plane)
    let (sin_r, cos_r) = rot_y.sin_cos();
    let fwd_flat = [-sin_r, 0.0, -cos_r];
    // Right = cross(forward, up), then normalize
    let right = v3_normalize(v3_cross(fwd_flat, up));
    // Recompute forward = cross(up, right) to ensure orthogonality
    let fwd = v3_cross(up, right);
    // Column-major 3x3: columns are right, up, -forward
    // det=+1 (proper rotation) — maps local -Z to world forward.
    // Model +X maps to world right, +Y to terrain normal, -Z to facing direction.
    [
        right[0],  right[1],  right[2],
        up[0],     up[1],     up[2],
       -fwd[0],   -fwd[1],   -fwd[2],
    ]
}

/// Apply a 3x3 rotation matrix (column-major) to a point
#[inline]
pub fn rot3x3_apply(m: &[f32; 9], p: Vec3) -> Vec3 {
    [
        m[0]*p[0] + m[3]*p[1] + m[6]*p[2],
        m[1]*p[0] + m[4]*p[1] + m[7]*p[2],
        m[2]*p[0] + m[5]*p[1] + m[8]*p[2],
    ]
}

/// Linearly interpolate between two Vec3 values
pub fn v3_lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    [a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, a[2] + (b[2]-a[2])*t]
}

/// Clamp a terrain normal so tilt from vertical never exceeds `max_deg` degrees.
/// Preserves horizontal direction, scales magnitude to match max angle.
pub fn clamp_normal_tilt(n: Vec3, max_deg: f32) -> Vec3 {
    let max_cos = max_deg.to_radians().cos(); // e.g. cos(35°) = 0.819
    if n[1] >= max_cos { return n; }
    let horiz_sq = n[0] * n[0] + n[2] * n[2];
    if horiz_sq < 0.0001 { return [0.0, 1.0, 0.0]; }
    let max_sin = max_deg.to_radians().sin();
    let scale = max_sin / horiz_sq.sqrt();
    [n[0] * scale, max_cos, n[2] * scale] // already unit: sin²+cos²=1
}

#[inline(always)]
pub fn dist_sq_2d(x1: f32, z1: f32, x2: f32, z2: f32) -> f32 {
    let dx = x1 - x2;
    let dz = z1 - z2;
    dx * dx + dz * dz
}

// ── Quaternion [x, y, z, w] ──────────────────────────────────────────────

pub type Quat = [f32; 4];

pub const QUAT_IDENTITY: Quat = [0.0, 0.0, 0.0, 1.0];

pub fn quat_from_axis_angle(axis: Vec3, angle: f32) -> Quat {
    let half = angle * 0.5;
    let (s, c) = half.sin_cos();
    [axis[0] * s, axis[1] * s, axis[2] * s, c]
}

pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    [
        a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
        a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
        a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
        a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2],
    ]
}

pub fn quat_conjugate(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Extract axis-angle from a quaternion. Returns (axis, angle_radians).
/// For identity quaternion, returns ([1,0,0], 0.0).
pub fn quat_to_axis_angle(q: Quat) -> (Vec3, f32) {
    let angle = 2.0 * q[3].clamp(-1.0, 1.0).acos();
    let s = (1.0 - q[3] * q[3]).sqrt();
    if s < 0.001 {
        ([1.0, 0.0, 0.0], angle)
    } else {
        ([q[0] / s, q[1] / s, q[2] / s], angle)
    }
}

pub fn quat_normalize(q: Quat) -> Quat {
    let len = (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]).sqrt();
    if len < 1e-10 { QUAT_IDENTITY } else { [q[0]/len, q[1]/len, q[2]/len, q[3]/len] }
}

/// Rotate a vector by a quaternion: q * v * q^-1
pub fn quat_rotate(q: Quat, v: Vec3) -> Vec3 {
    // Optimized: t = 2 * cross(q.xyz, v), result = v + w*t + cross(q.xyz, t)
    let qv = [q[0], q[1], q[2]];
    let t = v3_scale(v3_cross(qv, v), 2.0);
    v3_add(v3_add(v, v3_scale(t, q[3])), v3_cross(qv, t))
}

/// Spherical linear interpolation between two quaternions
pub fn quat_slerp(a: Quat, b: Quat, t: f32) -> Quat {
    let mut dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    let mut b = b;
    if dot < 0.0 {
        b = [-b[0], -b[1], -b[2], -b[3]];
        dot = -dot;
    }
    if dot > 0.9995 {
        // Linear interpolation for very close quaternions
        return quat_normalize([
            a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t,
            a[2] + (b[2]-a[2])*t, a[3] + (b[3]-a[3])*t,
        ]);
    }
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;
    [
        a[0]*wa + b[0]*wb, a[1]*wa + b[1]*wb,
        a[2]*wa + b[2]*wb, a[3]*wa + b[3]*wb,
    ]
}

/// Convert quaternion to 3x3 rotation matrix (column-major)
pub fn quat_to_mat3(q: Quat) -> [f32; 9] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let x2 = x+x; let y2 = y+y; let z2 = z+z;
    let xx = x*x2; let xy = x*y2; let xz = x*z2;
    let yy = y*y2; let yz = y*z2; let zz = z*z2;
    let wx = w*x2; let wy = w*y2; let wz = w*z2;
    // Column-major: col0 = right, col1 = up, col2 = forward
    [
        1.0-yy-zz,  xy+wz,      xz-wy,
        xy-wz,       1.0-xx-zz,  yz+wx,
        xz+wy,       yz-wx,      1.0-xx-yy,
    ]
}

/// Build quaternion from rotation matrix (column-major 3x3)
pub fn quat_from_mat3(m: &[f32; 9]) -> Quat {
    let trace = m[0] + m[4] + m[8];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [
            (m[5] - m[7]) / s,
            (m[6] - m[2]) / s,
            (m[1] - m[3]) / s,
            0.25 * s,
        ]
    } else if m[0] > m[4] && m[0] > m[8] {
        let s = (1.0 + m[0] - m[4] - m[8]).sqrt() * 2.0;
        [
            0.25 * s,
            (m[1] + m[3]) / s,
            (m[6] + m[2]) / s,
            (m[5] - m[7]) / s,
        ]
    } else if m[4] > m[8] {
        let s = (1.0 + m[4] - m[0] - m[8]).sqrt() * 2.0;
        [
            (m[1] + m[3]) / s,
            0.25 * s,
            (m[5] + m[7]) / s,
            (m[6] - m[2]) / s,
        ]
    } else {
        let s = (1.0 + m[8] - m[0] - m[4]).sqrt() * 2.0;
        [
            (m[6] + m[2]) / s,
            (m[5] + m[7]) / s,
            0.25 * s,
            (m[1] - m[3]) / s,
        ]
    }
}

/// Build quaternion from heading (rot_y) — rotation around Y axis
pub fn quat_from_rot_y(rot_y: f32) -> Quat {
    quat_from_axis_angle([0.0, 1.0, 0.0], rot_y)
}

/// Extract forward direction (-Z in local space) from quaternion
pub fn quat_forward(q: Quat) -> Vec3 {
    quat_rotate(q, [0.0, 0.0, -1.0])
}

/// Extract right direction (+X in local space) from quaternion
pub fn quat_right(q: Quat) -> Vec3 {
    quat_rotate(q, [1.0, 0.0, 0.0])
}

/// Extract up direction (+Y in local space) from quaternion
pub fn quat_up(q: Quat) -> Vec3 {
    quat_rotate(q, [0.0, 1.0, 0.0])
}

// ── 3x3 matrix utilities ─────────────────────────────────────────────────

/// Transpose a column-major 3x3 matrix (also its inverse for rotation matrices)
pub fn mat3_transpose(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

/// Compute the inverse inertia tensor in world space: R * I_inv_local * R^T
pub fn mat3_rotate_inertia(rot: &[f32; 9], i_inv: &[f32; 9]) -> [f32; 9] {
    let rt = mat3_transpose(rot);
    mat3_mul(&mat3_mul(rot, i_inv), &rt)
}

/// Multiply two column-major 3x3 matrices
pub fn mat3_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut r = [0.0f32; 9];
    for c in 0..3 {
        for row in 0..3 {
            r[c*3 + row] = a[0*3+row]*b[c*3+0] + a[1*3+row]*b[c*3+1] + a[2*3+row]*b[c*3+2];
        }
    }
    r
}

/// Diagonal 3x3 matrix from three values (for inertia tensors)
pub fn mat3_diagonal(x: f32, y: f32, z: f32) -> [f32; 9] {
    [x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, z]
}

// ── Mat4 utilities for skeletal animation ───────────────────────────────

pub const M4_IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

/// Build a 4x4 rotation matrix from Euler angles (XYZ order, radians).
/// Rotation order: first X, then Y, then Z — i.e. R = Rz * Ry * Rx.
pub fn m4_from_euler_xyz(rx: f32, ry: f32, rz: f32) -> Mat4 {
    let (sx, cx) = rx.sin_cos();
    let (sy, cy) = ry.sin_cos();
    let (sz, cz) = rz.sin_cos();
    // Column-major: m[col*4 + row]
    [
        cy*cz,               cy*sz,              -sy,    0.0,
        sx*sy*cz - cx*sz,    sx*sy*sz + cx*cz,    sx*cy, 0.0,
        cx*sy*cz + sx*sz,    cx*sy*sz - sx*cz,    cx*cy, 0.0,
        0.0,                 0.0,                 0.0,   1.0,
    ]
}

/// Build a 4x4 translation matrix
pub fn m4_from_translation(tx: f32, ty: f32, tz: f32) -> Mat4 {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        tx,  ty,  tz,  1.0,
    ]
}

/// Build a 4x4 matrix from a 3x3 rotation (column-major) + translation
pub fn m4_from_rot3_translation(rot: &[f32; 9], t: Vec3) -> Mat4 {
    [
        rot[0], rot[1], rot[2], 0.0,
        rot[3], rot[4], rot[5], 0.0,
        rot[6], rot[7], rot[8], 0.0,
        t[0],   t[1],   t[2],   1.0,
    ]
}

/// Transform a point by a 4x4 matrix (assumes w=1, no perspective divide)
#[inline]
pub fn m4_transform_point(m: &Mat4, p: Vec3) -> Vec3 {
    [
        m[0]*p[0] + m[4]*p[1] + m[8]*p[2]  + m[12],
        m[1]*p[0] + m[5]*p[1] + m[9]*p[2]  + m[13],
        m[2]*p[0] + m[6]*p[1] + m[10]*p[2] + m[14],
    ]
}

/// Transform a normal (direction) by a 4x4 matrix (ignores translation, uses upper-left 3x3)
#[inline]
pub fn m4_transform_normal(m: &Mat4, n: Vec3) -> Vec3 {
    v3_normalize([
        m[0]*n[0] + m[4]*n[1] + m[8]*n[2],
        m[1]*n[0] + m[5]*n[1] + m[9]*n[2],
        m[2]*n[0] + m[6]*n[1] + m[10]*n[2],
    ])
}

/// Invert a 4x4 affine matrix (rotation + translation, no scaling/shear).
/// For rigid transforms: inverse = transpose(R) with negated rotated translation.
pub fn m4_inverse_affine(m: &Mat4) -> Mat4 {
    // Transpose the 3x3 rotation part
    let r00 = m[0]; let r01 = m[4]; let r02 = m[8];
    let r10 = m[1]; let r11 = m[5]; let r12 = m[9];
    let r20 = m[2]; let r21 = m[6]; let r22 = m[10];
    let tx = m[12]; let ty = m[13]; let tz = m[14];
    // Inverse translation = -R^T * t
    let itx = -(r00*tx + r10*ty + r20*tz);
    let ity = -(r01*tx + r11*ty + r21*tz);
    let itz = -(r02*tx + r12*ty + r22*tz);
    [
        r00, r01, r02, 0.0,
        r10, r11, r12, 0.0,
        r20, r21, r22, 0.0,
        itx, ity, itz, 1.0,
    ]
}

/// Build a quaternion from Euler angles in XYZ order (radians)
pub fn quat_from_euler_xyz(rx: f32, ry: f32, rz: f32) -> Quat {
    let qx = quat_from_axis_angle([1.0, 0.0, 0.0], rx);
    let qy = quat_from_axis_angle([0.0, 1.0, 0.0], ry);
    let qz = quat_from_axis_angle([0.0, 0.0, 1.0], rz);
    quat_normalize(quat_mul(qz, quat_mul(qy, qx)))
}

/// Convert quaternion to a 4x4 rotation matrix (column-major)
pub fn quat_to_mat4(q: Quat) -> Mat4 {
    let m3 = quat_to_mat3(q);
    [
        m3[0], m3[1], m3[2], 0.0,
        m3[3], m3[4], m3[5], 0.0,
        m3[6], m3[7], m3[8], 0.0,
        0.0,   0.0,   0.0,   1.0,
    ]
}
