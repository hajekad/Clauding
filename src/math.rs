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
    // Column-major 3x3: columns are right, up, forward
    [
        right[0], right[1], right[2],
        up[0],    up[1],    up[2],
        fwd[0],   fwd[1],   fwd[2],
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
