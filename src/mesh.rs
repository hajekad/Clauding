// Parametric mesh primitives: cylinder, cone, icosphere, beveled box, wall with holes,
// raised strip, wave surface, extrude profile, lathe. All push WorldTri into a Vec.

use crate::state::WorldTri;
use std::cell::Cell;

thread_local! {
    static EXTRA_SUBDIVISIONS: Cell<u32> = const { Cell::new(0) };
    static SEGMENT_MULTIPLIER: Cell<u32> = const { Cell::new(1) };
}

/// Set mesh quality multipliers. extra_subdivs is added to icosphere subdivision count,
/// segment_mult multiplies parametric segment counts. Defaults: (0, 1) = no change.
pub fn set_mesh_quality(extra_subdivs: u32, segment_mult: u32) {
    EXTRA_SUBDIVISIONS.with(|c| c.set(extra_subdivs));
    SEGMENT_MULTIPLIER.with(|c| c.set(segment_mult));
}

fn quality_subdivisions(base: u32) -> u32 {
    base + EXTRA_SUBDIVISIONS.with(|c| c.get())
}

fn quality_segments(base: usize) -> usize {
    base * SEGMENT_MULTIPLIER.with(|c| c.get()) as usize
}

/// Compute normalized cross product of triangle edges
pub fn tri_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let e1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let e2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    let n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
    let l = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
    if l < 1e-10 { [0.0, 1.0, 0.0] } else { [n[0]/l, n[1]/l, n[2]/l] }
}

/// Push a single triangle. Central function for all triangle emission.
#[inline(always)]
pub fn push_tri(tris: &mut Vec<WorldTri>, a: [f32;3], b: [f32;3], c: [f32;3], normal: [f32;3], color: u32) {
    tris.push(WorldTri { v: [a, b, c], normal, color });
}

/// Push a quad as 2 tris (a,b,c) + (a,c,d). Normal derived from first triangle.
pub fn push_quad(tris: &mut Vec<WorldTri>, a: [f32;3], b: [f32;3], c: [f32;3], d: [f32;3], color: u32) {
    let normal = tri_normal(a, b, c);
    tris.push(WorldTri { v: [a, b, c], normal, color });
    tris.push(WorldTri { v: [a, c, d], normal, color });
}

/// Push a quad with an explicit normal: (a,b,c) + (a,c,d)
pub fn push_quad_n(tris: &mut Vec<WorldTri>, a: [f32;3], b: [f32;3], c: [f32;3], d: [f32;3], normal: [f32;3], color: u32) {
    tris.push(WorldTri { v: [a, b, c], normal, color });
    tris.push(WorldTri { v: [a, c, d], normal, color });
}

// ── Cylinder ────────────────────────────────────────────────────────────────

/// Cylinder centered at (cx, cy, cz) with radius r, height h, N side segments.
/// Y-axis aligned: bottom at cy - h/2, top at cy + h/2.
/// Generates N*4 tris (N*2 sides + N top fan + N bottom fan).
pub fn cylinder_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    r: f32, h: f32, segments: usize, color: u32,
) {
    let hh = h * 0.5;
    let n = quality_segments(segments).max(3);
    let step = std::f32::consts::TAU / n as f32;

    let top_center = [cx, cy + hh, cz];
    let bot_center = [cx, cy - hh, cz];

    for i in 0..n {
        let a0 = i as f32 * step;
        let a1 = (i + 1) as f32 * step;
        let (s0, c0) = (a0.sin(), a0.cos());
        let (s1, c1) = (a1.sin(), a1.cos());

        let bt0 = [cx + r*c0, cy - hh, cz + r*s0];
        let bt1 = [cx + r*c1, cy - hh, cz + r*s1];
        let tp0 = [cx + r*c0, cy + hh, cz + r*s0];
        let tp1 = [cx + r*c1, cy + hh, cz + r*s1];

        // Side quad (2 tris) — go up first, then around for outward normals
        push_quad(tris, bt0, tp0, tp1, bt1, color);

        // Top fan tri
        push_tri(tris, top_center, tp1, tp0, [0.0, 1.0, 0.0], color);
        // Bottom fan tri
        push_tri(tris, bot_center, bt0, bt1, [0.0, -1.0, 0.0], color);
    }
}

/// Cylinder between two arbitrary points (p0, p1) with radius r.
/// Uses Y-axis cylinder approximation rotated to align with the segment direction.
pub fn cylinder_between(
    tris: &mut Vec<WorldTri>, p0: [f32;3], p1: [f32;3],
    r: f32, segments: usize, color: u32,
) {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let dz = p1[2] - p0[2];
    let h = (dx*dx + dy*dy + dz*dz).sqrt();
    if h < 1e-6 { return; }

    // Direction vector
    let dir = [dx/h, dy/h, dz/h];

    // Find a perpendicular vector (use cross with world up, or world right if parallel)
    let up = if dir[1].abs() < 0.99 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
    let right = normalize3(cross3(dir, up));
    let fwd = cross3(right, dir);

    let n = quality_segments(segments).max(3);
    let step = std::f32::consts::TAU / n as f32;
    let center0 = p0;
    let center1 = p1;

    for i in 0..n {
        let a0 = i as f32 * step;
        let a1 = (i + 1) as f32 * step;
        let (s0, c0) = (a0.sin(), a0.cos());
        let (s1, c1) = (a1.sin(), a1.cos());

        let offset0 = [r*(c0*right[0] + s0*fwd[0]), r*(c0*right[1] + s0*fwd[1]), r*(c0*right[2] + s0*fwd[2])];
        let offset1 = [r*(c1*right[0] + s1*fwd[0]), r*(c1*right[1] + s1*fwd[1]), r*(c1*right[2] + s1*fwd[2])];

        let b0 = add3(center0, offset0);
        let b1 = add3(center0, offset1);
        let t0 = add3(center1, offset0);
        let t1 = add3(center1, offset1);

        // Side quad — go along axis first, then around for outward normals
        push_quad(tris, b0, t0, t1, b1, color);

        // End caps
        push_tri(tris, center0, b0, b1, [-dir[0], -dir[1], -dir[2]], color);
        push_tri(tris, center1, t1, t0, dir, color);
    }
}

// ── Cone ────────────────────────────────────────────────────────────────────

/// Cone with apex at top, base at bottom. Y-axis aligned.
/// Generates N*3 tris (N sides + N base fan).
pub fn cone_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    r: f32, h: f32, segments: usize, color: u32,
) {
    let hh = h * 0.5;
    let n = quality_segments(segments).max(3);
    let step = std::f32::consts::TAU / n as f32;

    let apex = [cx, cy + hh, cz];
    let bot_center = [cx, cy - hh, cz];

    for i in 0..n {
        let a0 = i as f32 * step;
        let a1 = (i + 1) as f32 * step;
        let (s0, c0) = (a0.sin(), a0.cos());
        let (s1, c1) = (a1.sin(), a1.cos());

        let b0 = [cx + r*c0, cy - hh, cz + r*s0];
        let b1 = [cx + r*c1, cy - hh, cz + r*s1];

        // Side tri
        let normal = tri_normal(b0, apex, b1);
        push_tri(tris, b0, apex, b1, normal, color);
        // Base tri
        push_tri(tris, bot_center, b0, b1, [0.0, -1.0, 0.0], color);
    }
}

// ── Icosphere ───────────────────────────────────────────────────────────────

const ICOSPHERE_FACES: [[usize; 3]; 20] = [
    [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
    [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
    [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
    [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
];

const ICOSPHERE_VERTS: [[f32; 3]; 12] = {
    const PHI: f32 = 1.618034; // (1 + sqrt(5)) / 2
    const A: f32 = 1.0;
    const B: f32 = PHI;
    [
        [-A, B, 0.0], [ A, B, 0.0], [-A,-B, 0.0], [ A,-B, 0.0],
        [0.0,-A, B], [0.0, A, B], [0.0,-A,-B], [0.0, A,-B],
        [ B, 0.0,-A], [ B, 0.0, A], [-B, 0.0,-A], [-B, 0.0, A],
    ]
};

fn icosphere_unit_verts() -> Vec<[f32; 3]> {
    ICOSPHERE_VERTS.iter().map(|v| {
        let l = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        [v[0]/l, v[1]/l, v[2]/l]
    }).collect()
}

fn subdivide_icosphere_mid(
    a_idx: usize, b_idx: usize,
    verts: &mut Vec<[f32;3]>,
    cache: &mut Vec<(usize, usize, usize)>,
) -> usize {
    let (lo, hi) = if a_idx < b_idx { (a_idx, b_idx) } else { (b_idx, a_idx) };
    for &(ca, cb, ci) in cache.iter() {
        if ca == lo && cb == hi { return ci; }
    }
    let va = verts[a_idx];
    let vb = verts[b_idx];
    let mid = [(va[0]+vb[0])*0.5, (va[1]+vb[1])*0.5, (va[2]+vb[2])*0.5];
    let l = (mid[0]*mid[0] + mid[1]*mid[1] + mid[2]*mid[2]).sqrt();
    let idx = verts.len();
    verts.push([mid[0]/l, mid[1]/l, mid[2]/l]);
    cache.push((lo, hi, idx));
    idx
}

fn subdivide_icosphere(verts: &mut Vec<[f32; 3]>, faces: &mut Vec<[usize; 3]>, subdivisions: u32) {
    for _ in 0..quality_subdivisions(subdivisions) {
        let mut new_faces = Vec::with_capacity(faces.len() * 4);
        let mut midpoint_cache: Vec<(usize, usize, usize)> = Vec::new();

        for f in faces.iter() {
            let m01 = subdivide_icosphere_mid(f[0], f[1], verts, &mut midpoint_cache);
            let m12 = subdivide_icosphere_mid(f[1], f[2], verts, &mut midpoint_cache);
            let m20 = subdivide_icosphere_mid(f[2], f[0], verts, &mut midpoint_cache);
            new_faces.push([f[0], m01, m20]);
            new_faces.push([f[1], m12, m01]);
            new_faces.push([f[2], m20, m12]);
            new_faces.push([m01, m12, m20]);
        }
        *faces = new_faces;
    }
}

/// Icosphere centered at (cx, cy, cz) with radius r.
/// subdivisions=0: 20 tris (icosahedron), 1: 80, 2: 320, 3: 1280.
pub fn sphere_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    r: f32, subdivisions: u32, color: u32,
) {
    let mut verts = icosphere_unit_verts();
    let mut faces = ICOSPHERE_FACES.to_vec();
    subdivide_icosphere(&mut verts, &mut faces, subdivisions);

    // Output tris scaled and translated
    for f in &faces {
        let v0 = [cx + verts[f[0]][0]*r, cy + verts[f[0]][1]*r, cz + verts[f[0]][2]*r];
        let v1 = [cx + verts[f[1]][0]*r, cy + verts[f[1]][1]*r, cz + verts[f[1]][2]*r];
        let v2 = [cx + verts[f[2]][0]*r, cy + verts[f[2]][1]*r, cz + verts[f[2]][2]*r];
        let normal = tri_normal(v0, v1, v2);
        push_tri(tris, v0, v1, v2, normal, color);
    }
}

/// Icosphere with vertex perturbation for rocks.
pub fn perturbed_sphere_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    r: f32, subdivisions: u32, perturbation: f32, seed: u64, color: u32,
) {
    let mut verts = icosphere_unit_verts();
    let mut faces = ICOSPHERE_FACES.to_vec();
    subdivide_icosphere(&mut verts, &mut faces, subdivisions);

    // Perturb each vertex radially using a hash of its direction
    for v in &mut verts {
        let hash = simple_hash(seed, v[0], v[1], v[2]);
        let offset = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0; // [-1, 1]
        let scale = 1.0 + offset * perturbation;
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
    }

    for f in &faces {
        let v0 = [cx + verts[f[0]][0]*r, cy + verts[f[0]][1]*r, cz + verts[f[0]][2]*r];
        let v1 = [cx + verts[f[1]][0]*r, cy + verts[f[1]][1]*r, cz + verts[f[1]][2]*r];
        let v2 = [cx + verts[f[2]][0]*r, cy + verts[f[2]][1]*r, cz + verts[f[2]][2]*r];
        let normal = tri_normal(v0, v1, v2);
        push_tri(tris, v0, v1, v2, normal, color);
    }
}

fn simple_hash(seed: u64, x: f32, y: f32, z: f32) -> u64 {
    let mut h = seed;
    h = h.wrapping_mul(6364136223846793005).wrapping_add(x.to_bits() as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(y.to_bits() as u64);
    h = h.wrapping_mul(6364136223846793005).wrapping_add(z.to_bits() as u64);
    h ^ (h >> 33)
}

// ── Beveled Box ─────────────────────────────────────────────────────────────

/// Axis-aligned box with chamfered edges. Centered at (cx,cy,cz) with full extents (w,h,d).
/// Bevel is the chamfer distance. Produces ~76 tris.
pub fn beveled_box_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    w: f32, h: f32, d: f32, bevel: f32, color: u32,
) {
    let b = bevel.min(w * 0.25).min(h * 0.25).min(d * 0.25);
    let hw = w * 0.5;
    let hh = h * 0.5;
    let hd = d * 0.5;

    // 6 main faces (inset by bevel)
    // Front face (z+): inset in x and y by bevel
    push_quad(tris,
        [cx - hw + b, cy - hh + b, cz + hd],
        [cx + hw - b, cy - hh + b, cz + hd],
        [cx + hw - b, cy + hh - b, cz + hd],
        [cx - hw + b, cy + hh - b, cz + hd],
        color,
    );
    // Back face (z-)
    push_quad(tris,
        [cx + hw - b, cy - hh + b, cz - hd],
        [cx - hw + b, cy - hh + b, cz - hd],
        [cx - hw + b, cy + hh - b, cz - hd],
        [cx + hw - b, cy + hh - b, cz - hd],
        color,
    );
    // Left face (x-)
    push_quad(tris,
        [cx - hw, cy - hh + b, cz - hd + b],
        [cx - hw, cy - hh + b, cz + hd - b],
        [cx - hw, cy + hh - b, cz + hd - b],
        [cx - hw, cy + hh - b, cz - hd + b],
        color,
    );
    // Right face (x+)
    push_quad(tris,
        [cx + hw, cy - hh + b, cz + hd - b],
        [cx + hw, cy - hh + b, cz - hd + b],
        [cx + hw, cy + hh - b, cz - hd + b],
        [cx + hw, cy + hh - b, cz + hd - b],
        color,
    );
    // Top face (y+)
    push_quad(tris,
        [cx - hw + b, cy + hh, cz + hd - b],
        [cx + hw - b, cy + hh, cz + hd - b],
        [cx + hw - b, cy + hh, cz - hd + b],
        [cx - hw + b, cy + hh, cz - hd + b],
        color,
    );
    // Bottom face (y-)
    push_quad(tris,
        [cx - hw + b, cy - hh, cz - hd + b],
        [cx + hw - b, cy - hh, cz - hd + b],
        [cx + hw - b, cy - hh, cz + hd - b],
        [cx - hw + b, cy - hh, cz + hd - b],
        color,
    );

    // 12 edge bevels (each a quad strip)
    // Vertical edges (4): connect front/back to left/right
    // Front-right vertical
    push_quad(tris,
        [cx + hw - b, cy - hh + b, cz + hd],
        [cx + hw, cy - hh + b, cz + hd - b],
        [cx + hw, cy + hh - b, cz + hd - b],
        [cx + hw - b, cy + hh - b, cz + hd],
        color,
    );
    // Front-left vertical
    push_quad(tris,
        [cx - hw, cy - hh + b, cz + hd - b],
        [cx - hw + b, cy - hh + b, cz + hd],
        [cx - hw + b, cy + hh - b, cz + hd],
        [cx - hw, cy + hh - b, cz + hd - b],
        color,
    );
    // Back-right vertical
    push_quad(tris,
        [cx + hw, cy - hh + b, cz - hd + b],
        [cx + hw - b, cy - hh + b, cz - hd],
        [cx + hw - b, cy + hh - b, cz - hd],
        [cx + hw, cy + hh - b, cz - hd + b],
        color,
    );
    // Back-left vertical
    push_quad(tris,
        [cx - hw + b, cy - hh + b, cz - hd],
        [cx - hw, cy - hh + b, cz - hd + b],
        [cx - hw, cy + hh - b, cz - hd + b],
        [cx - hw + b, cy + hh - b, cz - hd],
        color,
    );

    // Horizontal top edges (4)
    // Top-front
    push_quad(tris,
        [cx - hw + b, cy + hh - b, cz + hd],
        [cx + hw - b, cy + hh - b, cz + hd],
        [cx + hw - b, cy + hh, cz + hd - b],
        [cx - hw + b, cy + hh, cz + hd - b],
        color,
    );
    // Top-back
    push_quad(tris,
        [cx + hw - b, cy + hh - b, cz - hd],
        [cx - hw + b, cy + hh - b, cz - hd],
        [cx - hw + b, cy + hh, cz - hd + b],
        [cx + hw - b, cy + hh, cz - hd + b],
        color,
    );
    // Top-left
    push_quad(tris,
        [cx - hw, cy + hh - b, cz + hd - b],
        [cx - hw, cy + hh - b, cz - hd + b],
        [cx - hw + b, cy + hh, cz - hd + b],
        [cx - hw + b, cy + hh, cz + hd - b],
        color,
    );
    // Top-right
    push_quad(tris,
        [cx + hw, cy + hh - b, cz - hd + b],
        [cx + hw, cy + hh - b, cz + hd - b],
        [cx + hw - b, cy + hh, cz + hd - b],
        [cx + hw - b, cy + hh, cz - hd + b],
        color,
    );

    // Horizontal bottom edges (4)
    // Bottom-front
    push_quad(tris,
        [cx - hw + b, cy - hh, cz + hd - b],
        [cx + hw - b, cy - hh, cz + hd - b],
        [cx + hw - b, cy - hh + b, cz + hd],
        [cx - hw + b, cy - hh + b, cz + hd],
        color,
    );
    // Bottom-back
    push_quad(tris,
        [cx + hw - b, cy - hh, cz - hd + b],
        [cx - hw + b, cy - hh, cz - hd + b],
        [cx - hw + b, cy - hh + b, cz - hd],
        [cx + hw - b, cy - hh + b, cz - hd],
        color,
    );
    // Bottom-left
    push_quad(tris,
        [cx - hw + b, cy - hh, cz - hd + b],
        [cx - hw + b, cy - hh, cz + hd - b],
        [cx - hw, cy - hh + b, cz + hd - b],
        [cx - hw, cy - hh + b, cz - hd + b],
        color,
    );
    // Bottom-right
    push_quad(tris,
        [cx + hw - b, cy - hh, cz + hd - b],
        [cx + hw - b, cy - hh, cz - hd + b],
        [cx + hw, cy - hh + b, cz - hd + b],
        [cx + hw, cy - hh + b, cz + hd - b],
        color,
    );

    // 8 corner tris — each corner's normal points diagonally outward.
    // push_tri auto-corrects winding to match the outward normal.
    let corner = |tris: &mut Vec<WorldTri>, a: [f32;3], b: [f32;3], c: [f32;3]| {
        // Normal = average of the 3 face normals meeting at this corner (≈ outward diagonal)
        let n = [(a[0]-cx).signum() + (b[0]-cx).signum() + (c[0]-cx).signum(),
                 (a[1]-cy).signum() + (b[1]-cy).signum() + (c[1]-cy).signum(),
                 (a[2]-cz).signum() + (b[2]-cz).signum() + (c[2]-cz).signum()];
        let l = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).sqrt();
        let outward = if l > 0.01 { [n[0]/l, n[1]/l, n[2]/l] } else { tri_normal(a, b, c) };
        push_tri(tris, a, b, c, outward, color);
    };
    // Top corners
    corner(tris, [cx+hw-b,cy+hh-b,cz+hd], [cx+hw,cy+hh-b,cz+hd-b], [cx+hw-b,cy+hh,cz+hd-b]);
    corner(tris, [cx-hw+b,cy+hh-b,cz+hd], [cx-hw+b,cy+hh,cz+hd-b], [cx-hw,cy+hh-b,cz+hd-b]);
    corner(tris, [cx+hw-b,cy+hh-b,cz-hd], [cx+hw-b,cy+hh,cz-hd+b], [cx+hw,cy+hh-b,cz-hd+b]);
    corner(tris, [cx-hw+b,cy+hh-b,cz-hd], [cx-hw,cy+hh-b,cz-hd+b], [cx-hw+b,cy+hh,cz-hd+b]);
    // Bottom corners
    corner(tris, [cx+hw-b,cy-hh+b,cz+hd], [cx+hw-b,cy-hh,cz+hd-b], [cx+hw,cy-hh+b,cz+hd-b]);
    corner(tris, [cx-hw+b,cy-hh+b,cz+hd], [cx-hw,cy-hh+b,cz+hd-b], [cx-hw+b,cy-hh,cz+hd-b]);
    corner(tris, [cx+hw-b,cy-hh+b,cz-hd], [cx+hw,cy-hh+b,cz-hd+b], [cx+hw-b,cy-hh,cz-hd+b]);
    corner(tris, [cx-hw+b,cy-hh+b,cz-hd], [cx-hw+b,cy-hh,cz-hd+b], [cx-hw,cy-hh+b,cz-hd+b]);
}

// ── Wall with Holes (recessed windows/doors) ────────────────────────────────

/// A rectangular hole in a wall face. Position relative to wall's bottom-left corner.
pub struct WallHole {
    pub x: f32,      // left edge offset from wall left
    pub y: f32,      // bottom edge offset from wall bottom
    pub w: f32,      // hole width
    pub h: f32,      // hole height
}

/// Generate a wall face with recessed holes (windows/doors).
/// By default (swap_xz=false), the wall is in the XY plane at z=pos_z.
/// With swap_xz=true, x and z are swapped so the wall is in the YZ plane at x=pos_z.
/// `face_dir`: +1.0 for front, -1.0 for back (direction is swapped when swap_xz=true).
/// `left_dir`: +1.0 for increasing lateral, -1.0 for decreasing.
pub fn wall_with_holes_tris(
    tris: &mut Vec<WorldTri>,
    pos_x: f32, pos_y: f32, pos_z: f32,
    wall_w: f32, wall_h: f32,
    holes: &[WallHole],
    depth: f32,
    wall_color: u32,
    hole_colors: &[u32],
    face_dir: f32,
    left_dir: f32,
    swap_xz: bool,
) {
    // When swap_xz, negate depth so recess goes into the building
    let eff_depth = if swap_xz { -depth } else { depth };
    let fz = pos_z;
    let bz = pos_z - eff_depth * face_dir;
    let v = |x: f32, y: f32, z: f32| -> [f32;3] {
        if swap_xz { [z, y, x] } else { [x, y, z] }
    };

    let mut sorted_holes: Vec<(usize, &WallHole)> = holes.iter().enumerate().collect();
    sorted_holes.sort_by(|a, b| a.1.x.partial_cmp(&b.1.x).unwrap_or(std::cmp::Ordering::Equal));

    let mut y_cuts = vec![0.0_f32, wall_h];
    for (_, hole) in &sorted_holes {
        y_cuts.push(hole.y);
        y_cuts.push(hole.y + hole.h);
    }
    y_cuts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    y_cuts.dedup_by(|a, b| (*a - *b).abs() < 0.001);

    for yi in 0..y_cuts.len()-1 {
        let strip_bot = y_cuts[yi];
        let strip_top = y_cuts[yi + 1];
        if (strip_top - strip_bot) < 0.001 { continue; }

        let strip_holes: Vec<&(usize, &WallHole)> = sorted_holes.iter()
            .filter(|(_, h)| h.y < strip_top - 0.001 && h.y + h.h > strip_bot + 0.001)
            .collect();

        let mut x_start = 0.0_f32;
        for (_, hole) in &strip_holes {
            if hole.x > x_start + 0.001 {
                emit_wall_quad(tris, pos_x, pos_y, fz, x_start, strip_bot, hole.x, strip_top, wall_color, face_dir, left_dir, swap_xz);
            }
            x_start = hole.x + hole.w;
        }
        if x_start < wall_w - 0.001 {
            emit_wall_quad(tris, pos_x, pos_y, fz, x_start, strip_bot, wall_w, strip_top, wall_color, face_dir, left_dir, swap_xz);
        }
    }

    for &(orig_idx, hole) in &sorted_holes {
        let hc = hole_colors[orig_idx % hole_colors.len()];
        let hx0 = pos_x + hole.x * left_dir;
        let hx1 = pos_x + (hole.x + hole.w) * left_dir;
        let hy0 = pos_y + hole.y;
        let hy1 = pos_y + hole.y + hole.h;

        let (lx, rx) = if left_dir > 0.0 { (hx0, hx1) } else { (hx1, hx0) };

        // Back face of recess
        if face_dir > 0.0 {
            push_quad(tris, v(lx, hy0, bz), v(rx, hy0, bz), v(rx, hy1, bz), v(lx, hy1, bz), hc);
        } else {
            push_quad(tris, v(rx, hy0, bz), v(lx, hy0, bz), v(lx, hy1, bz), v(rx, hy1, bz), hc);
        }

        // Top side
        if face_dir > 0.0 {
            push_quad(tris, v(lx, hy1, fz), v(rx, hy1, fz), v(rx, hy1, bz), v(lx, hy1, bz), wall_color);
        } else {
            push_quad(tris, v(rx, hy1, fz), v(lx, hy1, fz), v(lx, hy1, bz), v(rx, hy1, bz), wall_color);
        }
        // Bottom side
        if face_dir > 0.0 {
            push_quad(tris, v(lx, hy0, bz), v(rx, hy0, bz), v(rx, hy0, fz), v(lx, hy0, fz), wall_color);
        } else {
            push_quad(tris, v(rx, hy0, bz), v(lx, hy0, bz), v(lx, hy0, fz), v(rx, hy0, fz), wall_color);
        }
        // Left side
        if face_dir > 0.0 {
            push_quad(tris, v(lx, hy0, bz), v(lx, hy0, fz), v(lx, hy1, fz), v(lx, hy1, bz), wall_color);
        } else {
            push_quad(tris, v(lx, hy0, fz), v(lx, hy0, bz), v(lx, hy1, bz), v(lx, hy1, fz), wall_color);
        }
        // Right side
        if face_dir > 0.0 {
            push_quad(tris, v(rx, hy0, fz), v(rx, hy0, bz), v(rx, hy1, bz), v(rx, hy1, fz), wall_color);
        } else {
            push_quad(tris, v(rx, hy0, bz), v(rx, hy0, fz), v(rx, hy1, fz), v(rx, hy1, bz), wall_color);
        }
    }

    // Overall wall side edges (left, right, top, back face for solid areas)
    {
        let wall_lx = pos_x;
        let wall_rx = pos_x + wall_w * left_dir;
        let (wlx, wrx) = if left_dir > 0.0 { (wall_lx, wall_rx) } else { (wall_rx, wall_lx) };
        let wy0 = pos_y;
        let wy1 = pos_y + wall_h;

        // Left side wall (depth edge)
        if face_dir > 0.0 {
            push_quad(tris, v(wlx, wy0, bz), v(wlx, wy0, fz), v(wlx, wy1, fz), v(wlx, wy1, bz), wall_color);
        } else {
            push_quad(tris, v(wlx, wy0, fz), v(wlx, wy0, bz), v(wlx, wy1, bz), v(wlx, wy1, fz), wall_color);
        }
        // Right side wall (depth edge)
        if face_dir > 0.0 {
            push_quad(tris, v(wrx, wy0, fz), v(wrx, wy0, bz), v(wrx, wy1, bz), v(wrx, wy1, fz), wall_color);
        } else {
            push_quad(tris, v(wrx, wy0, bz), v(wrx, wy0, fz), v(wrx, wy1, fz), v(wrx, wy1, bz), wall_color);
        }
        // Top edge (depth strip along top of wall)
        if face_dir > 0.0 {
            push_quad(tris, v(wlx, wy1, fz), v(wrx, wy1, fz), v(wrx, wy1, bz), v(wlx, wy1, bz), wall_color);
        } else {
            push_quad(tris, v(wrx, wy1, fz), v(wlx, wy1, fz), v(wlx, wy1, bz), v(wrx, wy1, bz), wall_color);
        }
        // Back face of wall (solid areas — render using same strip logic as front)
        for yi in 0..y_cuts.len()-1 {
            let strip_bot = y_cuts[yi];
            let strip_top = y_cuts[yi + 1];
            if (strip_top - strip_bot) < 0.001 { continue; }

            let strip_holes: Vec<&(usize, &WallHole)> = sorted_holes.iter()
                .filter(|(_, h)| h.y < strip_top - 0.001 && h.y + h.h > strip_bot + 0.001)
                .collect();

            let mut x_start = 0.0_f32;
            for (_, hole) in &strip_holes {
                if hole.x > x_start + 0.001 {
                    emit_wall_quad(tris, pos_x, pos_y, bz, x_start, strip_bot, hole.x, strip_top, wall_color, -face_dir, left_dir, swap_xz);
                }
                x_start = hole.x + hole.w;
            }
            if x_start < wall_w - 0.001 {
                emit_wall_quad(tris, pos_x, pos_y, bz, x_start, strip_bot, wall_w, strip_top, wall_color, -face_dir, left_dir, swap_xz);
            }
        }
    }
}

fn emit_wall_quad(
    tris: &mut Vec<WorldTri>,
    base_x: f32, base_y: f32, z: f32,
    x0: f32, y0: f32, x1: f32, y1: f32,
    color: u32,
    face_dir: f32, left_dir: f32,
    swap_xz: bool,
) {
    let wx0 = base_x + x0 * left_dir;
    let wx1 = base_x + x1 * left_dir;
    let wy0 = base_y + y0;
    let wy1 = base_y + y1;
    let (lx, rx) = if left_dir > 0.0 { (wx0, wx1) } else { (wx1, wx0) };
    let v = |x: f32, y: f32, z: f32| -> [f32;3] {
        if swap_xz { [z, y, x] } else { [x, y, z] }
    };

    if face_dir > 0.0 {
        push_quad(tris, v(lx, wy0, z), v(rx, wy0, z), v(rx, wy1, z), v(lx, wy1, z), color);
    } else {
        push_quad(tris, v(rx, wy0, z), v(lx, wy0, z), v(lx, wy1, z), v(rx, wy1, z), color);
    }
}

// ── Raised Strip ────────────────────────────────────────────────────────────

/// Generate a raised 3D strip along a polyline. Each segment is a box with
/// the given width and height, oriented along the segment direction.
/// This eliminates z-fighting vs flat surfaces beneath.
pub fn raised_strip_tris(
    tris: &mut Vec<WorldTri>, points: &[[f32; 3]],
    width: f32, height: f32, color: u32,
) {
    if points.len() < 2 { return; }
    let hw = width * 0.5;
    let hh = height * 0.5;

    for i in 0..points.len()-1 {
        let p0 = points[i];
        let p1 = points[i + 1];

        let dx = p1[0] - p0[0];
        let dz = p1[2] - p0[2];
        let len = (dx * dx + dz * dz).sqrt();
        if len < 0.001 { continue; }

        // Perpendicular in XZ plane
        let px = -dz / len * hw;
        let pz = dx / len * hw;

        // 8 corners of the segment box
        let y0_lo = p0[1] - hh;
        let y0_hi = p0[1] + hh;
        let y1_lo = p1[1] - hh;
        let y1_hi = p1[1] + hh;

        let bl0 = [p0[0] - px, y0_lo, p0[2] - pz];
        let br0 = [p0[0] + px, y0_lo, p0[2] + pz];
        let tl0 = [p0[0] - px, y0_hi, p0[2] - pz];
        let tr0 = [p0[0] + px, y0_hi, p0[2] + pz];
        let bl1 = [p1[0] - px, y1_lo, p1[2] - pz];
        let br1 = [p1[0] + px, y1_lo, p1[2] + pz];
        let tl1 = [p1[0] - px, y1_hi, p1[2] - pz];
        let tr1 = [p1[0] + px, y1_hi, p1[2] + pz];

        // Top face
        push_quad(tris, tl0, tr0, tr1, tl1, color);
        // Bottom face
        push_quad(tris, bl0, bl1, br1, br0, color);
        // Left side
        push_quad(tris, bl0, tl0, tl1, bl1, color);
        // Right side
        push_quad(tris, br0, br1, tr1, tr0, color);
        // Front cap (start)
        if i == 0 {
            push_quad(tris, bl0, br0, tr0, tl0, color);
        }
        // Back cap (end)
        if i == points.len() - 2 {
            push_quad(tris, bl1, tl1, tr1, br1, color);
        }
    }
}

// ── Wave Surface ────────────────────────────────────────────────────────────

/// Generate a sinusoidal wave surface over a rectangular area.
/// Never coplanar with flat surfaces — eliminates z-fighting with banks/terrain.
pub fn wave_surface_tris(
    tris: &mut Vec<WorldTri>,
    x_min: f32, x_max: f32, z_min: f32, z_max: f32,
    base_y: f32, amplitude: f32, freq: f32,
    subdivisions_x: usize, subdivisions_z: usize,
    color: u32,
) {
    let sx = subdivisions_x.max(1);
    let sz = subdivisions_z.max(1);
    let dx = (x_max - x_min) / sx as f32;
    let dz = (z_max - z_min) / sz as f32;

    for iz in 0..sz {
        for ix in 0..sx {
            let x0 = x_min + ix as f32 * dx;
            let x1 = x0 + dx;
            let z0 = z_min + iz as f32 * dz;
            let z1 = z0 + dz;

            let y00 = base_y + (x0 * freq).sin() * (z0 * freq * 0.7).cos() * amplitude
                + (x0 * freq * 2.3 + z0 * freq * 1.7).sin() * amplitude * 0.3;
            let y10 = base_y + (x1 * freq).sin() * (z0 * freq * 0.7).cos() * amplitude
                + (x1 * freq * 2.3 + z0 * freq * 1.7).sin() * amplitude * 0.3;
            let y01 = base_y + (x0 * freq).sin() * (z1 * freq * 0.7).cos() * amplitude
                + (x0 * freq * 2.3 + z1 * freq * 1.7).sin() * amplitude * 0.3;
            let y11 = base_y + (x1 * freq).sin() * (z1 * freq * 0.7).cos() * amplitude
                + (x1 * freq * 2.3 + z1 * freq * 1.7).sin() * amplitude * 0.3;

            let v00 = [x0, y00, z0];
            let v10 = [x1, y10, z0];
            let v01 = [x0, y01, z1];
            let v11 = [x1, y11, z1];

            // Per-tile color jitter for ripple variation
            let h = (ix as u32).wrapping_mul(73856093) ^ (iz as u32).wrapping_mul(19349663);
            let noise = (h % 20) as i32 - 10;
            let r = ((((color >> 16) & 0xFF) as i32 + noise).clamp(0, 255)) as u32;
            let g = ((((color >> 8) & 0xFF) as i32 + noise).clamp(0, 255)) as u32;
            let b = (((color & 0xFF) as i32 + noise).clamp(0, 255)) as u32;
            let c = (color & 0xFF000000) | (r << 16) | (g << 8) | b;

            let n1 = tri_normal(v00, v11, v10);
            push_tri(tris, v00, v11, v10, n1, c);
            let n2 = tri_normal(v00, v01, v11);
            push_tri(tris, v00, v01, v11, n2, c);
        }
    }
}

// ── Extrude Profile ─────────────────────────────────────────────────────────

/// Extrude a 2D profile along a 3D polyline path.
/// Profile points are in local (right, up) space perpendicular to the path direction.
/// Generates profile_len * 2 tris per path segment.
pub fn extrude_profile_tris(
    tris: &mut Vec<WorldTri>,
    profile: &[[f32; 2]],  // (right_offset, up_offset) pairs
    path: &[[f32; 3]],     // 3D path points
    color: u32,
) {
    if profile.len() < 2 || path.len() < 2 { return; }

    // For each path segment, compute a local frame and place profile rings
    let mut rings: Vec<Vec<[f32;3]>> = Vec::with_capacity(path.len());

    for pi in 0..path.len() {
        // Tangent direction
        let tangent = if pi == 0 {
            normalize3(sub3(path[1], path[0]))
        } else if pi == path.len() - 1 {
            normalize3(sub3(path[pi], path[pi-1]))
        } else {
            normalize3(add3(
                normalize3(sub3(path[pi], path[pi-1])),
                normalize3(sub3(path[pi+1], path[pi])),
            ))
        };

        let up = if tangent[1].abs() < 0.99 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let right = normalize3(cross3(tangent, up));
        let actual_up = cross3(right, tangent);

        let ring: Vec<[f32;3]> = profile.iter().map(|&[r, u]| {
            [
                path[pi][0] + right[0] * r + actual_up[0] * u,
                path[pi][1] + right[1] * r + actual_up[1] * u,
                path[pi][2] + right[2] * r + actual_up[2] * u,
            ]
        }).collect();
        rings.push(ring);
    }

    // Connect adjacent rings with quads
    for ri in 0..rings.len()-1 {
        for pi in 0..profile.len()-1 {
            let a = rings[ri][pi];
            let b = rings[ri][pi+1];
            let c = rings[ri+1][pi+1];
            let d = rings[ri+1][pi];
            push_quad(tris, a, b, c, d, color);
        }
    }
}

// ── Lathe (Revolution Surface) ──────────────────────────────────────────────

/// Lathe: revolve a 2D profile around Y axis at (cx, cy, cz).
/// Profile is array of (radius, y_offset) pairs.
/// Generates profile_len * 2 * segments tris.
pub fn lathe_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    profile: &[[f32; 2]],  // (radius, y_offset) pairs
    segments: usize, color: u32,
) {
    if profile.len() < 2 { return; }
    let n = quality_segments(segments).max(3);
    let step = std::f32::consts::TAU / n as f32;

    // Generate rings of vertices
    let mut rings: Vec<Vec<[f32;3]>> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let angle = i as f32 * step;
        let (s, c) = (angle.sin(), angle.cos());
        let ring: Vec<[f32;3]> = profile.iter().map(|&[r, y]| {
            [cx + r * c, cy + y, cz + r * s]
        }).collect();
        rings.push(ring);
    }

    // Connect adjacent rings
    for ri in 0..n {
        for pi in 0..profile.len()-1 {
            let a = rings[ri][pi];
            let b = rings[ri][pi+1];
            let c = rings[ri+1][pi+1];
            let d = rings[ri+1][pi];
            push_quad(tris, a, b, c, d, color);
        }
    }
}

// ── Axis-aligned box (kept for backward compatibility) ──────────────────────

/// Simple axis-aligned box. 12 tris.
pub fn box_tris(tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32, w: f32, h: f32, d: f32, color: u32) {
    let (hw, hh, hd) = (w * 0.5, h * 0.5, d * 0.5);
    let c = [
        [cx-hw, cy-hh, cz+hd], [cx+hw, cy-hh, cz+hd], [cx+hw, cy+hh, cz+hd], [cx-hw, cy+hh, cz+hd],
        [cx-hw, cy-hh, cz-hd], [cx+hw, cy-hh, cz-hd], [cx+hw, cy+hh, cz-hd], [cx-hw, cy+hh, cz-hd],
    ];
    let faces: [([usize; 4], [f32; 3]); 6] = [
        ([0,1,2,3], [0.0, 0.0, 1.0]), ([5,4,7,6], [0.0, 0.0,-1.0]),
        ([4,0,3,7], [-1.0,0.0,0.0]),  ([1,5,6,2], [1.0, 0.0, 0.0]),
        ([3,2,6,7], [0.0, 1.0, 0.0]), ([4,5,1,0], [0.0,-1.0, 0.0]),
    ];
    for (idx, normal) in faces {
        push_tri(tris, c[idx[0]], c[idx[1]], c[idx[2]], normal, color);
        push_tri(tris, c[idx[0]], c[idx[2]], c[idx[3]], normal, color);
    }
}

/// Rotated box in XZ plane. `rot_y` is rotation around Y axis in radians.
pub fn rotated_box_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    w: f32, h: f32, d: f32, rot_y: f32, color: u32,
) {
    let (hw, hh, hd) = (w * 0.5, h * 0.5, d * 0.5);
    let (sin_r, cos_r) = (rot_y.sin(), rot_y.cos());

    let rotate = |lx: f32, lz: f32| -> (f32, f32) {
        (cx + lx * cos_r - lz * sin_r, cz + lx * sin_r + lz * cos_r)
    };

    let (x0, z0) = rotate(-hw, hd);
    let (x1, z1) = rotate( hw, hd);
    let (x2, z2) = rotate( hw, -hd);
    let (x3, z3) = rotate(-hw, -hd);

    let c = [
        [x0, cy-hh, z0], [x1, cy-hh, z1], [x2, cy-hh, z2], [x3, cy-hh, z3],
        [x0, cy+hh, z0], [x1, cy+hh, z1], [x2, cy+hh, z2], [x3, cy+hh, z3],
    ];

    // Front: 0,1,5,4  Back: 2,3,7,6  Left: 3,0,4,7  Right: 1,2,6,5  Top: 4,5,6,7  Bot: 3,2,1,0
    let faces: [[usize; 4]; 6] = [
        [0,1,5,4], [2,3,7,6], [3,0,4,7], [1,2,6,5], [4,5,6,7], [3,2,1,0],
    ];
    for idx in faces {
        let normal = tri_normal(c[idx[0]], c[idx[1]], c[idx[2]]);
        push_tri(tris, c[idx[0]], c[idx[1]], c[idx[2]], normal, color);
        push_tri(tris, c[idx[0]], c[idx[2]], c[idx[3]], normal, color);
    }
}

// ── Pitched Roof ────────────────────────────────────────────────────────────

/// Pitched (gabled) roof. Ridge runs along X axis.
pub fn pitched_roof_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    w: f32, d: f32, peak_h: f32, color: u32,
) {
    let hw = w * 0.5;
    let hd = d * 0.5;

    // Ridge endpoints
    let r0 = [cx - hw, cy + peak_h, cz];
    let r1 = [cx + hw, cy + peak_h, cz];

    // Eave corners
    let e_fl = [cx - hw, cy, cz + hd]; // front-left
    let e_fr = [cx + hw, cy, cz + hd]; // front-right
    let e_bl = [cx - hw, cy, cz - hd]; // back-left
    let e_br = [cx + hw, cy, cz - hd]; // back-right

    // Front slope (z+ side)
    push_quad(tris, e_fl, e_fr, r1, r0, color);
    // Back slope (z- side)
    push_quad(tris, e_br, e_bl, r0, r1, color);

    // Gable ends (triangles, CCW from outside)
    let n_left = tri_normal(e_bl, e_fl, r0);
    push_tri(tris, e_bl, e_fl, r0, n_left, color);
    let n_right = tri_normal(e_fr, e_br, r1);
    push_tri(tris, e_fr, e_br, r1, n_right, color);
}

/// Hip roof: all four sides slope up to a shortened ridge.
pub fn hip_roof_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    w: f32, d: f32, peak_h: f32, color: u32,
) {
    let hw = w * 0.5;
    let hd = d * 0.5;
    let ridge_inset = hw * 0.4; // ridge shorter than base

    let r0 = [cx - hw + ridge_inset, cy + peak_h, cz];
    let r1 = [cx + hw - ridge_inset, cy + peak_h, cz];

    let e_fl = [cx - hw, cy, cz + hd];
    let e_fr = [cx + hw, cy, cz + hd];
    let e_bl = [cx - hw, cy, cz - hd];
    let e_br = [cx + hw, cy, cz - hd];

    // Front slope
    push_quad(tris, e_fl, e_fr, r1, r0, color);
    // Back slope
    push_quad(tris, e_br, e_bl, r0, r1, color);
    // Left hip (triangle)
    let n_left = tri_normal(e_bl, e_fl, r0);
    push_tri(tris, e_bl, e_fl, r0, n_left, color);
    // Right hip (triangle)
    let n_right = tri_normal(e_fr, e_br, r1);
    push_tri(tris, e_fr, e_br, r1, n_right, color);
}

// ── Cornice / Ledge ─────────────────────────────────────────────────────────

/// Horizontal ledge around a building at given height. Like a cornice or belt course.
/// Protrudes `depth` outward from the building face.
pub fn cornice_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    w: f32, d: f32, ledge_height: f32, depth: f32, color: u32,
) {
    let hw = w * 0.5 + depth;
    let hd = d * 0.5 + depth;
    let hh = ledge_height * 0.5;
    box_tris(tris, cx, cy, cz, hw * 2.0, ledge_height, hd * 2.0, color);
    let _ = hh; // dimensions handled by box_tris
}

// ── Individual Leaf ──────────────────────────────────────────────────────────

/// Single leaf: flat diamond shape (2 tris). `right`/`up` define the leaf plane.
/// Leaf is elongated along `up` (1.4x taller than wide).
fn leaf_tris(
    tris: &mut Vec<WorldTri>, center: [f32; 3],
    size: f32, right: [f32; 3], up: [f32; 3], color: u32,
) {
    let r = [right[0] * size, right[1] * size, right[2] * size];
    let u = [up[0] * size * 1.4, up[1] * size * 1.4, up[2] * size * 1.4];
    let tip = [center[0] + u[0], center[1] + u[1], center[2] + u[2]];
    let bot = [center[0] - u[0], center[1] - u[1], center[2] - u[2]];
    let lft = [center[0] - r[0], center[1] - r[1], center[2] - r[2]];
    let rgt = [center[0] + r[0], center[1] + r[1], center[2] + r[2]];
    let normal = tri_normal(lft, tip, rgt);
    push_tri(tris, lft, tip, rgt, normal, color);
    push_tri(tris, lft, rgt, bot, normal, color);
}

/// Scatter individual leaves on the surface shell of a sphere to form a canopy cluster.
/// Leaves face outward with random rotation. `leaf_count` leaves × 2 tris each.
pub fn leaf_canopy_tris(
    tris: &mut Vec<WorldTri>,
    cx: f32, cy: f32, cz: f32,
    radius: f32, leaf_count: usize, leaf_size: f32,
    seed: u64, colors: &[u32],
) {
    let mut h = seed;
    let next_f = |h: &mut u64| -> f32 {
        *h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*h >> 16) & 0xFFFF) as f32 / 65535.0
    };

    for _ in 0..leaf_count {
        // Uniform sphere surface: random azimuth + acos(uniform) for polar
        let theta = next_f(&mut h) * std::f32::consts::TAU;
        let cos_phi = next_f(&mut h) * 2.0 - 1.0;
        let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
        // Shell placement: 0.65r to 1.0r
        let r_frac = 0.65 + next_f(&mut h) * 0.35;

        let lx = cx + radius * r_frac * sin_phi * theta.cos();
        let ly = cy + radius * r_frac * cos_phi;
        let lz = cz + radius * r_frac * sin_phi * theta.sin();

        // Outward direction from center
        let dx = lx - cx;
        let dy = ly - cy;
        let dz = lz - cz;
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        let outward = if len > 0.001 { [dx / len, dy / len, dz / len] } else { [0.0, 1.0, 0.0] };

        // Build tangent frame on leaf surface
        let world_up = if outward[1].abs() < 0.99 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let right_base = normalize3(cross3(outward, world_up));
        let up_base = cross3(right_base, outward);

        // Random rotation around outward normal
        let rot = next_f(&mut h) * std::f32::consts::TAU;
        let (sr, cr) = (rot.sin(), rot.cos());
        let right = [
            right_base[0] * cr + up_base[0] * sr,
            right_base[1] * cr + up_base[1] * sr,
            right_base[2] * cr + up_base[2] * sr,
        ];
        let up = [
            -right_base[0] * sr + up_base[0] * cr,
            -right_base[1] * sr + up_base[1] * cr,
            -right_base[2] * sr + up_base[2] * cr,
        ];

        let color = colors[((h >> 8) as usize) % colors.len().max(1)];
        let sz = leaf_size * (0.7 + next_f(&mut h) * 0.6);
        leaf_tris(tris, [lx, ly, lz], sz, right, up, color);
    }
}

// ── Grass ───────────────────────────────────────────────────────────────────

/// Single grass blade: tapered quad (wide base, narrow mid, pointed tip). 3 tris.
/// `lean` in [0..1] controls how much it tilts, `lean_angle` is direction in radians.
fn grass_blade_tri(
    tris: &mut Vec<WorldTri>,
    x: f32, y: f32, z: f32,
    height: f32, width: f32,
    lean: f32, lean_angle: f32,
    color: u32,
) {
    let hw = width * 0.5;
    let (sl, cl) = (lean_angle.sin(), lean_angle.cos());
    // Base left/right perpendicular to lean direction
    let base_l = [x - hw * cl, y, z - hw * sl];
    let base_r = [x + hw * cl, y, z + hw * sl];
    // Mid-blade: wider taper (70% of base) for more visible blade body
    let mid_lean = lean * 0.4;
    let mid_hw = hw * 0.7;
    let mid_y = y + height * 0.45;
    let mid_x = x + mid_lean * height * 0.15 * sl;
    let mid_z = z - mid_lean * height * 0.15 * cl;
    let mid_l = [mid_x - mid_hw * cl, mid_y, mid_z - mid_hw * sl];
    let mid_r = [mid_x + mid_hw * cl, mid_y, mid_z + mid_hw * sl];
    // Tip leans in lean_angle direction
    let tip = [
        x + lean * height * 0.3 * sl,
        y + height,
        z - lean * height * 0.3 * cl,
    ];
    // Brighter yellow-green tip for sunlit gradient
    let tip_color = brighten_color(color, 30);
    // Lower quad: base to mid (2 tris)
    let n1 = tri_normal(base_l, mid_l, base_r);
    push_tri(tris, base_l, mid_l, base_r, n1, color);
    push_tri(tris, base_r, mid_l, mid_r, n1, color);
    // Upper triangle: mid to tip (1 tri)
    let n2 = tri_normal(mid_l, tip, mid_r);
    push_tri(tris, mid_l, tip, mid_r, n2, tip_color);
}

fn brighten_color(c: u32, delta: i32) -> u32 {
    let r = (((c >> 16) & 0xFF) as i32 + delta).clamp(0, 255) as u32;
    let g = (((c >> 8) & 0xFF) as i32 + delta + delta / 2).clamp(0, 255) as u32;
    let b = ((c & 0xFF) as i32 + delta).clamp(0, 255) as u32;
    (c & 0xFF000000) | (r << 16) | (g << 8) | b
}

/// Generate a patch of grass blades in a circular area. Each blade is 3 tris.
pub fn grass_patch_tris(
    tris: &mut Vec<WorldTri>,
    cx: f32, cy: f32, cz: f32,
    patch_radius: f32, blade_count: usize,
    blade_height: f32, blade_width: f32,
    seed: u64, colors: &[u32],
    height_fn: Option<&dyn Fn(f32, f32) -> f32>,
) {
    let mut h = seed;
    let next_f = |h: &mut u64| -> f32 {
        *h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*h >> 16) & 0xFFFF) as f32 / 65535.0
    };

    for _ in 0..blade_count {
        // Random position in circular patch
        let angle = next_f(&mut h) * std::f32::consts::TAU;
        let dist = next_f(&mut h).sqrt() * patch_radius; // sqrt for uniform distribution
        let bx = cx + angle.cos() * dist;
        let bz = cz + angle.sin() * dist;
        let by = if let Some(hf) = height_fn { hf(bx, bz) } else { cy };

        let bh = blade_height * (0.6 + next_f(&mut h) * 0.8);
        let bw = blade_width * (0.7 + next_f(&mut h) * 0.6);
        let lean = 0.2 + next_f(&mut h) * 0.5;
        let lean_dir = next_f(&mut h) * std::f32::consts::TAU;
        let color = colors[((h >> 10) as usize) % colors.len().max(1)];

        grass_blade_tri(tris, bx, by, bz, bh, bw, lean, lean_dir, color);
    }
}

// ── Bark Cylinder ───────────────────────────────────────────────────────────

/// Cylinder with vertical bark ridges. Creates a more organic trunk appearance.
/// `ridge_count` ridges protrude slightly from the base radius.
/// Each ridge adds 2 extra tris per segment height. Total ≈ segments × 4 + ridges × segments × 2.
pub fn bark_cylinder_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    r: f32, h: f32, segments: usize, ridge_depth: f32,
    seed: u64, color: u32, ridge_color: u32,
) {
    let hh = h * 0.5;
    let n = segments.max(6);
    let step = std::f32::consts::TAU / n as f32;

    // Base cylinder (smooth)
    let top_center = [cx, cy + hh, cz];
    let bot_center = [cx, cy - hh, cz];

    // Hash to determine which segments get ridges
    let mut rh = seed;
    let next_h = |rh: &mut u64| -> f32 {
        *rh = rh.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*rh >> 16) & 0xFFFF) as f32 / 65535.0
    };

    for i in 0..n {
        let a0 = i as f32 * step;
        let a1 = (i + 1) as f32 * step;
        let (s0, c0) = (a0.sin(), a0.cos());
        let (s1, c1) = (a1.sin(), a1.cos());

        // Determine if this segment has a ridge
        let has_ridge = next_h(&mut rh) < 0.4; // ~40% of segments get ridges
        let r_eff = if has_ridge { r + ridge_depth } else { r };
        let col = if has_ridge { ridge_color } else { color };

        let bt0 = [cx + r_eff * c0, cy - hh, cz + r_eff * s0];
        let bt1 = [cx + r * c1, cy - hh, cz + r * s1]; // next segment uses base r
        let tp0 = [cx + r_eff * c0, cy + hh, cz + r_eff * s0];
        let tp1 = [cx + r * c1, cy + hh, cz + r * s1];

        push_quad(tris, bt0, tp0, tp1, bt1, col);

        let n_top = [0.0, 1.0, 0.0];
        push_tri(tris, top_center, tp1, tp0, n_top, col);
        let n_bot = [0.0, -1.0, 0.0];
        push_tri(tris, bot_center, bt0, bt1, n_bot, col);
    }
}

// ── Bush ────────────────────────────────────────────────────────────────────

/// Generate a bush: short trunk + dense leaf clusters at low height.
pub fn bush_tris(
    tris: &mut Vec<WorldTri>,
    cx: f32, cy: f32, cz: f32,
    radius: f32, height: f32,
    seed: u64, leaf_colors: &[u32], trunk_color: u32,
) {
    // Short visible trunk/stem (only bottom portion visible)
    let trunk_h = height * 0.3;
    let trunk_r = radius * 0.08;
    cylinder_tris(tris, cx, cy + trunk_h * 0.5, cz, trunk_r, trunk_h, 5, trunk_color);

    // 2-4 overlapping leaf clusters forming the bush body
    let cluster_count = 2 + ((seed >> 4) as usize % 3);
    let mut h = seed;
    let next_f = |h: &mut u64| -> f32 {
        *h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*h >> 16) & 0xFFFF) as f32 / 65535.0
    };

    for ci in 0..cluster_count {
        let angle = (ci as f32 / cluster_count as f32) * std::f32::consts::TAU + next_f(&mut h) * 0.5;
        let spread = radius * 0.3;
        let clx = cx + angle.cos() * spread;
        let clz = cz + angle.sin() * spread;
        let cly = cy + height * (0.4 + next_f(&mut h) * 0.2);
        let cr = radius * (0.5 + next_f(&mut h) * 0.3);
        let leaves_per = 35 + ((h >> 8) as usize % 20);
        let leaf_sz = radius * 0.08;
        leaf_canopy_tris(tris, clx, cly, clz, cr, leaves_per, leaf_sz,
            h.wrapping_add(ci as u64), leaf_colors);
    }
}

// ── Helper math ─────────────────────────────────────────────────────────────

fn cross3(a: [f32;3], b: [f32;3]) -> [f32;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn normalize3(v: [f32;3]) -> [f32;3] {
    let l = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if l < 1e-10 { [0.0, 1.0, 0.0] } else { [v[0]/l, v[1]/l, v[2]/l] }
}

fn add3(a: [f32;3], b: [f32;3]) -> [f32;3] {
    [a[0]+b[0], a[1]+b[1], a[2]+b[2]]
}

fn sub3(a: [f32;3], b: [f32;3]) -> [f32;3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}

// ── Tapered Cylinder ────────────────────────────────────────────────────────

/// Cylinder with different top/bottom radii (for limbs, tapered shapes).
/// Y-axis aligned: bottom at cy - h/2, top at cy + h/2.
pub fn tapered_cylinder_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    r_bot: f32, r_top: f32, h: f32, segments: usize, color: u32,
) {
    let hh = h * 0.5;
    let n = quality_segments(segments).max(3);
    let step = std::f32::consts::TAU / n as f32;
    let top_c = [cx, cy + hh, cz];
    let bot_c = [cx, cy - hh, cz];

    for i in 0..n {
        let a0 = i as f32 * step;
        let a1 = (i + 1) as f32 * step;
        let (s0, c0) = (a0.sin(), a0.cos());
        let (s1, c1) = (a1.sin(), a1.cos());

        let bt0 = [cx + r_bot*c0, cy - hh, cz + r_bot*s0];
        let bt1 = [cx + r_bot*c1, cy - hh, cz + r_bot*s1];
        let tp0 = [cx + r_top*c0, cy + hh, cz + r_top*s0];
        let tp1 = [cx + r_top*c1, cy + hh, cz + r_top*s1];

        push_quad(tris, bt0, tp0, tp1, bt1, color);

        if r_top > 0.001 {
            push_tri(tris, top_c, tp1, tp0, [0.0, 1.0, 0.0], color);
        }
        push_tri(tris, bot_c, bt0, bt1, [0.0, -1.0, 0.0], color);
    }
}

// ── Ellipsoid ───────────────────────────────────────────────────────────────

/// Axis-aligned ellipsoid (stretched sphere). rx/ry/rz are the three radii.
/// Uses icosphere subdivision then scales.
pub fn ellipsoid_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    rx: f32, ry: f32, rz: f32, subdivisions: u32, color: u32,
) {
    // Generate unit icosphere vertices, then scale
    let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let a = 1.0;
    let b = 1.0 / phi;
    let base_verts: [[f32;3]; 12] = [
        [-b, a, 0.0],[b, a, 0.0],[-b,-a, 0.0],[b,-a, 0.0],
        [0.0,-b, a],[0.0, b, a],[0.0,-b,-a],[0.0, b,-a],
        [a, 0.0,-b],[a, 0.0, b],[-a, 0.0,-b],[-a, 0.0, b],
    ];
    let base_tris: [[usize;3]; 20] = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ];

    let mut verts: Vec<[f32;3]> = base_verts.to_vec();
    let mut faces: Vec<[usize;3]> = base_tris.to_vec();

    for _ in 0..quality_subdivisions(subdivisions) {
        let mut new_faces = Vec::with_capacity(faces.len() * 4);
        let mut midpoint_cache = std::collections::HashMap::new();
        for face in &faces {
            let mut mids = [0usize; 3];
            for e in 0..3 {
                let (a_idx, b_idx) = (face[e], face[(e+1)%3]);
                let key = if a_idx < b_idx { (a_idx, b_idx) } else { (b_idx, a_idx) };
                mids[e] = *midpoint_cache.entry(key).or_insert_with(|| {
                    let va = verts[a_idx];
                    let vb = verts[b_idx];
                    let mid = normalize3([(va[0]+vb[0])*0.5, (va[1]+vb[1])*0.5, (va[2]+vb[2])*0.5]);
                    verts.push(mid);
                    verts.len() - 1
                });
            }
            new_faces.push([face[0], mids[0], mids[2]]);
            new_faces.push([mids[0], face[1], mids[1]]);
            new_faces.push([mids[2], mids[1], face[2]]);
            new_faces.push([mids[0], mids[1], mids[2]]);
        }
        faces = new_faces;
    }

    // Scale vertices by ellipsoid radii and offset
    for face in &faces {
        let mut tv = [[0.0f32;3]; 3];
        for (i, &vi) in face.iter().enumerate() {
            let v = verts[vi];
            tv[i] = [cx + v[0]*rx, cy + v[1]*ry, cz + v[2]*rz];
        }
        let normal = tri_normal(tv[0], tv[1], tv[2]);
        push_tri(tris, tv[0], tv[1], tv[2], normal, color);
    }
}

// ── Oriented tapered cylinder ───────────────────────────────────────────────

/// Tapered cylinder between two arbitrary points with different radii at each end.
pub fn tapered_cylinder_between(
    tris: &mut Vec<WorldTri>, p0: [f32;3], p1: [f32;3],
    r0: f32, r1: f32, segments: usize, color: u32,
) {
    let dx = p1[0]-p0[0]; let dy = p1[1]-p0[1]; let dz = p1[2]-p0[2];
    let h = (dx*dx+dy*dy+dz*dz).sqrt();
    if h < 1e-6 { return; }
    let dir = [dx/h, dy/h, dz/h];
    let up = if dir[1].abs() < 0.99 { [0.0,1.0,0.0] } else { [1.0,0.0,0.0] };
    let right = normalize3(cross3(dir, up));
    let fwd = cross3(right, dir);
    let n = quality_segments(segments).max(3);
    let step = std::f32::consts::TAU / n as f32;
    let _mid = [(p0[0]+p1[0])*0.5, (p0[1]+p1[1])*0.5, (p0[2]+p1[2])*0.5];

    for i in 0..n {
        let a0 = i as f32 * step;
        let a1 = (i+1) as f32 * step;
        let (s0,c0) = (a0.sin(), a0.cos());
        let (s1,c1) = (a1.sin(), a1.cos());
        let bot0 = [p0[0]+r0*(right[0]*c0+fwd[0]*s0), p0[1]+r0*(right[1]*c0+fwd[1]*s0), p0[2]+r0*(right[2]*c0+fwd[2]*s0)];
        let bot1 = [p0[0]+r0*(right[0]*c1+fwd[0]*s1), p0[1]+r0*(right[1]*c1+fwd[1]*s1), p0[2]+r0*(right[2]*c1+fwd[2]*s1)];
        let top0 = [p1[0]+r1*(right[0]*c0+fwd[0]*s0), p1[1]+r1*(right[1]*c0+fwd[1]*s0), p1[2]+r1*(right[2]*c0+fwd[2]*s0)];
        let top1 = [p1[0]+r1*(right[0]*c1+fwd[0]*s1), p1[1]+r1*(right[1]*c1+fwd[1]*s1), p1[2]+r1*(right[2]*c1+fwd[2]*s1)];
        push_quad(tris, bot0, top0, top1, bot1, color);
        if r1 > 0.001 {
            push_tri(tris, p1, top1, top0, dir, color);
        }
        let neg_dir = [-dir[0],-dir[1],-dir[2]];
        push_tri(tris, p0, bot0, bot1, neg_dir, color);
    }
}

// ── Ring torus segment ──────────────────────────────────────────────────────

/// Partial torus/ring — useful for collar rims, boot cuffs, belt buckles.
/// Creates a ring of tube_segments around a circle of ring_segments at (cx,cy,cz).
pub fn ring_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    ring_r: f32, tube_r: f32, ring_segments: usize, tube_segments: usize, color: u32,
) {
    let rn = ring_segments.max(3);
    let tn = tube_segments.max(3);
    let rs = std::f32::consts::TAU / rn as f32;
    let ts = std::f32::consts::TAU / tn as f32;

    for i in 0..rn {
        let ra0 = i as f32 * rs;
        let ra1 = (i+1) as f32 * rs;
        for j in 0..tn {
            let ta0 = j as f32 * ts;
            let ta1 = (j+1) as f32 * ts;

            let point = |ra: f32, ta: f32| -> [f32;3] {
                let r = ring_r + tube_r * ta.cos();
                [cx + r * ra.cos(), cy + tube_r * ta.sin(), cz + r * ra.sin()]
            };

            let p00 = point(ra0, ta0);
            let p10 = point(ra1, ta0);
            let p11 = point(ra1, ta1);
            let p01 = point(ra0, ta1);
            push_quad(tris, p00, p10, p11, p01, color);
        }
    }
}

// ── Loft (cross-section skinning) ─────────────────────────────────────────

/// Loft: create a smooth surface by connecting cross-section rings at different Y heights.
/// Each ring has N (x, z) points forming a closed contour (counterclockwise from above).
/// All rings must have the same point count. Rings listed bottom to top.
/// Top and bottom are capped with triangle fans.
pub fn loft_y_tris(
    tris: &mut Vec<WorldTri>,
    rings: &[(f32, Vec<[f32; 2]>, u32)], // (y_height, [(x, z)], color)
) {
    loft_y_tris_caps(tris, rings, true, true);
}

/// Loft with cap control — skip bottom/top caps for tubes that disappear into other geometry
pub fn loft_y_tris_caps(
    tris: &mut Vec<WorldTri>,
    rings: &[(f32, Vec<[f32; 2]>, u32)],
    bottom_cap: bool,
    top_cap: bool,
) {
    if rings.len() < 2 { return; }
    let n = rings[0].1.len();

    // Connect adjacent rings with quads
    for hi in 0..rings.len() - 1 {
        let y0 = rings[hi].0;
        let pts0 = &rings[hi].1;
        let col = rings[hi].2;
        let y1 = rings[hi + 1].0;
        let pts1 = &rings[hi + 1].1;

        for pi in 0..n {
            let pn = (pi + 1) % n;
            let a = [pts0[pi][0], y0, pts0[pi][1]];
            let b = [pts1[pi][0], y1, pts1[pi][1]];
            let c = [pts1[pn][0], y1, pts1[pn][1]];
            let d = [pts0[pn][0], y0, pts0[pn][1]];
            push_quad(tris, a, b, c, d, col);
        }
    }

    if bottom_cap {
        let yb = rings[0].0;
        let bpts = &rings[0].1;
        let bcol = rings[0].2;
        let bcx: f32 = bpts.iter().map(|p| p[0]).sum::<f32>() / n as f32;
        let bcz: f32 = bpts.iter().map(|p| p[1]).sum::<f32>() / n as f32;
        let bc = [bcx, yb, bcz];
        for pi in 0..n {
            let pn = (pi + 1) % n;
            let a = [bpts[pi][0], yb, bpts[pi][1]];
            let b = [bpts[pn][0], yb, bpts[pn][1]];
            push_tri(tris, bc, a, b, [0.0, -1.0, 0.0], bcol);
        }
    }

    if top_cap {
        let Some(last) = rings.last() else { return; };
        let yt = last.0;
        let tpts = &last.1;
        let tcol = last.2;
        let tcx: f32 = tpts.iter().map(|p| p[0]).sum::<f32>() / n as f32;
        let tcz: f32 = tpts.iter().map(|p| p[1]).sum::<f32>() / n as f32;
        let tc = [tcx, yt, tcz];
        for pi in 0..n {
            let pn = (pi + 1) % n;
            let a = [tpts[pi][0], yt, tpts[pi][1]];
            let b = [tpts[pn][0], yt, tpts[pn][1]];
            push_tri(tris, tc, b, a, [0.0, 1.0, 0.0], tcol);
        }
    }
}

// ── Glow halo ────────────────────────────────────────────────────────────────

/// Emit a radial glow halo in a given plane around a center point.
/// The halo is a fan of triangles with the bright emissive color at center,
/// fading through 2 concentric rings to a dim outer edge.
///
/// `normal` defines the plane orientation (the disc is perpendicular to it).
/// `inner_r` is the solid bright core radius; `outer_r` is the dim edge radius.
/// `color` is the base emissive color (alpha=0x00, will be dimmed for outer rings).
/// `segments` controls smoothness of the disc (8-12 is typical).
fn glow_disc(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    normal: [f32; 3], inner_r: f32, outer_r: f32,
    segments: usize, color: u32,
) {
    let n = segments.max(6);
    let step = std::f32::consts::TAU / n as f32;

    // Build two tangent vectors perpendicular to normal
    let (tx, ty, tz) = (normal[0], normal[1], normal[2]);
    // Pick a non-parallel vector to cross with
    let (ax, ay, az) = if tx.abs() < 0.9 { (1.0, 0.0, 0.0) } else { (0.0, 1.0, 0.0) };
    // u = normalize(cross(normal, arbitrary))
    let (ux, uy, uz) = (ty * az - tz * ay, tz * ax - tx * az, tx * ay - ty * ax);
    let ul = (ux * ux + uy * uy + uz * uz).sqrt();
    let (ux, uy, uz) = (ux / ul, uy / ul, uz / ul);
    // v = normalize(cross(normal, u))
    let (vx, vy, vz) = (ty * uz - tz * uy, tz * ux - tx * uz, tx * uy - ty * ux);

    // Ring radii: core, mid, outer
    let mid_r = inner_r + (outer_r - inner_r) * 0.45;

    // Color dimming for rings (emissive colors have alpha=0x00)
    let r = (color >> 16) & 0xFF;
    let g = (color >> 8) & 0xFF;
    let b = color & 0xFF;
    // Mid ring: ~40% brightness
    let mid_color = ((r * 40 / 100) << 16) | ((g * 40 / 100) << 8) | (b * 40 / 100);
    // Outer ring: ~12% brightness
    let out_color = ((r * 12 / 100) << 16) | ((g * 12 / 100) << 8) | (b * 12 / 100);

    // Helper to get point on disc at angle and radius
    let pt = |angle: f32, radius: f32| -> [f32; 3] {
        let (sa, ca) = angle.sin_cos();
        let dx = ux * ca + vx * sa;
        let dy = uy * ca + vy * sa;
        let dz = uz * ca + vz * sa;
        [cx + dx * radius, cy + dy * radius, cz + dz * radius]
    };

    let center = [cx, cy, cz];

    for i in 0..n {
        let a0 = i as f32 * step;
        let a1 = (i + 1) as f32 * step;

        // Core: center to inner_r (full brightness, CCW winding)
        let p0 = pt(a0, inner_r);
        let p1 = pt(a1, inner_r);
        push_tri(tris, center, p0, p1, normal, color);

        // Mid ring: inner_r to mid_r
        let m0 = pt(a0, mid_r);
        let m1 = pt(a1, mid_r);
        push_tri(tris, p0, m0, m1, normal, mid_color);
        push_tri(tris, p0, m1, p1, normal, mid_color);

        // Outer ring: mid_r to outer_r
        let o0 = pt(a0, outer_r);
        let o1 = pt(a1, outer_r);
        push_tri(tris, m0, o0, o1, normal, out_color);
        push_tri(tris, m0, o1, m1, normal, out_color);
    }
}

/// Emit a multi-plane glow halo visible from all angles.
/// Creates glow discs in 3 perpendicular planes (XY, YZ, XZ) for omnidirectional visibility.
/// `core_r` = bright center radius, `glow_r` = dim outer edge radius.
pub fn glow_halo(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    core_r: f32, glow_r: f32, segments: usize, color: u32,
) {
    // XY plane (faces Z)
    glow_disc(tris, cx, cy, cz, [0.0, 0.0, 1.0], core_r, glow_r, segments, color);
    // YZ plane (faces X)
    glow_disc(tris, cx, cy, cz, [1.0, 0.0, 0.0], core_r, glow_r, segments, color);
    // XZ plane (faces Y) — horizontal disc (ground-visible glow)
    glow_disc(tris, cx, cy, cz, [0.0, 1.0, 0.0], core_r, glow_r, segments, color);
}

/// Emit a directional glow halo (for headlights/tail lights that face a specific direction).
/// Creates one disc facing the given direction plus a smaller perpendicular halo for side visibility.
pub fn glow_directional(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    dir: [f32; 3], core_r: f32, glow_r: f32, segments: usize, color: u32,
) {
    // Main disc facing the light direction
    glow_disc(tris, cx, cy, cz, dir, core_r, glow_r, segments, color);
    // Horizontal disc for top/bottom visibility (smaller)
    let side_r = glow_r * 0.6;
    let side_core = core_r * 0.7;
    glow_disc(tris, cx, cy, cz, [0.0, 1.0, 0.0], side_core, side_r, segments, color);
    // Cross-plane disc perpendicular to dir (vertical, rotated 90 degrees)
    // If dir is along Z, cross with Y to get X-axis disc
    let (dx, _dy, dz) = (dir[0], dir[1], dir[2]);
    let (cx2, cy2, cz2) = if dz.abs() > 0.5 {
        // Facing Z: add XY cross disc
        (1.0_f32, 0.0, 0.0)
    } else if dx.abs() > 0.5 {
        // Facing X: add YZ cross disc
        (0.0, 0.0, 1.0)
    } else {
        // Facing Y: add XZ cross disc
        (dx, 0.0, dz)
    };
    let l = (cx2 * cx2 + cy2 * cy2 + cz2 * cz2).sqrt();
    if l > 0.01 {
        glow_disc(tris, cx, cy, cz, [cx2/l, cy2/l, cz2/l], side_core, side_r, segments, color);
    }
}
