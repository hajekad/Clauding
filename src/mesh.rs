// Parametric mesh primitives: cylinder, cone, icosphere, beveled box, wall with holes,
// raised strip, wave surface, extrude profile, lathe. All push WorldTri into a Vec.

use crate::state::WorldTri;

/// Compute normalized cross product of triangle edges (CCW winding)
fn tri_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let e1 = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let e2 = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    let n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
    let l = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
    if l < 1e-10 { [0.0, 1.0, 0.0] } else { [n[0]/l, n[1]/l, n[2]/l] }
}

/// Push a quad as 2 tris (CCW winding: a-b-c, a-c-d)
fn push_quad(tris: &mut Vec<WorldTri>, a: [f32;3], b: [f32;3], c: [f32;3], d: [f32;3], color: u32) {
    let normal = tri_normal(a, b, c);
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
    let n = segments.max(3);
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

        // Top fan tri (CCW from above)
        let n_top = [0.0, 1.0, 0.0];
        tris.push(WorldTri { v: [top_center, tp1, tp0], normal: n_top, color });

        // Bottom fan tri (CCW from below)
        let n_bot = [0.0, -1.0, 0.0];
        tris.push(WorldTri { v: [bot_center, bt0, bt1], normal: n_bot, color });
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

    let n = segments.max(3);
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

        // End caps (CCW from outside)
        let n_cap0 = [-dir[0], -dir[1], -dir[2]];
        tris.push(WorldTri { v: [center0, b0, b1], normal: n_cap0, color });
        let n_cap1 = dir;
        tris.push(WorldTri { v: [center1, t1, t0], normal: n_cap1, color });
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
    let n = segments.max(3);
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

        // Side tri (CCW from outside)
        let normal = tri_normal(b0, apex, b1);
        tris.push(WorldTri { v: [b0, apex, b1], normal, color });

        // Base tri (CCW from below)
        let n_bot = [0.0, -1.0, 0.0];
        tris.push(WorldTri { v: [bot_center, b0, b1], normal: n_bot, color });
    }
}

// ── Icosphere ───────────────────────────────────────────────────────────────

/// Icosphere centered at (cx, cy, cz) with radius r.
/// subdivisions=0: 20 tris (icosahedron), 1: 80, 2: 320, 3: 1280.
pub fn sphere_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    r: f32, subdivisions: u32, color: u32,
) {
    // Golden ratio vertices for icosahedron
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let a = 1.0;
    let b = phi;

    let raw_verts: [[f32;3]; 12] = [
        [-a, b, 0.0], [ a, b, 0.0], [-a,-b, 0.0], [ a,-b, 0.0],
        [0.0,-a, b], [0.0, a, b], [0.0,-a,-b], [0.0, a,-b],
        [ b, 0.0,-a], [ b, 0.0, a], [-b, 0.0,-a], [-b, 0.0, a],
    ];

    // Normalize to unit sphere
    let mut verts: Vec<[f32;3]> = raw_verts.iter().map(|v| {
        let l = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        [v[0]/l, v[1]/l, v[2]/l]
    }).collect();

    let mut faces: Vec<[usize;3]> = vec![
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ];

    // Subdivide
    for _ in 0..subdivisions {
        let mut new_faces = Vec::with_capacity(faces.len() * 4);
        let mut midpoint_cache: Vec<(usize, usize, usize)> = Vec::new();

        let get_mid = |a_idx: usize, b_idx: usize, verts: &mut Vec<[f32;3]>, cache: &mut Vec<(usize, usize, usize)>| -> usize {
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
        };

        for f in &faces {
            let m01 = get_mid(f[0], f[1], &mut verts, &mut midpoint_cache);
            let m12 = get_mid(f[1], f[2], &mut verts, &mut midpoint_cache);
            let m20 = get_mid(f[2], f[0], &mut verts, &mut midpoint_cache);
            new_faces.push([f[0], m01, m20]);
            new_faces.push([f[1], m12, m01]);
            new_faces.push([f[2], m20, m12]);
            new_faces.push([m01, m12, m20]);
        }
        faces = new_faces;
    }

    // Output tris scaled and translated
    for f in &faces {
        let v0 = [cx + verts[f[0]][0]*r, cy + verts[f[0]][1]*r, cz + verts[f[0]][2]*r];
        let v1 = [cx + verts[f[1]][0]*r, cy + verts[f[1]][1]*r, cz + verts[f[1]][2]*r];
        let v2 = [cx + verts[f[2]][0]*r, cy + verts[f[2]][1]*r, cz + verts[f[2]][2]*r];
        let normal = tri_normal(v0, v1, v2);
        tris.push(WorldTri { v: [v0, v1, v2], normal, color });
    }
}

/// Icosphere with vertex perturbation for rocks.
pub fn perturbed_sphere_tris(
    tris: &mut Vec<WorldTri>, cx: f32, cy: f32, cz: f32,
    r: f32, subdivisions: u32, perturbation: f32, seed: u64, color: u32,
) {
    // Generate base icosphere vertices
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let a = 1.0;
    let b = phi;
    let raw_verts: [[f32;3]; 12] = [
        [-a, b, 0.0], [ a, b, 0.0], [-a,-b, 0.0], [ a,-b, 0.0],
        [0.0,-a, b], [0.0, a, b], [0.0,-a,-b], [0.0, a,-b],
        [ b, 0.0,-a], [ b, 0.0, a], [-b, 0.0,-a], [-b, 0.0, a],
    ];
    let mut verts: Vec<[f32;3]> = raw_verts.iter().map(|v| {
        let l = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
        [v[0]/l, v[1]/l, v[2]/l]
    }).collect();
    let mut faces: Vec<[usize;3]> = vec![
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ];
    for _ in 0..subdivisions {
        let mut new_faces = Vec::with_capacity(faces.len() * 4);
        let mut midpoint_cache: Vec<(usize, usize, usize)> = Vec::new();
        let get_mid = |a_idx: usize, b_idx: usize, verts: &mut Vec<[f32;3]>, cache: &mut Vec<(usize, usize, usize)>| -> usize {
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
        };
        for f in &faces {
            let m01 = get_mid(f[0], f[1], &mut verts, &mut midpoint_cache);
            let m12 = get_mid(f[1], f[2], &mut verts, &mut midpoint_cache);
            let m20 = get_mid(f[2], f[0], &mut verts, &mut midpoint_cache);
            new_faces.push([f[0], m01, m20]);
            new_faces.push([f[1], m12, m01]);
            new_faces.push([f[2], m20, m12]);
            new_faces.push([m01, m12, m20]);
        }
        faces = new_faces;
    }

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
        tris.push(WorldTri { v: [v0, v1, v2], normal, color });
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

    // 8 corner tris
    // Top-front-right
    tris.push(WorldTri {
        v: [[cx+hw-b, cy+hh-b, cz+hd], [cx+hw, cy+hh-b, cz+hd-b], [cx+hw-b, cy+hh, cz+hd-b]],
        normal: tri_normal([cx+hw-b, cy+hh-b, cz+hd], [cx+hw, cy+hh-b, cz+hd-b], [cx+hw-b, cy+hh, cz+hd-b]),
        color,
    });
    // Top-front-left
    tris.push(WorldTri {
        v: [[cx-hw+b, cy+hh-b, cz+hd], [cx-hw+b, cy+hh, cz+hd-b], [cx-hw, cy+hh-b, cz+hd-b]],
        normal: tri_normal([cx-hw+b, cy+hh-b, cz+hd], [cx-hw+b, cy+hh, cz+hd-b], [cx-hw, cy+hh-b, cz+hd-b]),
        color,
    });
    // Top-back-right
    tris.push(WorldTri {
        v: [[cx+hw-b, cy+hh-b, cz-hd], [cx+hw-b, cy+hh, cz-hd+b], [cx+hw, cy+hh-b, cz-hd+b]],
        normal: tri_normal([cx+hw-b, cy+hh-b, cz-hd], [cx+hw-b, cy+hh, cz-hd+b], [cx+hw, cy+hh-b, cz-hd+b]),
        color,
    });
    // Top-back-left
    tris.push(WorldTri {
        v: [[cx-hw+b, cy+hh-b, cz-hd], [cx-hw, cy+hh-b, cz-hd+b], [cx-hw+b, cy+hh, cz-hd+b]],
        normal: tri_normal([cx-hw+b, cy+hh-b, cz-hd], [cx-hw, cy+hh-b, cz-hd+b], [cx-hw+b, cy+hh, cz-hd+b]),
        color,
    });
    // Bottom-front-right
    tris.push(WorldTri {
        v: [[cx+hw-b, cy-hh+b, cz+hd], [cx+hw-b, cy-hh, cz+hd-b], [cx+hw, cy-hh+b, cz+hd-b]],
        normal: tri_normal([cx+hw-b, cy-hh+b, cz+hd], [cx+hw-b, cy-hh, cz+hd-b], [cx+hw, cy-hh+b, cz+hd-b]),
        color,
    });
    // Bottom-front-left
    tris.push(WorldTri {
        v: [[cx-hw+b, cy-hh+b, cz+hd], [cx-hw, cy-hh+b, cz+hd-b], [cx-hw+b, cy-hh, cz+hd-b]],
        normal: tri_normal([cx-hw+b, cy-hh+b, cz+hd], [cx-hw, cy-hh+b, cz+hd-b], [cx-hw+b, cy-hh, cz+hd-b]),
        color,
    });
    // Bottom-back-right
    tris.push(WorldTri {
        v: [[cx+hw-b, cy-hh+b, cz-hd], [cx+hw, cy-hh+b, cz-hd+b], [cx+hw-b, cy-hh, cz-hd+b]],
        normal: tri_normal([cx+hw-b, cy-hh+b, cz-hd], [cx+hw, cy-hh+b, cz-hd+b], [cx+hw-b, cy-hh, cz-hd+b]),
        color,
    });
    // Bottom-back-left
    tris.push(WorldTri {
        v: [[cx-hw+b, cy-hh+b, cz-hd], [cx-hw+b, cy-hh, cz-hd+b], [cx-hw, cy-hh+b, cz-hd+b]],
        normal: tri_normal([cx-hw+b, cy-hh+b, cz-hd], [cx-hw+b, cy-hh, cz-hd+b], [cx-hw, cy-hh+b, cz-hd+b]),
        color,
    });
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
    hole_color: u32,
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

    let mut sorted_holes: Vec<&WallHole> = holes.iter().collect();
    sorted_holes.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

    let mut y_cuts = vec![0.0_f32, wall_h];
    for hole in &sorted_holes {
        y_cuts.push(hole.y);
        y_cuts.push(hole.y + hole.h);
    }
    y_cuts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    y_cuts.dedup_by(|a, b| (*a - *b).abs() < 0.001);

    for yi in 0..y_cuts.len()-1 {
        let strip_bot = y_cuts[yi];
        let strip_top = y_cuts[yi + 1];
        if (strip_top - strip_bot) < 0.001 { continue; }

        let strip_holes: Vec<&&WallHole> = sorted_holes.iter()
            .filter(|h| h.y < strip_top - 0.001 && h.y + h.h > strip_bot + 0.001)
            .collect();

        let mut x_start = 0.0_f32;
        for hole in &strip_holes {
            if hole.x > x_start + 0.001 {
                emit_wall_quad(tris, pos_x, pos_y, fz, x_start, strip_bot, hole.x, strip_top, wall_color, face_dir, left_dir, swap_xz);
            }
            x_start = hole.x + hole.w;
        }
        if x_start < wall_w - 0.001 {
            emit_wall_quad(tris, pos_x, pos_y, fz, x_start, strip_bot, wall_w, strip_top, wall_color, face_dir, left_dir, swap_xz);
        }
    }

    for hole in &sorted_holes {
        let hx0 = pos_x + hole.x * left_dir;
        let hx1 = pos_x + (hole.x + hole.w) * left_dir;
        let hy0 = pos_y + hole.y;
        let hy1 = pos_y + hole.y + hole.h;

        let (lx, rx) = if left_dir > 0.0 { (hx0, hx1) } else { (hx1, hx0) };

        // Back face of recess
        if face_dir > 0.0 {
            push_quad(tris, v(lx, hy0, bz), v(rx, hy0, bz), v(rx, hy1, bz), v(lx, hy1, bz), hole_color);
        } else {
            push_quad(tris, v(rx, hy0, bz), v(lx, hy0, bz), v(lx, hy1, bz), v(rx, hy1, bz), hole_color);
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

            let y00 = base_y + (x0 * freq).sin() * (z0 * freq * 0.7).cos() * amplitude;
            let y10 = base_y + (x1 * freq).sin() * (z0 * freq * 0.7).cos() * amplitude;
            let y01 = base_y + (x0 * freq).sin() * (z1 * freq * 0.7).cos() * amplitude;
            let y11 = base_y + (x1 * freq).sin() * (z1 * freq * 0.7).cos() * amplitude;

            let v00 = [x0, y00, z0];
            let v10 = [x1, y10, z0];
            let v01 = [x0, y01, z1];
            let v11 = [x1, y11, z1];

            let n1 = tri_normal(v00, v11, v10);
            tris.push(WorldTri { v: [v00, v11, v10], normal: n1, color });
            let n2 = tri_normal(v00, v01, v11);
            tris.push(WorldTri { v: [v00, v01, v11], normal: n2, color });
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
    let n = segments.max(3);
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
        tris.push(WorldTri { v: [c[idx[0]], c[idx[1]], c[idx[2]]], normal, color });
        tris.push(WorldTri { v: [c[idx[0]], c[idx[2]], c[idx[3]]], normal, color });
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
        tris.push(WorldTri { v: [c[idx[0]], c[idx[1]], c[idx[2]]], normal, color });
        tris.push(WorldTri { v: [c[idx[0]], c[idx[2]], c[idx[3]]], normal, color });
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
    tris.push(WorldTri { v: [e_bl, e_fl, r0], normal: n_left, color });
    let n_right = tri_normal(e_fr, e_br, r1);
    tris.push(WorldTri { v: [e_fr, e_br, r1], normal: n_right, color });
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
    tris.push(WorldTri { v: [e_bl, e_fl, r0], normal: n_left, color });
    // Right hip (triangle)
    let n_right = tri_normal(e_fr, e_br, r1);
    tris.push(WorldTri { v: [e_fr, e_br, r1], normal: n_right, color });
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
