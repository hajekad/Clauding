//! Runtime GLTF mesh loader — reads .gltf + .bin into Vec<WorldTri>.
//! No crates, just std. Reuses JSON parsing patterns from gltf_study.rs.

use crate::state::WorldTri;
use std::fs;

/// A loaded character model ready for rendering.
pub struct LoadedModel {
    /// Triangles in model-local space, Y=0 at feet, centered on X/Z, facing -Z.
    pub tris: Vec<WorldTri>,
    /// Model name for debug display.
    pub name: String,
}

/// Load a GLTF model from a directory containing scene.gltf + scene.bin.
/// Returns triangles normalized to game scale (1.8m tall, centered, Y-up).
pub fn load_gltf_model(dir: &str, name: &str, skin_color: u32) -> LoadedModel {
    let json_str = fs::read_to_string(format!("{dir}/scene.gltf"))
        .unwrap_or_else(|e| panic!("Failed to read {dir}/scene.gltf: {e}"));
    let bin_data = fs::read(format!("{dir}/scene.bin"))
        .unwrap_or_else(|e| panic!("Failed to read {dir}/scene.bin: {e}"));

    let accessors = parse_accessors(&json_str);
    let buffer_views = parse_buffer_views(&json_str);

    // Extract all mesh primitives (positions, normals, indices)
    let primitives = parse_all_primitives(&json_str);

    let mut all_tris: Vec<WorldTri> = Vec::new();

    for prim in &primitives {
        let pos_acc = prim.position_accessor;
        let norm_acc = prim.normal_accessor;
        let idx_acc = prim.index_accessor;

        if pos_acc >= accessors.len() { continue; }

        // Extract positions
        let positions = extract_vec3(&accessors[pos_acc], &buffer_views, &bin_data);

        // Extract normals (if available)
        let normals = if let Some(na) = norm_acc {
            if na < accessors.len() {
                extract_vec3(&accessors[na], &buffer_views, &bin_data)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Extract indices
        let indices = if let Some(ia) = idx_acc {
            if ia < accessors.len() {
                extract_indices_flat(&accessors[ia], &buffer_views, &bin_data)
            } else {
                Vec::new()
            }
        } else {
            // No indices — sequential triangles
            (0..positions.len()).collect()
        };

        // Build triangles
        for tri_idx in indices.chunks_exact(3) {
            let i0 = tri_idx[0];
            let i1 = tri_idx[1];
            let i2 = tri_idx[2];
            if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                continue;
            }

            let v0 = positions[i0];
            let v1 = positions[i1];
            let v2 = positions[i2];

            // Compute face normal from vertices
            let normal = if !normals.is_empty() && i0 < normals.len() {
                normals[i0] // use vertex normal of first vertex
            } else {
                // Compute from cross product
                let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
                let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
                let nx = e1[1]*e2[2] - e1[2]*e2[1];
                let ny = e1[2]*e2[0] - e1[0]*e2[2];
                let nz = e1[0]*e2[1] - e1[1]*e2[0];
                let nl = (nx*nx + ny*ny + nz*nz).sqrt();
                if nl > 1e-10 { [nx/nl, ny/nl, nz/nl] } else { [0.0, 1.0, 0.0] }
            };

            all_tris.push(WorldTri {
                v: [v0, v1, v2],
                normal,
                color: skin_color,
            });
        }
    }

    // Normalize: center on X/Z, Y=0 at feet, scale to 1.8m tall
    normalize_model(&mut all_tris);

    eprintln!("gltf_loader: loaded '{}' from {}: {} tris", name, dir, all_tris.len());

    LoadedModel {
        tris: all_tris,
        name: name.to_string(),
    }
}

/// Normalize model to game coordinates: Y=0 at feet, centered X/Z, 1.8m tall.
fn normalize_model(tris: &mut [WorldTri]) {
    if tris.is_empty() { return; }

    // Find bounding box
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for tri in tris.iter() {
        for v in &tri.v {
            for i in 0..3 {
                min[i] = min[i].min(v[i]);
                max[i] = max[i].max(v[i]);
            }
        }
    }

    let dims = [max[0]-min[0], max[1]-min[1], max[2]-min[2]];

    // Detect up axis (tallest dimension)
    let up = if dims[1] >= dims[0] && dims[1] >= dims[2] { 1 }
             else if dims[2] >= dims[0] && dims[2] >= dims[1] { 2 }
             else { 1 };
    let height = dims[up];
    let scale = 1.8 / height;

    let depth_ax = if up == 1 { 2 } else { 1 };

    let cx = (min[0] + max[0]) * 0.5;
    let cy_base = min[up];
    let cz = (min[depth_ax] + max[depth_ax]) * 0.5;

    for tri in tris.iter_mut() {
        for v in &mut tri.v {
            let old = *v;
            v[0] = (old[0] - cx) * scale;
            v[1] = (old[up] - cy_base) * scale;
            v[2] = (old[depth_ax] - cz) * scale;
        }
        // Recompute normal for the transformed vertices
        let e1 = [tri.v[1][0]-tri.v[0][0], tri.v[1][1]-tri.v[0][1], tri.v[1][2]-tri.v[0][2]];
        let e2 = [tri.v[2][0]-tri.v[0][0], tri.v[2][1]-tri.v[0][1], tri.v[2][2]-tri.v[0][2]];
        let nx = e1[1]*e2[2] - e1[2]*e2[1];
        let ny = e1[2]*e2[0] - e1[0]*e2[2];
        let nz = e1[0]*e2[1] - e1[1]*e2[0];
        let nl = (nx*nx + ny*ny + nz*nz).sqrt();
        if nl > 1e-10 {
            tri.normal = [nx/nl, ny/nl, nz/nl];
        }
    }
}

// ══════════════════════════════════════════════════════════════
// GLTF PARSING — minimal JSON helpers (from gltf_study.rs)
// ══════════════════════════════════════════════════════════════

#[derive(Debug)]
struct Accessor {
    buffer_view: usize,
    byte_offset: usize,
    comp_type: u32,
    count: usize,
    acc_type: String, // "VEC3", "SCALAR", etc.
}

#[derive(Debug)]
struct BufferView {
    byte_offset: usize,
    byte_stride: usize,
    byte_length: usize,
}

struct MeshPrimitive {
    position_accessor: usize,
    normal_accessor: Option<usize>,
    index_accessor: Option<usize>,
}

fn extract_vec3(acc: &Accessor, bvs: &[BufferView], bin: &[u8]) -> Vec<[f32; 3]> {
    if acc.buffer_view >= bvs.len() { return Vec::new(); }
    let bv = &bvs[acc.buffer_view];
    let base = bv.byte_offset + acc.byte_offset;
    let stride = if bv.byte_stride > 0 { bv.byte_stride } else { 12 };
    let mut result = Vec::with_capacity(acc.count);
    for i in 0..acc.count {
        let off = base + i * stride;
        if off + 12 > bin.len() { break; }
        let x = f32::from_le_bytes([bin[off], bin[off+1], bin[off+2], bin[off+3]]);
        let y = f32::from_le_bytes([bin[off+4], bin[off+5], bin[off+6], bin[off+7]]);
        let z = f32::from_le_bytes([bin[off+8], bin[off+9], bin[off+10], bin[off+11]]);
        result.push([x, y, z]);
    }
    result
}

fn extract_indices_flat(acc: &Accessor, bvs: &[BufferView], bin: &[u8]) -> Vec<usize> {
    if acc.buffer_view >= bvs.len() { return Vec::new(); }
    let bv = &bvs[acc.buffer_view];
    let base = bv.byte_offset + acc.byte_offset;
    let mut result = Vec::with_capacity(acc.count);
    match acc.comp_type {
        5125 => { // u32
            for i in 0..acc.count {
                let off = base + i * 4;
                if off + 4 > bin.len() { break; }
                result.push(u32::from_le_bytes([bin[off], bin[off+1], bin[off+2], bin[off+3]]) as usize);
            }
        }
        5123 => { // u16
            for i in 0..acc.count {
                let off = base + i * 2;
                if off + 2 > bin.len() { break; }
                result.push(u16::from_le_bytes([bin[off], bin[off+1]]) as usize);
            }
        }
        5121 => { // u8
            for i in 0..acc.count {
                let off = base + i;
                if off >= bin.len() { break; }
                result.push(bin[off] as usize);
            }
        }
        _ => eprintln!("gltf_loader: unsupported index type {}", acc.comp_type),
    }
    result
}

fn parse_accessors(json: &str) -> Vec<Accessor> {
    let mut result = Vec::new();
    let Some(start) = find_array(json, "\"accessors\"") else { return result; };
    let arr = extract_array(json, start);
    for obj in iter_objects(&arr) {
        result.push(Accessor {
            buffer_view: find_int(&obj, "\"bufferView\"").unwrap_or(0) as usize,
            byte_offset: find_int(&obj, "\"byteOffset\"").unwrap_or(0) as usize,
            comp_type: find_int(&obj, "\"componentType\"").unwrap_or(0) as u32,
            count: find_int(&obj, "\"count\"").unwrap_or(0) as usize,
            acc_type: find_string(&obj, "\"type\"").unwrap_or_default(),
        });
    }
    result
}

fn parse_buffer_views(json: &str) -> Vec<BufferView> {
    let mut result = Vec::new();
    let Some(start) = find_array(json, "\"bufferViews\"") else { return result; };
    let arr = extract_array(json, start);
    for obj in iter_objects(&arr) {
        result.push(BufferView {
            byte_offset: find_int(&obj, "\"byteOffset\"").unwrap_or(0) as usize,
            byte_stride: find_int(&obj, "\"byteStride\"").unwrap_or(0) as usize,
            byte_length: find_int(&obj, "\"byteLength\"").unwrap_or(0) as usize,
        });
    }
    result
}

/// Parse all mesh primitives, extracting POSITION, NORMAL, and indices accessor indices.
fn parse_all_primitives(json: &str) -> Vec<MeshPrimitive> {
    let mut result = Vec::new();

    // Find all "primitives" arrays in meshes
    let Some(meshes_start) = find_array(json, "\"meshes\"") else { return result; };
    let meshes_arr = extract_array(json, meshes_start);

    for mesh_obj in iter_objects(&meshes_arr) {
        let Some(prims_start) = find_array(&mesh_obj, "\"primitives\"") else { continue; };
        let prims_arr = extract_array(&mesh_obj, prims_start);

        for prim_obj in iter_objects(&prims_arr) {
            // Find attributes object
            let pos = find_attr_accessor(&prim_obj, "POSITION");
            let norm = find_attr_accessor(&prim_obj, "NORMAL");
            let idx = find_int(&prim_obj, "\"indices\"").map(|v| v as usize);

            if let Some(pos_acc) = pos {
                result.push(MeshPrimitive {
                    position_accessor: pos_acc,
                    normal_accessor: norm,
                    index_accessor: idx,
                });
            }
        }
    }
    result
}

fn find_attr_accessor(prim: &str, attr: &str) -> Option<usize> {
    // Look for "POSITION": N or "NORMAL": N inside "attributes" object
    let needle = format!("\"{}\"", attr);
    let idx = prim.find(&needle)?;
    let after = &prim[idx + needle.len()..];
    let after = after.trim_start();
    let after = if after.starts_with(':') { &after[1..] } else { after };
    let after = after.trim_start();
    parse_leading_int(after).map(|v| v as usize)
}

// ── JSON helpers ──

fn find_array(json: &str, key: &str) -> Option<usize> {
    let idx = json.find(key)?;
    let after = &json[idx + key.len()..];
    let after = after.trim_start();
    let after = if after.starts_with(':') { &after[1..] } else { after };
    let after = after.trim_start();
    if after.starts_with('[') { Some(json.len() - after.len()) } else { None }
}

fn extract_array(json: &str, start: usize) -> String {
    let bytes = json.as_bytes();
    let mut depth = 0;
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'[' => depth += 1,
            b']' => { depth -= 1; if depth == 0 { return json[start..=i].to_string(); } }
            b'"' => { i += 1; while i < bytes.len() && bytes[i] != b'"' { if bytes[i] == b'\\' { i += 1; } i += 1; } }
            _ => {}
        }
        i += 1;
    }
    json[start..].to_string()
}

fn iter_objects(arr: &str) -> Vec<String> {
    let mut result = Vec::new();
    let bytes = arr.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i] != b'[' { i += 1; }
    i += 1;
    loop {
        while i < bytes.len() && bytes[i] != b'{' { i += 1; }
        if i >= bytes.len() { break; }
        let start = i;
        let mut depth = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'{' => depth += 1,
                b'}' => { depth -= 1; if depth == 0 { result.push(arr[start..=i].to_string()); i += 1; break; } }
                b'"' => { i += 1; while i < bytes.len() && bytes[i] != b'"' { if bytes[i] == b'\\' { i += 1; } i += 1; } }
                _ => {}
            }
            i += 1;
        }
    }
    result
}

fn find_int(obj: &str, key: &str) -> Option<i64> {
    let idx = obj.find(key)?;
    let after = &obj[idx + key.len()..];
    let after = after.trim_start();
    let after = if after.starts_with(':') { &after[1..] } else { after };
    parse_leading_int(after.trim_start())
}

fn find_string(obj: &str, key: &str) -> Option<String> {
    let idx = obj.find(key)?;
    let after = &obj[idx + key.len()..];
    let after = after.trim_start();
    let after = if after.starts_with(':') { &after[1..] } else { after };
    let after = after.trim_start();
    if !after.starts_with('"') { return None; }
    let content = &after[1..];
    let end = content.find('"')?;
    Some(content[..end].to_string())
}

fn parse_leading_int(s: &str) -> Option<i64> {
    let mut end = 0;
    let bytes = s.as_bytes();
    if end < bytes.len() && bytes[end] == b'-' { end += 1; }
    while end < bytes.len() && bytes[end].is_ascii_digit() { end += 1; }
    if end == 0 || (end == 1 && bytes[0] == b'-') { return None; }
    s[..end].parse().ok()
}
