//! Runtime GLTF mesh loader — reads .gltf + .bin into Vec<WorldTri>.
//! No crates, just std. Auto-discovers models from directory structure.

use crate::state::WorldTri;
use std::fs;

/// A loaded model ready for rendering (local space, Y=0 at base, centered X/Z).
pub struct LoadedModel {
    pub tris: Vec<WorldTri>,
    pub name: String,
}

/// Auto-discovered model library — scans directories for scene.gltf files.
/// Drop a new folder with scene.gltf + scene.bin anywhere under a category
/// and it gets picked up automatically on next load.
pub struct ModelLibrary {
    pub architecture: Vec<Vec<WorldTri>>,  // building models
    pub trees: Vec<Vec<WorldTri>>,         // tree/vegetation models
    pub characters: Vec<Vec<WorldTri>>,    // character models (1.8m tall humanoids)
    pub character_names: Vec<String>,      // display names for character selection menu
    pub cars: Vec<Vec<WorldTri>>,          // car models (~4.5m long)
    pub car_names: Vec<String>,            // display names for car models
}

impl ModelLibrary {
    pub fn empty() -> Self {
        ModelLibrary {
            architecture: Vec::new(), trees: Vec::new(),
            characters: Vec::new(), character_names: Vec::new(),
            cars: Vec::new(), car_names: Vec::new(),
        }
    }

    /// Scan models/v1/ and load all discovered GLTF models.
    /// Architecture models normalize to their natural proportions (1 unit = 1 meter).
    /// Tree models normalize to a game-appropriate height range.
    pub fn load_all() -> Self {
        let base = "models/v1";
        let mut lib = Self::empty();

        // Architecture: load at natural scale (assume GLTF is in meters)
        // We normalize so tallest dimension = model height, preserving aspect ratio
        let max_tris_arch: usize = 500;
        let max_tris_tree: usize = 50;

        let arch_dirs = discover_gltf_dirs(&format!("{base}/architecture"));
        for dir in &arch_dirs {
            let name = dir.rsplit('/').next().unwrap_or("?");
            if let Some(mut m) = try_load_gltf_scaled(dir, name, 0xFFCCBBAA, 8.0) {
                decimate_to_budget(&mut m.tris, max_tris_arch);
                lib.architecture.push(m.tris);
            }
        }

        let nature_dirs = discover_gltf_dirs(&format!("{base}/nature"));
        for dir in &nature_dirs {
            let name = dir.rsplit('/').next().unwrap_or("?");
            if let Some(mut m) = try_load_gltf_scaled(dir, name, 0xFF447733, 6.0) {
                decimate_to_budget(&mut m.tris, max_tris_tree);
                lib.trees.push(m.tris);
            }
        }

        // Characters: dynamic per-frame rendering — must be low poly
        // 100 NPCs × tris × 3 verts × 28 bytes = per-frame upload budget
        // At 2000 tris: 100 × 2000 × 3 × 28 = 16.8MB/frame — manageable
        let max_tris_char: usize = 2_000;
        let char_dirs = discover_gltf_dirs(&format!("{base}/characters"));
        for dir in &char_dirs {
            let name = dir.rsplit('/').next().unwrap_or("?");
            let default_skin: u32 = 0xFFBBA088;
            if let Some(mut m) = try_load_gltf_scaled(dir, name, default_skin, 1.8) {
                decimate_to_budget(&mut m.tris, max_tris_char);
                if !m.tris.is_empty() {
                    eprintln!("  character '{}': {} tris", name, m.tris.len());
                    lib.characters.push(m.tris);
                    lib.character_names.push(name.to_string());
                }
            }
        }

        // Cars: load at ~4.5m long (longest axis), max 20K tris
        let max_tris_car: usize = 20_000;
        let car_dirs = discover_gltf_dirs(&format!("{base}/cars"));
        for dir in &car_dirs {
            let name = dir.rsplit('/').next().unwrap_or("?");
            if let Some(mut m) = try_load_gltf_car(dir, name, 0xFF888888, 4.5) {
                decimate_to_budget(&mut m.tris, max_tris_car);
                if !m.tris.is_empty() {
                    eprintln!("  car '{}': {} tris", name, m.tris.len());
                    lib.cars.push(m.tris);
                    lib.car_names.push(name.to_string());
                }
            }
        }

        eprintln!("ModelLibrary: {} architecture, {} tree, {} character, {} car models",
            lib.architecture.len(), lib.trees.len(), lib.characters.len(), lib.cars.len());
        lib
    }
}

/// Vertex-clustering decimation: merges nearby vertices on a 3D grid.
/// Preserves mesh topology — adjacent triangles stay connected.
/// Degenerate triangles (collapsed to a line or point) are removed.
fn decimate_to_budget(tris: &mut Vec<WorldTri>, max_tris: usize) {
    if tris.len() <= max_tris { return; }
    let original = tris.len();

    // Binary search for grid resolution that gives ~target tri count
    let mut lo_res = 4u32;
    let mut hi_res = 512u32;
    let mut best_tris: Vec<WorldTri> = Vec::new();

    for _ in 0..12 { // max 12 iterations of binary search
        let mid = (lo_res + hi_res) / 2;
        let result = cluster_decimate(tris, mid);
        if result.len() > max_tris {
            hi_res = mid;
        } else {
            best_tris = result;
            lo_res = mid + 1;
        }
        if lo_res >= hi_res { break; }
    }

    if best_tris.is_empty() {
        best_tris = cluster_decimate(tris, lo_res);
    }

    eprintln!("  decimated {} -> {} tris (grid {})", original, best_tris.len(), lo_res);
    *tris = best_tris;
}

/// Cluster vertices into a 3D grid and rebuild triangles.
fn cluster_decimate(tris: &[WorldTri], grid_res: u32) -> Vec<WorldTri> {
    // Compute bounding box
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for tri in tris {
        for v in &tri.v {
            for i in 0..3 { min[i] = min[i].min(v[i]); max[i] = max[i].max(v[i]); }
        }
    }
    let dims = [max[0]-min[0]+1e-6, max[1]-min[1]+1e-6, max[2]-min[2]+1e-6];
    let inv_cell = [grid_res as f32 / dims[0], grid_res as f32 / dims[1], grid_res as f32 / dims[2]];

    // Map vertex to grid cell index
    let to_cell = |v: [f32; 3]| -> u64 {
        let cx = ((v[0] - min[0]) * inv_cell[0]).min(grid_res as f32 - 1.0).max(0.0) as u64;
        let cy = ((v[1] - min[1]) * inv_cell[1]).min(grid_res as f32 - 1.0).max(0.0) as u64;
        let cz = ((v[2] - min[2]) * inv_cell[2]).min(grid_res as f32 - 1.0).max(0.0) as u64;
        cx + cy * grid_res as u64 + cz * grid_res as u64 * grid_res as u64
    };

    // Accumulate vertex positions per cell
    use std::collections::HashMap;
    let mut cell_sum: HashMap<u64, ([f64; 3], u32)> = HashMap::new();
    for tri in tris {
        for v in &tri.v {
            let cell = to_cell(*v);
            let entry = cell_sum.entry(cell).or_insert(([0.0; 3], 0));
            entry.0[0] += v[0] as f64;
            entry.0[1] += v[1] as f64;
            entry.0[2] += v[2] as f64;
            entry.1 += 1;
        }
    }

    // Compute cell representative positions (centroid of all vertices in cell)
    let cell_pos: HashMap<u64, [f32; 3]> = cell_sum.iter().map(|(&cell, &(sum, count))| {
        let c = count as f64;
        (cell, [(sum[0]/c) as f32, (sum[1]/c) as f32, (sum[2]/c) as f32])
    }).collect();

    // Rebuild triangles using cell representatives, skip degenerate
    let mut result = Vec::new();
    for tri in tris {
        let c0 = to_cell(tri.v[0]);
        let c1 = to_cell(tri.v[1]);
        let c2 = to_cell(tri.v[2]);

        // Skip degenerate: any two vertices collapsed to same cell
        if c0 == c1 || c1 == c2 || c0 == c2 { continue; }

        let v0 = cell_pos[&c0];
        let v1 = cell_pos[&c1];
        let v2 = cell_pos[&c2];

        // Recompute normal
        let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
        let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
        let nx = e1[1]*e2[2] - e1[2]*e2[1];
        let ny = e1[2]*e2[0] - e1[0]*e2[2];
        let nz = e1[0]*e2[1] - e1[1]*e2[0];
        let nl = (nx*nx + ny*ny + nz*nz).sqrt();
        let normal = if nl > 1e-10 { [nx/nl, ny/nl, nz/nl] } else { tri.normal };

        result.push(WorldTri { v: [v0, v1, v2], normal, color: tri.color });
    }

    // Deduplicate identical triangles (same 3 cell indices)
    let mut seen: HashMap<(u64,u64,u64), bool> = HashMap::new();
    let mut deduped = Vec::with_capacity(result.len());
    for tri in &result {
        let key = (to_cell(tri.v[0]), to_cell(tri.v[1]), to_cell(tri.v[2]));
        if seen.insert(key, true).is_none() {
            deduped.push(WorldTri { v: tri.v, normal: tri.normal, color: tri.color });
        }
    }

    deduped
}

/// Recursively discover directories containing scene.gltf files.
fn discover_gltf_dirs(root: &str) -> Vec<String> {
    let mut result = Vec::new();
    scan_for_gltf(root, &mut result);
    result.sort();
    result
}

fn scan_for_gltf(dir: &str, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    let mut has_gltf = false;
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path.to_string_lossy().to_string());
        } else if path.file_name().map_or(false, |n| n == "scene.gltf") {
            has_gltf = true;
        }
    }
    if has_gltf {
        out.push(dir.to_string());
    }
    // Recurse into subdirectories (finds nested models)
    for sub in subdirs {
        scan_for_gltf(&sub, out);
    }
}

/// Load a GLTF model and normalize to a specific target height.
/// Returns None if files don't exist or can't be parsed.
pub fn try_load_gltf_scaled(dir: &str, name: &str, color: u32, target_height: f32) -> Option<LoadedModel> {
    let gltf_path = format!("{dir}/scene.gltf");
    let bin_path = format!("{dir}/scene.bin");
    let json_str = fs::read_to_string(&gltf_path).ok()?;
    let bin_data = fs::read(&bin_path).ok()?;

    let accessors = parse_accessors(&json_str);
    let buffer_views = parse_buffer_views(&json_str);
    let primitives = parse_all_primitives(&json_str);

    let mut all_tris: Vec<WorldTri> = Vec::new();
    for prim in &primitives {
        let pos_acc = prim.position_accessor;
        if pos_acc >= accessors.len() { continue; }
        let positions = extract_vec3(&accessors[pos_acc], &buffer_views, &bin_data);
        let normals = if let Some(na) = prim.normal_accessor {
            if na < accessors.len() { extract_vec3(&accessors[na], &buffer_views, &bin_data) }
            else { Vec::new() }
        } else { Vec::new() };
        let indices = if let Some(ia) = prim.index_accessor {
            if ia < accessors.len() { extract_indices_flat(&accessors[ia], &buffer_views, &bin_data) }
            else { Vec::new() }
        } else { (0..positions.len()).collect() };

        for tri_idx in indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri_idx[0], tri_idx[1], tri_idx[2]);
            if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() { continue; }
            let normal = if !normals.is_empty() && i0 < normals.len() { normals[i0] }
            else {
                let e1 = [positions[i1][0]-positions[i0][0], positions[i1][1]-positions[i0][1], positions[i1][2]-positions[i0][2]];
                let e2 = [positions[i2][0]-positions[i0][0], positions[i2][1]-positions[i0][1], positions[i2][2]-positions[i0][2]];
                let (nx,ny,nz) = (e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]);
                let nl = (nx*nx+ny*ny+nz*nz).sqrt();
                if nl > 1e-10 { [nx/nl, ny/nl, nz/nl] } else { [0.0, 1.0, 0.0] }
            };
            all_tris.push(WorldTri { v: [positions[i0], positions[i1], positions[i2]], normal, color });
        }
    }
    if all_tris.is_empty() { return None; }
    normalize_model_to_height(&mut all_tris, target_height);
    // Filter out tris with NaN/Inf vertices (corrupt data from some GLTF files)
    all_tris.retain(|tri| {
        tri.v.iter().all(|v| v.iter().all(|c| c.is_finite()))
            && tri.normal.iter().all(|c| c.is_finite())
    });
    if all_tris.is_empty() { return None; }
    eprintln!("gltf_loader: loaded '{}' ({} tris, {:.1}m)", name, all_tris.len(), target_height);
    Some(LoadedModel { tris: all_tris, name: name.to_string() })
}

/// Load a car GLTF model, normalizing by longest dimension to target_length.
/// Unlike characters, cars keep Y as the up axis (not tallest-as-up).
fn try_load_gltf_car(dir: &str, name: &str, color: u32, target_length: f32) -> Option<LoadedModel> {
    let gltf_path = format!("{dir}/scene.gltf");
    let bin_path = format!("{dir}/scene.bin");
    let json_str = fs::read_to_string(&gltf_path).ok()?;
    let bin_data = fs::read(&bin_path).ok()?;

    let accessors = parse_accessors(&json_str);
    let buffer_views = parse_buffer_views(&json_str);
    let primitives = parse_all_primitives(&json_str);

    let mut all_tris: Vec<WorldTri> = Vec::new();
    for prim in &primitives {
        let pos_acc = prim.position_accessor;
        if pos_acc >= accessors.len() { continue; }
        let positions = extract_vec3(&accessors[pos_acc], &buffer_views, &bin_data);
        let normals = if let Some(na) = prim.normal_accessor {
            if na < accessors.len() { extract_vec3(&accessors[na], &buffer_views, &bin_data) }
            else { Vec::new() }
        } else { Vec::new() };
        let indices = if let Some(ia) = prim.index_accessor {
            if ia < accessors.len() { extract_indices_flat(&accessors[ia], &buffer_views, &bin_data) }
            else { Vec::new() }
        } else { (0..positions.len()).collect() };

        for tri_idx in indices.chunks_exact(3) {
            let (i0, i1, i2) = (tri_idx[0], tri_idx[1], tri_idx[2]);
            if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() { continue; }
            let normal = if !normals.is_empty() && i0 < normals.len() { normals[i0] }
            else {
                let e1 = [positions[i1][0]-positions[i0][0], positions[i1][1]-positions[i0][1], positions[i1][2]-positions[i0][2]];
                let e2 = [positions[i2][0]-positions[i0][0], positions[i2][1]-positions[i0][1], positions[i2][2]-positions[i0][2]];
                let (nx,ny,nz) = (e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]);
                let nl = (nx*nx+ny*ny+nz*nz).sqrt();
                if nl > 1e-10 { [nx/nl, ny/nl, nz/nl] } else { [0.0, 1.0, 0.0] }
            };
            all_tris.push(WorldTri { v: [positions[i0], positions[i1], positions[i2]], normal, color });
        }
    }
    if all_tris.is_empty() { return None; }
    normalize_car_to_length(&mut all_tris, target_length);
    all_tris.retain(|tri| {
        tri.v.iter().all(|v| v.iter().all(|c| c.is_finite()))
            && tri.normal.iter().all(|c| c.is_finite())
    });
    if all_tris.is_empty() { return None; }
    eprintln!("gltf_loader: loaded car '{}' ({} tris, {:.1}m)", name, all_tris.len(), target_length);
    Some(LoadedModel { tris: all_tris, name: name.to_string() })
}

/// Normalize car model: scale so longest horizontal axis = target_length,
/// center X/Z, Y=0 at bottom. Keeps Y as the up axis (cars are wide/long, not tall).
fn normalize_car_to_length(tris: &mut [WorldTri], target_length: f32) {
    if tris.is_empty() { return; }
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for tri in tris.iter() {
        for v in &tri.v {
            for i in 0..3 { min[i] = min[i].min(v[i]); max[i] = max[i].max(v[i]); }
        }
    }
    let dims = [max[0]-min[0], max[1]-min[1], max[2]-min[2]];
    // Longest dimension (could be X, Y, or Z) — scale uniformly so longest = target_length
    let longest = dims[0].max(dims[1]).max(dims[2]);
    if longest < 1e-6 { return; }
    let scale = target_length / longest;
    let cx = (min[0] + max[0]) * 0.5;
    let cy_base = min[1]; // Y=0 at bottom of car
    let cz = (min[2] + max[2]) * 0.5;
    for tri in tris.iter_mut() {
        for v in &mut tri.v {
            v[0] = (v[0] - cx) * scale;
            v[1] = (v[1] - cy_base) * scale;
            v[2] = (v[2] - cz) * scale;
        }
        let e1 = [tri.v[1][0]-tri.v[0][0], tri.v[1][1]-tri.v[0][1], tri.v[1][2]-tri.v[0][2]];
        let e2 = [tri.v[2][0]-tri.v[0][0], tri.v[2][1]-tri.v[0][1], tri.v[2][2]-tri.v[0][2]];
        let nx = e1[1]*e2[2] - e1[2]*e2[1];
        let ny = e1[2]*e2[0] - e1[0]*e2[2];
        let nz = e1[0]*e2[1] - e1[1]*e2[0];
        let nl = (nx*nx + ny*ny + nz*nz).sqrt();
        if nl > 1e-10 { tri.normal = [nx/nl, ny/nl, nz/nl]; }
    }
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

/// Normalize model to game coordinates: Y=0 at base, centered X/Z, target height.
fn normalize_model_to_height(tris: &mut [WorldTri], target_height: f32) {
    if tris.is_empty() { return; }
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for tri in tris.iter() {
        for v in &tri.v {
            for i in 0..3 { min[i] = min[i].min(v[i]); max[i] = max[i].max(v[i]); }
        }
    }
    let dims = [max[0]-min[0], max[1]-min[1], max[2]-min[2]];
    let up = if dims[1] >= dims[0] && dims[1] >= dims[2] { 1 }
             else if dims[2] >= dims[0] && dims[2] >= dims[1] { 2 }
             else { 1 };
    let height = dims[up];
    if height < 1e-6 { return; }
    let scale = target_height / height;
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
        let e1 = [tri.v[1][0]-tri.v[0][0], tri.v[1][1]-tri.v[0][1], tri.v[1][2]-tri.v[0][2]];
        let e2 = [tri.v[2][0]-tri.v[0][0], tri.v[2][1]-tri.v[0][1], tri.v[2][2]-tri.v[0][2]];
        let nx = e1[1]*e2[2] - e1[2]*e2[1];
        let ny = e1[2]*e2[0] - e1[0]*e2[2];
        let nz = e1[0]*e2[1] - e1[1]*e2[0];
        let nl = (nx*nx + ny*ny + nz*nz).sqrt();
        if nl > 1e-10 { tri.normal = [nx/nl, ny/nl, nz/nl]; }
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
