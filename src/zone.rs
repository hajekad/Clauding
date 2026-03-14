// Zone-based world classification
// Analyzes terrain to create a zone map that drives asset placement
// Zones emerge from terrain features — no hardcoded positions

use crate::state::*;
use crate::noise;
use crate::rng::Rng;

pub const ZONE_GRID: usize = 50;
pub const ZONE_CELL_SIZE: f32 = WORLD_SIZE / ZONE_GRID as f32; // 10.0m cells
pub const NUM_ZONE_KINDS: usize = 7;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ZoneKind {
    Wilderness = 0,
    Farmland = 1,
    Village = 2,
    Town = 3,
    Industrial = 4,
    Waterfront = 5,
    Parkland = 6,
}

#[derive(Clone)]
pub struct ZoneCell {
    pub kind: ZoneKind,
    pub density: f32,     // 0.0-1.0 — how developed/populated
    pub elevation: f32,   // average terrain height in cell
    pub flatness: f32,    // 0.0-1.0 (1.0 = perfectly flat)
    pub water_dist: f32,  // distance to nearest river segment
}

pub struct ZoneMap {
    pub cells: Vec<ZoneCell>,
    pub grid_size: usize,
    pub cell_size: f32,
    pub settlement_centers: Vec<[f32; 2]>,
}

impl ZoneMap {
    pub fn generate(terrain: &Terrain, river_segments: &[RiverSegment], seed: u64) -> Self {
        let grid_size = ZONE_GRID;
        let cell_size = ZONE_CELL_SIZE;
        let n = grid_size * grid_size;
        let mut cells = Vec::with_capacity(n);
        let mut rng = Rng::new(seed.wrapping_add(54321));

        // Step 1: Analyze terrain for each cell
        for gz in 0..grid_size {
            for gx in 0..grid_size {
                let wx = (gx as f32 + 0.5) * cell_size - WORLD_HALF;
                let wz = (gz as f32 + 0.5) * cell_size - WORLD_HALF;

                // Sample terrain at center + 4 corners for flatness
                let h_c = terrain.height_at(wx, wz);
                let off = cell_size * 0.4;
                let h_nw = terrain.height_at(wx - off, wz - off);
                let h_ne = terrain.height_at(wx + off, wz - off);
                let h_sw = terrain.height_at(wx - off, wz + off);
                let h_se = terrain.height_at(wx + off, wz + off);
                let avg = (h_c + h_nw + h_ne + h_sw + h_se) * 0.2;
                let variance = ((h_c - avg).powi(2) + (h_nw - avg).powi(2)
                    + (h_ne - avg).powi(2) + (h_sw - avg).powi(2)
                    + (h_se - avg).powi(2)) * 0.2;
                let flatness = 1.0 / (1.0 + variance * 10.0);

                // Water distance
                let mut water_dist = f32::MAX;
                for seg in river_segments {
                    let d = crate::world::point_to_segment_dist(
                        wx, wz, seg.x1, seg.z1, seg.x2, seg.z2,
                    );
                    if d < water_dist { water_dist = d; }
                }

                cells.push(ZoneCell {
                    kind: ZoneKind::Wilderness,
                    density: 0.0,
                    elevation: avg,
                    flatness,
                    water_dist,
                });
            }
        }

        // Step 2: Score cells for settlement potential
        let mut scores: Vec<(f32, usize)> = cells.iter().enumerate().map(|(i, cell)| {
            let gx = i % grid_size;
            let gz = i / grid_size;
            let wx = (gx as f32 + 0.5) * cell_size - WORLD_HALF;
            let wz = (gz as f32 + 0.5) * cell_size - WORLD_HALF;

            // Flat terrain is essential for buildings
            let flat_score = cell.flatness * 3.0;
            // Prefer mid-elevation (not deep valleys or mountain peaks)
            let elev_score = 1.0 - (cell.elevation.abs() / 10.0).min(1.0);
            // Slight preference for water proximity (trade, fishing, transport)
            let water_score = if cell.water_dist < 5.0 { 0.0 } // in river
                else if cell.water_dist < 60.0 { 0.5 * (1.0 - cell.water_dist / 60.0) }
                else { 0.0 };
            // Penalize world edges (settlements shouldn't be at the very edge)
            let edge_dist = (wx.abs() / WORLD_HALF).max(wz.abs() / WORLD_HALF);
            let edge_penalty = if edge_dist > 0.6 { (edge_dist - 0.6) * 6.0 } else { 0.0 };

            (flat_score + elev_score + water_score - edge_penalty, i)
        }).collect();
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Step 3: Pick settlement seeds — emergent from terrain (no artificial cap)
        let mut settlement_centers = Vec::new();
        let min_separation = 120.0;

        for &(score, idx) in &scores {
            if score < 1.5 { break; }

            let gx = idx % grid_size;
            let gz = idx / grid_size;
            let wx = (gx as f32 + 0.5) * cell_size - WORLD_HALF;
            let wz = (gz as f32 + 0.5) * cell_size - WORLD_HALF;

            let too_close = settlement_centers.iter().any(|c: &[f32; 2]| {
                ((wx - c[0]).powi(2) + (wz - c[1]).powi(2)).sqrt() < min_separation
            });
            if too_close { continue; }

            settlement_centers.push([wx, wz]);
        }

        // Fallback: ensure at least one settlement
        if settlement_centers.is_empty() {
            settlement_centers.push([rng.range(-20.0, 20.0), rng.range(-20.0, 20.0)]);
        }

        // Step 4: Assign zones radiating from settlements
        for (si, center) in settlement_centers.iter().enumerate() {
            let is_main = si == 0;
            let core_radius = if is_main { 55.0 } else { 25.0 };
            let town_radius = if is_main { 85.0 } else { 45.0 };
            let farm_radius = town_radius + 50.0;

            for gz in 0..grid_size {
                for gx in 0..grid_size {
                    let idx = gz * grid_size + gx;
                    let wx = (gx as f32 + 0.5) * cell_size - WORLD_HALF;
                    let wz = (gz as f32 + 0.5) * cell_size - WORLD_HALF;
                    let dist = ((wx - center[0]).powi(2) + (wz - center[1]).powi(2)).sqrt();

                    // Noise-distorted radius for organic zone boundaries
                    let noise_val = noise::value_noise_2d(
                        wx * 0.015, wz * 0.015,
                        seed.wrapping_add(5555 + si as u64 * 7777),
                    );
                    let noisy_dist = dist - noise_val * 25.0;

                    if noisy_dist < town_radius {
                        let kind = if is_main { ZoneKind::Town } else { ZoneKind::Village };
                        let density = if noisy_dist < core_radius {
                            0.7 + 0.3 * (1.0 - noisy_dist / core_radius)
                        } else {
                            0.7 * (1.0 - (noisy_dist - core_radius)
                                / (town_radius - core_radius))
                        };
                        let density = density.max(0.05);
                        if density > cells[idx].density {
                            cells[idx].kind = kind;
                            cells[idx].density = density;
                        }
                    } else if noisy_dist < farm_radius
                        && cells[idx].kind == ZoneKind::Wilderness
                    {
                        let density = 0.3
                            * (1.0
                                - (noisy_dist - town_radius) / (farm_radius - town_radius));
                        if density > cells[idx].density {
                            cells[idx].kind = ZoneKind::Farmland;
                            cells[idx].density = density.max(0.05);
                        }
                    }
                }
            }
        }

        // Step 5: Classify special zones based on terrain features
        let main_center = settlement_centers[0];
        for i in 0..n {
            let gx = i % grid_size;
            let gz = i / grid_size;
            let wx = (gx as f32 + 0.5) * cell_size - WORLD_HALF;
            let wz = (gz as f32 + 0.5) * cell_size - WORLD_HALF;

            // Near water + near settlement → Waterfront/Industrial
            if cells[i].water_dist < 25.0 && cells[i].water_dist > 3.0 {
                let dist_to_main =
                    ((wx - main_center[0]).powi(2) + (wz - main_center[1]).powi(2)).sqrt();
                if dist_to_main < 140.0 {
                    let kind = if dist_to_main > 80.0 && cells[i].density < 0.3 {
                        ZoneKind::Industrial
                    } else {
                        ZoneKind::Waterfront
                    };
                    let wd = 0.4 * (1.0 - cells[i].water_dist / 25.0);
                    if wd > cells[i].density || cells[i].kind == ZoneKind::Wilderness {
                        cells[i].kind = kind;
                        cells[i].density = wd.max(cells[i].density).max(0.1);
                    }
                }
            }

            // Parkland: noise-driven patches near town edges on flat terrain
            if cells[i].kind == ZoneKind::Wilderness || cells[i].kind == ZoneKind::Farmland {
                let dist_to_main =
                    ((wx - main_center[0]).powi(2) + (wz - main_center[1]).powi(2)).sqrt();
                let park_noise = noise::value_noise_2d(wx * 0.03, wz * 0.03, seed + 8888);
                if dist_to_main > 70.0
                    && dist_to_main < 130.0
                    && park_noise > 0.5
                    && cells[i].flatness > 0.5
                {
                    cells[i].kind = ZoneKind::Parkland;
                    cells[i].density = 0.3 + 0.2 * (park_noise - 0.5) * 2.0;
                }
            }
        }

        // Wilderness cells get density based on terrain interest
        for cell in &mut cells {
            if cell.kind == ZoneKind::Wilderness && cell.density < 0.01 {
                cell.density = 0.3 + (1.0 - cell.flatness) * 0.4;
            }
        }

        ZoneMap { cells, grid_size, cell_size, settlement_centers }
    }

    /// Empty zone map (used before world generation runs)
    pub fn empty() -> Self {
        ZoneMap {
            cells: Vec::new(),
            grid_size: 0,
            cell_size: ZONE_CELL_SIZE,
            settlement_centers: Vec::new(),
        }
    }

    /// Look up zone cell at world coordinates
    pub fn zone_at(&self, x: f32, z: f32) -> &ZoneCell {
        let gx = ((x + WORLD_HALF) / self.cell_size) as usize;
        let gz = ((z + WORLD_HALF) / self.cell_size) as usize;
        let gx = gx.min(self.grid_size - 1);
        let gz = gz.min(self.grid_size - 1);
        &self.cells[gz * self.grid_size + gx]
    }

    /// Find the centroid of all cells matching a zone kind
    pub fn find_zone_center(&self, kind: ZoneKind) -> Option<[f32; 2]> {
        let mut sum_x = 0.0f32;
        let mut sum_z = 0.0f32;
        let mut count = 0u32;
        for gz in 0..self.grid_size {
            for gx in 0..self.grid_size {
                if self.cells[gz * self.grid_size + gx].kind == kind {
                    sum_x += (gx as f32 + 0.5) * self.cell_size - WORLD_HALF;
                    sum_z += (gz as f32 + 0.5) * self.cell_size - WORLD_HALF;
                    count += 1;
                }
            }
        }
        if count > 0 {
            Some([sum_x / count as f32, sum_z / count as f32])
        } else {
            None
        }
    }

    /// Main settlement center (first seed)
    pub fn main_settlement(&self) -> [f32; 2] {
        self.settlement_centers[0]
    }
}
