// Data-driven asset placement engine
// Distributes assets across the world based on zone weights, spacing rules,
// and terrain constraints. Adding a new asset type = adding a PlacementRule const.

use crate::state::*;
use crate::zone::{ZoneMap, NUM_ZONE_KINDS};
use crate::rng::Rng;
use crate::world;

#[derive(Clone, Copy)]
pub enum RoadRelation {
    Roadside, // Must be alongside a road (street lights, trash bins)
    NearRoad, // Within a distance range of road (buildings)
    OffRoad,  // Must not be on/near roads (trees, rocks)
    Any,      // No preference (grass)
}

#[derive(Clone, Copy)]
pub struct PlacementRule {
    pub zone_weights: [f32; NUM_ZONE_KINDS],
    pub density: f32,         // instances per zone cell at density=1.0 and weight=1.0
    pub min_spacing: f32,     // minimum distance between instances of this type
    pub road_relation: RoadRelation,
    pub road_offset_min: f32, // min distance from road center
    pub road_offset_max: f32, // max distance from road center
    pub slope_max: f32,       // minimum terrain normal_y (higher = require flatter)
    pub cluster_min: u32,     // min items per cluster
    pub cluster_max: u32,     // max items per cluster
    pub cluster_radius: f32,  // scatter radius within a cluster
    pub avoid_water: bool,
    pub avoid_buildings: bool,
    pub size_min: f32,        // min scale factor
    pub size_max: f32,        // max scale factor
}

pub struct PlacedAsset {
    pub x: f32,
    pub z: f32,
    pub scale: f32,
    pub variant: u32,
}

/// Spatial grid for efficient minimum-spacing enforcement
struct SpacingGrid {
    cells: Vec<Vec<[f32; 2]>>,
    grid_size: usize,
    cell_size: f32,
}

impl SpacingGrid {
    fn new(min_spacing: f32) -> Self {
        let cell_size = min_spacing.max(2.0);
        let grid_size = (WORLD_SIZE / cell_size).ceil() as usize + 1;
        SpacingGrid {
            cells: vec![Vec::new(); grid_size * grid_size],
            grid_size,
            cell_size,
        }
    }

    fn insert(&mut self, x: f32, z: f32) {
        let gx = ((x + WORLD_HALF) / self.cell_size) as usize;
        let gz = ((z + WORLD_HALF) / self.cell_size) as usize;
        if gx >= self.grid_size || gz >= self.grid_size { return; }
        self.cells[gz * self.grid_size + gx].push([x, z]);
    }

    fn too_close(&self, x: f32, z: f32, min_dist: f32) -> bool {
        let gx = ((x + WORLD_HALF) / self.cell_size) as usize;
        let gz = ((z + WORLD_HALF) / self.cell_size) as usize;
        if gx >= self.grid_size || gz >= self.grid_size { return true; }
        let min_sq = min_dist * min_dist;
        let search = (min_dist / self.cell_size).ceil() as i32 + 1;
        for dz in -search..=search {
            for dx in -search..=search {
                let nx = gx as i32 + dx;
                let nz = gz as i32 + dz;
                if nx < 0 || nz < 0
                    || nx >= self.grid_size as i32
                    || nz >= self.grid_size as i32
                {
                    continue;
                }
                for p in &self.cells[nz as usize * self.grid_size + nx as usize] {
                    let ddx = x - p[0];
                    let ddz = z - p[1];
                    if ddx * ddx + ddz * ddz < min_sq {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Place assets across the world according to a placement rule.
/// Returns positions with scale and variant for mesh generation.
pub fn place_assets(
    rule: &PlacementRule,
    zone_map: &ZoneMap,
    terrain: &Terrain,
    roads: &RoadNetwork,
    river_segments: &[RiverSegment],
    buildings: &[Building],
    rng: &mut Rng,
) -> Vec<PlacedAsset> {
    let mut result = Vec::new();
    let mut grid = SpacingGrid::new(rule.min_spacing);
    let max_attempts = 8;

    for gz in 0..zone_map.grid_size {
        for gx in 0..zone_map.grid_size {
            let cell = &zone_map.cells[gz * zone_map.grid_size + gx];
            let weight = rule.zone_weights[cell.kind as usize];
            if weight < 0.001 { continue; }

            // Expected count for this cell — density is per-cell (tuned for 10m ref cells)
            // Don't scale with cell area: density means "trees per cell"
            let expected = rule.density * weight * cell.density;
            // Probabilistic rounding
            let frac = expected - expected.floor();
            let count = expected as u32
                + if rng.range(0.0, 1.0) < frac { 1 } else { 0 };
            if count == 0 { continue; }

            let cx = (gx as f32 + 0.5) * zone_map.cell_size - WORLD_HALF;
            let cz = (gz as f32 + 0.5) * zone_map.cell_size - WORLD_HALF;

            for _ in 0..count {
                // Cluster size
                let cluster_n = if rule.cluster_min == rule.cluster_max {
                    rule.cluster_min
                } else {
                    rule.cluster_min
                        + rng.next() as u32 % (rule.cluster_max - rule.cluster_min + 1)
                };

                for _ in 0..cluster_n {
                    let mut placed = false;
                    for _ in 0..max_attempts {
                        let scatter = zone_map.cell_size * 0.5 + rule.cluster_radius;
                        let x = cx + rng.range(-scatter, scatter);
                        let z = cz + rng.range(-scatter, scatter);

                        // World bounds
                        if x.abs() > WORLD_HALF - 3.0 || z.abs() > WORLD_HALF - 3.0 {
                            continue;
                        }

                        // Slope check
                        let normal = terrain.normal_at(x, z);
                        if normal[1] < rule.slope_max { continue; }

                        // Water avoidance
                        if rule.avoid_water
                            && world::on_river(x, z, river_segments)
                        {
                            continue;
                        }

                        // Building avoidance
                        if rule.avoid_buildings
                            && world::overlaps_building(x, z, 2.0, buildings)
                        {
                            continue;
                        }

                        // Road relation
                        match rule.road_relation {
                            RoadRelation::Roadside | RoadRelation::NearRoad => {
                                let (dist, _) = world::road_dist_network(x, z, roads);
                                if dist < rule.road_offset_min
                                    || dist > rule.road_offset_max
                                {
                                    continue;
                                }
                            }
                            RoadRelation::OffRoad => {
                                if world::on_any_road(x, z, roads) { continue; }
                            }
                            RoadRelation::Any => {}
                        }

                        // Spacing check
                        if grid.too_close(x, z, rule.min_spacing) { continue; }

                        let scale = rng.range(rule.size_min, rule.size_max);
                        let variant = rng.next() as u32;

                        grid.insert(x, z);
                        result.push(PlacedAsset { x, z, scale, variant });
                        placed = true;
                        break;
                    }
                    if !placed { break; } // stop cluster if can't find valid spot
                }
            }
        }
    }

    result
}

// ==================== Placement Rule Tables ====================
// Zone weight order: [Wilderness, Farmland, Village, Town, Industrial, Waterfront, Parkland]

pub const TREE_RULE: PlacementRule = PlacementRule {
    zone_weights: [1.0, 0.15, 0.25, 0.03, 0.0, 0.08, 0.5],
    density: 0.04,
    min_spacing: 15.0,
    road_relation: RoadRelation::OffRoad,
    road_offset_min: 0.0,
    road_offset_max: 0.0,
    slope_max: 0.5,
    cluster_min: 1,
    cluster_max: 4,
    cluster_radius: 6.0,
    avoid_water: true,
    avoid_buildings: true,
    size_min: 0.7,
    size_max: 1.3,
};

pub const ROCK_RULE: PlacementRule = PlacementRule {
    zone_weights: [0.7, 0.05, 0.03, 0.0, 0.1, 0.15, 0.05],
    density: 0.03,
    min_spacing: 12.0,
    road_relation: RoadRelation::OffRoad,
    road_offset_min: 0.0,
    road_offset_max: 0.0,
    slope_max: 0.3,
    cluster_min: 1,
    cluster_max: 3,
    cluster_radius: 4.0,
    avoid_water: true,
    avoid_buildings: true,
    size_min: 0.5,
    size_max: 1.5,
};

pub const BUILDING_RULE: PlacementRule = PlacementRule {
    zone_weights: [0.0, 0.02, 0.5, 1.0, 0.0, 0.15, 0.0],
    density: 0.8,
    min_spacing: 5.0,
    road_relation: RoadRelation::NearRoad,
    road_offset_min: 7.0,
    road_offset_max: 25.0,
    slope_max: 0.7,
    cluster_min: 1,
    cluster_max: 1,
    cluster_radius: 2.0,
    avoid_water: true,
    avoid_buildings: true,
    size_min: 1.0,
    size_max: 1.0,
};

pub const STREET_LIGHT_RULE: PlacementRule = PlacementRule {
    zone_weights: [0.0, 0.0, 0.3, 1.0, 0.4, 0.2, 0.3],
    density: 1.2,
    min_spacing: 15.0,
    road_relation: RoadRelation::Roadside,
    road_offset_min: 6.5,
    road_offset_max: 10.0,
    slope_max: 0.7,
    cluster_min: 1,
    cluster_max: 1,
    cluster_radius: 0.0,
    avoid_water: true,
    avoid_buildings: true,
    size_min: 1.0,
    size_max: 1.0,
};

pub const TRASH_BIN_RULE: PlacementRule = PlacementRule {
    zone_weights: [0.0, 0.0, 0.2, 0.8, 0.3, 0.1, 0.4],
    density: 0.25,
    min_spacing: 10.0,
    road_relation: RoadRelation::Roadside,
    road_offset_min: 6.0,
    road_offset_max: 10.0,
    slope_max: 0.7,
    cluster_min: 1,
    cluster_max: 1,
    cluster_radius: 0.0,
    avoid_water: true,
    avoid_buildings: false,
    size_min: 1.0,
    size_max: 1.0,
};

pub const BUSH_RULE: PlacementRule = PlacementRule {
    zone_weights: [0.4, 0.25, 0.3, 0.08, 0.0, 0.1, 0.6],
    density: 0.2,
    min_spacing: 2.0,
    road_relation: RoadRelation::OffRoad,
    road_offset_min: 0.0,
    road_offset_max: 0.0,
    slope_max: 0.5,
    cluster_min: 1,
    cluster_max: 3,
    cluster_radius: 3.0,
    avoid_water: true,
    avoid_buildings: true,
    size_min: 0.6,
    size_max: 1.2,
};

pub const GRASS_RULE: PlacementRule = PlacementRule {
    zone_weights: [0.6, 0.5, 0.2, 0.03, 0.0, 0.05, 0.4],
    density: 0.3,
    min_spacing: 2.0,
    road_relation: RoadRelation::OffRoad,
    road_offset_min: 0.0,
    road_offset_max: 0.0,
    slope_max: 0.5,
    cluster_min: 1,
    cluster_max: 2,
    cluster_radius: 3.0,
    avoid_water: true,
    avoid_buildings: true,
    size_min: 0.8,
    size_max: 1.5,
};
