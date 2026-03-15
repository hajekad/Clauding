// Navigation mesh: walkability grid, A* pathfinding, reachability analysis
//
// Replaces all NPC stuck/teleport band-aids with proper pathfinding.
// Grid-based approach: 1m resolution over 500m world = 500x500 cells.

use crate::state::*;

/// Walk grid resolution in cells per meter (0.5 = 2m cells for 3km world → 1500×1500)
const CELLS_PER_METER: f32 = 0.5;
/// Margin around obstacles (meters) — prevents NPCs from rubbing against walls
const OBSTACLE_MARGIN: f32 = 0.5;
/// Max A* iterations before giving up (prevents infinite loops on huge searches)
const MAX_ASTAR_ITERS: usize = 100_000;

pub struct WalkGrid {
    /// Walkable bitmap: true = can walk, false = blocked
    pub cells: Vec<bool>,
    pub grid_w: usize,
    pub grid_h: usize,
    pub cell_size: f32,
    pub origin_x: f32, // world X of cell (0,0)
    pub origin_z: f32, // world Z of cell (0,0)
    /// Connected component IDs per cell (0 = blocked, 1+ = component ID)
    pub components: Vec<u16>,
    /// Which component contains the most walkable cells (main landmass)
    pub main_component: u16,
}

impl WalkGrid {
    pub fn empty() -> Self {
        WalkGrid {
            cells: Vec::new(), grid_w: 0, grid_h: 0,
            cell_size: 1.0, origin_x: 0.0, origin_z: 0.0,
            components: Vec::new(), main_component: 0,
        }
    }

    /// Convert world coordinates to grid cell indices
    #[inline]
    pub fn world_to_cell(&self, wx: f32, wz: f32) -> (i32, i32) {
        let cx = ((wx - self.origin_x) / self.cell_size) as i32;
        let cz = ((wz - self.origin_z) / self.cell_size) as i32;
        (cx, cz)
    }

    /// Convert grid cell to world coordinates (center of cell)
    #[inline]
    pub fn cell_to_world(&self, cx: i32, cz: i32) -> (f32, f32) {
        let wx = self.origin_x + (cx as f32 + 0.5) * self.cell_size;
        let wz = self.origin_z + (cz as f32 + 0.5) * self.cell_size;
        (wx, wz)
    }

    #[inline]
    fn in_bounds(&self, cx: i32, cz: i32) -> bool {
        cx >= 0 && cz >= 0 && (cx as usize) < self.grid_w && (cz as usize) < self.grid_h
    }

    #[inline]
    fn idx(&self, cx: i32, cz: i32) -> usize {
        cz as usize * self.grid_w + cx as usize
    }

    /// Check if a world position is walkable
    pub fn is_walkable(&self, wx: f32, wz: f32) -> bool {
        let (cx, cz) = self.world_to_cell(wx, wz);
        if !self.in_bounds(cx, cz) { return false; }
        self.cells[self.idx(cx, cz)]
    }

    /// Check if two positions are in the same connected component (reachable from each other)
    pub fn is_reachable(&self, from_x: f32, from_z: f32, to_x: f32, to_z: f32) -> bool {
        let (fx, fz) = self.world_to_cell(from_x, from_z);
        let (tx, tz) = self.world_to_cell(to_x, to_z);
        if !self.in_bounds(fx, fz) || !self.in_bounds(tx, tz) { return false; }
        let ca = self.components[self.idx(fx, fz)];
        let cb = self.components[self.idx(tx, tz)];
        ca != 0 && ca == cb
    }

    /// A* pathfinding from (from_x, from_z) to (to_x, to_z).
    /// Returns a smoothed path of world-space waypoints, or empty vec if unreachable.
    pub fn find_path(&self, from_x: f32, from_z: f32, to_x: f32, to_z: f32) -> Vec<[f32; 2]> {
        let (sx, sz) = self.world_to_cell(from_x, from_z);
        let (gx, gz) = self.world_to_cell(to_x, to_z);

        // Snap start/goal to nearest walkable cell if they're in blocked cells
        let (sx, sz) = self.nearest_walkable(sx, sz);
        let (gx, gz) = self.nearest_walkable(gx, gz);

        if !self.in_bounds(sx, sz) || !self.in_bounds(gx, gz) { return Vec::new(); }
        let si = self.idx(sx, sz);
        let gi = self.idx(gx, gz);
        if !self.cells[si] || !self.cells[gi] { return Vec::new(); }

        // Quick reachability check
        if self.components[si] != self.components[gi] { return Vec::new(); }

        // Trivial case: already at goal
        if sx == gx && sz == gz {
            let (wx, wz) = self.cell_to_world(gx, gz);
            return vec![[wx, wz]];
        }

        // A* with binary heap
        let n = self.grid_w * self.grid_h;
        let mut g_score = vec![f32::MAX; n];
        let mut came_from = vec![u32::MAX; n];
        g_score[si] = 0.0;

        // Min-heap: (f_score_fixed, index)
        // Use fixed-point f_score (multiply by 1024) to avoid f32 in BinaryHeap
        let mut open = BinaryMinHeap::new();
        let h = heuristic(sx, sz, gx, gz);
        open.push(h, si as u32);

        let mut iters = 0u32;

        while let Some((_f, ci)) = open.pop() {
            if ci as usize == gi {
                // Reconstruct path as cell indices, convert to world coords
                let idx_path = reconstruct(came_from.as_slice(), si, gi, self.grid_w);
                let world_path: Vec<[f32; 2]> = idx_path.iter().map(|&i| {
                    let cx = (i % self.grid_w) as i32;
                    let cz = (i / self.grid_w) as i32;
                    let (wx, wz) = self.cell_to_world(cx, cz);
                    [wx, wz]
                }).collect();
                return smooth_path(&world_path, self);
            }

            if iters >= MAX_ASTAR_ITERS as u32 { break; }
            iters += 1;

            let cx = (ci as usize % self.grid_w) as i32;
            let cz = (ci as usize / self.grid_w) as i32;
            let cur_g = g_score[ci as usize];

            // 8-directional neighbors
            for &(dx, dz, cost) in &NEIGHBORS {
                let nx = cx + dx;
                let nz = cz + dz;
                if !self.in_bounds(nx, nz) { continue; }
                let ni = self.idx(nx, nz);
                if !self.cells[ni] { continue; }

                // Diagonal movement: check both adjacent cells to prevent corner-cutting
                if dx != 0 && dz != 0 {
                    if !self.cells[self.idx(cx + dx, cz)] || !self.cells[self.idx(cx, cz + dz)] {
                        continue;
                    }
                }

                let new_g = cur_g + cost;
                if new_g < g_score[ni] {
                    g_score[ni] = new_g;
                    came_from[ni] = ci;
                    let f = new_g + heuristic(nx, nz, gx, gz);
                    open.push(f, ni as u32);
                }
            }
        }

        Vec::new() // unreachable
    }

    /// Find nearest walkable cell to (cx, cz) via spiral search
    fn nearest_walkable(&self, cx: i32, cz: i32) -> (i32, i32) {
        if self.in_bounds(cx, cz) && self.cells[self.idx(cx, cz)] {
            return (cx, cz);
        }
        for r in 1..20 {
            for dx in -r..=r {
                for &dz in &[-r, r] {
                    let nx = cx + dx;
                    let nz = cz + dz;
                    if self.in_bounds(nx, nz) && self.cells[self.idx(nx, nz)] {
                        return (nx, nz);
                    }
                }
            }
            for dz in (-r + 1)..r {
                for &dx in &[-r, r] {
                    let nx = cx + dx;
                    let nz = cz + dz;
                    if self.in_bounds(nx, nz) && self.cells[self.idx(nx, nz)] {
                        return (nx, nz);
                    }
                }
            }
        }
        (cx, cz) // fallback
    }

}

// 8-directional neighbors: (dx, dz, cost)
const DIAG: f32 = 1.414;
const NEIGHBORS: [(i32, i32, f32); 8] = [
    ( 0, -1, 1.0), ( 0,  1, 1.0), (-1,  0, 1.0), ( 1,  0, 1.0),
    (-1, -1, DIAG), ( 1, -1, DIAG), (-1,  1, DIAG), ( 1,  1, DIAG),
];

#[inline]
fn heuristic(ax: i32, az: i32, bx: i32, bz: i32) -> f32 {
    // Octile distance (admissible for 8-directional grid)
    let dx = (ax - bx).unsigned_abs() as f32;
    let dz = (az - bz).unsigned_abs() as f32;
    let (lo, hi) = if dx < dz { (dx, dz) } else { (dz, dx) };
    hi + lo * (DIAG - 1.0)
}

fn reconstruct(came_from: &[u32], start: usize, goal: usize, _grid_w: usize) -> Vec<usize> {
    let mut path = Vec::new();
    let mut cur = goal;
    let max_len = came_from.len();
    let mut count = 0;
    while cur != start && count < max_len {
        path.push(cur);
        if came_from[cur] == u32::MAX { break; }
        cur = came_from[cur] as usize;
        count += 1;
    }
    // Don't include start (NPC is already there)
    path.reverse();
    path
}

/// Build the walkability grid from world data
pub fn build_walk_grid(
    buildings: &[Building],
    rocks: &[Rock],
    trees: &[Tree],
    walls: &[Wall],
    clutter: &[[f32; 3]],
    river_segments: &[RiverSegment],
    bridges: &[Bridge],
    terrain: &Terrain,
) -> WalkGrid {
    let cell_size = 1.0 / CELLS_PER_METER;
    let grid_w = (WORLD_SIZE / cell_size) as usize;
    let grid_h = grid_w;
    let origin_x = -WORLD_HALF;
    let origin_z = -WORLD_HALF;
    let n = grid_w * grid_h;
    let mut cells = vec![true; n];

    // Mark cells blocked by buildings (AABB + margin)
    for b in buildings {
        let hw = b.w * 0.5 + OBSTACLE_MARGIN;
        let hd = b.d * 0.5 + OBSTACLE_MARGIN;
        let min_x = b.x - hw;
        let max_x = b.x + hw;
        let min_z = b.z - hd;
        let max_z = b.z + hd;
        fill_rect_blocked(&mut cells, grid_w, grid_h, cell_size, origin_x, origin_z,
            min_x, max_x, min_z, max_z);
    }

    // Walls (AABB + margin)
    for w in walls {
        let min_x = w.x - w.hw - OBSTACLE_MARGIN;
        let max_x = w.x + w.hw + OBSTACLE_MARGIN;
        let min_z = w.z - w.hd - OBSTACLE_MARGIN;
        let max_z = w.z + w.hd + OBSTACLE_MARGIN;
        fill_rect_blocked(&mut cells, grid_w, grid_h, cell_size, origin_x, origin_z,
            min_x, max_x, min_z, max_z);
    }

    // Rocks (circle + margin)
    for r in rocks {
        let radius = r.size + OBSTACLE_MARGIN;
        fill_circle_blocked(&mut cells, grid_w, grid_h, cell_size, origin_x, origin_z,
            r.x, r.z, radius);
    }

    // Trees (trunk only — small circle)
    for t in trees {
        let radius = t.trunk_radius + OBSTACLE_MARGIN;
        fill_circle_blocked(&mut cells, grid_w, grid_h, cell_size, origin_x, origin_z,
            t.x, t.z, radius);
    }

    // Clutter (barrels, crates, sacks, handcarts)
    for c in clutter {
        let radius = c[2] + OBSTACLE_MARGIN;
        fill_circle_blocked(&mut cells, grid_w, grid_h, cell_size, origin_x, origin_z,
            c[0], c[1], radius);
    }

    // River segments (blocked, except bridges)
    for seg in river_segments {
        let dx = seg.x2 - seg.x1;
        let dz = seg.z2 - seg.z1;
        let len = (dx * dx + dz * dz).sqrt();
        if len < 0.01 { continue; }
        let hw = seg.width * 0.5 + OBSTACLE_MARGIN;

        // Walk along segment, marking cells within half-width as blocked
        let steps = (len / cell_size).ceil() as usize + 1;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let cx = seg.x1 + dx * t;
            let cz = seg.z1 + dz * t;
            fill_circle_blocked(&mut cells, grid_w, grid_h, cell_size, origin_x, origin_z,
                cx, cz, hw);
        }
    }

    // Unblock bridge areas (bridges let you cross rivers)
    for br in bridges {
        // Bridge is an oriented rectangle: center (cx,cz), direction (dir_x, dir_z), half-width hw, half-length hl
        let perp_x = -br.dir_z;
        let perp_z = br.dir_x;
        // Sample points inside the bridge rectangle
        let steps_l = (br.hl * 2.0 / cell_size).ceil() as usize + 1;
        let steps_w = (br.hw * 2.0 / cell_size).ceil() as usize + 1;
        for sl in 0..=steps_l {
            let tl = -1.0 + 2.0 * sl as f32 / steps_l as f32;
            for sw in 0..=steps_w {
                let tw = -1.0 + 2.0 * sw as f32 / steps_w as f32;
                let px = br.cx + br.dir_x * br.hl * tl + perp_x * br.hw * tw;
                let pz = br.cz + br.dir_z * br.hl * tl + perp_z * br.hw * tw;
                let (gx, gz) = (
                    ((px - origin_x) / cell_size) as i32,
                    ((pz - origin_z) / cell_size) as i32,
                );
                if gx >= 0 && gz >= 0 && (gx as usize) < grid_w && (gz as usize) < grid_h {
                    cells[gz as usize * grid_w + gx as usize] = true;
                }
            }
        }
    }

    // Mark steep terrain as blocked (slope > 45°)
    for iz in 0..grid_h {
        for ix in 0..grid_w {
            if !cells[iz * grid_w + ix] { continue; }
            let (wx, wz) = (
                origin_x + (ix as f32 + 0.5) * cell_size,
                origin_z + (iz as f32 + 0.5) * cell_size,
            );
            let normal = terrain.normal_at(wx, wz);
            let slope = 1.0 - normal[1]; // 0=flat, 0.29=45°, 0.5=60°
            if slope > 0.29 {
                cells[iz * grid_w + ix] = false;
            }
        }
    }

    // Mark world boundary cells as blocked (5m margin)
    let margin_cells = (5.0 / cell_size) as usize;
    for iz in 0..grid_h {
        for ix in 0..grid_w {
            if ix < margin_cells || ix >= grid_w - margin_cells
                || iz < margin_cells || iz >= grid_h - margin_cells
            {
                cells[iz * grid_w + ix] = false;
            }
        }
    }

    // Compute connected components (flood fill)
    let mut components = vec![0u16; n];
    let mut component_id = 0u16;
    let mut component_sizes: Vec<usize> = Vec::new();
    for start in 0..n {
        if !cells[start] || components[start] != 0 { continue; }
        component_id += 1;
        let size = flood_fill(&cells, &mut components, grid_w, grid_h, start, component_id);
        component_sizes.push(size);
    }

    // Find main component (largest)
    let main_component = if component_sizes.is_empty() {
        0
    } else {
        let mut best = 0;
        let mut best_size = 0;
        for (i, &s) in component_sizes.iter().enumerate() {
            if s > best_size {
                best_size = s;
                best = i;
            }
        }
        (best + 1) as u16 // component IDs start at 1
    };

    WalkGrid {
        cells,
        grid_w,
        grid_h,
        cell_size,
        origin_x,
        origin_z,
        components,
        main_component,
    }
}

fn fill_rect_blocked(
    cells: &mut [bool], grid_w: usize, grid_h: usize,
    cell_size: f32, origin_x: f32, origin_z: f32,
    min_x: f32, max_x: f32, min_z: f32, max_z: f32,
) {
    let cx0 = ((min_x - origin_x) / cell_size).floor() as i32;
    let cx1 = ((max_x - origin_x) / cell_size).ceil() as i32;
    let cz0 = ((min_z - origin_z) / cell_size).floor() as i32;
    let cz1 = ((max_z - origin_z) / cell_size).ceil() as i32;
    for cz in cz0..=cz1 {
        if cz < 0 || cz as usize >= grid_h { continue; }
        for cx in cx0..=cx1 {
            if cx < 0 || cx as usize >= grid_w { continue; }
            cells[cz as usize * grid_w + cx as usize] = false;
        }
    }
}

fn fill_circle_blocked(
    cells: &mut [bool], grid_w: usize, grid_h: usize,
    cell_size: f32, origin_x: f32, origin_z: f32,
    wx: f32, wz: f32, radius: f32,
) {
    let cx0 = ((wx - radius - origin_x) / cell_size).floor() as i32;
    let cx1 = ((wx + radius - origin_x) / cell_size).ceil() as i32;
    let cz0 = ((wz - radius - origin_z) / cell_size).floor() as i32;
    let cz1 = ((wz + radius - origin_z) / cell_size).ceil() as i32;
    let r2 = radius * radius;
    for cz in cz0..=cz1 {
        if cz < 0 || cz as usize >= grid_h { continue; }
        for cx in cx0..=cx1 {
            if cx < 0 || cx as usize >= grid_w { continue; }
            let px = origin_x + (cx as f32 + 0.5) * cell_size;
            let pz = origin_z + (cz as f32 + 0.5) * cell_size;
            let dx = px - wx;
            let dz = pz - wz;
            if dx * dx + dz * dz <= r2 {
                cells[cz as usize * grid_w + cx as usize] = false;
            }
        }
    }
}

fn flood_fill(
    cells: &[bool], components: &mut [u16],
    grid_w: usize, grid_h: usize, start: usize, id: u16,
) -> usize {
    let mut stack = vec![start];
    let mut count = 0;
    while let Some(idx) = stack.pop() {
        if components[idx] != 0 { continue; }
        if !cells[idx] { continue; }
        components[idx] = id;
        count += 1;

        let x = (idx % grid_w) as i32;
        let z = (idx / grid_w) as i32;
        for &(dx, dz) in &[(0, -1), (0, 1), (-1, 0), (1, 0)] {
            let nx = x + dx;
            let nz = z + dz;
            if nx >= 0 && nz >= 0 && (nx as usize) < grid_w && (nz as usize) < grid_h {
                let ni = nz as usize * grid_w + nx as usize;
                if cells[ni] && components[ni] == 0 {
                    stack.push(ni);
                }
            }
        }
    }
    count
}

/// Simple binary min-heap for A* open set
struct BinaryMinHeap {
    data: Vec<(u32, u32)>, // (f_score_fixed, node_index) — f_score as fixed-point (×1024)
}

impl BinaryMinHeap {
    fn new() -> Self {
        BinaryMinHeap { data: Vec::with_capacity(1024) }
    }

    fn push(&mut self, f_score: f32, node: u32) {
        let f_fixed = (f_score * 1024.0) as u32;
        self.data.push((f_fixed, node));
        self.sift_up(self.data.len() - 1);
    }

    fn pop(&mut self) -> Option<(u32, u32)> {
        if self.data.is_empty() { return None; }
        let top = self.data[0];
        let last = self.data.len() - 1;
        self.data.swap(0, last);
        self.data.pop();
        if !self.data.is_empty() {
            self.sift_down(0);
        }
        Some(top)
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.data[idx].0 < self.data[parent].0 {
                self.data.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.data.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut smallest = idx;
            if left < len && self.data[left].0 < self.data[smallest].0 {
                smallest = left;
            }
            if right < len && self.data[right].0 < self.data[smallest].0 {
                smallest = right;
            }
            if smallest != idx {
                self.data.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }
}

/// Smooth a grid path by removing unnecessary waypoints using line-of-sight checks
fn smooth_path(path: &[[f32; 2]], grid: &WalkGrid) -> Vec<[f32; 2]> {
    if path.len() <= 2 { return path.to_vec(); }

    let mut result = vec![path[0]];
    let mut anchor = 0;

    while anchor < path.len() - 1 {
        let mut furthest = anchor + 1;
        // Find the furthest point we can see from anchor
        for i in (anchor + 2)..path.len() {
            if line_walkable(grid, path[anchor][0], path[anchor][1], path[i][0], path[i][1]) {
                furthest = i;
            }
        }
        result.push(path[furthest]);
        anchor = furthest;
    }

    result
}

/// Check if a straight line between two points crosses only walkable cells
fn line_walkable(grid: &WalkGrid, x0: f32, z0: f32, x1: f32, z1: f32) -> bool {
    let dx = x1 - x0;
    let dz = z1 - z0;
    let dist = (dx * dx + dz * dz).sqrt();
    let steps = (dist / (grid.cell_size * 0.5)).ceil() as usize;
    if steps == 0 { return true; }
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let px = x0 + dx * t;
        let pz = z0 + dz * t;
        if !grid.is_walkable(px, pz) { return false; }
    }
    true
}
