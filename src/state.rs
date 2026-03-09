// GameState: global mutable game state, entity stores, constants

use crate::rng::Rng;
use crate::input::KeyBinds;
use crate::menu::MenuData;

pub const DEFAULT_WIDTH: usize = 1920;
pub const DEFAULT_HEIGHT: usize = 1080;
pub const MAX_FPS: f32 = 500.0;
pub const FRAME_TIME_MIN: f32 = 1.0 / MAX_FPS;

pub const WORLD_SIZE: f32 = 500.0;
pub const WORLD_HALF: f32 = WORLD_SIZE * 0.5;
pub const NUM_BUILDINGS: usize = 125;
pub const NUM_TREES: usize = 200;
pub const NUM_ROCKS: usize = 50;
pub const NUM_STREET_LIGHTS: usize = 60;
pub const CAR_ROAD_WIDTH: f32 = 6.0;
pub const SIDEWALK_WIDTH: f32 = 1.5;
pub const FIELD_ROAD_WIDTH: f32 = 2.5;
pub const NPC_SPEED_SIDEWALK: f32 = 3.5;
pub const NPC_SPEED_FIELD_ROAD: f32 = 2.8;
pub const NPC_SPEED_CAR_ROAD: f32 = 2.0;
pub const NPC_SPEED_TERRAIN: f32 = 1.8;
pub const NPC_SPEED_STEEP: f32 = 1.0;
pub const VEHICLE_SPEED: f32 = 15.0;
pub const VEHICLE_ACCEL: f32 = 12.0;
pub const VEHICLE_BRAKE: f32 = 20.0;
pub const VEHICLE_TURN_SPEED: f32 = 2.5;
pub const VEHICLE_ENTER_DIST: f32 = 3.0;
pub const LANE_OFFSET: f32 = CAR_ROAD_WIDTH * 0.25; // 1.5m — center of each 3m lane
pub const INTERSECTION_APPROACH_DIST: f32 = 15.0;
pub const INTERSECTION_STOP_DIST: f32 = 5.0;
pub const INTERSECTION_WAIT_MAX: f32 = 4.0;
pub const FOLLOW_DISTANCE: f32 = 20.0;
pub const MIN_FOLLOW_DISTANCE: f32 = 6.0;
pub const PARKING_SPOT_LENGTH: f32 = 5.0;
pub const PARKING_SPOT_WIDTH: f32 = 2.5;
pub const FOG_DIST: f32 = 375.0;
pub const PLAYER_SPEED: f32 = 5.0;
pub const SPRINT_SPEED: f32 = 9.0;
pub const PLAYER_RADIUS: f32 = 0.4;
pub const DAY_LENGTH: f32 = 1440.0; // 1 game-minute = 1 real second (24 real minutes per day)
pub const HEADLESS_DT: f32 = 1.0 / 30.0; // shared timestep for headless simulation (observe, etc.)
pub const NUM_NPCS: usize = 100;
pub const NUM_ITEMS: usize = 250;
pub const NPC_SPEED: f32 = 2.5;
pub const TERRAIN_GRID: usize = 250;
pub const TERRAIN_CELL: f32 = WORLD_SIZE / TERRAIN_GRID as f32; // 2m per cell
pub const GRAVITY: f32 = 20.0;
pub const JUMP_VELOCITY: f32 = 8.0;

// Collision/vehicle damage constants
pub const VEHICLE_HIT_DAMAGE_MULT: f32 = 5.0;  // damage = speed * this
pub const VEHICLE_HIT_LAUNCH_UP: f32 = 5.0;    // upward velocity on hit
pub const VEHICLE_CRASH_SELF_DAMAGE: f32 = 0.3; // fraction of speed as damage to vehicle occupant
pub const SPEED_LIMIT: f32 = 10.0;              // speeding threshold on CarRoad

// Ragdoll constants
pub const RAGDOLL_DURATION: f32 = 3.0;
pub const RAGDOLL_POINT_COUNT: usize = 7;
// (idx_a, idx_b, rest_length): hips=0, chest=1, head=2, l_hand=3, r_hand=4, l_foot=5, r_foot=6
pub const RAGDOLL_CONSTRAINTS: [(usize, usize, f32); 6] = [
    (0, 1, 0.7),  // hips-chest
    (1, 2, 0.45), // chest-head
    (1, 3, 0.65), // chest-l_hand
    (1, 4, 0.65), // chest-r_hand
    (0, 5, 0.65), // hips-l_foot
    (0, 6, 0.65), // hips-r_foot
];

// Combat constants
pub const ATTACK_RANGE: f32 = 2.0;
pub const ATTACK_CONE_COS: f32 = 0.5; // cos(60°) — 120° cone
pub const ATTACK_DAMAGE: f32 = 15.0;
pub const NPC_ATTACK_DAMAGE: f32 = 10.0;
pub const ATTACK_COOLDOWN: f32 = 0.5;
pub const ATTACK_ANIM_DURATION: f32 = 0.3;
pub const KNOCKBACK_FORCE: f32 = 6.0;
pub const KNOCKBACK_UP: f32 = 3.0;
pub const KNOCKBACK_FRICTION: f32 = 5.0;
pub const KNOCKOUT_TIME: f32 = 10.0;
pub const KNOCKOUT_REGEN_HP: f32 = 30.0;
pub const NPC_HEALTH_MAX: f32 = 100.0;
pub const HEALTH_REGEN_RATE: f32 = 2.0;
pub const CAMERA_SHAKE_DECAY: f32 = 8.0;
pub const CAMERA_SHAKE_INTENSITY: f32 = 0.15;
pub const HIT_FLASH_DURATION: f32 = 0.15;

// Hunger/thirst constants
pub const HUNGER_DRAIN_RATE: f32 = 0.014;
pub const THIRST_DRAIN_RATE: f32 = 0.028;
pub const STARVATION_DAMAGE: f32 = 5.0;
pub const DEHYDRATION_DAMAGE: f32 = 8.0;
pub const FOOD_RESTORE: f32 = 35.0;
pub const WATER_RESTORE: f32 = 40.0;
pub const VENDING_FOOD_COST: f32 = 1.0;
pub const VENDING_DRINK_COST: f32 = 1.0;
pub const VENDING_FOOD_RESTORE: f32 = 40.0;
pub const VENDING_WATER_RESTORE: f32 = 40.0;
pub const NEWSPAPER_FOOD_RESTORE: f32 = 20.0;
pub const PLAYER_MIN_HEALTH_STARVE: f32 = 10.0;
pub const HUNGER_AUTOPILOT: f32 = 30.0;
pub const THIRST_AUTOPILOT: f32 = 30.0;
pub const NPC_STARTING_MONEY: f32 = 0.0;

// NPC sensory communication constants
pub const SOUND_RANGE: f32 = 30.0;
pub const VISION_RANGE: f32 = 30.0;
pub const VISION_CONE_COS: f32 = 0.5; // cos(60°) = 120° full cone
#[allow(dead_code)]
pub const SOUND_CHANNELS: usize = 3;

// NPC life simulation constants
pub const NUM_TRASH_BINS: usize = 60;
pub const NPC_JUMP_VELOCITY: f32 = 6.0;
pub const NPC_DRIVE_THRESHOLD: f32 = 15.0;
pub const NPC_PICKUP_DIST: f32 = 4.0;  // generous range for 500m world
pub const NPC_BIN_DIST: f32 = 1.5;
pub const INTERACT_DIST: f32 = 2.0;
pub const NIGHT_SPAWN_START: f32 = 20.0; // 8 PM
pub const NIGHT_SPAWN_END: f32 = 4.0;    // 4 AM
pub const DOCK_Z_START: f32 = 175.0;
pub const WATER_Y: f32 = -1.0;
pub const RIVER_WIDTH: f32 = 12.0;
pub const RIVER_DEPTH: f32 = 3.0;
pub const RIVER_CURRENT: f32 = 2.0;
pub const DROWN_DAMAGE: f32 = 8.0;
pub const PARKING_LOT_COUNT: usize = 4;

pub struct Wall { pub x: f32, pub z: f32, pub hw: f32, pub hd: f32, pub height: f32 }

pub struct RiverSegment { pub x1: f32, pub z1: f32, pub x2: f32, pub z2: f32, pub width: f32 }

/// Bridge zone: oriented rectangle where NPCs can walk across river.
/// dir = road direction unit vector, hw = half-width, hl = half-length.
pub struct Bridge { pub cx: f32, pub cz: f32, pub dir_x: f32, pub dir_z: f32, pub hw: f32, pub hl: f32 }

#[derive(Clone, Copy, PartialEq)]
pub enum RoadTier { CarRoad, FieldRoad }

#[derive(Clone, Copy, PartialEq)]
pub enum Surface { Sidewalk, CarRoad, FieldRoad, Terrain }

#[derive(Clone)]
pub struct RoadSegment {
    pub x0: f32, pub z0: f32,
    pub x1: f32, pub z1: f32,
    pub tier: RoadTier,
}

pub struct RoadNetwork {
    pub segments: Vec<RoadSegment>,
    pub nodes: Vec<[f32; 2]>,
    pub graph: RoadGraph,
    pub parking_spots: Vec<ParkingSpot>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum LaneDirection { Forward, Reverse }

#[derive(Clone, Copy)]
pub struct PathWaypoint { pub node_idx: usize, pub segment_idx: usize }

#[derive(Clone, Copy, PartialEq)]
pub enum IntersectionState { Cruising, Approaching, Waiting, Turning }

pub struct ParkingSpot {
    pub x: f32, pub z: f32, pub rot_y: f32,
    pub occupied_by: Option<usize>, // vehicle index
}

pub struct RoadGraph {
    pub adjacency: Vec<Vec<(usize, usize, f32)>>, // per-node: (neighbor, seg_idx, dist)
    pub segment_nodes: Vec<(usize, usize)>,        // per-segment: (node_a, node_b)
}

impl RoadNetwork {
    pub fn new() -> Self {
        RoadNetwork {
            segments: Vec::new(), nodes: Vec::new(),
            graph: RoadGraph { adjacency: Vec::new(), segment_nodes: Vec::new() },
            parking_spots: Vec::new(),
        }
    }
}

pub struct Terrain {
    pub heights: Vec<f32>,   // (TERRAIN_GRID+1)^2 height samples
    pub grid: usize,
    pub cell_size: f32,
}

impl Terrain {
    pub fn new() -> Self {
        let n = TERRAIN_GRID + 1;
        Terrain {
            heights: vec![0.0; n * n],
            grid: TERRAIN_GRID,
            cell_size: TERRAIN_CELL,
        }
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let gx = (x + WORLD_HALF) / self.cell_size;
        let gz = (z + WORLD_HALF) / self.cell_size;
        let ix = (gx as usize).min(self.grid - 1);
        let iz = (gz as usize).min(self.grid - 1);
        let fx = gx - ix as f32;
        let fz = gz - iz as f32;
        let fx = fx.clamp(0.0, 1.0);
        let fz = fz.clamp(0.0, 1.0);
        let stride = self.grid + 1;
        let h00 = self.heights[iz * stride + ix];
        let h10 = self.heights[iz * stride + ix + 1];
        let h01 = self.heights[(iz + 1) * stride + ix];
        let h11 = self.heights[(iz + 1) * stride + ix + 1];
        let h0 = h00 + (h10 - h00) * fx;
        let h1 = h01 + (h11 - h01) * fx;
        h0 + (h1 - h0) * fz
    }

    pub fn normal_at(&self, x: f32, z: f32) -> [f32; 3] {
        let d = self.cell_size * 0.5;
        let hx0 = self.height_at(x - d, z);
        let hx1 = self.height_at(x + d, z);
        let hz0 = self.height_at(x, z - d);
        let hz1 = self.height_at(x, z + d);
        let nx = hx0 - hx1;
        let nz = hz0 - hz1;
        let ny = 2.0 * d;
        let l = (nx * nx + ny * ny + nz * nz).sqrt();
        if l < 1e-10 { [0.0, 1.0, 0.0] } else { [nx / l, ny / l, nz / l] }
    }
}

#[allow(dead_code)]
pub struct Building {
    pub x: f32, pub z: f32,
    pub w: f32, pub d: f32,
    pub h: f32, pub ground_y: f32,
}

pub struct Rock {
    pub x: f32, pub z: f32,
    pub size: f32,
}

pub struct Tree {
    pub x: f32, pub z: f32,
    pub trunk_radius: f32,
}

pub struct StreetLight {
    pub x: f32, pub z: f32,
}

pub struct TrashBin {
    pub x: f32, pub y: f32, pub z: f32,
    pub items_held: u32,
    pub carried_by: Option<usize>, // NPC index carrying this bin
    pub terrain_normal: [f32; 3],
}

pub struct Vehicle {
    pub x: f32, pub y: f32, pub z: f32,
    pub rot_y: f32,
    pub speed: f32,
    pub terrain_normal: [f32; 3],
    pub color: u32,
    pub occupied: bool,       // player is driving
    pub ai_active: bool,      // AI drives when not occupied and on road
    pub ai_target_x: f32,
    pub ai_target_z: f32,
    pub rng: Rng,
    pub owner_npc: Option<usize>, // None = ambient traffic, Some(i) = NPC-owned
    // Traffic AI fields
    pub path: Vec<PathWaypoint>,
    pub path_idx: usize,
    pub current_segment: Option<usize>,
    pub lane_dir: LaneDirection,
    pub intersection_state: IntersectionState,
    pub intersection_wait_timer: f32,
    pub cruise_speed: f32,       // per-vehicle 7.0–12.0 m/s
    pub target_speed: f32,
    pub parking_target: Option<usize>,
    pub parked: bool,
    pub idle_timer: f32,          // tracks how long vehicle has been at speed < 0.5
}

pub const VEHICLE_COLORS: [u32; 6] = [
    0xFFCC3333, 0xFF3333CC, 0xFF33CC33, 0xFFCCCC33, 0xFFCC33CC, 0xFFFFFFFF,
];

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NpcJob {
    Collector,
    GarbageCollector,
    TaxiDriver,
    DeliveryCourier,
    MailCarrier,
    Paramedic,
    Firefighter,
    PolicePatrol,
    StreetVendor,
    Mechanic,
    ConstructionWorker,
    Fisherman,
    Farmer,
    Lumberjack,
    Scavenger,
}

pub const NPC_JOB_COUNT: usize = 15;

#[derive(Clone, Copy, PartialEq)]
pub enum PlayerJobType {
    None,
    GarbageCollector,
    TaxiDriver,
    DeliveryCourier,
    MailCarrier,
    Paramedic,
    Firefighter,
    PolicePatrol,
    StreetVendor,
    Mechanic,
    ConstructionWorker,
    Fisherman,
    Farmer,
    Lumberjack,
    Scavenger,
}

pub struct PlayerJob {
    pub job_type: PlayerJobType,
    pub time_remaining: f32,
    pub items_done: u32,
    pub items_needed: u32,
}

impl PlayerJob {
    pub fn none() -> Self {
        PlayerJob {
            job_type: PlayerJobType::None,
            time_remaining: 0.0, items_done: 0, items_needed: 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum InteractibleKind {
    VendingMachine,
    ParkBench,
    Dumpster,
    Atm,
    PhoneBooth,
    FireHydrant,
    NewspaperStand,
    Mailbox,
    Payphone,
}

pub struct Interactible {
    pub x: f32, pub y: f32, pub z: f32,
    pub kind: InteractibleKind,
    pub cooldown: f32,
    pub state_val: f32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NpcState {
    Sleeping,
    HomeTask,
    GoingToWork,
    Working,
    GoingHome,
    Driving,
    Interacting,
    KnockedOut,
}

pub struct Npc {
    // Existing
    pub x: f32, pub y: f32, pub z: f32,
    pub rot_y: f32,
    pub walk_phase: f32,
    pub target_x: f32, pub target_z: f32,
    pub shirt_color: u32,
    pub pants_color: u32,
    pub rng: Rng,
    // Physics
    pub vel_y: f32,
    pub on_ground: bool,
    pub terrain_normal: [f32; 3],
    // Life simulation
    pub state: NpcState,
    pub home_idx: usize,         // index into buildings
    pub car_idx: usize,          // index into vehicles
    pub wake_hour: f32,          // 5.0–9.0
    pub state_timer: f32,        // seconds spent in current state
    pub money: f32,
    // Work
    pub carrying_item: bool,
    pub carrying_bin: Option<usize>,
    pub target_item: Option<usize>,
    pub target_bin: Option<usize>,
    pub items_deposited_today: u32,
    // Driving
    pub in_vehicle: bool,
    pub parked_x: f32, pub parked_z: f32,
    // Pathfinding
    pub stuck_timer: f32,
    pub stuck_count: u8, // consecutive stuck events — escalates recovery
    pub detour_x: f32, pub detour_z: f32,
    pub detouring: bool,
    // Job system
    pub job: NpcJob,
    pub job_timer: f32,
    pub job_target_x: f32, pub job_target_z: f32,
    pub interaction_target: Option<usize>, // index into interactibles
    // Social interactions
    pub interacting_with: Option<usize>, // other NPC index
    pub interaction_timer: f32,
    // NEAT fitness tracking
    pub brain_idx: usize,
    pub fitness_money_earned: f32,
    pub fitness_items_picked: u32,
    pub fitness_interactions: u32,
    pub fitness_distance: f32,
    pub fitness_stuck_time: f32,
    pub prev_x: f32, pub prev_z: f32,
    // Combat
    pub health: f32,
    pub attack_cooldown: f32,
    pub attack_phase: f32,
    pub hit_flash: f32,
    pub knockout_timer: f32,
    pub knockback_vx: f32,
    pub knockback_vz: f32,
    pub attack_intent: u8, // 0=none, 1=player, 2=npc
    pub fitness_knockouts: u32,
    pub fitness_hits_landed: u32,
    // Hunger/thirst
    pub hunger: f32,
    pub thirst: f32,
    pub starving_dead: bool,
    pub fitness_starve_time: f32,
    // Sound/vision communication
    pub sound: [f32; 3],
    pub fitness_sounds_made: u32,
    pub fitness_npcs_heard: u32,
    // Proximity tracking (continuous reward for being near items)
    pub fitness_proximity: f32,
    // Ragdoll
    pub ragdoll_active: bool,
    pub ragdoll_points: [[f32; 3]; 7],  // hips, chest, head, l_hand, r_hand, l_foot, r_foot
    pub ragdoll_prev: [[f32; 3]; 7],
    pub ragdoll_timer: f32,
    // Law system
    pub wanted: bool,
    pub bounty: f32,
    pub violation_timer: f32,
    // Police
    pub police_target: Option<usize>,
    // Target cooldown — prevents re-targeting same unreachable item after stuck recovery
    pub wander_cooldown: f32,
}

pub const NPC_SHIRT_COLORS: [u32; 6] = [
    0xFFAA3333, 0xFF33AA33, 0xFF3333AA, 0xFFAAAA33, 0xFF33AAAA, 0xFFAA33AA,
];
pub const NPC_PANTS_COLORS: [u32; 4] = [
    0xFF333355, 0xFF443322, 0xFF334433, 0xFF444444,
];

#[derive(Clone, Copy, PartialEq)]
pub enum ItemKind { Health, Money, Stamina, Food, Water }

pub struct Item {
    pub x: f32, pub y: f32, pub z: f32,
    pub kind: ItemKind,
    pub active: bool,
    pub spin_phase: f32,
    pub falling: bool,
    pub vel_y: f32,
    pub claimed_by: Option<usize>, // NPC index heading for this
    pub skip_until: f32, // countdown — NPCs skip this item while > 0 (marked unreachable)
}

pub struct Player {
    pub x: f32, pub y: f32, pub z: f32,
    pub rot_y: f32,
    pub health: f32,
    pub stamina: f32,
    pub money: f32,
    pub vel_y: f32,
    pub on_ground: bool,
    pub terrain_normal: [f32; 3],
    pub walk_phase: f32,
    pub sprinting: bool,
    pub in_vehicle: Option<usize>, // index into vehicles vec
    pub carrying_item: bool,
    pub carrying_bin: Option<usize>,
    // Job system
    pub active_job: PlayerJob,
    pub sitting: bool,
    pub bank_balance: f32,
    pub job_menu_open: bool,
    pub job_menu_cursor: usize,
    // Combat
    pub attack_cooldown: f32,
    pub attack_phase: f32,
    pub hit_flash: f32,
    pub damage_shake: f32,
    // Hunger/thirst
    pub hunger: f32,
    pub thirst: f32,
    // Law system
    pub wanted_vehicle_hit: bool,
    pub bounty: f32,
    // Body type
    pub is_female: bool,
}

pub struct Camera {
    pub x: f32, pub y: f32, pub z: f32,
    pub tx: f32, pub ty: f32, pub tz: f32,
    pub yaw: f32,   // orbit yaw around player (radians)
    pub pitch: f32,  // orbit pitch above player (radians)
}

// WorldTri: a single triangle in world space with flat color
#[derive(Clone)]
pub struct WorldTri {
    pub v: [[f32; 3]; 3],
    pub normal: [f32; 3],
    pub color: u32,
}

pub struct WorldData {
    pub static_tris: Vec<WorldTri>,
    pub buildings: Vec<Building>,
    pub rocks: Vec<Rock>,
    pub vehicles: Vec<Vehicle>,
    pub npcs: Vec<Npc>,
    pub items: Vec<Item>,
    pub trees: Vec<Tree>,
    pub street_lights: Vec<StreetLight>,
    pub trash_bins: Vec<TrashBin>,
    pub interactibles: Vec<Interactible>,
    pub walls: Vec<Wall>,
    pub river_segments: Vec<RiverSegment>,
    pub bridges: Vec<Bridge>,
}

pub struct GameState {
    pub keys: [bool; 256],
    pub prev_keys: [bool; 256],
    pub should_quit: bool,
    pub width: usize,
    pub height: usize,
    pub time_of_day: f32, // 0.0 - 24.0 hours
    pub world_seed: u64,
    pub road_network: RoadNetwork,
    pub terrain: Terrain,
    pub frame_counter: u64,
    pub player: Player,
    pub camera: Camera,
    pub world: WorldData,
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub keybinds: KeyBinds,
    pub menu: MenuData,
    pub mouse_sensitivity: f32,
    pub invert_mouse_x: bool,
    pub invert_mouse_y: bool,
    pub spawn_rng: Rng,
    pub day_count: u32,
    pub neat_population: crate::neat::Population,
    pub neat_brains: Vec<crate::neat::NeatBrain>,
    pub time_speed: u32, // 1, 10, 100, 1000
}

impl GameState {
    pub fn new(w: usize, h: usize, seed: u64) -> Self {
        GameState {
            keys: [false; 256],
            prev_keys: [false; 256],
            should_quit: false,
            time_of_day: 10.0, // start at 10 AM
            world_seed: seed,
            road_network: RoadNetwork::new(), // filled by generate_world
            terrain: Terrain::new(),
            frame_counter: 0,
            width: w,
            height: h,
            player: Player {
                x: 0.0, y: 0.0, z: 10.0,
                rot_y: 0.0,
                health: 100.0, stamina: 100.0, money: 0.0,
                vel_y: 0.0, on_ground: true,
                walk_phase: 0.0, sprinting: false, in_vehicle: None,
                carrying_item: false, carrying_bin: None,
                active_job: PlayerJob::none(), sitting: false, bank_balance: 0.0,
                job_menu_open: false, job_menu_cursor: 0,
                attack_cooldown: 0.0, attack_phase: 0.0, hit_flash: 0.0, damage_shake: 0.0,
                hunger: 100.0, thirst: 100.0,
                wanted_vehicle_hit: false, bounty: 0.0,
                is_female: false,
                terrain_normal: [0.0, 1.0, 0.0],
            },
            camera: Camera {
                x: 0.0, y: 8.0, z: 18.0,
                tx: 0.0, ty: 1.0, tz: 10.0,
                yaw: 0.0,
                pitch: 0.35, // ~20 degrees above horizontal
            },
            world: WorldData {
                static_tris: Vec::new(),
                buildings: Vec::new(),
                rocks: Vec::new(),
                vehicles: Vec::new(),
                npcs: Vec::new(),
                items: Vec::new(),
                trees: Vec::new(),
                street_lights: Vec::new(),
                trash_bins: Vec::new(),
                interactibles: Vec::new(),
                walls: Vec::new(),
                river_segments: Vec::new(),
                bridges: Vec::new(),
            },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            keybinds: KeyBinds::default_binds(),
            menu: MenuData::new(),
            mouse_sensitivity: 1.0,
            invert_mouse_x: false,
            invert_mouse_y: false,
            spawn_rng: Rng::new(seed.wrapping_add(0xDEAD)),
            day_count: 1,
            neat_population: crate::neat::Population::new(NUM_NPCS, seed.wrapping_add(0xAE47)),
            neat_brains: Vec::new(),
            time_speed: 1,
        }
    }
}
