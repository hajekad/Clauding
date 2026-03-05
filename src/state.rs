// GameState: global mutable game state, entity stores, constants

use crate::rng::Rng;
use crate::input::KeyBinds;
use crate::menu::MenuData;

pub const DEFAULT_WIDTH: usize = 1920;
pub const DEFAULT_HEIGHT: usize = 1080;
pub const MAX_FPS: f32 = 500.0;
pub const FRAME_TIME_MIN: f32 = 1.0 / MAX_FPS;

pub const WORLD_SIZE: f32 = 200.0;
pub const WORLD_HALF: f32 = WORLD_SIZE * 0.5;
pub const NUM_BUILDINGS: usize = 50;
pub const NUM_TREES: usize = 80;
pub const NUM_ROCKS: usize = 20;
pub const NUM_STREET_LIGHTS: usize = 25;
pub const NUM_VEHICLES: usize = 30;
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
pub const FOG_DIST: f32 = 150.0;
pub const PLAYER_SPEED: f32 = 5.0;
pub const SPRINT_SPEED: f32 = 9.0;
pub const PLAYER_RADIUS: f32 = 0.4;
pub const DAY_LENGTH: f32 = 1440.0; // 1 game-minute = 1 real second (24 real minutes per day)
pub const NUM_NPCS: usize = 40;
pub const NUM_ITEMS: usize = 100;
pub const NPC_SPEED: f32 = 2.5;
pub const TERRAIN_GRID: usize = 100;
pub const TERRAIN_CELL: f32 = WORLD_SIZE / TERRAIN_GRID as f32; // 2m per cell
pub const GRAVITY: f32 = 20.0;
pub const JUMP_VELOCITY: f32 = 8.0;

// NPC life simulation constants
pub const NUM_TRASH_BINS: usize = 25;
pub const NPC_JUMP_VELOCITY: f32 = 6.0;
pub const NPC_DRIVE_THRESHOLD: f32 = 30.0;
pub const NPC_PICKUP_DIST: f32 = 1.2;
pub const NPC_BIN_DIST: f32 = 1.5;
pub const INTERACT_DIST: f32 = 2.0;
pub const NIGHT_SPAWN_START: f32 = 20.0; // 8 PM
pub const NIGHT_SPAWN_END: f32 = 4.0;    // 4 AM
pub const DOCK_Z_START: f32 = 70.0;
pub const WATER_Y: f32 = -1.0;

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
}

impl RoadNetwork {
    pub fn new() -> Self {
        RoadNetwork { segments: Vec::new(), nodes: Vec::new() }
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
}

pub struct Vehicle {
    pub x: f32, pub y: f32, pub z: f32,
    pub rot_y: f32,
    pub speed: f32,
    pub color: u32,
    pub occupied: bool,       // player is driving
    pub ai_active: bool,      // AI drives when not occupied and on road
    pub ai_target_x: f32,
    pub ai_target_z: f32,
    pub rng: Rng,
    pub owner_npc: Option<usize>, // None = ambient traffic, Some(i) = NPC-owned
}

pub const VEHICLE_COLORS: [u32; 6] = [
    0xFFCC3333, 0xFF3333CC, 0xFF33CC33, 0xFFCCCC33, 0xFFCC33CC, 0xFFFFFFFF,
];

#[derive(Clone, Copy, PartialEq)]
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
    pub objective_x: f32,
    pub objective_z: f32,
    pub progress: f32,
    pub earnings: f32,
    pub time_remaining: f32,
    pub items_done: u32,
    pub items_needed: u32,
}

impl PlayerJob {
    pub fn none() -> Self {
        PlayerJob {
            job_type: PlayerJobType::None,
            objective_x: 0.0, objective_z: 0.0,
            progress: 0.0, earnings: 0.0,
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
    pub rot_y: f32,
    pub cooldown: f32,
    pub state_val: f32,
    pub used_by: Option<usize>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum NpcState {
    Sleeping,
    HomeTask,
    GoingToWork,
    Working,
    GoingHome,
    Driving,
    Interacting,
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
}

pub const NPC_SHIRT_COLORS: [u32; 6] = [
    0xFFAA3333, 0xFF33AA33, 0xFF3333AA, 0xFFAAAA33, 0xFF33AAAA, 0xFFAA33AA,
];
pub const NPC_PANTS_COLORS: [u32; 4] = [
    0xFF333355, 0xFF443322, 0xFF334433, 0xFF444444,
];

#[derive(Clone, Copy, PartialEq)]
pub enum ItemKind { Health, Money, Stamina }

pub struct Item {
    pub x: f32, pub y: f32, pub z: f32,
    pub kind: ItemKind,
    pub active: bool,
    #[allow(dead_code)]
    pub respawn_timer: f32,
    pub spin_phase: f32,
    pub falling: bool,
    pub vel_y: f32,
    pub claimed_by: Option<usize>, // NPC index heading for this
}

pub struct Player {
    pub x: f32, pub y: f32, pub z: f32,
    pub rot_y: f32,
    pub health: f32,
    pub stamina: f32,
    pub money: f32,
    pub vel_y: f32,
    pub on_ground: bool,
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
        }
    }
}
