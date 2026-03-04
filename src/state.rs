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
pub const ROAD_WIDTH: f32 = 6.0;
pub const VEHICLE_SPEED: f32 = 15.0;
pub const VEHICLE_ACCEL: f32 = 12.0;
pub const VEHICLE_BRAKE: f32 = 20.0;
pub const VEHICLE_TURN_SPEED: f32 = 2.5;
pub const VEHICLE_ENTER_DIST: f32 = 3.0;
pub const FOG_DIST: f32 = 150.0;
pub const PLAYER_SPEED: f32 = 5.0;
pub const SPRINT_SPEED: f32 = 9.0;
pub const PLAYER_RADIUS: f32 = 0.4;
pub const DAY_LENGTH: f32 = 120.0; // 120 seconds per full day cycle
pub const NUM_NPCS: usize = 40;
pub const NUM_ITEMS: usize = 20;
pub const NPC_SPEED: f32 = 2.5;
pub const ITEM_PICKUP_DIST: f32 = 1.5;
pub const ITEM_RESPAWN_TIME: f32 = 30.0;

pub struct Building {
    pub x: f32, pub z: f32,
    pub w: f32, pub d: f32,
}

pub struct Rock {
    pub x: f32, pub z: f32,
    pub size: f32,
}

pub struct Vehicle {
    pub x: f32, pub z: f32,
    pub rot_y: f32,
    pub speed: f32,
    pub color: u32,
    pub occupied: bool,       // player is driving
    pub ai_active: bool,      // AI drives when not occupied and on road
    pub ai_target_x: f32,
    pub ai_target_z: f32,
    pub rng: Rng,
}

pub const VEHICLE_COLORS: [u32; 6] = [
    0xFFCC3333, 0xFF3333CC, 0xFF33CC33, 0xFFCCCC33, 0xFFCC33CC, 0xFFFFFFFF,
];

pub struct Npc {
    pub x: f32, pub z: f32,
    pub rot_y: f32,
    pub walk_phase: f32,
    pub target_x: f32, pub target_z: f32,
    pub shirt_color: u32,
    pub pants_color: u32,
    pub rng: Rng,
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
    pub x: f32, pub z: f32,
    pub kind: ItemKind,
    pub active: bool,
    pub respawn_timer: f32,
    pub spin_phase: f32,
}

pub struct Player {
    pub x: f32, pub y: f32, pub z: f32,
    pub rot_y: f32,
    pub health: f32,
    pub stamina: f32,
    pub money: f32,
    pub score: u32,
    pub walk_phase: f32,
    pub sprinting: bool,
    pub in_vehicle: Option<usize>, // index into vehicles vec
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
}

pub struct GameState {
    pub keys: [bool; 256],
    pub prev_keys: [bool; 256],
    pub should_quit: bool,
    pub width: usize,
    pub height: usize,
    pub time_of_day: f32, // 0.0 - 24.0 hours
    pub world_seed: u64,
    pub road_positions: Vec<f32>,
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
}

impl GameState {
    pub fn new(w: usize, h: usize, seed: u64) -> Self {
        GameState {
            keys: [false; 256],
            prev_keys: [false; 256],
            should_quit: false,
            time_of_day: 10.0, // start at 10 AM
            world_seed: seed,
            road_positions: Vec::new(), // filled by generate_world
            frame_counter: 0,
            width: w,
            height: h,
            player: Player {
                x: 0.0, y: 0.0, z: 10.0,
                rot_y: 0.0,
                health: 100.0, stamina: 100.0, money: 50.0,
                score: 0,
                walk_phase: 0.0, sprinting: false, in_vehicle: None,
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
            },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
            keybinds: KeyBinds::default_binds(),
            menu: MenuData::new(),
            mouse_sensitivity: 1.0,
            invert_mouse_x: false,
            invert_mouse_y: false,
        }
    }
}
