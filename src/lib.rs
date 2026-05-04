//! Clauding — a Rust 3D city/NPC simulation built with **zero external crates**.
//!
//! `Cargo.toml` has no dependencies. Vulkan, Wayland, and `libc` are reached
//! through `dlopen`; GLTF, FBX, NEAT, navmesh, the job system, and the software
//! rasterizer are all written from scratch. Linux/Wayland only.
//!
//! See the module docs for a tour: [`gpu`] (Vulkan FFI), [`raster`] + [`render`]
//! (software rasterizer), [`gltf_loader`], [`fbx_anim`], [`neat`], [`jobs`].

#![allow(unsafe_op_in_unsafe_fn)]

pub mod anatomy;
pub mod camera;
pub mod collision;
pub mod color;
pub mod combat;
pub mod deform;
pub mod fbx_anim;
pub mod gltf_loader;
pub mod gpu;
pub mod gpu_kernels;
pub mod gpu_shaders;
pub mod gpu_spirv;
pub mod hud;
pub mod image;
pub mod input;
pub mod jobs;
pub mod material;
pub mod math;
pub mod menu;
pub mod mesh;
pub mod navmesh;
pub mod neat;
pub mod noise;
pub mod npc;
pub mod particle;
pub mod physics;
pub mod placement;
pub mod platform;
pub mod player;
pub mod player_jobs;
pub mod raster;
pub mod render;
pub mod rng;
pub mod skeleton;
pub mod skeleton_anim;
pub mod state;
pub mod suspension;
pub mod telemetry;
pub mod tire;
pub mod vehicle;
pub mod vehicle_physics;
pub mod world;
pub mod zone;
