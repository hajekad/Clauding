# Clauding

A Rust 3D city/NPC simulation built without external crates.

Real-time graphics via raw Vulkan FFI (`dlopen` of `libvulkan.so.1`) plus a
parallel software rasterizer fallback. NPCs with jobs, vehicle physics with a
suspension/Pacejka tire model, and a NEAT (NeuroEvolution of Augmenting
Topologies) implementation that evolves NPC decision-making.

## The constraint

`Cargo.toml` declares **zero dependencies**. Nothing in `[dependencies]`,
nothing transitive. The constraint is the project's defining property:

- **Vulkan** is loaded with `dlopen("libvulkan.so.1")` and every entry point
  resolved through `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr`. SPIR-V
  shaders are emitted as `Vec<u32>` directly from Rust — no `glslang`, no
  `naga`.
- **Wayland** is driven the same way: `dlopen("libwayland-client.so")`,
  hand-rolled `xdg-shell` interface definitions, shm framebuffers.
- **GLTF and FBX** parsers are written by hand (FBX includes a tiny inline
  DEFLATE decoder for the zlib-compressed array streams).
- **NEAT, navmesh A\*, the parallel job system, frame-time percentile
  telemetry, the PNG writer, the procedural-noise terrain generator, and the
  software rasterizer** are all from scratch on top of `std`.

This is a deliberate choice for one personal project. It is not a
recommendation for production code. Don't `dlopen` Vulkan in your day job.

## What's in here

A short tour of the more interesting source files:

| File | LOC | What it does |
| --- | ---: | --- |
| `src/gpu.rs` | ~2,400 | Vulkan compute + graphics pipelines via raw FFI; offscreen render → readback to `HOST_CACHED` buffer → Wayland present. |
| `src/render.rs` + `src/raster.rs` | ~5,000 combined | Software rasterizer (edge-function fill, z-buffer) and the procedural mesh authoring used by every entity. |
| `src/gltf_loader.rs` | ~1,160 | GLTF + .bin parser, no crate. |
| `src/fbx_anim.rs` | ~1,070 | FBX binary parser with an inline DEFLATE decoder for skeletal animation streams (Mixamo). |
| `src/neat.rs` | ~1,240 | NEAT neuroevolution — speciation, crossover, structural mutations, save/load. |
| `src/jobs.rs` | ~1,020 | Per-job NPC behavior dispatch. |
| `src/navmesh.rs` | ~570 | 1m-cell walkability grid + A\*. |
| `src/main.rs` | ~480 | Fixed-timestep main loop, 500 FPS render cap, percentile frame-time telemetry. |
| `src/gpu_kernels.rs` + `src/gpu_shaders.rs` | ~700 | SPIR-V compute and graphics shaders, emitted as Rust `Vec<u32>`. |
| `src/world.rs` | ~4,060 | Heightmap terrain, building/road/tree/rock generation. |

## Status

Functional prototype. Around 37k LOC. Built for Linux/Wayland on an RTX 3080
Ti. There are no tests; this is exploratory work, not production code. Release
builds use `lto = "fat"` and `codegen-units = 1` — `cargo build --release`
takes a while.

## Why publish it

It exists as evidence that one person can do raw graphics, FFI, parsers, and
simulation primitives in pure Rust on a tight time budget. Useful as a
portfolio artifact; not useful as a starting point for derivative work — you
almost certainly want `wgpu`, `glam`, `winit`, and `gltf` instead.

## Building and running

```sh
cargo run --release            # game, fullscreen 1920×1080
cargo run --release --bin train -- 42 200   # headless NEAT trainer (seed, generations)
```

Requirements:

- Linux with a Wayland compositor.
- `libvulkan.so.1` and `libwayland-client.so.0` reachable through the dynamic
  loader (any standard distro install).
- A Vulkan-capable GPU. The software rasterizer is preserved as a fallback,
  but the GPU path is what's been kept current.

`ESC` opens the pause menu. The renderer reads a previously trained NEAT
population from `neat_trained.bin` on startup if present; otherwise it starts
from a fresh population.

## Assets — not included

The repo contains code only. The 3D asset library the author used is **not
shipped** with the source: it is roughly 3 GB and a mix of Mixamo, Sketchfab,
and other third-party content whose licenses do not permit redistribution. You
must supply your own.

The game scans `models/v1/` at startup and loads whatever it finds. The
expected layout is:

```
models/v1/
├── architecture/<name>/scene.gltf      # buildings (any number)
├── nature/<name>/scene.gltf            # trees, rocks
├── characters/<name>/scene.gltf        # NPC + player meshes
│   └── <name>.obj                      # optional, used for skinning fallback
├── cars/<name>/scene.gltf              # vehicles
├── animations/walking.fbx              # Mixamo clip (required for NPC anim)
├── animations/run_forward.fbx
├── animations/picking_up.fbx
├── animations/sitting_pose.fbx
├── animations/elbow_punch.fbx
├── animations/hook_punch.fbx
├── animations/drop_kick.fbx
└── animations/roundhouse_kick.fbx
```

Each `<name>/` directory needs `scene.gltf` plus its `.bin` and any textures
it references. Naming is otherwise arbitrary — the loader walks the trees and
filters by triangle count and bounding-box proportions, so you can drop any
sensibly-scaled model in.

Suggested sources:

- **Animations:** [Mixamo](https://www.mixamo.com/) — the eight clip names
  above are exact filenames the FBX loader looks for. Download as **FBX
  Binary**, no skin.
- **Buildings, props, vehicles:** [Sketchfab](https://sketchfab.com/) (filter
  by GLTF + downloadable + permissive license), [Poly Haven](https://polyhaven.com/),
  [Quaternius](https://quaternius.com/).
- **Trees / rocks:** Quaternius nature packs work well at the scales the
  loader expects.

The game runs without `models/v1/` present — terrain, roads, and procedural
geometry all generate fine — but NPCs and buildings will be invisible until
you supply at least a few GLTFs.

## License

MIT — see [LICENSE](LICENSE). Applies to source code only. Any 3D assets you
add to `models/v1/` carry whatever license the original author attached.
