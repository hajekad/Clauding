//! Particle system — SoA arrays, GPU-accelerated update with CPU fallback.
//! Emitters: vehicle exhaust, sprint dust, item pickup sparkles.

use crate::color::darken;
use crate::gpu::*;
use crate::math::*;
use crate::raster::*;
use crate::render::clip_to_screen;
use crate::rng::Rng;
use crate::state::*;

const MAX_PARTICLES: usize = 4096;
const GRAVITY: f32 = -9.8;

pub struct ParticleSystem {
    // SoA layout for GPU compute
    pub pos_x: Vec<f32>,
    pub pos_y: Vec<f32>,
    pub pos_z: Vec<f32>,
    pub vel_x: Vec<f32>,
    pub vel_y: Vec<f32>,
    pub vel_z: Vec<f32>,
    pub lifetime: Vec<f32>,
    pub color: Vec<u32>,
    pub count: usize,
    // GPU buffers (None if no GPU)
    gpu_bufs: Option<GpuParticleBufs>,
    pub emission_rng: Rng,
}

struct GpuParticleBufs {
    buf_px: GpuBuf,
    buf_py: GpuBuf,
    buf_pz: GpuBuf,
    buf_vx: GpuBuf,
    buf_vy: GpuBuf,
    buf_vz: GpuBuf,
    buf_lt: GpuBuf,
}

impl ParticleSystem {
    pub fn new(gpu: &mut Option<GpuContext>, seed: u64) -> Self {
        let buf_size = MAX_PARTICLES * 4;
        let gpu_bufs = gpu.as_mut().map(|ctx| GpuParticleBufs {
            buf_px: ctx.create_buffer(buf_size),
            buf_py: ctx.create_buffer(buf_size),
            buf_pz: ctx.create_buffer(buf_size),
            buf_vx: ctx.create_buffer(buf_size),
            buf_vy: ctx.create_buffer(buf_size),
            buf_vz: ctx.create_buffer(buf_size),
            buf_lt: ctx.create_buffer(buf_size),
        });

        ParticleSystem {
            pos_x: vec![0.0; MAX_PARTICLES],
            pos_y: vec![0.0; MAX_PARTICLES],
            pos_z: vec![0.0; MAX_PARTICLES],
            vel_x: vec![0.0; MAX_PARTICLES],
            vel_y: vec![0.0; MAX_PARTICLES],
            vel_z: vec![0.0; MAX_PARTICLES],
            lifetime: vec![0.0; MAX_PARTICLES],
            color: vec![0; MAX_PARTICLES],
            count: 0,
            gpu_bufs,
            emission_rng: Rng::new(seed),
        }
    }

    pub fn emit(&mut self, x: f32, y: f32, z: f32, vx: f32, vy: f32, vz: f32, lt: f32, color: u32) {
        // Find a dead particle to reuse, or use next slot
        let idx = if self.count < MAX_PARTICLES {
            let i = self.count;
            self.count += 1;
            i
        } else {
            // Find first dead particle
            match self.lifetime.iter().position(|&l| l <= 0.0) {
                Some(i) => i,
                None => return, // all full
            }
        };

        self.pos_x[idx] = x;
        self.pos_y[idx] = y;
        self.pos_z[idx] = z;
        self.vel_x[idx] = vx;
        self.vel_y[idx] = vy;
        self.vel_z[idx] = vz;
        self.lifetime[idx] = lt;
        self.color[idx] = color;
    }

    pub fn update(&mut self, gpu: &mut Option<GpuContext>, dt: f32) {
        if self.count == 0 {
            return;
        }

        let n = self.count;

        if let (Some(ctx), Some(bufs)) = (gpu.as_mut(), self.gpu_bufs.as_ref()) {
            // GPU path: upload → dispatch → download
            ctx.upload(&bufs.buf_px, bytemuck_cast(&self.pos_x[..n]));
            ctx.upload(&bufs.buf_py, bytemuck_cast(&self.pos_y[..n]));
            ctx.upload(&bufs.buf_pz, bytemuck_cast(&self.pos_z[..n]));
            ctx.upload(&bufs.buf_vx, bytemuck_cast(&self.vel_x[..n]));
            ctx.upload(&bufs.buf_vy, bytemuck_cast(&self.vel_y[..n]));
            ctx.upload(&bufs.buf_vz, bytemuck_cast(&self.vel_z[..n]));
            ctx.upload(&bufs.buf_lt, bytemuck_cast(&self.lifetime[..n]));

            // Push constants: count(u32), dt(f32), gravity(f32)
            let count = n as u32;
            let mut push = [0u8; 12];
            push[0..4].copy_from_slice(&count.to_ne_bytes());
            push[4..8].copy_from_slice(&dt.to_ne_bytes());
            push[8..12].copy_from_slice(&GRAVITY.to_ne_bytes());

            ctx.dispatch(
                "particle_update",
                &[
                    &bufs.buf_px,
                    &bufs.buf_py,
                    &bufs.buf_pz,
                    &bufs.buf_vx,
                    &bufs.buf_vy,
                    &bufs.buf_vz,
                    &bufs.buf_lt,
                ],
                &push,
                count,
            );

            // Download results
            ctx.download(&bufs.buf_px, bytemuck_cast_mut(&mut self.pos_x[..n]));
            ctx.download(&bufs.buf_py, bytemuck_cast_mut(&mut self.pos_y[..n]));
            ctx.download(&bufs.buf_pz, bytemuck_cast_mut(&mut self.pos_z[..n]));
            ctx.download(&bufs.buf_vy, bytemuck_cast_mut(&mut self.vel_y[..n]));
            ctx.download(&bufs.buf_lt, bytemuck_cast_mut(&mut self.lifetime[..n]));
        } else {
            // CPU fallback
            for i in 0..n {
                if self.lifetime[i] <= 0.0 {
                    continue;
                }
                self.vel_y[i] += GRAVITY * dt;
                self.pos_x[i] += self.vel_x[i] * dt;
                self.pos_y[i] += self.vel_y[i] * dt;
                self.pos_z[i] += self.vel_z[i] * dt;
                self.lifetime[i] -= dt;
            }
        }
    }
}

// Emit particles from game entities using frame_counter for determinism
pub fn sys_emit_particles(ps: &mut ParticleSystem, game: &GameState, _dt: f32) {
    let frame = game.frame_counter;

    // Vehicle exhaust
    for v in &game.world.vehicles {
        if v.speed.abs() < 1.0 {
            continue;
        }
        let (sin_r, cos_r) = v.rot_y.sin_cos();
        let ex = v.x + sin_r * 1.8; // behind vehicle
        let ez = v.z + cos_r * 1.8;
        // Emit every 3rd frame per vehicle
        if frame % 3 == 0 {
            let spread = 0.3;
            let vx = sin_r * 1.0 + ps.emission_rng.range(-0.3, 0.3);
            let vz = cos_r * 1.0 + ps.emission_rng.range(-0.2, 0.2);
            ps.emit(
                ex,
                v.y + 0.3,
                ez,
                vx * spread,
                0.5,
                vz * spread,
                0.8,
                0xFF666666,
            );
        }
    }

    // Fire hydrant water bursts
    for inter in &game.world.interactibles {
        if inter.kind != InteractibleKind::FireHydrant {
            continue;
        }
        if inter.state_val <= 0.0 {
            continue;
        }
        if frame % 2 == 0 {
            let vx = ps.emission_rng.range(-1.5, 1.5);
            let vz = ps.emission_rng.range(-1.5, 1.5);
            let vy = 6.0 + ps.emission_rng.range(0.0, 3.0);
            ps.emit(inter.x, inter.y + 0.5, inter.z, vx, vy, vz, 1.0, 0xFF3388FF);
        }
    }

    // Smokestack smoke (dockyard)
    // Smokestacks are tall structures in the dockyard; emit from top
    if frame % 4 == 0 {
        // Two smokestacks at known positions (matching world.rs)
        let stacks = [(-25.0f32, DOCK_Z_START + 20.0), (25.0, DOCK_Z_START + 20.0)];
        for &(sx, sz) in &stacks {
            let vx = ps.emission_rng.range(-0.3, 0.3);
            let vz = ps.emission_rng.range(-0.3, 0.3);
            ps.emit(sx, 12.0, sz, vx, 1.5, vz, 2.5, 0xFF444444);
        }
    }

    // Sprint dust
    let p = &game.player;
    if p.sprinting && p.in_vehicle.is_none() {
        if frame % 2 == 0 {
            let dx = ps.emission_rng.range(-0.75, 0.75);
            let dz = ps.emission_rng.range(-0.45, 0.45);
            ps.emit(p.x, p.y + 0.05, p.z, dx, 0.3, dz, 0.5, 0xFF998866);
        }
    }
}

// Emit pickup sparkles (call when item is picked up)
pub fn emit_pickup_sparkle(ps: &mut ParticleSystem, x: f32, z: f32, color: u32) {
    for i in 0..8 {
        let angle = i as f32 * std::f32::consts::TAU / 8.0;
        let speed = 2.0;
        let vx = angle.cos() * speed;
        let vz = angle.sin() * speed;
        ps.emit(x, 0.5, z, vx, 3.0, vz, 0.6, color);
    }
}

// Render particles as small screen-space quads
pub fn sys_render_particles(fb: &mut Framebuffer, ps: &ParticleSystem, cam: &Camera) {
    let aspect = fb.w as f32 / fb.h as f32;
    let eye = v3(cam.x, cam.y, cam.z);
    let target = v3(cam.tx, cam.ty, cam.tz);
    let view = m4_look_at(eye, target, v3(0.0, 1.0, 0.0));
    let proj = m4_perspective(60.0_f32.to_radians(), aspect, 0.1, 200.0);
    let vp = m4_mul(&proj, &view);
    let w = fb.w as f32;
    let h = fb.h as f32;

    for i in 0..ps.count {
        if ps.lifetime[i] <= 0.0 {
            continue;
        }

        let pos = [ps.pos_x[i], ps.pos_y[i], ps.pos_z[i]];
        let clip = m4_transform_no_div(&vp, pos);
        if clip[3] <= 0.01 {
            continue;
        }

        let scr = clip_to_screen(clip, w, h);
        if scr[0] < 0.0
            || scr[0] >= w
            || scr[1] < 0.0
            || scr[1] >= h
            || scr[2] < 0.0
            || scr[2] > 1.0
        {
            continue;
        }

        // Particle size based on distance (2-4 pixels)
        let size = ((3.0 / clip[3]).clamp(1.0, 4.0)) as usize;
        let px = scr[0] as usize;
        let py = scr[1] as usize;

        // Fade alpha based on remaining lifetime
        let alpha = (ps.lifetime[i] * 2.0).clamp(0.0, 1.0);
        let color = darken(ps.color[i], alpha);

        for dy in 0..size {
            for dx in 0..size {
                let x = px + dx;
                let y = py + dy;
                if x < fb.w && y < fb.h {
                    fb.put_pixel_overlay(x, y, color);
                }
            }
        }
    }
}
