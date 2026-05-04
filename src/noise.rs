//! Pure Rust procedural noise for terrain and world generation.
//! Hash-based value noise with fBm, ridged multifractal, and domain warping.

/// Deterministic hash of 2D integer coordinates + seed → float in [-1, 1]
pub fn hash_2d(ix: i32, iy: i32, seed: u64) -> f32 {
    let mut h = seed
        .wrapping_add((ix as i64 as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add((iy as i64 as u64).wrapping_mul(0x517CC1B727220A95));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31;
    (h as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Quintic Hermite interpolation (C2 continuous — no visible grid artifacts)
fn quintic(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// 2D value noise with quintic interpolation. Returns value in [-1, 1].
pub fn value_noise_2d(x: f32, y: f32, seed: u64) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let sx = quintic(fx);
    let sy = quintic(fy);

    let v00 = hash_2d(ix, iy, seed);
    let v10 = hash_2d(ix + 1, iy, seed);
    let v01 = hash_2d(ix, iy + 1, seed);
    let v11 = hash_2d(ix + 1, iy + 1, seed);

    let a = v00 + (v10 - v00) * sx;
    let b = v01 + (v11 - v01) * sx;
    a + (b - a) * sy
}

/// Fractal Brownian motion — layered noise at increasing frequency.
/// Returns value approximately in [-1, 1].
pub fn fbm(
    x: f32,
    y: f32,
    octaves: u32,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    seed: u64,
) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = frequency;
    let mut max_amp = 0.0;
    for i in 0..octaves {
        sum += value_noise_2d(x * freq, y * freq, seed.wrapping_add(i as u64 * 31337)) * amp;
        max_amp += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    sum / max_amp
}

/// Ridged multifractal — produces sharp ridges and valleys.
/// Returns value approximately in [0, 1].
pub fn ridged(
    x: f32,
    y: f32,
    octaves: u32,
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    seed: u64,
) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut freq = frequency;
    let mut max_amp = 0.0;
    for i in 0..octaves {
        let n = value_noise_2d(x * freq, y * freq, seed.wrapping_add(i as u64 * 31337));
        sum += (1.0 - n.abs()) * amp;
        max_amp += amp;
        freq *= lacunarity;
        amp *= gain;
    }
    sum / max_amp
}

/// Domain warping — displaces input coordinates by noise for organic distortion.
/// Returns warped (x, y) coordinates.
pub fn warp_2d(x: f32, y: f32, strength: f32, frequency: f32, seed: u64) -> (f32, f32) {
    let wx = value_noise_2d(x * frequency, y * frequency, seed) * strength;
    let wy = value_noise_2d(x * frequency, y * frequency, seed.wrapping_add(99999)) * strength;
    (x + wx, y + wy)
}
