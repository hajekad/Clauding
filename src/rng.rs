//! Deterministic PRNG — LCG with 64-bit state.
//! All game randomness flows through this to keep runs reproducible from a seed.

#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    pub fn f32(&mut self) -> f32 {
        self.next() as f32 / u32::MAX as f32
    }

    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        min + self.f32() * (max - min)
    }

    pub fn pick<T: Copy>(&mut self, arr: &[T]) -> T {
        arr[self.next() as usize % arr.len()]
    }

    /// Create a child RNG with a deterministic seed derived from this RNG's state and an index.
    /// Used to give each entity its own independent stream.
    pub fn fork(&mut self, index: u64) -> Rng {
        let seed = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(index.wrapping_mul(2891336453));
        Rng::new(seed)
    }
}
