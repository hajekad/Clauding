//! NEAT — NeuroEvolution of Augmenting Topologies — drives NPC high-level decisions.
//! The evolved network observes the world, handles combat/sounds, and picks
//! the daily job; job-specific work behavior is delegated to `jobs.rs`.

use crate::npc::npc_walk_toward;
use crate::rng::Rng;
use crate::state::*;
use crate::world::surface_at;

// ---- Constants ----

pub const NUM_INPUTS: u16 = 53;
pub const NUM_OUTPUTS: u16 = 14; // 0-12: movement/combat/sound, 13: job_choice

pub const ALL_JOBS: [NpcJob; NPC_JOB_COUNT] = [
    NpcJob::Collector,
    NpcJob::GarbageCollector,
    NpcJob::TaxiDriver,
    NpcJob::DeliveryCourier,
    NpcJob::MailCarrier,
    NpcJob::Paramedic,
    NpcJob::Firefighter,
    NpcJob::PolicePatrol,
    NpcJob::StreetVendor,
    NpcJob::Mechanic,
    NpcJob::ConstructionWorker,
    NpcJob::Fisherman,
    NpcJob::Farmer,
    NpcJob::Lumberjack,
    NpcJob::Scavenger,
];
const WEIGHT_CLAMP: f32 = 8.0;

// Compatibility distance coefficients
const C1_EXCESS: f32 = 1.0;
const C2_DISJOINT: f32 = 1.0;
const C3_WEIGHT: f32 = 0.4;

// Mutation rates — conservative to preserve seeded brain behavior
const RATE_WEIGHT_PERTURB: f32 = 0.30;
const RATE_WEIGHT_RESET: f32 = 0.02;
const RATE_ADD_CONNECTION: f32 = 0.02;
const RATE_ADD_NODE: f32 = 0.005;
const RATE_TOGGLE: f32 = 0.005;

// Evolution parameters
const ELITE_FRACTION: f32 = 0.50; // 50% survive unmutated
const CROSSOVER_FRACTION: f32 = 0.75;
const TARGET_SPECIES_MIN: usize = 3;
const TARGET_SPECIES_MAX: usize = 6;
const STAGNATION_LIMIT: u16 = 15;

// Fitness weights
const FIT_PICKUP: f32 = 15.0; // dominant reward — item pickup is the primary goal
const FIT_INTERACT: f32 = 0.5;
const FIT_DISTANCE: f32 = 0.001; // minimal: just breaks ties
const FIT_STUCK_PENALTY: f32 = 0.5;
const FIT_KNOCKOUT_PENALTY: f32 = 2.0;
const FIT_HITS_LANDED: f32 = 0.5;

// ---- Activation functions ----

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-4.9 * x).exp())
}

fn fast_tanh(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

// ---- Gene types ----

#[derive(Clone, Copy, PartialEq)]
pub enum NodeKind {
    Input,
    Output,
    Hidden,
}

#[derive(Clone, Copy)]
pub struct NodeGene {
    pub id: u16,
    pub kind: NodeKind,
}

#[derive(Clone, Copy)]
pub struct ConnectionGene {
    pub in_node: u16,
    pub out_node: u16,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: u32,
}

#[derive(Clone)]
pub struct Genome {
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    pub fitness: f32,
    pub species_id: u16,
    pub adjusted_fitness: f32,
}

impl Genome {
    fn new_minimal(rng: &mut Rng, innovation: &mut InnovationTracker) -> Self {
        let mut nodes = Vec::with_capacity((NUM_INPUTS + NUM_OUTPUTS) as usize);
        for i in 0..NUM_INPUTS {
            nodes.push(NodeGene {
                id: i,
                kind: NodeKind::Input,
            });
        }
        for i in 0..NUM_OUTPUTS {
            nodes.push(NodeGene {
                id: NUM_INPUTS + i,
                kind: NodeKind::Output,
            });
        }
        // Random sparse connections: each output gets 2-4 random input connections
        let mut connections = Vec::new();
        for o in 0..NUM_OUTPUTS {
            let out_id = NUM_INPUTS + o;
            let num_conns = 2 + (rng.next() % 3) as u16;
            for _ in 0..num_conns {
                let in_id = rng.next() as u16 % NUM_INPUTS;
                let innov = innovation.next(in_id, out_id);
                connections.push(ConnectionGene {
                    in_node: in_id,
                    out_node: out_id,
                    weight: rng.range(-2.0, 2.0),
                    enabled: true,
                    innovation: innov,
                });
            }
        }
        Genome {
            nodes,
            connections,
            fitness: 0.0,
            species_id: 0,
            adjusted_fitness: 0.0,
        }
    }

    fn new_seeded_collector(rng: &mut Rng, innovation: &mut InnovationTracker) -> Self {
        let mut g = Self::new_minimal(rng, innovation);
        g.connections.clear();

        // Input indices (from gather_inputs):
        // 0=carrying_item, 6-7=nearest_item_dx/dz, 12-13=nearest_bin_dx/dz
        // Outputs: 0=walk_dx, 1=walk_dz, 2=walk_mag, 3=pickup, 4=deposit, ..., 13=job_choice
        // Direction inputs normalized by /250.0 (WORLD_HALF), weights scaled accordingly

        // Walk toward nearest item (inputs 6,7 -> outputs 0,1)
        let conns = [
            (6, 0, 3.75), // item_dx -> walk_dx (scaled for /250 normalization)
            (7, 1, 3.75), // item_dz -> walk_dz
            (27, 2, 1.0), // bias -> walk_magnitude (always walk)
            (27, 3, 2.0), // bias -> pickup tendency (strong: always try to pick up)
            (0, 4, 2.0),  // carrying_item -> deposit
            // When carrying, walk toward bin instead
            (0, 0, -3.75), // carrying_item suppresses item direction
            (12, 0, 3.75), // bin_dx -> walk_dx (when carrying, bin signal dominates)
            (13, 1, 3.75), // bin_dz -> walk_dz
            // Weak defensive: low health slightly inhibits attacking
            (28, 8, -0.3), // self_health -> attack_player (low health = less attack)
            (28, 9, -0.3), // self_health -> attack_npc
            // (hunger/thirst survival handled by traditional AI autopilot)
            // Sound/vision communication
            (43, 0, 0.5),   // hear0_dx -> walk_dx (approach sound sources)
            (44, 1, 0.5),   // hear0_dz -> walk_dz
            (27, 10, -0.5), // bias -> sound_0 (suppress constant noise)
            (35, 0, 0.3),   // vis0_dx -> walk_dx (approach visible NPCs)
            (36, 1, 0.3),   // vis0_dz -> walk_dz
            (38, 2, -0.3),  // vis0_health -> walk_mag (avoid injured NPCs)
            // Job choice (output 13): neutral start, NN learns which jobs earn most
            (27, 13, 0.0), // bias -> job_choice (neutral)
        ];
        for &(inp, out, w) in &conns {
            let innov = innovation.next(inp, NUM_INPUTS + out);
            g.connections.push(ConnectionGene {
                in_node: inp,
                out_node: NUM_INPUTS + out,
                weight: w + rng.range(-0.3, 0.3),
                enabled: true,
                innovation: innov,
            });
        }
        g
    }

    fn next_node_id(&self) -> u16 {
        self.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1
    }
}

// ---- Innovation tracker ----

pub struct InnovationTracker {
    counter: u32,
    history: Vec<(u16, u16, u32)>,
}

impl InnovationTracker {
    pub fn new() -> Self {
        InnovationTracker {
            counter: 0,
            history: Vec::new(),
        }
    }

    pub fn next(&mut self, in_node: u16, out_node: u16) -> u32 {
        // Check if this structural innovation already happened this generation
        for &(i, o, innov) in &self.history {
            if i == in_node && o == out_node {
                return innov;
            }
        }
        let innov = self.counter;
        self.counter += 1;
        self.history.push((in_node, out_node, innov));
        innov
    }

    pub fn reset_generation(&mut self) {
        self.history.clear();
    }
}

// ---- Compiled brain for fast forward pass ----

pub struct NeatBrain {
    pub node_values: Vec<f32>,
    node_activations: Vec<u8>, // 0=linear(input), 1=sigmoid, 2=tanh
    conn_from: Vec<u16>,
    conn_to: Vec<u16>,
    conn_weight: Vec<f32>,
    num_inputs: u16,
    _num_outputs: u16,
}

impl NeatBrain {
    pub fn compile(genome: &Genome) -> Self {
        // Build node id -> index map
        let mut node_ids: Vec<u16> = genome.nodes.iter().map(|n| n.id).collect();
        node_ids.sort();

        let n = node_ids.len();
        let node_values = vec![0.0f32; n];
        let mut node_activations = vec![0u8; n];

        let id_to_idx = |id: u16| -> Option<u16> {
            node_ids.iter().position(|&nid| nid == id).map(|p| p as u16)
        };

        // Set activations: inputs=linear, first 2 outputs=tanh, rest=sigmoid, hidden=sigmoid
        for (idx, &nid) in node_ids.iter().enumerate() {
            let node = genome.nodes.iter().find(|n| n.id == nid).unwrap();
            node_activations[idx] = match node.kind {
                NodeKind::Input => 0,
                NodeKind::Output => {
                    if nid < NUM_INPUTS + 2 { 2 } else { 1 } // walk dx/dz use tanh
                }
                NodeKind::Hidden => 1,
            };
        }

        // Collect enabled connections, mapped to indices
        let mut conn_from = Vec::new();
        let mut conn_to = Vec::new();
        let mut conn_weight = Vec::new();

        for c in &genome.connections {
            if !c.enabled {
                continue;
            }
            if let (Some(fi), Some(ti)) = (id_to_idx(c.in_node), id_to_idx(c.out_node)) {
                conn_from.push(fi);
                conn_to.push(ti);
                conn_weight.push(c.weight);
            }
        }

        NeatBrain {
            node_values,
            node_activations,
            conn_from,
            conn_to,
            conn_weight,
            num_inputs: NUM_INPUTS,
            _num_outputs: NUM_OUTPUTS,
        }
    }

    pub fn activate(&mut self, inputs: &[f32]) -> [f32; NUM_OUTPUTS as usize] {
        let n = self.node_values.len();
        let ni = self.num_inputs as usize;

        // Clear non-input nodes
        for i in ni..n {
            self.node_values[i] = 0.0;
        }
        // Set inputs
        for i in 0..ni.min(inputs.len()) {
            self.node_values[i] = inputs[i];
        }
        // Sum connections
        let nc = self.conn_from.len();
        for c in 0..nc {
            let from_val = self.node_values[self.conn_from[c] as usize];
            self.node_values[self.conn_to[c] as usize] += from_val * self.conn_weight[c];
        }
        // Apply activations
        for i in ni..n {
            self.node_values[i] = match self.node_activations[i] {
                2 => fast_tanh(self.node_values[i]),
                1 => sigmoid(self.node_values[i]),
                _ => self.node_values[i],
            };
        }
        // Extract outputs (nodes at indices ni..ni+num_outputs)
        let no = NUM_OUTPUTS as usize;
        let mut out = [0.0f32; NUM_OUTPUTS as usize];
        out.copy_from_slice(&self.node_values[ni..ni + no]);
        out
    }
}

// ---- Species ----

pub struct Species {
    pub id: u16,
    pub representative: usize, // genome index
    pub members: Vec<usize>,
    pub best_fitness: f32,
    pub stagnation: u16,
}

// ---- Population ----

pub struct Population {
    pub genomes: Vec<Genome>,
    pub species: Vec<Species>,
    pub innovation: InnovationTracker,
    pub generation: u32,
    pub rng: Rng,
    pub compat_threshold: f32,
    next_species_id: u16,
}

impl Population {
    pub fn new(pop_size: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut innovation = InnovationTracker::new();

        let mut genomes = Vec::with_capacity(pop_size);
        for _ in 0..pop_size {
            // All genomes start as seeded collectors for reliable initial behavior
            genomes.push(Genome::new_seeded_collector(&mut rng, &mut innovation));
        }

        Population {
            genomes,
            species: Vec::new(),
            innovation,
            generation: 0,
            rng,
            compat_threshold: 3.0,
            next_species_id: 1,
        }
    }

    pub fn evolve(&mut self, fitnesses: &[f32]) {
        let pop_size = self.genomes.len();

        // 1. Assign fitness
        for (i, g) in self.genomes.iter_mut().enumerate() {
            g.fitness = if i < fitnesses.len() {
                fitnesses[i]
            } else {
                0.0
            };
        }

        // 2. Speciate
        self.speciate();

        // 3. Adjust fitness (explicit fitness sharing)
        for sp in &self.species {
            let size = sp.members.len().max(1) as f32;
            for &mi in &sp.members {
                self.genomes[mi].adjusted_fitness = self.genomes[mi].fitness / size;
            }
        }

        // 4. Calculate offspring allocation
        let total_adj: f32 = self
            .genomes
            .iter()
            .map(|g| g.adjusted_fitness.max(0.0))
            .sum();
        let mut offspring_counts: Vec<usize> = Vec::new();
        let mut total_allocated = 0;

        for sp in &self.species {
            if sp.stagnation >= STAGNATION_LIMIT && self.species.len() > 1 {
                offspring_counts.push(0);
                continue;
            }
            let sp_adj: f32 = sp
                .members
                .iter()
                .map(|&mi| self.genomes[mi].adjusted_fitness.max(0.0))
                .sum();
            let count = if total_adj > 0.0 {
                ((sp_adj / total_adj) * pop_size as f32).round() as usize
            } else {
                1
            };
            let count = count.max(1);
            offspring_counts.push(count);
            total_allocated += count;
        }

        // Adjust to match pop_size exactly
        while total_allocated > pop_size {
            // Remove from largest species
            if let Some(max_idx) = offspring_counts
                .iter()
                .enumerate()
                .filter(|&(_, c)| *c > 1)
                .max_by_key(|&(_, c)| *c)
                .map(|(i, _)| i)
            {
                offspring_counts[max_idx] -= 1;
                total_allocated -= 1;
            } else {
                break;
            }
        }
        while total_allocated < pop_size {
            if let Some(idx) = offspring_counts.iter().position(|&c| c > 0) {
                offspring_counts[idx] += 1;
                total_allocated += 1;
            } else {
                break;
            }
        }

        // 5. Reproduce
        self.innovation.reset_generation();
        let mut new_genomes = Vec::with_capacity(pop_size);

        for (si, sp) in self.species.iter().enumerate() {
            let count = offspring_counts[si];
            if count == 0 {
                continue;
            }

            // Sort members by fitness (descending)
            let mut sorted_members = sp.members.clone();
            sorted_members.sort_by(|&a, &b| {
                self.genomes[b]
                    .fitness
                    .partial_cmp(&self.genomes[a].fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Elites
            let num_elites = ((count as f32 * ELITE_FRACTION).ceil() as usize)
                .min(sorted_members.len())
                .max(1);
            for &mi in sorted_members.iter().take(num_elites) {
                if new_genomes.len() < pop_size {
                    new_genomes.push(self.genomes[mi].clone());
                }
            }

            // Remaining offspring
            let remaining = count.saturating_sub(num_elites);
            for _ in 0..remaining {
                if new_genomes.len() >= pop_size {
                    break;
                }

                let use_crossover =
                    self.rng.f32() < CROSSOVER_FRACTION && sorted_members.len() >= 2;
                let child = if use_crossover {
                    let pi1 = sorted_members[self.rng.next() as usize % sorted_members.len()];
                    let pi2 = sorted_members[self.rng.next() as usize % sorted_members.len()];
                    let (p1, p2) = if self.genomes[pi1].fitness >= self.genomes[pi2].fitness {
                        (&self.genomes[pi1], &self.genomes[pi2])
                    } else {
                        (&self.genomes[pi2], &self.genomes[pi1])
                    };
                    let mut c = crossover(p1, p2, &mut self.rng);
                    mutate(&mut c, &mut self.rng, &mut self.innovation);
                    c
                } else {
                    let pi = sorted_members[self.rng.next() as usize % sorted_members.len()];
                    let mut c = self.genomes[pi].clone();
                    mutate(&mut c, &mut self.rng, &mut self.innovation);
                    c
                };
                new_genomes.push(child);
            }
        }

        // Fill any remaining slots
        while new_genomes.len() < pop_size {
            let i = self.rng.next() as usize % self.genomes.len();
            new_genomes.push(self.genomes[i].clone());
        }
        new_genomes.truncate(pop_size);

        self.genomes = new_genomes;
        self.generation += 1;

        // Dynamic threshold adjustment
        let num_species = self.species.len();
        if num_species > TARGET_SPECIES_MAX {
            self.compat_threshold += 0.3;
        } else if num_species < TARGET_SPECIES_MIN {
            self.compat_threshold = (self.compat_threshold - 0.3).max(0.5);
        }
    }

    fn speciate(&mut self) {
        // Update stagnation for existing species
        for sp in &mut self.species {
            let prev_best = sp.best_fitness;
            sp.best_fitness = sp
                .members
                .iter()
                .map(|&mi| self.genomes[mi].fitness)
                .fold(f32::MIN, f32::max);
            if sp.best_fitness <= prev_best {
                sp.stagnation += 1;
            } else {
                sp.stagnation = 0;
            }
            sp.members.clear();
        }

        // Assign each genome to a species
        for gi in 0..self.genomes.len() {
            let mut placed = false;
            for sp in &mut self.species {
                let rep_idx = sp.representative;
                if rep_idx < self.genomes.len() {
                    let dist = compatibility_distance(&self.genomes[gi], &self.genomes[rep_idx]);
                    if dist < self.compat_threshold {
                        sp.members.push(gi);
                        self.genomes[gi].species_id = sp.id;
                        placed = true;
                        break;
                    }
                }
            }
            if !placed {
                let sid = self.next_species_id;
                self.next_species_id += 1;
                self.species.push(Species {
                    id: sid,
                    representative: gi,
                    members: vec![gi],
                    best_fitness: self.genomes[gi].fitness,
                    stagnation: 0,
                });
                self.genomes[gi].species_id = sid;
            }
        }

        // Remove empty species, update representatives
        self.species.retain(|sp| !sp.members.is_empty());
        for sp in &mut self.species {
            let ri = sp.members[self.rng.next() as usize % sp.members.len()];
            sp.representative = ri;
        }
    }
}

// ---- Genetic operators ----

fn compatibility_distance(g1: &Genome, g2: &Genome) -> f32 {
    let c1 = &g1.connections;
    let c2 = &g2.connections;
    if c1.is_empty() && c2.is_empty() {
        return 0.0;
    }

    let mut i = 0usize;
    let mut j = 0usize;
    let mut matching = 0u32;
    let mut disjoint = 0u32;
    let mut weight_diff_sum = 0.0f32;

    while i < c1.len() && j < c2.len() {
        if c1[i].innovation == c2[j].innovation {
            matching += 1;
            weight_diff_sum += (c1[i].weight - c2[j].weight).abs();
            i += 1;
            j += 1;
        } else if c1[i].innovation < c2[j].innovation {
            disjoint += 1;
            i += 1;
        } else {
            disjoint += 1;
            j += 1;
        }
    }

    let excess = (c1.len() - i + c2.len() - j) as u32;
    let n = c1.len().max(c2.len()).max(1) as f32;
    let w_avg = if matching > 0 {
        weight_diff_sum / matching as f32
    } else {
        0.0
    };

    C1_EXCESS * excess as f32 / n + C2_DISJOINT * disjoint as f32 / n + C3_WEIGHT * w_avg
}

fn crossover(parent1: &Genome, parent2: &Genome, rng: &mut Rng) -> Genome {
    // parent1 is the fitter parent
    let mut child_connections = Vec::new();
    let c1 = &parent1.connections;
    let c2 = &parent2.connections;

    let mut i = 0usize;
    let mut j = 0usize;

    while i < c1.len() && j < c2.len() {
        if c1[i].innovation == c2[j].innovation {
            // Matching: random parent
            let c = if rng.next() % 2 == 0 { c1[i] } else { c2[j] };
            child_connections.push(c);
            i += 1;
            j += 1;
        } else if c1[i].innovation < c2[j].innovation {
            // Disjoint from fitter parent
            child_connections.push(c1[i]);
            i += 1;
        } else {
            // Disjoint from less fit parent — skip
            j += 1;
        }
    }
    // Excess from fitter parent
    while i < c1.len() {
        child_connections.push(c1[i]);
        i += 1;
    }

    // Collect nodes referenced by child connections
    let mut node_ids: Vec<u16> = Vec::new();
    for c in &child_connections {
        if !node_ids.contains(&c.in_node) {
            node_ids.push(c.in_node);
        }
        if !node_ids.contains(&c.out_node) {
            node_ids.push(c.out_node);
        }
    }
    // Ensure all input/output nodes exist
    for id in 0..NUM_INPUTS + NUM_OUTPUTS {
        if !node_ids.contains(&id) {
            node_ids.push(id);
        }
    }
    node_ids.sort();

    let nodes: Vec<NodeGene> = node_ids
        .iter()
        .map(|&id| {
            // Find node in parent1 first, then parent2
            parent1
                .nodes
                .iter()
                .find(|n| n.id == id)
                .or_else(|| parent2.nodes.iter().find(|n| n.id == id))
                .copied()
                .unwrap_or(NodeGene {
                    id,
                    kind: NodeKind::Hidden,
                })
        })
        .collect();

    Genome {
        nodes,
        connections: child_connections,
        fitness: 0.0,
        species_id: parent1.species_id,
        adjusted_fitness: 0.0,
    }
}

fn mutate(genome: &mut Genome, rng: &mut Rng, innovation: &mut InnovationTracker) {
    // Weight perturbation
    for c in &mut genome.connections {
        let r = rng.f32();
        if r < RATE_WEIGHT_PERTURB {
            c.weight += rng.range(-0.1, 0.1);
            c.weight = c.weight.clamp(-WEIGHT_CLAMP, WEIGHT_CLAMP);
        } else if r < RATE_WEIGHT_PERTURB + RATE_WEIGHT_RESET {
            c.weight = rng.range(-2.0, 2.0);
        }
    }

    // Add connection
    if rng.f32() < RATE_ADD_CONNECTION {
        mutate_add_connection(genome, rng, innovation);
    }

    // Add node
    if rng.f32() < RATE_ADD_NODE {
        mutate_add_node(genome, rng, innovation);
    }

    // Toggle connection
    if rng.f32() < RATE_TOGGLE && !genome.connections.is_empty() {
        let idx = rng.next() as usize % genome.connections.len();
        genome.connections[idx].enabled = !genome.connections[idx].enabled;
    }
}

fn mutate_add_connection(genome: &mut Genome, rng: &mut Rng, innovation: &mut InnovationTracker) {
    // Try up to 20 times to find a valid new connection
    for _ in 0..20 {
        let from = genome.nodes[rng.next() as usize % genome.nodes.len()];
        let to = genome.nodes[rng.next() as usize % genome.nodes.len()];
        if from.id == to.id {
            continue;
        }
        if to.kind == NodeKind::Input {
            continue;
        }
        if from.kind == NodeKind::Output && to.kind == NodeKind::Output {
            continue;
        }
        // Check if connection already exists
        let exists = genome
            .connections
            .iter()
            .any(|c| c.in_node == from.id && c.out_node == to.id);
        if exists {
            continue;
        }

        let innov = innovation.next(from.id, to.id);
        genome.connections.push(ConnectionGene {
            in_node: from.id,
            out_node: to.id,
            weight: rng.range(-2.0, 2.0),
            enabled: true,
            innovation: innov,
        });
        // Keep connections sorted by innovation
        genome.connections.sort_by_key(|c| c.innovation);
        return;
    }
}

fn mutate_add_node(genome: &mut Genome, rng: &mut Rng, innovation: &mut InnovationTracker) {
    let enabled: Vec<usize> = genome
        .connections
        .iter()
        .enumerate()
        .filter(|(_, c)| c.enabled)
        .map(|(i, _)| i)
        .collect();
    if enabled.is_empty() {
        return;
    }

    let ci = enabled[rng.next() as usize % enabled.len()];
    let old_conn = genome.connections[ci];
    genome.connections[ci].enabled = false;

    let new_id = genome.next_node_id();
    genome.nodes.push(NodeGene {
        id: new_id,
        kind: NodeKind::Hidden,
    });

    let innov1 = innovation.next(old_conn.in_node, new_id);
    let innov2 = innovation.next(new_id, old_conn.out_node);

    genome.connections.push(ConnectionGene {
        in_node: old_conn.in_node,
        out_node: new_id,
        weight: 1.0,
        enabled: true,
        innovation: innov1,
    });
    genome.connections.push(ConnectionGene {
        in_node: new_id,
        out_node: old_conn.out_node,
        weight: old_conn.weight,
        enabled: true,
        innovation: innov2,
    });
    genome.connections.sort_by_key(|c| c.innovation);
}

// ---- Input gathering ----

pub fn gather_inputs(
    world: &WorldData,
    i: usize,
    net: &RoadNetwork,
    time_of_day: f32,
    player_x: f32,
    player_z: f32,
) -> [f32; NUM_INPUTS as usize] {
    let npc = &world.npcs[i];
    let mut inp = [0.0f32; NUM_INPUTS as usize];

    // Self state (0-5)
    inp[0] = if npc.carrying_item { 1.0 } else { 0.0 };
    inp[1] = if npc.carrying_bin.is_some() { 1.0 } else { 0.0 };
    inp[2] = (npc.money / 50.0).min(1.0);
    inp[3] = npc.state_timer / WORK_DURATION;
    inp[4] = (npc.stuck_timer / 5.0).min(1.0);
    inp[5] = if npc.on_ground { 1.0 } else { 0.0 };

    // Nearest 3 items (6-11) — inline top-3 tracking, no allocation
    let mut top3_idx = [usize::MAX; 3];
    let mut top3_d2 = [f32::MAX; 3];
    for (ii, item) in world.items.iter().enumerate() {
        if !item.active || item.falling {
            continue;
        }
        let dx = item.x - npc.x;
        let dz = item.z - npc.z;
        let d2 = dx * dx + dz * dz;
        if d2 < top3_d2[2] {
            if d2 < top3_d2[0] {
                top3_d2[2] = top3_d2[1];
                top3_idx[2] = top3_idx[1];
                top3_d2[1] = top3_d2[0];
                top3_idx[1] = top3_idx[0];
                top3_d2[0] = d2;
                top3_idx[0] = ii;
            } else if d2 < top3_d2[1] {
                top3_d2[2] = top3_d2[1];
                top3_idx[2] = top3_idx[1];
                top3_d2[1] = d2;
                top3_idx[1] = ii;
            } else {
                top3_d2[2] = d2;
                top3_idx[2] = ii;
            }
        }
    }
    for k in 0..3 {
        if top3_idx[k] != usize::MAX {
            let item = &world.items[top3_idx[k]];
            inp[6 + k * 2] = ((item.x - npc.x) / 250.0).clamp(-1.0, 1.0);
            inp[7 + k * 2] = ((item.z - npc.z) / 250.0).clamp(-1.0, 1.0);
        }
    }

    // Nearest bin (12-14): dx, dz, items_held
    let mut best_bin_dist = f32::MAX;
    let mut best_bin = None;
    for (bi, bin) in world.trash_bins.iter().enumerate() {
        if bin.carried_by.is_some() {
            continue;
        }
        let dx = bin.x - npc.x;
        let dz = bin.z - npc.z;
        let d = dx * dx + dz * dz;
        if d < best_bin_dist {
            best_bin_dist = d;
            best_bin = Some(bi);
        }
    }
    if let Some(bi) = best_bin {
        inp[12] = ((world.trash_bins[bi].x - npc.x) / 250.0).clamp(-1.0, 1.0);
        inp[13] = ((world.trash_bins[bi].z - npc.z) / 250.0).clamp(-1.0, 1.0);
        inp[14] = (world.trash_bins[bi].items_held as f32 / 10.0).min(1.0);
    }

    // Nearest item active flag (15) — is there a pickable item within 2m?
    inp[15] = if top3_idx[0] != usize::MAX && top3_d2[0] < NPC_PICKUP_DIST * NPC_PICKUP_DIST {
        1.0
    } else {
        0.0
    };

    // Nearest job-relevant interactible (16-17)
    let relevant_kind = job_relevant_interactible(npc.job);
    if let Some(kind) = relevant_kind {
        let mut best_dist = f32::MAX;
        let mut best_ix = None;
        for (ii, inter) in world.interactibles.iter().enumerate() {
            if inter.kind != kind {
                continue;
            }
            let dx = inter.x - npc.x;
            let dz = inter.z - npc.z;
            let d = dx * dx + dz * dz;
            if d < best_dist {
                best_dist = d;
                best_ix = Some(ii);
            }
        }
        if let Some(ii) = best_ix {
            inp[16] = ((world.interactibles[ii].x - npc.x) / 250.0).clamp(-1.0, 1.0);
            inp[17] = ((world.interactibles[ii].z - npc.z) / 250.0).clamp(-1.0, 1.0);
        }
    }

    // Single-pass NPC scan: nearest NPC (18-19), 2 visible (35-42), 2 audible (43-52)
    let (sin_r, cos_r) = npc.rot_y.sin_cos();
    let fwd_x = -sin_r;
    let fwd_z = -cos_r;
    let vis_range_sq = VISION_RANGE * VISION_RANGE;
    let hear_range_sq = SOUND_RANGE * SOUND_RANGE;
    let mut best_npc_dist = f32::MAX;
    let mut best_npc: Option<usize> = None;
    let mut vis0_dist = f32::MAX;
    let mut vis1_dist = f32::MAX;
    let mut vis0_idx: Option<usize> = None;
    let mut vis1_idx: Option<usize> = None;
    let mut hear0_dist = f32::MAX;
    let mut hear1_dist = f32::MAX;
    let mut hear0_idx: Option<usize> = None;
    let mut hear1_idx: Option<usize> = None;
    for j in 0..world.npcs.len() {
        if j == i {
            continue;
        }
        let other = &world.npcs[j];
        let dx = other.x - npc.x;
        let dz = other.z - npc.z;
        let d2 = dx * dx + dz * dz;
        // Nearest NPC (any state)
        if d2 < best_npc_dist {
            best_npc_dist = d2;
            best_npc = Some(j);
        }
        if other.state == NpcState::Sleeping {
            continue;
        }
        if d2 < 0.01 {
            continue;
        }
        // Vision: within range + forward cone
        if !other.in_vehicle && d2 <= vis_range_sq {
            let dist = d2.sqrt();
            let dot = (dx / dist) * fwd_x + (dz / dist) * fwd_z;
            if dot >= VISION_CONE_COS {
                if dist < vis0_dist {
                    vis1_dist = vis0_dist;
                    vis1_idx = vis0_idx;
                    vis0_dist = dist;
                    vis0_idx = Some(j);
                } else if dist < vis1_dist {
                    vis1_dist = dist;
                    vis1_idx = Some(j);
                }
            }
        }
        // Hearing: within range + making sound
        if d2 <= hear_range_sq {
            let total_sound = other.sound[0] + other.sound[1] + other.sound[2];
            if total_sound >= 0.01 {
                let dist = d2.sqrt();
                if dist < hear0_dist {
                    hear1_dist = hear0_dist;
                    hear1_idx = hear0_idx;
                    hear0_dist = dist;
                    hear0_idx = Some(j);
                } else if dist < hear1_dist {
                    hear1_dist = dist;
                    hear1_idx = Some(j);
                }
            }
        }
    }
    if let Some(j) = best_npc {
        inp[18] = ((world.npcs[j].x - npc.x) / 250.0).clamp(-1.0, 1.0);
        inp[19] = ((world.npcs[j].z - npc.z) / 250.0).clamp(-1.0, 1.0);
    }

    // Own vehicle (20-21)
    let car_idx = npc.car_idx.min(world.vehicles.len().saturating_sub(1));
    if car_idx < world.vehicles.len() {
        inp[20] = ((world.vehicles[car_idx].x - npc.x) / 250.0).clamp(-1.0, 1.0);
        inp[21] = ((world.vehicles[car_idx].z - npc.z) / 250.0).clamp(-1.0, 1.0);
    }

    // Home (22-23)
    let home = &world.buildings[npc.home_idx % world.buildings.len()];
    inp[22] = ((home.x - npc.x) / 250.0).clamp(-1.0, 1.0);
    inp[23] = ((home.z - npc.z) / 250.0).clamp(-1.0, 1.0);

    // Time, surface, job, bias (24-27)
    inp[24] = time_of_day / 24.0;
    inp[25] = match surface_at(npc.x, npc.z, net) {
        Surface::Sidewalk => 0.0,
        Surface::CarRoad => 0.33,
        Surface::FieldRoad => 0.66,
        Surface::Terrain => 1.0,
    };
    inp[26] = npc.job.index() as f32 / 14.0;
    inp[27] = 1.0; // bias

    // Combat inputs (28-32)
    inp[28] = npc.health / NPC_HEALTH_MAX;
    inp[29] = (npc.knockout_timer / KNOCKOUT_TIME).min(1.0);
    let pdx = player_x - npc.x;
    let pdz = player_z - npc.z;
    inp[30] = (pdx / 125.0).clamp(-1.0, 1.0);
    inp[31] = (pdz / 125.0).clamp(-1.0, 1.0);
    let player_dist = (pdx * pdx + pdz * pdz).sqrt();
    inp[32] = (player_dist / (ATTACK_RANGE * 5.0)).min(1.0);

    // Hunger/thirst (33-34)
    inp[33] = npc.hunger / 100.0;
    inp[34] = npc.thirst / 100.0;

    // Write vision results (35-42)
    if let Some(j) = vis0_idx {
        let other = &world.npcs[j];
        let dx = other.x - npc.x;
        let dz = other.z - npc.z;
        let dist = vis0_dist.max(0.01);
        inp[35] = (dx / dist).clamp(-1.0, 1.0);
        inp[36] = (dz / dist).clamp(-1.0, 1.0);
        inp[37] = if other.carrying_item { 1.0 } else { 0.0 };
        inp[38] = other.health / NPC_HEALTH_MAX;
    }
    if let Some(j) = vis1_idx {
        let other = &world.npcs[j];
        let dx = other.x - npc.x;
        let dz = other.z - npc.z;
        let dist = vis1_dist.max(0.01);
        inp[39] = (dx / dist).clamp(-1.0, 1.0);
        inp[40] = (dz / dist).clamp(-1.0, 1.0);
        inp[41] = if other.carrying_item { 1.0 } else { 0.0 };
        inp[42] = other.health / NPC_HEALTH_MAX;
    }

    // Write hearing results (43-52)
    if let Some(j) = hear0_idx {
        let other = &world.npcs[j];
        let dx = other.x - npc.x;
        let dz = other.z - npc.z;
        let dist = hear0_dist.max(0.01);
        let atten = 1.0 - dist / SOUND_RANGE;
        inp[43] = (dx / dist).clamp(-1.0, 1.0);
        inp[44] = (dz / dist).clamp(-1.0, 1.0);
        inp[45] = other.sound[0] * atten;
        inp[46] = other.sound[1] * atten;
        inp[47] = other.sound[2] * atten;
    }
    if let Some(j) = hear1_idx {
        let other = &world.npcs[j];
        let dx = other.x - npc.x;
        let dz = other.z - npc.z;
        let dist = hear1_dist.max(0.01);
        let atten = 1.0 - dist / SOUND_RANGE;
        inp[48] = (dx / dist).clamp(-1.0, 1.0);
        inp[49] = (dz / dist).clamp(-1.0, 1.0);
        inp[50] = other.sound[0] * atten;
        inp[51] = other.sound[1] * atten;
        inp[52] = other.sound[2] * atten;
    }

    inp
}

fn job_relevant_interactible(job: NpcJob) -> Option<InteractibleKind> {
    match job {
        NpcJob::GarbageCollector | NpcJob::Scavenger => Some(InteractibleKind::Dumpster),
        NpcJob::MailCarrier => Some(InteractibleKind::Mailbox),
        NpcJob::Firefighter => Some(InteractibleKind::FireHydrant),
        NpcJob::StreetVendor => Some(InteractibleKind::NewspaperStand),
        NpcJob::Mechanic => None, // uses vehicle inputs
        _ => None,
    }
}

// ---- Output execution ----

pub fn execute_outputs(
    world: &mut WorldData,
    i: usize,
    outputs: &[f32],
    net: &RoadNetwork,
    terrain: &Terrain,
    dt: f32,
    walk_grid: &crate::navmesh::WalkGrid,
) {
    if outputs.len() < 13 {
        return;
    }

    // Sound outputs (10-12) → NPC sound channels
    world.npcs[i].sound[0] = outputs[10];
    world.npcs[i].sound[1] = outputs[11];
    world.npcs[i].sound[2] = outputs[12];
    if outputs[10] > 0.1 || outputs[11] > 0.1 || outputs[12] > 0.1 {
        world.npcs[i].fitness_sounds_made += 1;
    }

    let walk_dx = outputs[0] * 2.0 - 1.0; // sigmoid 0-1 → signed -1..+1
    let walk_dz = outputs[1] * 2.0 - 1.0;
    let walk_mag = outputs[2];
    let do_pickup = outputs[3] > 0.5;
    let do_deposit = outputs[4] > 0.5;
    let do_pickup_bin = outputs[5] > 0.5;
    let do_place_bin = outputs[6] > 0.5;
    let do_interact = outputs[7] > 0.5;
    let do_attack_player = outputs[8] > 0.5;
    let do_attack_npc = outputs[9] > 0.5;

    // Action priority: deposit > place_bin > pickup > pickup_bin > interact > attack > walk

    // Deposit item at nearest bin
    if do_deposit && world.npcs[i].carrying_item {
        let mut best_d = NPC_BIN_DIST * NPC_BIN_DIST;
        let mut best_bi = None;
        for (bi, bin) in world.trash_bins.iter().enumerate() {
            if bin.carried_by.is_some() {
                continue;
            }
            let dx = world.npcs[i].x - bin.x;
            let dz = world.npcs[i].z - bin.z;
            let d = dx * dx + dz * dz;
            if d < best_d {
                best_d = d;
                best_bi = Some(bi);
            }
        }
        if let Some(bi) = best_bi {
            world.npcs[i].carrying_item = false;
            world.npcs[i].items_deposited_today += 1;
            world.npcs[i].money += 1.0;
            world.npcs[i].fitness_money_earned += 1.0;
            world.trash_bins[bi].items_held += 1;
            return;
        }
    }

    // Place bin
    if do_place_bin {
        if let Some(bi) = world.npcs[i].carrying_bin {
            if bi < world.trash_bins.len() {
                world.trash_bins[bi].x = world.npcs[i].x;
                world.trash_bins[bi].z = world.npcs[i].z;
                world.trash_bins[bi].y = terrain.height_at(world.npcs[i].x, world.npcs[i].z);
                world.trash_bins[bi].terrain_normal =
                    terrain.normal_at(world.npcs[i].x, world.npcs[i].z);
                world.trash_bins[bi].carried_by = None;
                world.npcs[i].carrying_bin = None;
            }
            return;
        }
    }

    // Pick up item
    if do_pickup && !world.npcs[i].carrying_item && world.npcs[i].carrying_bin.is_none() {
        let mut best_d = NPC_PICKUP_DIST * NPC_PICKUP_DIST;
        let mut best_ii = None;
        for (ii, item) in world.items.iter().enumerate() {
            if !item.active || item.falling {
                continue;
            }
            let dx = world.npcs[i].x - item.x;
            let dz = world.npcs[i].z - item.z;
            let d = dx * dx + dz * dz;
            if d < best_d {
                best_d = d;
                best_ii = Some(ii);
            }
        }
        if let Some(ii) = best_ii {
            let kind = world.items[ii].kind;
            world.items[ii].active = false;
            world.items[ii].claimed_by = None;
            world.npcs[i].fitness_items_picked += 1;
            // Food/Water: auto-consume, no carrying
            match kind {
                ItemKind::Food => {
                    world.npcs[i].hunger = (world.npcs[i].hunger + FOOD_RESTORE).min(100.0);
                }
                ItemKind::Water => {
                    world.npcs[i].thirst = (world.npcs[i].thirst + WATER_RESTORE).min(100.0);
                }
                _ => {
                    world.npcs[i].carrying_item = true;
                }
            }
            return;
        }
    }

    // Pick up bin
    if do_pickup_bin && world.npcs[i].carrying_bin.is_none() && !world.npcs[i].carrying_item {
        let mut best_d = NPC_BIN_DIST * NPC_BIN_DIST;
        let mut best_bi = None;
        for (bi, bin) in world.trash_bins.iter().enumerate() {
            if bin.carried_by.is_some() {
                continue;
            }
            let dx = world.npcs[i].x - bin.x;
            let dz = world.npcs[i].z - bin.z;
            let d = dx * dx + dz * dz;
            if d < best_d {
                best_d = d;
                best_bi = Some(bi);
            }
        }
        if let Some(bi) = best_bi {
            world.npcs[i].carrying_bin = Some(bi);
            world.trash_bins[bi].carried_by = Some(i);
            return;
        }
    }

    // Interact with nearest interactible
    if do_interact {
        let mut best_d = INTERACT_DIST * INTERACT_DIST;
        let mut best_ii = None;
        for (ii, inter) in world.interactibles.iter().enumerate() {
            if inter.cooldown > 0.0 {
                continue;
            }
            let dx = world.npcs[i].x - inter.x;
            let dz = world.npcs[i].z - inter.z;
            let d = dx * dx + dz * dz;
            if d < best_d {
                best_d = d;
                best_ii = Some(ii);
            }
        }
        if let Some(ii) = best_ii {
            world.interactibles[ii].cooldown = 5.0;
            world.npcs[i].money += 1.0;
            world.npcs[i].fitness_money_earned += 1.0;
            world.npcs[i].fitness_interactions += 1;
            return;
        }
    }

    // Attack intent (processed by combat.rs)
    if do_attack_player {
        world.npcs[i].attack_intent = 1;
        return;
    }
    if do_attack_npc {
        world.npcs[i].attack_intent = 2;
        return;
    }

    // Walk
    if walk_mag > 0.1 {
        let tx = (world.npcs[i].x + walk_dx * 15.0).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
        let tz = (world.npcs[i].z + walk_dz * 15.0).clamp(-WORLD_HALF + 5.0, WORLD_HALF - 5.0);
        npc_walk_toward(world, i, tx, tz, net, terrain, dt, walk_grid);
    }
}

// ---- Fitness evaluation ----

pub fn evaluate_fitness(npc: &Npc) -> f32 {
    // Primary: survival (keeping hunger/thirst bars full)
    let survival_bonus = (npc.hunger / 100.0 + npc.thirst / 100.0) * 15.0; // up to 30 for full bars
    let food_bonus = npc.fitness_items_picked as f32 * FIT_PICKUP;
    let money_bonus = npc.fitness_money_earned * 1.0;
    let deposit_bonus = npc.items_deposited_today as f32 * 2.0;
    let interact_bonus = npc.fitness_interactions as f32 * FIT_INTERACT;
    let distance_bonus = npc.fitness_distance * FIT_DISTANCE;
    let starve_penalty = npc.fitness_starve_time * 2.0;
    let stuck_penalty = (npc.fitness_stuck_time * FIT_STUCK_PENALTY).min(20.0); // capped
    let ko_penalty = npc.fitness_knockouts as f32 * FIT_KNOCKOUT_PENALTY;
    let hits_bonus = npc.fitness_hits_landed as f32 * FIT_HITS_LANDED;

    let comm_bonus = (npc.fitness_npcs_heard as f32 * 0.001).min(1.0);

    survival_bonus
        + food_bonus
        + money_bonus
        + deposit_bonus
        + interact_bonus
        + distance_bonus
        + hits_bonus
        + comm_bonus
        - starve_penalty
        - stuck_penalty
        - ko_penalty
}

// ---- Save/Load population (raw binary, no crates) ----

const NEAT_MAGIC: [u8; 4] = [b'N', b'E', b'A', b'T'];

pub fn save_population(path: &str, pop: &Population) {
    let mut buf: Vec<u8> = Vec::new();

    // Header: magic + generation + compat_threshold + innovation counter
    buf.extend_from_slice(&NEAT_MAGIC);
    buf.extend_from_slice(&pop.generation.to_le_bytes());
    buf.extend_from_slice(&pop.compat_threshold.to_le_bytes());
    buf.extend_from_slice(&pop.innovation.counter.to_le_bytes());
    buf.extend_from_slice(&(pop.genomes.len() as u32).to_le_bytes());

    for genome in &pop.genomes {
        // Node count + nodes
        buf.extend_from_slice(&(genome.nodes.len() as u32).to_le_bytes());
        for node in &genome.nodes {
            buf.extend_from_slice(&node.id.to_le_bytes());
            let kind_byte: u8 = match node.kind {
                NodeKind::Input => 0,
                NodeKind::Output => 1,
                NodeKind::Hidden => 2,
            };
            buf.push(kind_byte);
        }
        // Connection count + connections
        buf.extend_from_slice(&(genome.connections.len() as u32).to_le_bytes());
        for conn in &genome.connections {
            buf.extend_from_slice(&conn.in_node.to_le_bytes());
            buf.extend_from_slice(&conn.out_node.to_le_bytes());
            buf.extend_from_slice(&conn.weight.to_le_bytes());
            buf.push(if conn.enabled { 1 } else { 0 });
            buf.extend_from_slice(&conn.innovation.to_le_bytes());
        }
    }

    let _ = std::fs::write(path, &buf);
}

pub fn load_population(path: &str, pop_size: usize) -> Option<Population> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 20 {
        return None;
    }
    if &data[0..4] != &NEAT_MAGIC {
        return None;
    }

    let mut pos = 4;

    let generation = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
    pos += 4;
    let compat_threshold = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
    pos += 4;
    let innov_counter = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
    pos += 4;
    let genome_count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
    pos += 4;

    let mut genomes = Vec::with_capacity(genome_count);

    for _ in 0..genome_count {
        if pos + 4 > data.len() {
            return None;
        }
        let node_count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            if pos + 3 > data.len() {
                return None;
            }
            let id = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?);
            pos += 2;
            let kind = match data[pos] {
                0 => NodeKind::Input,
                1 => NodeKind::Output,
                _ => NodeKind::Hidden,
            };
            pos += 1;
            nodes.push(NodeGene { id, kind });
        }

        if pos + 4 > data.len() {
            return None;
        }
        let conn_count = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;

        let mut connections = Vec::with_capacity(conn_count);
        for _ in 0..conn_count {
            if pos + 11 > data.len() {
                return None;
            }
            let in_node = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?);
            pos += 2;
            let out_node = u16::from_le_bytes(data[pos..pos + 2].try_into().ok()?);
            pos += 2;
            let weight = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
            pos += 4;
            let enabled = data[pos] != 0;
            pos += 1;
            let innovation = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
            pos += 4;
            connections.push(ConnectionGene {
                in_node,
                out_node,
                weight,
                enabled,
                innovation,
            });
        }

        genomes.push(Genome {
            nodes,
            connections,
            fitness: 0.0,
            species_id: 0,
            adjusted_fitness: 0.0,
        });
    }

    // Pad or truncate to match pop_size
    while genomes.len() < pop_size {
        let src = genomes[genomes.len() % genome_count.max(1)].clone();
        genomes.push(src);
    }
    genomes.truncate(pop_size);

    let mut pop = Population {
        genomes,
        species: Vec::new(),
        innovation: InnovationTracker {
            counter: innov_counter,
            history: Vec::new(),
        },
        generation,
        rng: Rng::new(generation as u64 ^ 0xBEEF),
        compat_threshold,
        next_species_id: 1,
    };
    // Re-speciate loaded genomes
    pop.speciate();
    Some(pop)
}
