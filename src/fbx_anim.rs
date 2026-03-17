//! FBX Binary parser — extracts skeleton + animation data from Mixamo FBX files.
//! Pure Rust, no crates. Includes a minimal DEFLATE decoder for zlib-compressed arrays.

use std::collections::HashMap;

// ── Public types ────────────────────────────────────────────────────────

/// A single bone in the skeleton hierarchy
pub struct FbxBone {
    pub name: String,
    pub parent: Option<usize>,
    pub bind_translation: [f32; 3],
    pub bind_rotation: [f32; 3], // Euler degrees (XYZ order)
    pub pre_rotation: [f32; 3],  // FBX PreRotation (degrees, applied before Lcl Rotation)
}

/// The skeleton: ordered list of bones with parent indices
pub struct FbxSkeleton {
    pub bones: Vec<FbxBone>,
}

/// Per-bone animation channel: keyframe times + rotations (and optionally translations)
pub struct BoneChannel {
    pub bone_index: usize,
    pub times: Vec<f32>,                  // seconds
    pub translations: Option<Vec<[f32; 3]>>, // only Hips typically
    pub rotations: Vec<[f32; 3]>,         // Euler degrees per keyframe
}

/// A complete animation clip parsed from one FBX file
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub fps: f32,
    pub bone_channels: Vec<BoneChannel>,
    pub looping: bool,
}

// ── FBX time conversion ─────────────────────────────────────────────────

const FBX_TICKS_PER_SECOND: f64 = 46186158000.0;

// ── Internal FBX node representation ────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
enum FbxProp {
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Str(String),
    Raw(Vec<u8>),
    ArrI32(Vec<i32>),
    ArrI64(Vec<i64>),
    ArrF32(Vec<f32>),
    ArrF64(Vec<f64>),
}

struct FbxNode {
    name: String,
    props: Vec<FbxProp>,
    children: Vec<FbxNode>,
}

impl FbxNode {
    fn child(&self, name: &str) -> Option<&FbxNode> {
        self.children.iter().find(|c| c.name == name)
    }

    fn children_named(&self, name: &str) -> impl Iterator<Item = &FbxNode> {
        self.children.iter().filter(move |c| c.name == name)
    }

    fn prop_i64(&self, idx: usize) -> i64 {
        match &self.props[idx] {
            FbxProp::I64(v) => *v,
            FbxProp::I32(v) => *v as i64,
            _ => 0,
        }
    }

    fn prop_f64(&self, idx: usize) -> f64 {
        match &self.props[idx] {
            FbxProp::F64(v) => *v,
            FbxProp::F32(v) => *v as f64,
            FbxProp::I64(v) => *v as f64,
            FbxProp::I32(v) => *v as f64,
            _ => 0.0,
        }
    }

    fn prop_str(&self, idx: usize) -> &str {
        match &self.props[idx] {
            FbxProp::Str(s) => s,
            _ => "",
        }
    }

    fn prop_arr_i32(&self, idx: usize) -> &[i32] {
        match &self.props[idx] { FbxProp::ArrI32(v) => v, _ => &[] }
    }

    fn prop_arr_f64(&self, idx: usize) -> &[f64] {
        match &self.props[idx] { FbxProp::ArrF64(v) => v, _ => &[] }
    }

    fn prop_arr_f32(&self, idx: usize) -> &[f32] {
        match &self.props[idx] { FbxProp::ArrF32(v) => v, _ => &[] }
    }
}

// ── DEFLATE / zlib decompression ────────────────────────────────────────

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,   // byte position
    bit: u32,     // bits remaining in buffer
    buf: u32,     // bit buffer
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0, bit: 0, buf: 0 }
    }

    fn read_bits(&mut self, n: u32) -> u32 {
        while self.bit < n {
            if self.pos < self.data.len() {
                self.buf |= (self.data[self.pos] as u32) << self.bit;
                self.pos += 1;
                self.bit += 8;
            } else {
                break;
            }
        }
        let mask = (1u32 << n) - 1;
        let val = self.buf & mask;
        self.buf >>= n;
        self.bit -= n;
        val
    }
}

/// Decode a Huffman code from the bit stream using a table of (code_len, symbol) pairs
struct HuffTree {
    // Table: for each bit length, sorted symbols
    max_bits: u32,
    counts: [u16; 16],     // number of codes of each length
    symbols: Vec<u16>,     // symbols sorted by code length then value
}

impl HuffTree {
    fn from_lengths(lengths: &[u8]) -> Self {
        let mut counts = [0u16; 16];
        let mut max_bits = 0u32;
        for &l in lengths {
            if l > 0 {
                counts[l as usize] += 1;
                if l as u32 > max_bits { max_bits = l as u32; }
            }
        }

        // Build sorted symbol table (RFC 1951 algorithm)
        let mut offsets = [0u16; 16];
        for i in 1..15 {
            offsets[i + 1] = offsets[i] + counts[i];
        }
        let total: usize = offsets[max_bits as usize] as usize + counts[max_bits as usize] as usize;
        let mut symbols = vec![0u16; total];
        for (sym, &l) in lengths.iter().enumerate() {
            if l > 0 {
                symbols[offsets[l as usize] as usize] = sym as u16;
                offsets[l as usize] += 1;
            }
        }

        HuffTree { max_bits, counts, symbols }
    }

    fn decode(&self, br: &mut BitReader) -> u16 {
        let mut code = 0u32;
        let mut first = 0u32;
        let mut index = 0u32;
        for bits in 1..=self.max_bits {
            code |= br.read_bits(1);
            let count = self.counts[bits as usize] as u32;
            if code < first + count {
                return self.symbols[(index + code - first) as usize];
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        0 // should not reach here for valid data
    }
}

// Length/distance extra bits tables
const LEN_BASE: [u16; 29] = [
    3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258,
];
const LEN_EXTRA: [u8; 29] = [
    0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,
];
const DIST_BASE: [u16; 30] = [
    1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,
    1025,1537,2049,3073,4097,6145,8193,12289,16385,24577,
];
const DIST_EXTRA: [u8; 30] = [
    0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,
];

/// Fixed Huffman trees (BTYPE=1)
fn fixed_lit_tree() -> HuffTree {
    let mut lengths = [0u8; 288];
    for i in 0..=143   { lengths[i] = 8; }
    for i in 144..=255 { lengths[i] = 9; }
    for i in 256..=279 { lengths[i] = 7; }
    for i in 280..=287 { lengths[i] = 8; }
    HuffTree::from_lengths(&lengths)
}

fn fixed_dist_tree() -> HuffTree {
    let lengths = [5u8; 32];
    HuffTree::from_lengths(&lengths)
}

/// Code length order for dynamic Huffman tables (RFC 1951)
const CL_ORDER: [usize; 19] = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15];

fn inflate(compressed: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(compressed.len() * 4);
    let mut br = BitReader::new(compressed);

    loop {
        let bfinal = br.read_bits(1);
        let btype = br.read_bits(2);

        match btype {
            0 => {
                // Uncompressed block
                // Skip remaining bits in current byte
                br.bit = 0;
                br.buf = 0;
                if br.pos + 4 > br.data.len() { break; }
                let len = br.data[br.pos] as usize | ((br.data[br.pos + 1] as usize) << 8);
                br.pos += 4; // skip len + nlen
                output.extend_from_slice(&br.data[br.pos..br.pos + len]);
                br.pos += len;
            }
            1 => {
                // Fixed Huffman codes
                let lit_tree = fixed_lit_tree();
                let dist_tree = fixed_dist_tree();
                inflate_block(&mut br, &lit_tree, &dist_tree, &mut output);
            }
            2 => {
                // Dynamic Huffman codes
                let hlit = br.read_bits(5) as usize + 257;
                let hdist = br.read_bits(5) as usize + 1;
                let hclen = br.read_bits(4) as usize + 4;

                let mut cl_lengths = [0u8; 19];
                for i in 0..hclen {
                    cl_lengths[CL_ORDER[i]] = br.read_bits(3) as u8;
                }
                let cl_tree = HuffTree::from_lengths(&cl_lengths);

                // Decode literal/length + distance code lengths
                let mut all_lengths = vec![0u8; hlit + hdist];
                let mut i = 0;
                while i < hlit + hdist {
                    let sym = cl_tree.decode(&mut br) as usize;
                    if sym < 16 {
                        all_lengths[i] = sym as u8;
                        i += 1;
                    } else if sym == 16 {
                        let rep = br.read_bits(2) as usize + 3;
                        let prev = if i > 0 { all_lengths[i - 1] } else { 0 };
                        for _ in 0..rep { all_lengths[i] = prev; i += 1; }
                    } else if sym == 17 {
                        let rep = br.read_bits(3) as usize + 3;
                        for _ in 0..rep { all_lengths[i] = 0; i += 1; }
                    } else {
                        // sym == 18
                        let rep = br.read_bits(7) as usize + 11;
                        for _ in 0..rep { all_lengths[i] = 0; i += 1; }
                    }
                }

                let lit_tree = HuffTree::from_lengths(&all_lengths[..hlit]);
                let dist_tree = HuffTree::from_lengths(&all_lengths[hlit..]);
                inflate_block(&mut br, &lit_tree, &dist_tree, &mut output);
            }
            _ => break, // invalid
        }

        if bfinal == 1 { break; }
    }

    output
}

fn inflate_block(br: &mut BitReader, lit_tree: &HuffTree, dist_tree: &HuffTree, output: &mut Vec<u8>) {
    loop {
        let sym = lit_tree.decode(br) as usize;
        if sym < 256 {
            output.push(sym as u8);
        } else if sym == 256 {
            break; // end of block
        } else {
            // Length
            let li = sym - 257;
            let length = LEN_BASE[li] as usize + br.read_bits(LEN_EXTRA[li] as u32) as usize;
            // Distance
            let di = dist_tree.decode(br) as usize;
            let distance = DIST_BASE[di] as usize + br.read_bits(DIST_EXTRA[di] as u32) as usize;
            // Copy from back-reference
            let start = output.len() - distance;
            for j in 0..length {
                output.push(output[start + j]);
            }
        }
    }
}

/// Decompress zlib data (2-byte header + deflate + 4-byte adler32 checksum)
fn zlib_decompress(data: &[u8]) -> Vec<u8> {
    if data.len() < 6 { return Vec::new(); }
    // Skip 2-byte zlib header, decompress the deflate stream
    inflate(&data[2..])
}

// ── FBX Binary Parser ──────────────────────────────────────────────────

fn read_u8(data: &[u8], off: usize) -> u8 { data[off] }
fn read_u32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]])
}
fn read_u64(data: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        data[off], data[off+1], data[off+2], data[off+3],
        data[off+4], data[off+5], data[off+6], data[off+7],
    ])
}
fn read_i32(data: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]])
}
fn read_i64(data: &[u8], off: usize) -> i64 {
    i64::from_le_bytes([
        data[off], data[off+1], data[off+2], data[off+3],
        data[off+4], data[off+5], data[off+6], data[off+7],
    ])
}
fn read_f32(data: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([data[off], data[off+1], data[off+2], data[off+3]])
}
fn read_f64(data: &[u8], off: usize) -> f64 {
    f64::from_le_bytes([
        data[off], data[off+1], data[off+2], data[off+3],
        data[off+4], data[off+5], data[off+6], data[off+7],
    ])
}

/// Parse FBX properties from the data stream, returns (props, bytes_consumed)
fn parse_props(data: &[u8], offset: usize, num_props: u64) -> (Vec<FbxProp>, usize) {
    let mut props = Vec::with_capacity(num_props as usize);
    let mut off = offset;
    for _ in 0..num_props {
        let ptype = data[off] as char;
        off += 1;
        match ptype {
            'C' => { props.push(FbxProp::Bool(data[off] != 0)); off += 1; }
            'I' => { props.push(FbxProp::I32(read_i32(data, off))); off += 4; }
            'L' => { props.push(FbxProp::I64(read_i64(data, off))); off += 8; }
            'F' => { props.push(FbxProp::F32(read_f32(data, off))); off += 4; }
            'D' => { props.push(FbxProp::F64(read_f64(data, off))); off += 8; }
            'S' | 'R' => {
                let len = read_u32(data, off) as usize;
                off += 4;
                if ptype == 'S' {
                    let s = String::from_utf8_lossy(&data[off..off+len]).into_owned();
                    props.push(FbxProp::Str(s));
                } else {
                    props.push(FbxProp::Raw(data[off..off+len].to_vec()));
                }
                off += len;
            }
            'i' | 'l' | 'f' | 'd' => {
                let arr_len = read_u32(data, off) as usize;
                let encoding = read_u32(data, off + 4);
                let comp_len = read_u32(data, off + 8) as usize;
                off += 12;
                let raw = if encoding == 1 {
                    zlib_decompress(&data[off..off+comp_len])
                } else {
                    data[off..off+comp_len].to_vec()
                };
                off += comp_len;
                match ptype {
                    'i' => {
                        let mut arr = Vec::with_capacity(arr_len);
                        for j in 0..arr_len {
                            arr.push(read_i32(&raw, j * 4));
                        }
                        props.push(FbxProp::ArrI32(arr));
                    }
                    'l' => {
                        let mut arr = Vec::with_capacity(arr_len);
                        for j in 0..arr_len {
                            arr.push(read_i64(&raw, j * 8));
                        }
                        props.push(FbxProp::ArrI64(arr));
                    }
                    'f' => {
                        let mut arr = Vec::with_capacity(arr_len);
                        for j in 0..arr_len {
                            arr.push(read_f32(&raw, j * 4));
                        }
                        props.push(FbxProp::ArrF32(arr));
                    }
                    'd' => {
                        let mut arr = Vec::with_capacity(arr_len);
                        for j in 0..arr_len {
                            arr.push(read_f64(&raw, j * 8));
                        }
                        props.push(FbxProp::ArrF64(arr));
                    }
                    _ => unreachable!(),
                }
            }
            _ => {
                // Unknown property type — skip remaining
                break;
            }
        }
    }
    (props, off - offset)
}

/// Parse FBX nodes recursively. FBX 7500+ uses 64-bit offsets.
fn parse_nodes(data: &[u8], mut offset: usize, end_limit: usize) -> Vec<FbxNode> {
    let mut nodes = Vec::new();
    while offset + 25 <= end_limit {
        let end_offset = read_u64(data, offset) as usize;
        let num_props = read_u64(data, offset + 8);
        let prop_list_len = read_u64(data, offset + 16) as usize;
        let name_len = read_u8(data, offset + 24) as usize;
        if end_offset == 0 { break; }

        let name = String::from_utf8_lossy(&data[offset + 25..offset + 25 + name_len]).into_owned();
        let prop_start = offset + 25 + name_len;
        let (props, _consumed) = parse_props(data, prop_start, num_props);
        let children_start = prop_start + prop_list_len;
        let children = if children_start < end_offset {
            parse_nodes(data, children_start, end_offset)
        } else {
            Vec::new()
        };

        nodes.push(FbxNode { name, props, children });
        offset = end_offset;
    }
    nodes
}

/// Parse a complete FBX binary file into a tree of nodes
fn parse_fbx(data: &[u8]) -> Vec<FbxNode> {
    // Verify magic
    if data.len() < 27 || &data[0..21] != b"Kaydara FBX Binary  \x00" {
        eprintln!("[fbx_anim] Not a valid FBX binary file");
        return Vec::new();
    }
    // Version at offset 23 (u32) — we support 7700 (FBX 2020)
    let _version = read_u32(data, 23);
    parse_nodes(data, 27, data.len())
}

// ── Skeleton + Animation extraction ─────────────────────────────────────

/// Extract skeleton and animation clip from parsed FBX nodes
fn extract_skeleton_and_clip(nodes: &[FbxNode], clip_name: &str, looping: bool) -> (FbxSkeleton, AnimationClip) {
    let objects = nodes.iter().find(|n| n.name == "Objects");
    let connections = nodes.iter().find(|n| n.name == "Connections");

    if objects.is_none() || connections.is_none() {
        return (FbxSkeleton { bones: Vec::new() }, AnimationClip {
            name: clip_name.to_string(), duration: 0.0, fps: 60.0,
            bone_channels: Vec::new(), looping,
        });
    }
    let objects = objects.unwrap();
    let connections = connections.unwrap();

    // Step 1: Collect all Model nodes (bones) with their IDs, names, and bind poses
    struct ModelInfo {
        id: i64,
        name: String,
        bind_translation: [f32; 3],
        bind_rotation: [f32; 3],
        pre_rotation: [f32; 3],  // FBX PreRotation (applied before Lcl Rotation)
    }

    let mut models: Vec<ModelInfo> = Vec::new();
    let mut model_id_to_idx: HashMap<i64, usize> = HashMap::new();

    for node in objects.children_named("Model") {
        if node.props.len() < 3 { continue; }
        let id = node.prop_i64(0);
        let raw_name = node.prop_str(1);
        // Name format: "mixamorig:Hips\x00\x01Model" — take part before \x00
        let name = raw_name.split('\x00').next().unwrap_or(raw_name).to_string();

        // Only process skeleton bones (LimbNode type)
        let node_type = node.prop_str(2);
        if node_type != "LimbNode" { continue; }

        let mut trans = [0.0f32; 3];
        let mut rot = [0.0f32; 3];
        let mut pre_rot = [0.0f32; 3];

        if let Some(props70) = node.child("Properties70") {
            for p in props70.children_named("P") {
                if p.props.len() >= 7 {
                    let pname = p.prop_str(0);
                    if pname == "Lcl Translation" {
                        trans[0] = p.prop_f64(4) as f32;
                        trans[1] = p.prop_f64(5) as f32;
                        trans[2] = p.prop_f64(6) as f32;
                    } else if pname == "Lcl Rotation" {
                        rot[0] = p.prop_f64(4) as f32;
                        rot[1] = p.prop_f64(5) as f32;
                        rot[2] = p.prop_f64(6) as f32;
                    } else if pname == "PreRotation" {
                        pre_rot[0] = p.prop_f64(4) as f32;
                        pre_rot[1] = p.prop_f64(5) as f32;
                        pre_rot[2] = p.prop_f64(6) as f32;
                    }
                }
            }
        }

        let idx = models.len();
        model_id_to_idx.insert(id, idx);
        models.push(ModelInfo { id, name, bind_translation: trans, bind_rotation: rot, pre_rotation: pre_rot });
    }

    // Step 2: Parse connections to build parent-child hierarchy
    // Connection format: C { "OO"/"OP", source_id, dest_id [, property] }
    let mut parent_map: HashMap<i64, i64> = HashMap::new(); // child_model_id -> parent_model_id

    // Also track AnimationCurveNode -> Model connections (OP with Lcl Translation/Rotation)
    // And AnimationCurve -> AnimationCurveNode connections (OP with d|X/d|Y/d|Z or OO)
    let mut acn_to_model: HashMap<i64, (i64, String)> = HashMap::new(); // acn_id -> (model_id, prop_name)
    let mut ac_to_acn: HashMap<i64, (i64, String)> = HashMap::new(); // curve_id -> (acn_id, channel)

    // Collect all AnimationCurveNode and AnimationCurve IDs
    let mut acn_ids: HashMap<i64, String> = HashMap::new(); // id -> name (T/R/S)
    let mut ac_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();

    for node in objects.children_named("AnimationCurveNode") {
        if node.props.len() >= 2 {
            let id = node.prop_i64(0);
            let raw_name = node.prop_str(1).split('\x00').next().unwrap_or("").to_string();
            acn_ids.insert(id, raw_name);
        }
    }
    for node in objects.children_named("AnimationCurve") {
        if !node.props.is_empty() {
            ac_ids.insert(node.prop_i64(0));
        }
    }

    for conn in connections.children_named("C") {
        if conn.props.len() < 3 { continue; }
        let ctype = conn.prop_str(0);
        let src = conn.prop_i64(1);
        let dst = conn.prop_i64(2);
        let prop = if conn.props.len() > 3 { conn.prop_str(3).to_string() } else { String::new() };

        match ctype {
            "OO" => {
                // Model -> Model parent
                if model_id_to_idx.contains_key(&src) && (model_id_to_idx.contains_key(&dst) || dst == 0) {
                    parent_map.insert(src, dst);
                }
                // AnimationCurve -> AnimationCurveNode (no property specified)
                if ac_ids.contains(&src) && acn_ids.contains_key(&dst) {
                    ac_to_acn.insert(src, (dst, String::new()));
                }
            }
            "OP" => {
                // AnimationCurveNode -> Model with property (Lcl Translation/Rotation)
                if acn_ids.contains_key(&src) && model_id_to_idx.contains_key(&dst) {
                    acn_to_model.insert(src, (dst, prop));
                }
                // AnimationCurve -> AnimationCurveNode with channel (d|X, d|Y, d|Z)
                else if ac_ids.contains(&src) && acn_ids.contains_key(&dst) {
                    ac_to_acn.insert(src, (dst, prop));
                }
            }
            _ => {}
        }
    }

    // Step 3: Build skeleton bones with parent indices
    let mut bones: Vec<FbxBone> = Vec::new();
    for model in &models {
        let parent_idx = if let Some(&parent_id) = parent_map.get(&model.id) {
            if parent_id == 0 { None } else { model_id_to_idx.get(&parent_id).copied() }
        } else {
            None
        };
        bones.push(FbxBone {
            name: model.name.clone(),
            parent: parent_idx,
            bind_translation: model.bind_translation,
            bind_rotation: model.bind_rotation,
            pre_rotation: model.pre_rotation,
        });
    }

    // Step 4: Extract animation curves
    // For each AnimationCurve, get its KeyTime + KeyValueFloat
    struct CurveData {
        times: Vec<i64>,
        values: Vec<f32>,
    }

    let mut curves: HashMap<i64, CurveData> = HashMap::new();
    for node in objects.children_named("AnimationCurve") {
        if node.props.is_empty() { continue; }
        let id = node.prop_i64(0);
        let mut times = Vec::new();
        let mut values = Vec::new();

        for child in &node.children {
            if child.name == "KeyTime" {
                if let Some(FbxProp::ArrI64(arr)) = child.props.first() {
                    times = arr.clone();
                }
            } else if child.name == "KeyValueFloat" {
                if let Some(FbxProp::ArrF32(arr)) = child.props.first() {
                    values = arr.clone();
                } else if let Some(FbxProp::ArrF64(arr)) = child.props.first() {
                    values = arr.iter().map(|&v| v as f32).collect();
                }
            }
        }

        if !times.is_empty() && !values.is_empty() {
            curves.insert(id, CurveData { times, values });
        }
    }

    // Step 5: Build bone channels by following the connection chain:
    //   AnimationCurve --(d|X/Y/Z)--> AnimationCurveNode --(Lcl Translation/Rotation)--> Model (bone)
    // For each bone, we need up to 6 curves: tx,ty,tz, rx,ry,rz
    struct BoneChannelBuilder {
        bone_idx: usize,
        tx: Option<i64>, ty: Option<i64>, tz: Option<i64>,
        rx: Option<i64>, ry: Option<i64>, rz: Option<i64>,
    }

    let mut channel_builders: HashMap<usize, BoneChannelBuilder> = HashMap::new();

    for (&curve_id, &(acn_id, ref channel)) in &ac_to_acn {
        if let Some(&(model_id, ref prop)) = acn_to_model.get(&acn_id) {
            if let Some(&bone_idx) = model_id_to_idx.get(&model_id) {
                let builder = channel_builders.entry(bone_idx).or_insert(BoneChannelBuilder {
                    bone_idx,
                    tx: None, ty: None, tz: None,
                    rx: None, ry: None, rz: None,
                });

                let is_translation = prop == "Lcl Translation";
                let is_rotation = prop == "Lcl Rotation";

                match channel.as_str() {
                    "d|X" => { if is_translation { builder.tx = Some(curve_id); } else if is_rotation { builder.rx = Some(curve_id); } }
                    "d|Y" => { if is_translation { builder.ty = Some(curve_id); } else if is_rotation { builder.ry = Some(curve_id); } }
                    "d|Z" => { if is_translation { builder.tz = Some(curve_id); } else if is_rotation { builder.rz = Some(curve_id); } }
                    _ => {}
                }
            }
        }
    }

    // Step 6: Assemble BoneChannels from the builders
    let mut bone_channels: Vec<BoneChannel> = Vec::new();
    let mut max_time: f64 = 0.0;

    for builder in channel_builders.values() {
        // Get rotation curves (required)
        let rx_curve = builder.rx.and_then(|id| curves.get(&id));
        let ry_curve = builder.ry.and_then(|id| curves.get(&id));
        let rz_curve = builder.rz.and_then(|id| curves.get(&id));

        // Determine number of keyframes from any available rotation curve
        let n_keys = rx_curve.map(|c| c.times.len())
            .or(ry_curve.map(|c| c.times.len()))
            .or(rz_curve.map(|c| c.times.len()))
            .unwrap_or(0);

        if n_keys == 0 { continue; }

        // Get time array from any curve
        let time_source = rx_curve.or(ry_curve).or(rz_curve).unwrap();
        let times: Vec<f32> = time_source.times.iter()
            .map(|&t| t as f64 / FBX_TICKS_PER_SECOND)
            .map(|t| { if t > max_time { max_time = t; } t as f32 })
            .collect();

        // Build rotation keyframes
        let mut rotations = Vec::with_capacity(n_keys);
        for i in 0..n_keys {
            rotations.push([
                rx_curve.map(|c| c.values.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0),
                ry_curve.map(|c| c.values.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0),
                rz_curve.map(|c| c.values.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0),
            ]);
        }

        // Build translation keyframes (only for bones that have them, typically Hips)
        let tx_curve = builder.tx.and_then(|id| curves.get(&id));
        let ty_curve = builder.ty.and_then(|id| curves.get(&id));
        let tz_curve = builder.tz.and_then(|id| curves.get(&id));
        let translations = if tx_curve.is_some() || ty_curve.is_some() || tz_curve.is_some() {
            let mut trans = Vec::with_capacity(n_keys);
            for i in 0..n_keys {
                trans.push([
                    tx_curve.map(|c| c.values.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0),
                    ty_curve.map(|c| c.values.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0),
                    tz_curve.map(|c| c.values.get(i).copied().unwrap_or(0.0)).unwrap_or(0.0),
                ]);
            }
            Some(trans)
        } else {
            None
        };

        bone_channels.push(BoneChannel {
            bone_index: builder.bone_idx,
            times,
            translations,
            rotations,
        });
    }

    let duration = max_time as f32;
    let fps = if duration > 0.0 && !bone_channels.is_empty() {
        let max_keys = bone_channels.iter().map(|c| c.times.len()).max().unwrap_or(1);
        if max_keys > 1 {
            (max_keys - 1) as f32 / duration
        } else {
            60.0
        }
    } else {
        60.0
    };

    let skeleton = FbxSkeleton { bones };
    let clip = AnimationClip {
        name: clip_name.to_string(),
        duration,
        fps,
        bone_channels,
        looping,
    };

    (skeleton, clip)
}

// ── Public API ──────────────────────────────────────────────────────────

/// Load an FBX file and extract skeleton + animation clip.
/// Returns (skeleton, clip). All FBX files share the same Mixamo skeleton.
pub fn load_fbx(path: &str, clip_name: &str, looping: bool) -> (FbxSkeleton, AnimationClip) {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[fbx_anim] Failed to read {}: {}", path, e);
            return (FbxSkeleton { bones: Vec::new() }, AnimationClip {
                name: clip_name.to_string(), duration: 0.0, fps: 60.0,
                bone_channels: Vec::new(), looping,
            });
        }
    };

    let nodes = parse_fbx(&data);
    extract_skeleton_and_clip(&nodes, clip_name, looping)
}

/// Load all 8 animation FBX files and return skeleton + all clips.
/// The skeleton is taken from the first file (all share the same Mixamo skeleton).
pub fn load_all_animations(dir: &str) -> (FbxSkeleton, Vec<AnimationClip>) {
    let files = [
        ("walking.fbx",         "walking",        true),
        ("run_forward.fbx",     "run_forward",    true),
        ("elbow_punch.fbx",     "elbow_punch",    false),
        ("hook_punch.fbx",      "hook_punch",     false),
        ("roundhouse_kick.fbx", "roundhouse_kick", false),
        ("drop_kick.fbx",       "drop_kick",      false),
        ("picking_up.fbx",      "picking_up",     false),
        ("sitting_pose.fbx",    "sitting_pose",    false),
    ];

    let mut skeleton = FbxSkeleton { bones: Vec::new() };
    let mut clips = Vec::new();

    for (i, (file, name, looping)) in files.iter().enumerate() {
        let path = format!("{}/{}", dir, file);
        let (skel, clip) = load_fbx(&path, name, *looping);
        let bone_count = skel.bones.len();
        if i == 0 {
            skeleton = skel;
        }
        eprintln!("[fbx_anim] Loaded {}: {} bones, {} channels, {:.2}s, {} keys/chan",
            name,
            bone_count,
            clip.bone_channels.len(),
            clip.duration,
            clip.bone_channels.first().map(|c| c.times.len()).unwrap_or(0),
        );
        clips.push(clip);
    }

    (skeleton, clips)
}

// ── FBX inspection + skin data extraction ──────────────────────────────

/// Per-bone skin cluster data extracted from FBX Deformers
pub struct SkinCluster {
    pub bone_name: String,
    pub vertex_indices: Vec<i32>,
    pub weights: Vec<f64>,
    pub transform: [f64; 16],       // inverse bind matrix (row-major)
    pub transform_link: [f64; 16],  // bind pose matrix (row-major)
}

/// Complete skin data from an FBX file
pub struct FbxSkinData {
    pub clusters: Vec<SkinCluster>,
    pub vertex_count: usize,  // total vertices in the FBX mesh
}

/// Inspect FBX file structure — returns human-readable summary for studio
pub fn inspect_fbx(path: &str) -> String {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => return format!("Failed to read {}: {}", path, e),
    };
    let nodes = parse_fbx(&data);
    let mut out = String::new();

    // Find Objects
    let objects = match nodes.iter().find(|n| n.name == "Objects") {
        Some(n) => n,
        None => return "No Objects node found".to_string(),
    };

    // Count node types
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for child in &objects.children {
        *type_counts.entry(child.name.clone()).or_insert(0) += 1;
    }
    out.push_str(&format!("FBX Objects: {:?}\n", type_counts));

    // List Deformers
    for node in objects.children.iter().filter(|n| n.name == "Deformer") {
        let id = if !node.props.is_empty() { node.prop_i64(0) } else { 0 };
        let name = if node.props.len() > 1 { node.prop_str(1) } else { "" };
        let dtype = if node.props.len() > 2 { node.prop_str(2) } else { "" };
        out.push_str(&format!("  Deformer ID={} name='{}' type='{}'\n", id, name, dtype));

        // Show children of Cluster deformers
        if dtype == "Cluster" {
            for child in &node.children {
                let desc = match child.name.as_str() {
                    "Indexes" if !child.props.is_empty() => {
                        match &child.props[0] {
                            FbxProp::ArrI32(v) => format!("int[{}]", v.len()),
                            _ => format!("{} props", child.props.len()),
                        }
                    }
                    "Weights" if !child.props.is_empty() => {
                        match &child.props[0] {
                            FbxProp::ArrF64(v) => format!("f64[{}]", v.len()),
                            FbxProp::ArrF32(v) => format!("f32[{}]", v.len()),
                            _ => format!("{} props", child.props.len()),
                        }
                    }
                    "Transform" | "TransformLink" if !child.props.is_empty() => {
                        match &child.props[0] {
                            FbxProp::ArrF64(v) => format!("mat4x4 ({} doubles)", v.len()),
                            _ => format!("{} props", child.props.len()),
                        }
                    }
                    _ => format!("{} children, {} props", child.children.len(), child.props.len()),
                };
                out.push_str(&format!("    {}: {}\n", child.name, desc));
            }
        }
    }

    // List Geometry nodes
    for node in objects.children.iter().filter(|n| n.name == "Geometry") {
        let id = if !node.props.is_empty() { node.prop_i64(0) } else { 0 };
        let name = if node.props.len() > 1 { node.prop_str(1) } else { "" };
        let gtype = if node.props.len() > 2 { node.prop_str(2) } else { "" };
        let mut vert_count = 0;
        let mut idx_count = 0;
        for child in &node.children {
            if child.name == "Vertices" && !child.props.is_empty() {
                if let FbxProp::ArrF64(v) = &child.props[0] { vert_count = v.len() / 3; }
            }
            if child.name == "PolygonVertexIndex" && !child.props.is_empty() {
                if let FbxProp::ArrI32(v) = &child.props[0] { idx_count = v.len(); }
            }
        }
        out.push_str(&format!("  Geometry ID={} '{}' type='{}' verts={} indices={}\n",
            id, name, gtype, vert_count, idx_count));
    }

    // Connections summary
    if let Some(conns) = nodes.iter().find(|n| n.name == "Connections") {
        out.push_str(&format!("Connections: {} total\n", conns.children.len()));
    }

    out
}

/// Extract skin data (bone weights + bind matrices) from an FBX file.
/// Returns None if no skin deformers found.
pub fn extract_skin_data(path: &str) -> Option<FbxSkinData> {
    let data = std::fs::read(path).ok()?;
    let nodes = parse_fbx(&data);

    let objects = nodes.iter().find(|n| n.name == "Objects")?;
    let connections = nodes.iter().find(|n| n.name == "Connections")?;

    // Build connection map: child_id -> parent_id
    let mut conn_child_to_parent: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut conn_parent_to_child: HashMap<i64, Vec<i64>> = HashMap::new();
    for c in &connections.children {
        if c.props.len() >= 3 {
            let child_id = c.prop_i64(1);
            let parent_id = c.prop_i64(2);
            conn_child_to_parent.entry(child_id).or_default().push(parent_id);
            conn_parent_to_child.entry(parent_id).or_default().push(child_id);
        }
    }

    // Map Model IDs to bone names
    let mut model_id_to_name: HashMap<i64, String> = HashMap::new();
    for node in objects.children.iter().filter(|n| n.name == "Model") {
        if node.props.len() >= 2 {
            let id = node.prop_i64(0);
            let raw_name = node.prop_str(1);
            let name = raw_name.split('\x00').next().unwrap_or(raw_name).to_string();
            model_id_to_name.insert(id, name);
        }
    }

    // Find mesh vertex count from Geometry node
    let mut vertex_count = 0;
    for node in objects.children.iter().filter(|n| n.name == "Geometry") {
        for child in &node.children {
            if child.name == "Vertices" && !child.props.is_empty() {
                if let FbxProp::ArrF64(v) = &child.props[0] { vertex_count = v.len() / 3; }
            }
        }
    }

    // Extract Cluster deformers
    let mut clusters: Vec<SkinCluster> = Vec::new();

    for node in objects.children.iter().filter(|n| n.name == "Deformer") {
        if node.props.len() < 3 { continue; }
        let dtype = node.prop_str(2);
        if dtype != "Cluster" { continue; }

        let cluster_id = node.prop_i64(0);

        // Get vertex indices and weights
        let mut indices: Vec<i32> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut transform = [0.0f64; 16];
        let mut transform_link = [0.0f64; 16];

        for child in &node.children {
            match child.name.as_str() {
                "Indexes" if !child.props.is_empty() => {
                    if let FbxProp::ArrI32(v) = &child.props[0] {
                        indices = v.clone();
                    }
                }
                "Weights" if !child.props.is_empty() => {
                    match &child.props[0] {
                        FbxProp::ArrF64(v) => weights = v.clone(),
                        FbxProp::ArrF32(v) => weights = v.iter().map(|&f| f as f64).collect(),
                        _ => {}
                    }
                }
                "Transform" if !child.props.is_empty() => {
                    if let FbxProp::ArrF64(v) = &child.props[0] {
                        if v.len() >= 16 {
                            transform.copy_from_slice(&v[..16]);
                        }
                    }
                }
                "TransformLink" if !child.props.is_empty() => {
                    if let FbxProp::ArrF64(v) = &child.props[0] {
                        if v.len() >= 16 {
                            transform_link.copy_from_slice(&v[..16]);
                        }
                    }
                }
                _ => {}
            }
        }

        // Find which bone this cluster connects to via Connections
        // Cluster -> SubDeformer(Skin) -> Geometry, but also Cluster -> Model(bone)
        let mut bone_name = String::new();
        if let Some(parents) = conn_child_to_parent.get(&cluster_id) {
            for &pid in parents {
                if let Some(name) = model_id_to_name.get(&pid) {
                    bone_name = name.clone();
                    break;
                }
            }
        }
        // Also check children connected TO this cluster
        if bone_name.is_empty() {
            if let Some(children) = conn_parent_to_child.get(&cluster_id) {
                for &cid in children {
                    if let Some(name) = model_id_to_name.get(&cid) {
                        bone_name = name.clone();
                        break;
                    }
                }
            }
        }

        if !indices.is_empty() {
            clusters.push(SkinCluster {
                bone_name,
                vertex_indices: indices,
                weights,
                transform,
                transform_link,
            });
        }
    }

    if clusters.is_empty() {
        return None;
    }

    eprintln!("[fbx_anim] Extracted skin data: {} clusters, {} mesh verts",
        clusters.len(), vertex_count);

    Some(FbxSkinData { clusters, vertex_count })
}
