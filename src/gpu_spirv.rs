// Shared SPIR-V utilities: constants, macros, and helpers used by both
// gpu_kernels.rs (compute shaders) and gpu_shaders.rs (graphics shaders)

#![allow(unused)]

pub fn encode_spirv_string(s: &str) -> Vec<u32> {
    let mut bytes: Vec<u8> = s.bytes().collect();
    bytes.push(0);
    while bytes.len() % 4 != 0 { bytes.push(0); }
    bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[macro_export]
macro_rules! emit {
    ($s:expr, $opcode:expr $(, $op:expr)*) => {{
        let ops: &[u32] = &[$($op),*];
        $s.push((((ops.len() as u32 + 1) << 16) | ($opcode as u32)));
        $s.extend_from_slice(ops);
    }};
}

#[macro_export]
macro_rules! emit_str {
    ($s:expr, $opcode:expr, [$($pre:expr),*], $str:expr, [$($post:expr),*]) => {{
        let pre: &[u32] = &[$($pre),*];
        let post: &[u32] = &[$($post),*];
        let sw = $crate::gpu_spirv::encode_spirv_string($str);
        let wc = (1 + pre.len() + sw.len() + post.len()) as u32;
        $s.push((wc << 16) | ($opcode as u32));
        $s.extend_from_slice(pre);
        $s.extend_from_slice(&sw);
        $s.extend_from_slice(post);
    }};
}

// SPIR-V opcodes (shared between compute and graphics shaders)
pub const OP_CAP: u16 = 17;
pub const OP_MEM_MODEL: u16 = 14;
pub const OP_ENTRY: u16 = 15;
pub const OP_EXEC_MODE: u16 = 16;
pub const OP_DECORATE: u16 = 71;
pub const OP_MEMBER_DEC: u16 = 72;
pub const OP_TYPE_VOID: u16 = 19;
pub const OP_TYPE_FN: u16 = 33;
pub const OP_TYPE_INT: u16 = 21;
pub const OP_TYPE_FLOAT: u16 = 22;
pub const OP_TYPE_VEC: u16 = 23;
pub const OP_TYPE_PTR: u16 = 32;
pub const OP_TYPE_STRUCT: u16 = 30;
pub const OP_CONST: u16 = 43;
pub const OP_VAR: u16 = 59;
pub const OP_FN: u16 = 54;
pub const OP_FN_END: u16 = 56;
pub const OP_LABEL: u16 = 248;
pub const OP_ACCESS: u16 = 65;
pub const OP_LOAD: u16 = 61;
pub const OP_STORE: u16 = 62;
pub const OP_RETURN: u16 = 253;
pub const OP_FMUL: u16 = 133;
pub const OP_FADD: u16 = 129;
pub const OP_FSUB: u16 = 131;

// Decoration values
pub const DEC_BUILTIN: u32 = 11;
pub const DEC_BLOCK: u32 = 2;
pub const DEC_OFFSET: u32 = 35;

// Storage classes
pub const SC_INPUT: u32 = 1;
pub const SC_PUSH_CONST: u32 = 9;

pub fn header(s: &mut Vec<u32>, bound: u32) {
    s[0] = 0x07230203;
    s[1] = 0x00010300; // SPIR-V 1.3
    s[2] = 0;
    s[3] = bound;
    s[4] = 0;
}
