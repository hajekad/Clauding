// SPIR-V graphics shader binaries (vertex + fragment), built programmatically
// Vertex: push-constant VP matrix transform + color pass-through
// Fragment: writes interpolated color to attachment

#![allow(unused)]

fn encode_spirv_string(s: &str) -> Vec<u32> {
    let mut bytes: Vec<u8> = s.bytes().collect();
    bytes.push(0);
    while bytes.len() % 4 != 0 { bytes.push(0); }
    bytes.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

macro_rules! emit {
    ($s:expr, $opcode:expr $(, $op:expr)*) => {{
        let ops: &[u32] = &[$($op),*];
        $s.push((((ops.len() as u32 + 1) << 16) | ($opcode as u32)));
        $s.extend_from_slice(ops);
    }};
}

macro_rules! emit_str {
    ($s:expr, $opcode:expr, [$($pre:expr),*], $str:expr, [$($post:expr),*]) => {{
        let pre: &[u32] = &[$($pre),*];
        let post: &[u32] = &[$($post),*];
        let sw = encode_spirv_string($str);
        let wc = (1 + pre.len() + sw.len() + post.len()) as u32;
        $s.push((wc << 16) | ($opcode as u32));
        $s.extend_from_slice(pre);
        $s.extend_from_slice(&sw);
        $s.extend_from_slice(post);
    }};
}

// SPIR-V opcodes
const OP_CAP: u16 = 17;
const OP_MEM_MODEL: u16 = 14;
const OP_ENTRY: u16 = 15;
const OP_EXEC_MODE: u16 = 16;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DEC: u16 = 72;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_FN: u16 = 33;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VEC: u16 = 23;
const OP_TYPE_MAT: u16 = 24;
const OP_TYPE_PTR: u16 = 32;
const OP_TYPE_STRUCT: u16 = 30;
const OP_CONST: u16 = 43;
const OP_VAR: u16 = 59;
const OP_FN: u16 = 54;
const OP_FN_END: u16 = 56;
const OP_LABEL: u16 = 248;
const OP_ACCESS: u16 = 65;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_RETURN: u16 = 253;
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_MAT_TIMES_VEC: u16 = 145;

// Decoration values
const DEC_BUILTIN: u32 = 11;
const DEC_BLOCK: u32 = 2;
const DEC_OFFSET: u32 = 35;
const DEC_LOCATION: u32 = 30;
const DEC_COL_MAJOR: u32 = 5;
const DEC_MAT_STRIDE: u32 = 7;

// Storage classes
const SC_INPUT: u32 = 1;
const SC_OUTPUT: u32 = 3;
const SC_PUSH_CONST: u32 = 9;

// Built-in values
const BI_POSITION: u32 = 0;

// Execution mode values
const EM_ORIGIN_UPPER_LEFT: u32 = 7;

fn header(s: &mut Vec<u32>, bound: u32) {
    s[0] = 0x07230203;
    s[1] = 0x00010300; // SPIR-V 1.3
    s[2] = 0;
    s[3] = bound;
    s[4] = 0;
}

/// Vertex shader: VP matrix transform + color pass-through
///
/// Inputs:  location 0 = vec3 position, location 1 = vec4 color (B8G8R8A8_UNORM auto-unpacked)
/// Outputs: gl_Position (BuiltIn), location 0 = vec4 fragColor
/// Push constants: mat4 VP (64 bytes)
pub fn build_vertex_shader() -> Vec<u32> {
    let mut s: Vec<u32> = vec![0; 5];

    // ID assignments
    let ty_void = 1u32;
    let ty_fn_void = 2;
    let ty_float = 3;
    let ty_uint = 4;
    let ty_vec3 = 5;
    let ty_vec4 = 6;
    let ty_mat4 = 7;
    let ty_ptr_in_vec3 = 8;
    let ty_ptr_in_vec4 = 9;
    let ty_ptr_out_vec4 = 10;
    let ty_pc_struct = 11;
    let ty_ptr_pc = 12;
    let ty_ptr_pc_mat4 = 13;
    let in_pos = 14;
    let in_color = 15;
    let gl_position = 16;
    let out_frag_color = 17;
    let pc_var = 18;
    let c_0u = 19;
    let c_1f = 20;
    let main_fn = 21;
    let lbl_entry = 22;
    // Temps
    let r_pos = 23u32;       // loaded vec3 position
    let r_pos4 = 24;         // vec4(pos, 1.0)
    let r_vp_ptr = 25;       // pointer to VP matrix
    let r_vp = 26;           // loaded VP matrix
    let r_clip_pos = 27;     // VP * pos4
    let r_color = 28;        // loaded color
    let bound = 29u32;

    // Capability: Shader
    emit!(s, OP_CAP, 1);

    // Memory model: Logical, GLSL450
    emit!(s, OP_MEM_MODEL, 0, 1);

    // Entry point: Vertex shader
    // ExecutionModel 0 = Vertex
    emit_str!(s, OP_ENTRY, [0, main_fn], "main", [in_pos, in_color, gl_position, out_frag_color]);

    // Decorations
    emit!(s, OP_DECORATE, in_pos, DEC_LOCATION, 0);
    emit!(s, OP_DECORATE, in_color, DEC_LOCATION, 1);
    emit!(s, OP_DECORATE, gl_position, DEC_BUILTIN, BI_POSITION);
    emit!(s, OP_DECORATE, out_frag_color, DEC_LOCATION, 0);
    emit!(s, OP_DECORATE, ty_pc_struct, DEC_BLOCK);
    emit!(s, OP_MEMBER_DEC, ty_pc_struct, 0, DEC_OFFSET, 0);
    emit!(s, OP_MEMBER_DEC, ty_pc_struct, 0, DEC_COL_MAJOR);
    emit!(s, OP_MEMBER_DEC, ty_pc_struct, 0, DEC_MAT_STRIDE, 16);

    // Types
    emit!(s, OP_TYPE_VOID, ty_void);
    emit!(s, OP_TYPE_FN, ty_fn_void, ty_void);
    emit!(s, OP_TYPE_FLOAT, ty_float, 32);
    emit!(s, OP_TYPE_INT, ty_uint, 32, 0);
    emit!(s, OP_TYPE_VEC, ty_vec3, ty_float, 3);
    emit!(s, OP_TYPE_VEC, ty_vec4, ty_float, 4);
    emit!(s, OP_TYPE_MAT, ty_mat4, ty_vec4, 4);
    emit!(s, OP_TYPE_PTR, ty_ptr_in_vec3, SC_INPUT, ty_vec3);
    emit!(s, OP_TYPE_PTR, ty_ptr_in_vec4, SC_INPUT, ty_vec4);
    emit!(s, OP_TYPE_PTR, ty_ptr_out_vec4, SC_OUTPUT, ty_vec4);
    emit!(s, OP_TYPE_STRUCT, ty_pc_struct, ty_mat4);
    emit!(s, OP_TYPE_PTR, ty_ptr_pc, SC_PUSH_CONST, ty_pc_struct);
    emit!(s, OP_TYPE_PTR, ty_ptr_pc_mat4, SC_PUSH_CONST, ty_mat4);

    // Variables
    emit!(s, OP_VAR, ty_ptr_in_vec3, in_pos, SC_INPUT);
    emit!(s, OP_VAR, ty_ptr_in_vec4, in_color, SC_INPUT);
    emit!(s, OP_VAR, ty_ptr_out_vec4, gl_position, SC_OUTPUT);
    emit!(s, OP_VAR, ty_ptr_out_vec4, out_frag_color, SC_OUTPUT);
    emit!(s, OP_VAR, ty_ptr_pc, pc_var, SC_PUSH_CONST);

    // Constants
    emit!(s, OP_CONST, ty_uint, c_0u, 0);
    emit!(s, OP_CONST, ty_float, c_1f, 0x3F800000u32); // 1.0f

    // Function
    emit!(s, OP_FN, ty_void, main_fn, 0, ty_fn_void);
    emit!(s, OP_LABEL, lbl_entry);

    // Load position
    emit!(s, OP_LOAD, ty_vec3, r_pos, in_pos);

    // Construct vec4(pos, 1.0)
    emit!(s, OP_COMPOSITE_CONSTRUCT, ty_vec4, r_pos4, r_pos, c_1f);

    // Load VP matrix from push constants
    emit!(s, OP_ACCESS, ty_ptr_pc_mat4, r_vp_ptr, pc_var, c_0u);
    emit!(s, OP_LOAD, ty_mat4, r_vp, r_vp_ptr);

    // gl_Position = VP * vec4(pos, 1.0)
    emit!(s, OP_MAT_TIMES_VEC, ty_vec4, r_clip_pos, r_vp, r_pos4);
    emit!(s, OP_STORE, gl_position, r_clip_pos);

    // fragColor = inColor
    emit!(s, OP_LOAD, ty_vec4, r_color, in_color);
    emit!(s, OP_STORE, out_frag_color, r_color);

    emit!(s, OP_RETURN);
    emit!(s, OP_FN_END);

    header(&mut s, bound);
    s
}

/// Fragment shader: pass-through color
///
/// Input:  location 0 = vec4 fragColor (interpolated from vertex)
/// Output: location 0 = vec4 outColor (to color attachment)
pub fn build_fragment_shader() -> Vec<u32> {
    let mut s: Vec<u32> = vec![0; 5];

    let ty_void = 1u32;
    let ty_fn_void = 2;
    let ty_float = 3;
    let ty_vec4 = 4;
    let ty_ptr_in_vec4 = 5;
    let ty_ptr_out_vec4 = 6;
    let in_color = 7;
    let out_color = 8;
    let main_fn = 9;
    let lbl_entry = 10;
    let r_color = 11;
    let bound = 12u32;

    // Capability: Shader
    emit!(s, OP_CAP, 1);

    // Memory model: Logical, GLSL450
    emit!(s, OP_MEM_MODEL, 0, 1);

    // Entry point: Fragment shader (ExecutionModel 4)
    emit_str!(s, OP_ENTRY, [4, main_fn], "main", [in_color, out_color]);

    // Execution mode: OriginUpperLeft
    emit!(s, OP_EXEC_MODE, main_fn, EM_ORIGIN_UPPER_LEFT);

    // Decorations
    emit!(s, OP_DECORATE, in_color, DEC_LOCATION, 0);
    emit!(s, OP_DECORATE, out_color, DEC_LOCATION, 0);

    // Types
    emit!(s, OP_TYPE_VOID, ty_void);
    emit!(s, OP_TYPE_FN, ty_fn_void, ty_void);
    emit!(s, OP_TYPE_FLOAT, ty_float, 32);
    emit!(s, OP_TYPE_VEC, ty_vec4, ty_float, 4);
    emit!(s, OP_TYPE_PTR, ty_ptr_in_vec4, SC_INPUT, ty_vec4);
    emit!(s, OP_TYPE_PTR, ty_ptr_out_vec4, SC_OUTPUT, ty_vec4);

    // Variables
    emit!(s, OP_VAR, ty_ptr_in_vec4, in_color, SC_INPUT);
    emit!(s, OP_VAR, ty_ptr_out_vec4, out_color, SC_OUTPUT);

    // Function
    emit!(s, OP_FN, ty_void, main_fn, 0, ty_fn_void);
    emit!(s, OP_LABEL, lbl_entry);

    emit!(s, OP_LOAD, ty_vec4, r_color, in_color);
    emit!(s, OP_STORE, out_color, r_color);

    emit!(s, OP_RETURN);
    emit!(s, OP_FN_END);

    header(&mut s, bound);
    s
}
