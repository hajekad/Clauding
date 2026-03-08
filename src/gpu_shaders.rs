// SPIR-V graphics shader binaries (vertex + fragment), built programmatically
// Vertex: VP transform + GPU-side lighting (sun, ambient, fog, emissive) via push constants
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
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
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
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_VECTOR_SHUFFLE: u16 = 79;
const OP_MAT_TIMES_VEC: u16 = 145;
const OP_VEC_TIMES_SCALAR: u16 = 142;
const OP_DOT: u16 = 148;
const OP_FMUL: u16 = 133;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;

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

/// Vertex shader: VP transform + GPU-side lighting/fog + emissive glow
///
/// Inputs:  location 0 = vec3 position, location 1 = vec4 color (B8G8R8A8_UNORM), location 2 = vec3 normal
/// Outputs: gl_Position (BuiltIn), location 0 = vec4 fragColor (lit + fogged)
/// Push constants (128 bytes):
///   mat4 VP                          (offset 0,  64 bytes)
///   vec4 light_dir_ambient           (offset 64, xyz=light_dir, w=ambient)
///   vec4 sun_fog_params              (offset 80, x=sun_strength, y=fog_dist_sq_inv, z=fwd_x, w=fwd_z)
///   vec4 fog_color                   (offset 96, xyz=fog_color normalized 0-1)
///   vec4 eye_pos                     (offset 112, xyz=camera position)
///
/// Emissive system: alpha < 1.0 signals emissive geometry (windows, lamp globes, vehicle lights).
/// Emissive verts bypass lighting and glow brighter at night, with reduced fog.
pub fn build_vertex_shader() -> Vec<u32> {
    let mut s: Vec<u32> = vec![0; 5];

    // GLSL.std.450 import
    let glsl = 1u32;

    // Type IDs
    let ty_void = 2u32;
    let ty_fn_void = 3;
    let ty_float = 4;
    let ty_uint = 5;
    let ty_vec3 = 6;
    let ty_vec4 = 7;
    let ty_mat4 = 8;
    let ty_ptr_in_vec3 = 9;
    let ty_ptr_in_vec4 = 10;
    let ty_ptr_out_vec4 = 11;
    let ty_pc_struct = 12;
    let ty_ptr_pc = 13;
    let ty_ptr_pc_mat4 = 14;
    let ty_ptr_pc_vec4 = 15;

    // Variable IDs
    let in_pos = 16;
    let in_color = 17;
    let in_normal = 18;
    let gl_position = 19;
    let out_frag_color = 20;
    let pc_var = 21;

    // Constant IDs
    let c_0u = 22;
    let c_1u = 23;
    let c_2u = 24;
    let c_3u = 25;
    let c_4u = 26;
    let c_0f = 27;
    let c_1f = 28;
    let c_01f = 29;
    let c_13f = 30;
    // New constants for emissive
    let c_07f = 70;   // 0.7
    let c_20f = 71;   // 2.0
    let c_25f = 72;   // 2.5

    // Function IDs
    let main_fn = 31;
    let lbl_entry = 32;

    // Temp result IDs
    let r_pos = 33u32;
    let r_color = 34;
    let r_normal = 35;
    let r_pos4 = 36;
    let r_vp_ptr = 37;
    let r_vp = 38;
    let r_clip_pos = 39;
    // Push constant loads
    let r_ld_ptr = 40;
    let r_ld_vec = 41;
    let r_light_dir = 42;
    let r_ambient = 43;
    let r_sf_ptr = 44;
    let r_sf_vec = 45;
    let r_sun_str = 46;
    let r_fog_inv = 47;
    let r_fc_ptr = 48;
    let r_fc_vec = 49;
    let r_fog_rgb = 50;
    let r_ep_ptr = 51;
    let r_ep_vec = 52;
    let r_eye_xyz = 53;
    // Lighting
    let r_sun_dot = 54;
    let r_sun_max = 55;
    let r_sun_lit = 56;
    let r_int_raw = 57;
    let r_intensity = 58;
    // Fog
    let r_to_eye = 59;
    let r_dist_sq = 60;
    let r_fog_raw = 61;
    let r_fog_sq = 62;
    // Color mixing
    let r_color_rgb = 63;
    let r_lit_rgb = 64;
    let r_inv_fog = 65;
    let r_lit_scaled = 66;
    let r_fog_scaled = 67;
    let r_final_rgb = 68;
    let r_final = 69;
    // Emissive path (IDs 73+)
    let r_alpha = 73;
    let r_emissive_f = 74;    // 1.0 - alpha (0=normal, 1=emissive)
    let r_sun_x2 = 75;        // sun_strength * 2.0
    let r_boost_raw = 76;     // 2.0 - sun_x2
    let r_boost = 77;         // clamp(boost_raw, 1.0, 2.5) — night glow intensity
    let r_int_alpha = 78;     // intensity * alpha
    let r_boost_ef = 79;      // boost * emissive_f
    let r_mixed_int = 80;     // final intensity (normal+emissive blended)
    let r_ef_07 = 81;         // emissive_f * 0.7
    let r_fog_reduce = 82;    // 1.0 - ef_07 (fog reduction factor)
    let r_eff_fog = 83;       // fog_sq * fog_reduce (effective fog for this vertex)

    let bound = 84u32;

    // --- Preamble (must follow SPIR-V logical layout order) ---

    // 1. Capability
    emit!(s, OP_CAP, 1); // Shader

    // 2. ExtInstImport
    emit_str!(s, OP_EXT_INST_IMPORT, [glsl], "GLSL.std.450", []);

    // 3. Memory model: Logical, GLSL450
    emit!(s, OP_MEM_MODEL, 0, 1);

    // 4. Entry point: Vertex shader
    emit_str!(s, OP_ENTRY, [0, main_fn], "main",
        [in_pos, in_color, in_normal, gl_position, out_frag_color]);

    // --- Decorations ---
    emit!(s, OP_DECORATE, in_pos, DEC_LOCATION, 0);
    emit!(s, OP_DECORATE, in_color, DEC_LOCATION, 1);
    emit!(s, OP_DECORATE, in_normal, DEC_LOCATION, 2);
    emit!(s, OP_DECORATE, gl_position, DEC_BUILTIN, BI_POSITION);
    emit!(s, OP_DECORATE, out_frag_color, DEC_LOCATION, 0);
    emit!(s, OP_DECORATE, ty_pc_struct, DEC_BLOCK);
    // Member 0: mat4 VP
    emit!(s, OP_MEMBER_DEC, ty_pc_struct, 0, DEC_OFFSET, 0);
    emit!(s, OP_MEMBER_DEC, ty_pc_struct, 0, DEC_COL_MAJOR);
    emit!(s, OP_MEMBER_DEC, ty_pc_struct, 0, DEC_MAT_STRIDE, 16);
    // Member 1: vec4 light_dir_ambient
    emit!(s, OP_MEMBER_DEC, ty_pc_struct, 1, DEC_OFFSET, 64);
    // Member 2: vec4 sun_fog_params
    emit!(s, OP_MEMBER_DEC, ty_pc_struct, 2, DEC_OFFSET, 80);
    // Member 3: vec4 fog_color
    emit!(s, OP_MEMBER_DEC, ty_pc_struct, 3, DEC_OFFSET, 96);
    // Member 4: vec4 eye_pos
    emit!(s, OP_MEMBER_DEC, ty_pc_struct, 4, DEC_OFFSET, 112);

    // --- Types ---
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
    emit!(s, OP_TYPE_STRUCT, ty_pc_struct, ty_mat4, ty_vec4, ty_vec4, ty_vec4, ty_vec4);
    emit!(s, OP_TYPE_PTR, ty_ptr_pc, SC_PUSH_CONST, ty_pc_struct);
    emit!(s, OP_TYPE_PTR, ty_ptr_pc_mat4, SC_PUSH_CONST, ty_mat4);
    emit!(s, OP_TYPE_PTR, ty_ptr_pc_vec4, SC_PUSH_CONST, ty_vec4);

    // --- Variables ---
    emit!(s, OP_VAR, ty_ptr_in_vec3, in_pos, SC_INPUT);
    emit!(s, OP_VAR, ty_ptr_in_vec4, in_color, SC_INPUT);
    emit!(s, OP_VAR, ty_ptr_in_vec3, in_normal, SC_INPUT);
    emit!(s, OP_VAR, ty_ptr_out_vec4, gl_position, SC_OUTPUT);
    emit!(s, OP_VAR, ty_ptr_out_vec4, out_frag_color, SC_OUTPUT);
    emit!(s, OP_VAR, ty_ptr_pc, pc_var, SC_PUSH_CONST);

    // --- Constants ---
    emit!(s, OP_CONST, ty_uint, c_0u, 0);
    emit!(s, OP_CONST, ty_uint, c_1u, 1);
    emit!(s, OP_CONST, ty_uint, c_2u, 2);
    emit!(s, OP_CONST, ty_uint, c_3u, 3);
    emit!(s, OP_CONST, ty_uint, c_4u, 4);
    emit!(s, OP_CONST, ty_float, c_0f, 0u32);                   // 0.0
    emit!(s, OP_CONST, ty_float, c_1f, 0x3F800000u32);          // 1.0
    emit!(s, OP_CONST, ty_float, c_01f, 0.1_f32.to_bits());     // 0.1
    emit!(s, OP_CONST, ty_float, c_13f, 1.3_f32.to_bits());     // 1.3
    emit!(s, OP_CONST, ty_float, c_07f, 0.7_f32.to_bits());     // 0.7
    emit!(s, OP_CONST, ty_float, c_20f, 2.0_f32.to_bits());     // 2.0
    emit!(s, OP_CONST, ty_float, c_25f, 2.5_f32.to_bits());     // 2.5

    // --- Function ---
    emit!(s, OP_FN, ty_void, main_fn, 0, ty_fn_void);
    emit!(s, OP_LABEL, lbl_entry);

    // Load vertex inputs
    emit!(s, OP_LOAD, ty_vec3, r_pos, in_pos);
    emit!(s, OP_LOAD, ty_vec4, r_color, in_color);
    emit!(s, OP_LOAD, ty_vec3, r_normal, in_normal);

    // Transform: gl_Position = VP * vec4(pos, 1.0)
    emit!(s, OP_COMPOSITE_CONSTRUCT, ty_vec4, r_pos4, r_pos, c_1f);
    emit!(s, OP_ACCESS, ty_ptr_pc_mat4, r_vp_ptr, pc_var, c_0u);
    emit!(s, OP_LOAD, ty_mat4, r_vp, r_vp_ptr);
    emit!(s, OP_MAT_TIMES_VEC, ty_vec4, r_clip_pos, r_vp, r_pos4);
    emit!(s, OP_STORE, gl_position, r_clip_pos);

    // Load push constants: lighting params
    emit!(s, OP_ACCESS, ty_ptr_pc_vec4, r_ld_ptr, pc_var, c_1u);
    emit!(s, OP_LOAD, ty_vec4, r_ld_vec, r_ld_ptr);
    emit!(s, OP_VECTOR_SHUFFLE, ty_vec3, r_light_dir, r_ld_vec, r_ld_vec, 0, 1, 2);
    emit!(s, OP_COMPOSITE_EXTRACT, ty_float, r_ambient, r_ld_vec, 3);

    emit!(s, OP_ACCESS, ty_ptr_pc_vec4, r_sf_ptr, pc_var, c_2u);
    emit!(s, OP_LOAD, ty_vec4, r_sf_vec, r_sf_ptr);
    emit!(s, OP_COMPOSITE_EXTRACT, ty_float, r_sun_str, r_sf_vec, 0);
    emit!(s, OP_COMPOSITE_EXTRACT, ty_float, r_fog_inv, r_sf_vec, 1);

    emit!(s, OP_ACCESS, ty_ptr_pc_vec4, r_fc_ptr, pc_var, c_3u);
    emit!(s, OP_LOAD, ty_vec4, r_fc_vec, r_fc_ptr);
    emit!(s, OP_VECTOR_SHUFFLE, ty_vec3, r_fog_rgb, r_fc_vec, r_fc_vec, 0, 1, 2);

    emit!(s, OP_ACCESS, ty_ptr_pc_vec4, r_ep_ptr, pc_var, c_4u);
    emit!(s, OP_LOAD, ty_vec4, r_ep_vec, r_ep_ptr);
    emit!(s, OP_VECTOR_SHUFFLE, ty_vec3, r_eye_xyz, r_ep_vec, r_ep_vec, 0, 1, 2);

    // === Lighting ===
    // Normal lighting: intensity = clamp(max(dot(normal, light_dir), 0) * sun + ambient, 0.1, 1.3)
    emit!(s, OP_DOT, ty_float, r_sun_dot, r_normal, r_light_dir);
    emit!(s, OP_EXT_INST, ty_float, r_sun_max, glsl, 40, r_sun_dot, c_0f);   // FMax
    emit!(s, OP_FMUL, ty_float, r_sun_lit, r_sun_max, r_sun_str);
    emit!(s, OP_FADD, ty_float, r_int_raw, r_sun_lit, r_ambient);
    emit!(s, OP_EXT_INST, ty_float, r_intensity, glsl, 43, r_int_raw, c_01f, c_13f); // FClamp

    // === Fog ===
    // fog_sq = min(dot(eye-pos, eye-pos) * fog_dist_sq_inv, 1.0)
    emit!(s, OP_FSUB, ty_vec3, r_to_eye, r_eye_xyz, r_pos);
    emit!(s, OP_DOT, ty_float, r_dist_sq, r_to_eye, r_to_eye);
    emit!(s, OP_FMUL, ty_float, r_fog_raw, r_dist_sq, r_fog_inv);
    emit!(s, OP_EXT_INST, ty_float, r_fog_sq, glsl, 37, r_fog_raw, c_1f);    // FMin

    // === Emissive system (branchless) ===
    // alpha = color.w (1.0 = normal, 0.0 = emissive)
    // emissive_f = 1.0 - alpha (0.0 = normal, 1.0 = emissive)
    emit!(s, OP_COMPOSITE_EXTRACT, ty_float, r_alpha, r_color, 3);
    emit!(s, OP_FSUB, ty_float, r_emissive_f, c_1f, r_alpha);

    // Emissive boost: clamp(2.0 - sun_strength*2.0, 1.0, 2.5)
    // Night (sun=0): boost=2.0, Day (sun=0.65): boost=0.7→clamped to 1.0
    emit!(s, OP_FMUL, ty_float, r_sun_x2, r_sun_str, c_20f);
    emit!(s, OP_FSUB, ty_float, r_boost_raw, c_20f, r_sun_x2);
    emit!(s, OP_EXT_INST, ty_float, r_boost, glsl, 43, r_boost_raw, c_1f, c_25f); // FClamp

    // Blend intensity: intensity*alpha + boost*emissive_f
    // Normal verts (alpha=1): intensity*1 + boost*0 = intensity
    // Emissive verts (alpha=0): intensity*0 + boost*1 = boost
    emit!(s, OP_FMUL, ty_float, r_int_alpha, r_intensity, r_alpha);
    emit!(s, OP_FMUL, ty_float, r_boost_ef, r_boost, r_emissive_f);
    emit!(s, OP_FADD, ty_float, r_mixed_int, r_int_alpha, r_boost_ef);

    // Reduce fog for emissive (lights visible through haze):
    // eff_fog = fog_sq * (1.0 - emissive_f * 0.7)
    // Normal: fog_sq * 1.0 = full fog
    // Emissive: fog_sq * 0.3 = 30% fog
    emit!(s, OP_FMUL, ty_float, r_ef_07, r_emissive_f, c_07f);
    emit!(s, OP_FSUB, ty_float, r_fog_reduce, c_1f, r_ef_07);
    emit!(s, OP_FMUL, ty_float, r_eff_fog, r_fog_sq, r_fog_reduce);

    // === Final color: mix(color.rgb * mixed_intensity, fog_color, eff_fog) ===
    emit!(s, OP_VECTOR_SHUFFLE, ty_vec3, r_color_rgb, r_color, r_color, 0, 1, 2);
    emit!(s, OP_VEC_TIMES_SCALAR, ty_vec3, r_lit_rgb, r_color_rgb, r_mixed_int);
    emit!(s, OP_FSUB, ty_float, r_inv_fog, c_1f, r_eff_fog);
    emit!(s, OP_VEC_TIMES_SCALAR, ty_vec3, r_lit_scaled, r_lit_rgb, r_inv_fog);
    emit!(s, OP_VEC_TIMES_SCALAR, ty_vec3, r_fog_scaled, r_fog_rgb, r_eff_fog);
    emit!(s, OP_FADD, ty_vec3, r_final_rgb, r_lit_scaled, r_fog_scaled);
    emit!(s, OP_COMPOSITE_CONSTRUCT, ty_vec4, r_final, r_final_rgb, c_1f);

    emit!(s, OP_STORE, out_frag_color, r_final);

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
