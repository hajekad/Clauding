// SPIR-V compute shader binaries, built programmatically
// Each shader is a function returning Vec<u32> (SPIR-V words)

#![allow(unused)]

use crate::gpu_spirv::*;
use crate::{emit, emit_str};

// SPIR-V opcodes (compute-specific)
const OP_TYPE_RT_ARR: u16 = 29;
const OP_TYPE_BOOL: u16 = 20;
const OP_UGTEQ: u16 = 174;
const OP_SEL_MERGE: u16 = 247;
const OP_BR_COND: u16 = 250;
const OP_BR: u16 = 249;

// Decoration values (compute-specific)
const DEC_DESC_SET: u32 = 34;
const DEC_BINDING: u32 = 33;
const DEC_STRIDE: u32 = 6;

// Storage classes (compute-specific)
const SC_STORAGE_BUF: u32 = 12;

// Built-in values
const BI_GID: u32 = 28;

// Execution mode values
const EM_LOCAL_SIZE: u32 = 17;

// Test shader: buf[gid] *= 2.0
// 1 storage buffer (set=0, binding=0), push constant = { count: u32 }
pub fn build_test_multiply() -> Vec<u32> {
    let mut s: Vec<u32> = vec![0; 5];

    // IDs
    let (ty_void, ty_fn) = (1u32, 2);
    let (ty_u32, ty_f32) = (3, 4);
    let (ty_uvec3, ty_ptr_in_uvec3) = (5, 6);
    let gid_var = 7u32;
    let ty_ptr_in_u32 = 8u32;
    let (c_0, c_2f) = (9u32, 10);
    let ty_rt_arr = 11u32;
    let (ty_buf, ty_ptr_sb_buf) = (12, 13);
    let buf_var = 14u32;
    let ty_ptr_sb_f32 = 15u32;
    let ty_bool = 16u32;
    let (ty_pc, ty_ptr_pc, pc_var, ty_ptr_pc_u32) = (17, 18, 19, 20);
    let main_fn = 21u32;
    let (lbl_entry, lbl_merge, lbl_ret, lbl_body) = (22, 23, 24, 25);
    let (r_gp, r_gid, r_cp, r_cnt, r_cmp) = (26, 27, 28, 29, 30);
    let (r_dp, r_val, r_res) = (31, 32, 33);
    let bound = 34u32;

    // Capability
    emit!(s, OP_CAP, 1); // Shader

    // Memory model
    emit!(s, OP_MEM_MODEL, 0, 1); // Logical, GLSL450

    // Entry point
    emit_str!(s, OP_ENTRY, [5, main_fn], "main", [gid_var]);

    // Execution mode
    emit!(s, OP_EXEC_MODE, main_fn, EM_LOCAL_SIZE, 64, 1, 1);

    // Decorations
    emit!(s, OP_DECORATE, gid_var, DEC_BUILTIN, BI_GID);
    emit!(s, OP_DECORATE, ty_buf, DEC_BLOCK);
    emit!(s, OP_MEMBER_DEC, ty_buf, 0, DEC_OFFSET, 0);
    emit!(s, OP_DECORATE, buf_var, DEC_DESC_SET, 0);
    emit!(s, OP_DECORATE, buf_var, DEC_BINDING, 0);
    emit!(s, OP_DECORATE, ty_rt_arr, DEC_STRIDE, 4);
    emit!(s, OP_DECORATE, ty_pc, DEC_BLOCK);
    emit!(s, OP_MEMBER_DEC, ty_pc, 0, DEC_OFFSET, 0);

    // Types
    emit!(s, OP_TYPE_VOID, ty_void);
    emit!(s, OP_TYPE_FN, ty_fn, ty_void);
    emit!(s, OP_TYPE_INT, ty_u32, 32, 0);
    emit!(s, OP_TYPE_FLOAT, ty_f32, 32);
    emit!(s, OP_TYPE_VEC, ty_uvec3, ty_u32, 3);
    emit!(s, OP_TYPE_PTR, ty_ptr_in_uvec3, SC_INPUT, ty_uvec3);
    emit!(s, OP_VAR, ty_ptr_in_uvec3, gid_var, SC_INPUT);
    emit!(s, OP_TYPE_PTR, ty_ptr_in_u32, SC_INPUT, ty_u32);
    emit!(s, OP_CONST, ty_u32, c_0, 0);
    emit!(s, OP_CONST, ty_f32, c_2f, 0x40000000u32); // 2.0f
    emit!(s, OP_TYPE_RT_ARR, ty_rt_arr, ty_f32);
    emit!(s, OP_TYPE_STRUCT, ty_buf, ty_rt_arr);
    emit!(s, OP_TYPE_PTR, ty_ptr_sb_buf, SC_STORAGE_BUF, ty_buf);
    emit!(s, OP_VAR, ty_ptr_sb_buf, buf_var, SC_STORAGE_BUF);
    emit!(s, OP_TYPE_PTR, ty_ptr_sb_f32, SC_STORAGE_BUF, ty_f32);
    emit!(s, OP_TYPE_BOOL, ty_bool);
    emit!(s, OP_TYPE_STRUCT, ty_pc, ty_u32);
    emit!(s, OP_TYPE_PTR, ty_ptr_pc, SC_PUSH_CONST, ty_pc);
    emit!(s, OP_VAR, ty_ptr_pc, pc_var, SC_PUSH_CONST);
    emit!(s, OP_TYPE_PTR, ty_ptr_pc_u32, SC_PUSH_CONST, ty_u32);

    // Function
    emit!(s, OP_FN, ty_void, main_fn, 0, ty_fn);
    emit!(s, OP_LABEL, lbl_entry);

    emit!(s, OP_ACCESS, ty_ptr_in_u32, r_gp, gid_var, c_0);
    emit!(s, OP_LOAD, ty_u32, r_gid, r_gp);
    emit!(s, OP_ACCESS, ty_ptr_pc_u32, r_cp, pc_var, c_0);
    emit!(s, OP_LOAD, ty_u32, r_cnt, r_cp);

    emit!(s, OP_UGTEQ, ty_bool, r_cmp, r_gid, r_cnt);
    emit!(s, OP_SEL_MERGE, lbl_merge, 0);
    emit!(s, OP_BR_COND, r_cmp, lbl_ret, lbl_body);

    emit!(s, OP_LABEL, lbl_ret);
    emit!(s, OP_RETURN);

    emit!(s, OP_LABEL, lbl_body);
    emit!(s, OP_ACCESS, ty_ptr_sb_f32, r_dp, buf_var, c_0, r_gid);
    emit!(s, OP_LOAD, ty_f32, r_val, r_dp);
    emit!(s, OP_FMUL, ty_f32, r_res, r_val, c_2f);
    emit!(s, OP_STORE, r_dp, r_res);
    emit!(s, OP_BR, lbl_merge);

    emit!(s, OP_LABEL, lbl_merge);
    emit!(s, OP_RETURN);
    emit!(s, OP_FN_END);

    header(&mut s, bound);
    s
}

// Particle update shader: pos += vel * dt, lifetime -= dt
// Binding 0: pos_x (f32[]), Binding 1: pos_y (f32[]), Binding 2: pos_z (f32[])
// Binding 3: vel_x (f32[]), Binding 4: vel_y (f32[]), Binding 5: vel_z (f32[])
// Binding 6: lifetime (f32[])
// Push constants: { count: u32, dt: f32, gravity: f32 }
pub fn build_particle_update() -> Vec<u32> {
    let mut s: Vec<u32> = vec![0; 5];

    // Types: reuse a single buffer struct type, bind 7 buffers
    let ty_void = 1u32;
    let ty_fn = 2;
    let (ty_u32, ty_f32) = (3, 4);
    let (ty_uvec3, ty_ptr_in_uvec3) = (5, 6);
    let gid_var = 7u32;
    let ty_ptr_in_u32 = 8;
    let (c_0, c_1, c_2) = (9u32, 10, 11); // uint constants for indexing push constants
    let ty_rt_arr = 12;
    let ty_buf = 13;
    let ty_ptr_sb_buf = 14;
    // 7 buffer variables: 15-21
    let buf_px = 15u32; let buf_py = 16; let buf_pz = 17;
    let buf_vx = 18; let buf_vy = 19; let buf_vz = 20;
    let buf_lt = 21;
    let ty_ptr_sb_f32 = 22;
    let ty_bool = 23;
    // Push constant struct: { count: u32, dt: f32, gravity: f32 }
    let ty_pc = 24;
    let ty_ptr_pc = 25;
    let pc_var = 26;
    let ty_ptr_pc_u32 = 27;
    let ty_ptr_pc_f32 = 28;
    let main_fn = 29;
    let (lbl_entry, lbl_merge, lbl_ret, lbl_body) = (30, 31, 32, 33);

    let mut next = 34u32;
    macro_rules! nid { () => {{ let id = next; next += 1; id }} }

    emit!(s, OP_CAP, 1);
    emit!(s, OP_MEM_MODEL, 0, 1);
    emit_str!(s, OP_ENTRY, [5, main_fn], "main", [gid_var]);
    emit!(s, OP_EXEC_MODE, main_fn, EM_LOCAL_SIZE, 64, 1, 1);

    // Decorations
    emit!(s, OP_DECORATE, gid_var, DEC_BUILTIN, BI_GID);
    emit!(s, OP_DECORATE, ty_buf, DEC_BLOCK);
    emit!(s, OP_MEMBER_DEC, ty_buf, 0, DEC_OFFSET, 0);
    emit!(s, OP_DECORATE, ty_rt_arr, DEC_STRIDE, 4);
    for (i, &var) in [buf_px, buf_py, buf_pz, buf_vx, buf_vy, buf_vz, buf_lt].iter().enumerate() {
        emit!(s, OP_DECORATE, var, DEC_DESC_SET, 0);
        emit!(s, OP_DECORATE, var, DEC_BINDING, i as u32);
    }
    emit!(s, OP_DECORATE, ty_pc, DEC_BLOCK);
    emit!(s, OP_MEMBER_DEC, ty_pc, 0, DEC_OFFSET, 0);
    emit!(s, OP_MEMBER_DEC, ty_pc, 1, DEC_OFFSET, 4);
    emit!(s, OP_MEMBER_DEC, ty_pc, 2, DEC_OFFSET, 8);

    // Types
    emit!(s, OP_TYPE_VOID, ty_void);
    emit!(s, OP_TYPE_FN, ty_fn, ty_void);
    emit!(s, OP_TYPE_INT, ty_u32, 32, 0);
    emit!(s, OP_TYPE_FLOAT, ty_f32, 32);
    emit!(s, OP_TYPE_VEC, ty_uvec3, ty_u32, 3);
    emit!(s, OP_TYPE_PTR, ty_ptr_in_uvec3, SC_INPUT, ty_uvec3);
    emit!(s, OP_VAR, ty_ptr_in_uvec3, gid_var, SC_INPUT);
    emit!(s, OP_TYPE_PTR, ty_ptr_in_u32, SC_INPUT, ty_u32);
    emit!(s, OP_CONST, ty_u32, c_0, 0);
    emit!(s, OP_CONST, ty_u32, c_1, 1);
    emit!(s, OP_CONST, ty_u32, c_2, 2);
    emit!(s, OP_TYPE_RT_ARR, ty_rt_arr, ty_f32);
    emit!(s, OP_TYPE_STRUCT, ty_buf, ty_rt_arr);
    emit!(s, OP_TYPE_PTR, ty_ptr_sb_buf, SC_STORAGE_BUF, ty_buf);
    for &var in &[buf_px, buf_py, buf_pz, buf_vx, buf_vy, buf_vz, buf_lt] {
        emit!(s, OP_VAR, ty_ptr_sb_buf, var, SC_STORAGE_BUF);
    }
    emit!(s, OP_TYPE_PTR, ty_ptr_sb_f32, SC_STORAGE_BUF, ty_f32);
    emit!(s, OP_TYPE_BOOL, ty_bool);
    emit!(s, OP_TYPE_STRUCT, ty_pc, ty_u32, ty_f32, ty_f32); // count, dt, gravity
    emit!(s, OP_TYPE_PTR, ty_ptr_pc, SC_PUSH_CONST, ty_pc);
    emit!(s, OP_VAR, ty_ptr_pc, pc_var, SC_PUSH_CONST);
    emit!(s, OP_TYPE_PTR, ty_ptr_pc_u32, SC_PUSH_CONST, ty_u32);
    emit!(s, OP_TYPE_PTR, ty_ptr_pc_f32, SC_PUSH_CONST, ty_f32);

    // Function
    emit!(s, OP_FN, ty_void, main_fn, 0, ty_fn);
    emit!(s, OP_LABEL, lbl_entry);

    let r_gp = nid!(); let r_gid = nid!();
    emit!(s, OP_ACCESS, ty_ptr_in_u32, r_gp, gid_var, c_0);
    emit!(s, OP_LOAD, ty_u32, r_gid, r_gp);

    let r_cp = nid!(); let r_cnt = nid!();
    emit!(s, OP_ACCESS, ty_ptr_pc_u32, r_cp, pc_var, c_0);
    emit!(s, OP_LOAD, ty_u32, r_cnt, r_cp);

    let r_cmp = nid!();
    emit!(s, OP_UGTEQ, ty_bool, r_cmp, r_gid, r_cnt);
    emit!(s, OP_SEL_MERGE, lbl_merge, 0);
    emit!(s, OP_BR_COND, r_cmp, lbl_ret, lbl_body);

    emit!(s, OP_LABEL, lbl_ret);
    emit!(s, OP_RETURN);

    emit!(s, OP_LABEL, lbl_body);

    // Load dt and gravity
    let r_dtp = nid!(); let r_dt = nid!();
    emit!(s, OP_ACCESS, ty_ptr_pc_f32, r_dtp, pc_var, c_1);
    emit!(s, OP_LOAD, ty_f32, r_dt, r_dtp);

    let r_grvp = nid!(); let r_grv = nid!();
    emit!(s, OP_ACCESS, ty_ptr_pc_f32, r_grvp, pc_var, c_2);
    emit!(s, OP_LOAD, ty_f32, r_grv, r_grvp);

    // Load velocities
    let r_vxp = nid!(); let r_vx = nid!();
    emit!(s, OP_ACCESS, ty_ptr_sb_f32, r_vxp, buf_vx, c_0, r_gid);
    emit!(s, OP_LOAD, ty_f32, r_vx, r_vxp);

    let r_vyp = nid!(); let r_vy = nid!();
    emit!(s, OP_ACCESS, ty_ptr_sb_f32, r_vyp, buf_vy, c_0, r_gid);
    emit!(s, OP_LOAD, ty_f32, r_vy, r_vyp);

    let r_vzp = nid!(); let r_vz = nid!();
    emit!(s, OP_ACCESS, ty_ptr_sb_f32, r_vzp, buf_vz, c_0, r_gid);
    emit!(s, OP_LOAD, ty_f32, r_vz, r_vzp);

    // vel_y += gravity * dt
    let r_gdt = nid!();
    emit!(s, OP_FMUL, ty_f32, r_gdt, r_grv, r_dt);
    let r_vy2 = nid!();
    emit!(s, OP_FADD, ty_f32, r_vy2, r_vy, r_gdt);
    emit!(s, OP_STORE, r_vyp, r_vy2);

    // Load positions and update: pos += vel * dt
    let r_pxp = nid!(); let r_px = nid!();
    emit!(s, OP_ACCESS, ty_ptr_sb_f32, r_pxp, buf_px, c_0, r_gid);
    emit!(s, OP_LOAD, ty_f32, r_px, r_pxp);
    let r_dx = nid!(); let r_px2 = nid!();
    emit!(s, OP_FMUL, ty_f32, r_dx, r_vx, r_dt);
    emit!(s, OP_FADD, ty_f32, r_px2, r_px, r_dx);
    emit!(s, OP_STORE, r_pxp, r_px2);

    let r_pyp = nid!(); let r_py = nid!();
    emit!(s, OP_ACCESS, ty_ptr_sb_f32, r_pyp, buf_py, c_0, r_gid);
    emit!(s, OP_LOAD, ty_f32, r_py, r_pyp);
    let r_dy = nid!(); let r_py2 = nid!();
    emit!(s, OP_FMUL, ty_f32, r_dy, r_vy2, r_dt);
    emit!(s, OP_FADD, ty_f32, r_py2, r_py, r_dy);
    emit!(s, OP_STORE, r_pyp, r_py2);

    let r_pzp = nid!(); let r_pz = nid!();
    emit!(s, OP_ACCESS, ty_ptr_sb_f32, r_pzp, buf_pz, c_0, r_gid);
    emit!(s, OP_LOAD, ty_f32, r_pz, r_pzp);
    let r_dz = nid!(); let r_pz2 = nid!();
    emit!(s, OP_FMUL, ty_f32, r_dz, r_vz, r_dt);
    emit!(s, OP_FADD, ty_f32, r_pz2, r_pz, r_dz);
    emit!(s, OP_STORE, r_pzp, r_pz2);

    // lifetime -= dt
    let r_ltp = nid!(); let r_lt = nid!();
    emit!(s, OP_ACCESS, ty_ptr_sb_f32, r_ltp, buf_lt, c_0, r_gid);
    emit!(s, OP_LOAD, ty_f32, r_lt, r_ltp);
    let r_lt2 = nid!();
    emit!(s, OP_FSUB, ty_f32, r_lt2, r_lt, r_dt);
    emit!(s, OP_STORE, r_ltp, r_lt2);

    emit!(s, OP_BR, lbl_merge);
    emit!(s, OP_LABEL, lbl_merge);
    emit!(s, OP_RETURN);
    emit!(s, OP_FN_END);

    header(&mut s, next);
    s
}
