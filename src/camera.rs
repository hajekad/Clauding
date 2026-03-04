// sys_camera: mouse-driven orbit camera (Elden Ring style)
// Yaw/pitch orbit around the player, smooth follow

use crate::state::*;

const CAM_DIST_WALK: f32 = 8.0;
const CAM_DIST_DRIVE: f32 = 14.0;
const CAM_HEIGHT_OFFSET: f32 = 1.5; // extra height above orbit center
const CAM_LERP_WALK: f32 = 0.1;
const CAM_LERP_DRIVE: f32 = 0.12;
const PITCH_MIN: f32 = 0.05;  // ~3 degrees (nearly level)
const PITCH_MAX: f32 = 1.2;   // ~69 degrees (steep overhead)
const MOUSE_SPEED: f32 = 0.003;

pub fn sys_camera(
    cam: &mut Camera, player: &Player,
    mouse_dx: f32, mouse_dy: f32,
    sensitivity: f32, invert_x: bool, invert_y: bool,
    _dt: f32,
) {
    // Apply mouse input to yaw/pitch
    // Default: mouse right = look right (yaw decreases = camera orbits left = world moves right)
    let dx = if invert_x { mouse_dx } else { -mouse_dx };
    let dy = if invert_y { -mouse_dy } else { mouse_dy };

    cam.yaw += dx * MOUSE_SPEED * sensitivity;
    cam.pitch = (cam.pitch + dy * MOUSE_SPEED * sensitivity).clamp(PITCH_MIN, PITCH_MAX);

    // Keep yaw in [-PI, PI]
    while cam.yaw > std::f32::consts::PI { cam.yaw -= 2.0 * std::f32::consts::PI; }
    while cam.yaw < -std::f32::consts::PI { cam.yaw += 2.0 * std::f32::consts::PI; }

    let (dist, lerp) = if player.in_vehicle.is_some() {
        (CAM_DIST_DRIVE, CAM_LERP_DRIVE)
    } else {
        (CAM_DIST_WALK, CAM_LERP_WALK)
    };

    // Spherical offset from player
    let cos_p = cam.pitch.cos();
    let sin_p = cam.pitch.sin();
    let target_x = player.x + cam.yaw.sin() * cos_p * dist;
    let target_y = player.y + sin_p * dist + CAM_HEIGHT_OFFSET;
    let target_z = player.z + cam.yaw.cos() * cos_p * dist;

    // Smooth follow
    cam.x += (target_x - cam.x) * lerp;
    cam.y += (target_y - cam.y) * lerp;
    cam.z += (target_z - cam.z) * lerp;

    // Look at player chest
    cam.tx = player.x;
    cam.ty = player.y + 1.2;
    cam.tz = player.z;
}
