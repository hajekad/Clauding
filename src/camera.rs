// sys_camera: smooth follow behind player/vehicle, interpolation

use crate::state::*;

const CAM_BEHIND_WALK: f32 = 8.0;
const CAM_HEIGHT_WALK: f32 = 5.0;
const CAM_BEHIND_DRIVE: f32 = 14.0;
const CAM_HEIGHT_DRIVE: f32 = 7.0;
const CAM_LERP_WALK: f32 = 0.06;
const CAM_LERP_DRIVE: f32 = 0.08;

pub fn sys_camera(cam: &mut Camera, player: &Player, _dt: f32) {
    let (behind, height, lerp) = if player.in_vehicle.is_some() {
        (CAM_BEHIND_DRIVE, CAM_HEIGHT_DRIVE, CAM_LERP_DRIVE)
    } else {
        (CAM_BEHIND_WALK, CAM_HEIGHT_WALK, CAM_LERP_WALK)
    };

    let target_x = player.x + player.rot_y.sin() * behind;
    let target_z = player.z + player.rot_y.cos() * behind;
    let target_y = player.y + height;

    cam.x += (target_x - cam.x) * lerp;
    cam.y += (target_y - cam.y) * lerp;
    cam.z += (target_z - cam.z) * lerp;

    cam.tx = player.x;
    cam.ty = player.y + 1.2;
    cam.tz = player.z;
}
