// Wayland platform: window via libwayland-client.so dlopen, shm framebuffer, keyboard input
// Uses raw Wayland protocol through libwayland-client function pointers.
// xdg-shell interfaces defined manually (not in libwayland-client).

#![allow(non_upper_case_globals, unsafe_op_in_unsafe_fn)]

use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr;

// --- libc FFI (always linked in Rust programs) ---

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn memfd_create(name: *const c_char, flags: c_int) -> c_int;
    fn ftruncate(fd: c_int, length: i64) -> c_int;
    fn mmap(addr: *mut c_void, length: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn poll(fds: *mut PollFd, nfds: u64, timeout: c_int) -> c_int;
}

const RTLD_LAZY: c_int = 1;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_SHARED: c_int = 1;
const MFD_CLOEXEC: c_int = 1;
const POLLIN: i16 = 1;
const WL_SHM_FORMAT_XRGB8888: u32 = 1;
const WL_SEAT_CAPABILITY_POINTER: u32 = 1;
const WL_SEAT_CAPABILITY_KEYBOARD: u32 = 2;

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: i16,
    revents: i16,
}

// --- Wayland FFI types ---

#[repr(C)]
pub struct WlInterface {
    name: *const c_char,
    version: c_int,
    method_count: c_int,
    methods: *const WlMessage,
    event_count: c_int,
    events: *const WlMessage,
}
unsafe impl Sync for WlInterface {}

#[repr(C)]
struct WlMessage {
    name: *const c_char,
    signature: *const c_char,
    types: *const *const WlInterface,
}
unsafe impl Sync for WlMessage {}

#[repr(C)]
struct WlArray {
    size: usize,
    alloc: usize,
    data: *mut c_void,
}

// --- Function pointer types ---

type FnDisplayConnect = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type FnDisplayDisconnect = unsafe extern "C" fn(*mut c_void);
type FnDisplayRoundtrip = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnDisplayFlush = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnDisplayGetFd = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnDisplayPrepareRead = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnDisplayReadEvents = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnDisplayCancelRead = unsafe extern "C" fn(*mut c_void);
type FnDisplayDispatchPending = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnProxyMarshalFlags = unsafe extern "C" fn(*mut c_void, u32, *const WlInterface, u32, u32, ...) -> *mut c_void;
type FnProxyAddListener = unsafe extern "C" fn(*mut c_void, *const c_void, *mut c_void) -> c_int;
type FnProxyGetVersion = unsafe extern "C" fn(*mut c_void) -> u32;
type FnProxyDestroy = unsafe extern "C" fn(*mut c_void);

struct WlFns {
    display_connect: FnDisplayConnect,
    display_disconnect: FnDisplayDisconnect,
    display_roundtrip: FnDisplayRoundtrip,
    display_flush: FnDisplayFlush,
    display_get_fd: FnDisplayGetFd,
    display_prepare_read: FnDisplayPrepareRead,
    display_read_events: FnDisplayReadEvents,
    display_cancel_read: FnDisplayCancelRead,
    display_dispatch_pending: FnDisplayDispatchPending,
    proxy_marshal_flags: FnProxyMarshalFlags,
    proxy_add_listener: FnProxyAddListener,
    proxy_get_version: FnProxyGetVersion,
    #[allow(dead_code)]
    proxy_destroy: FnProxyDestroy,
}

// --- xdg-shell interface definitions ---
// These are not in libwayland-client.so, so we define them manually.
// Only need correct signatures and counts; types arrays are all-null.

// Wrapper for raw pointer to make it Sync for static context
#[repr(transparent)]
#[derive(Clone, Copy)]
struct IfacePtr(*const WlInterface);
unsafe impl Sync for IfacePtr {}
const NP: IfacePtr = IfacePtr(ptr::null());

static NULL_TYPES: [IfacePtr; 8] = [NP; 8];

// xdg_wm_base: 4 requests (destroy, create_positioner, get_xdg_surface, pong), 1 event (ping)
static XDG_WM_BASE_REQUESTS: [WlMessage; 4] = [
    WlMessage { name: c"destroy".as_ptr(),            signature: c"".as_ptr(),  types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"create_positioner".as_ptr(),  signature: c"n".as_ptr(), types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"get_xdg_surface".as_ptr(),    signature: c"no".as_ptr(),types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"pong".as_ptr(),               signature: c"u".as_ptr(), types: NULL_TYPES.as_ptr() as *const *const WlInterface },
];
static XDG_WM_BASE_EVENTS: [WlMessage; 1] = [
    WlMessage { name: c"ping".as_ptr(), signature: c"u".as_ptr(), types: NULL_TYPES.as_ptr() as *const *const WlInterface },
];
static XDG_WM_BASE_INTERFACE: WlInterface = WlInterface {
    name: c"xdg_wm_base".as_ptr(), version: 2, method_count: 4,
    methods: XDG_WM_BASE_REQUESTS.as_ptr(), event_count: 1,
    events: XDG_WM_BASE_EVENTS.as_ptr(),
};

// xdg_surface: 5 requests (destroy, get_toplevel, get_popup, set_window_geometry, ack_configure), 1 event (configure)
static XDG_SURFACE_REQUESTS: [WlMessage; 5] = [
    WlMessage { name: c"destroy".as_ptr(),              signature: c"".as_ptr(),    types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"get_toplevel".as_ptr(),         signature: c"n".as_ptr(),   types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"get_popup".as_ptr(),            signature: c"n?oo".as_ptr(),types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"set_window_geometry".as_ptr(),  signature: c"iiii".as_ptr(),types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"ack_configure".as_ptr(),        signature: c"u".as_ptr(),   types: NULL_TYPES.as_ptr() as *const *const WlInterface },
];
static XDG_SURFACE_EVENTS: [WlMessage; 1] = [
    WlMessage { name: c"configure".as_ptr(), signature: c"u".as_ptr(), types: NULL_TYPES.as_ptr() as *const *const WlInterface },
];
static XDG_SURFACE_INTERFACE: WlInterface = WlInterface {
    name: c"xdg_surface".as_ptr(), version: 2, method_count: 5,
    methods: XDG_SURFACE_REQUESTS.as_ptr(), event_count: 1,
    events: XDG_SURFACE_EVENTS.as_ptr(),
};

// xdg_toplevel: 14 requests, 2 events (configure, close) for v1
static XDG_TOPLEVEL_REQUESTS: [WlMessage; 14] = [
    WlMessage { name: c"destroy".as_ptr(),          signature: c"".as_ptr(),    types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"set_parent".as_ptr(),       signature: c"?o".as_ptr(),  types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"set_title".as_ptr(),        signature: c"s".as_ptr(),   types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"set_app_id".as_ptr(),       signature: c"s".as_ptr(),   types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"show_window_menu".as_ptr(), signature: c"ouii".as_ptr(),types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"move".as_ptr(),             signature: c"ou".as_ptr(),  types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"resize".as_ptr(),           signature: c"ouu".as_ptr(), types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"set_max_size".as_ptr(),     signature: c"ii".as_ptr(),  types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"set_min_size".as_ptr(),     signature: c"ii".as_ptr(),  types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"set_maximized".as_ptr(),    signature: c"".as_ptr(),    types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"unset_maximized".as_ptr(),  signature: c"".as_ptr(),    types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"set_fullscreen".as_ptr(),   signature: c"?o".as_ptr(),  types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"unset_fullscreen".as_ptr(), signature: c"".as_ptr(),    types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"set_minimized".as_ptr(),    signature: c"".as_ptr(),    types: NULL_TYPES.as_ptr() as *const *const WlInterface },
];
static XDG_TOPLEVEL_EVENTS: [WlMessage; 4] = [
    WlMessage { name: c"configure".as_ptr(),       signature: c"iia".as_ptr(), types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"close".as_ptr(),           signature: c"".as_ptr(),    types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"configure_bounds".as_ptr(),signature: c"ii".as_ptr(),  types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"wm_capabilities".as_ptr(), signature: c"a".as_ptr(),   types: NULL_TYPES.as_ptr() as *const *const WlInterface },
];
static XDG_TOPLEVEL_INTERFACE: WlInterface = WlInterface {
    name: c"xdg_toplevel".as_ptr(), version: 2, method_count: 14,
    methods: XDG_TOPLEVEL_REQUESTS.as_ptr(), event_count: 4,
    events: XDG_TOPLEVEL_EVENTS.as_ptr(),
};

// --- zwp_relative_pointer_manager_v1 / zwp_relative_pointer_v1 interfaces ---

static ZWP_REL_PTR_MGR_REQUESTS: [WlMessage; 2] = [
    WlMessage { name: c"destroy".as_ptr(), signature: c"".as_ptr(), types: NULL_TYPES.as_ptr() as *const *const WlInterface },
    WlMessage { name: c"get_relative_pointer".as_ptr(), signature: c"no".as_ptr(), types: NULL_TYPES.as_ptr() as *const *const WlInterface },
];
static ZWP_REL_PTR_MGR_INTERFACE: WlInterface = WlInterface {
    name: c"zwp_relative_pointer_manager_v1".as_ptr(), version: 1,
    method_count: 2, methods: ZWP_REL_PTR_MGR_REQUESTS.as_ptr(),
    event_count: 0, events: ptr::null(),
};

static ZWP_REL_PTR_REQUESTS: [WlMessage; 1] = [
    WlMessage { name: c"destroy".as_ptr(), signature: c"".as_ptr(), types: NULL_TYPES.as_ptr() as *const *const WlInterface },
];
static ZWP_REL_PTR_EVENTS: [WlMessage; 1] = [
    WlMessage { name: c"relative_motion".as_ptr(), signature: c"uuffff".as_ptr(), types: NULL_TYPES.as_ptr() as *const *const WlInterface },
];
static ZWP_REL_PTR_INTERFACE: WlInterface = WlInterface {
    name: c"zwp_relative_pointer_v1".as_ptr(), version: 1,
    method_count: 1, methods: ZWP_REL_PTR_REQUESTS.as_ptr(),
    event_count: 1, events: ZWP_REL_PTR_EVENTS.as_ptr(),
};

// --- Listener structs ---

#[repr(C)]
struct RegistryListener {
    global: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const c_char, u32),
    global_remove: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
}

#[repr(C)]
struct XdgWmBaseListener {
    ping: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
}

#[repr(C)]
struct XdgSurfaceListener {
    configure: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
}

#[repr(C)]
struct XdgToplevelListener {
    configure: unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i32, *mut WlArray),
    close: unsafe extern "C" fn(*mut c_void, *mut c_void),
    configure_bounds: unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i32),
    wm_capabilities: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut WlArray),
}

#[repr(C)]
struct SeatListener {
    capabilities: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
    name: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_char),
}

#[repr(C)]
struct KeyboardListener {
    keymap: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, i32, u32),
    enter: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *mut c_void, *mut WlArray),
    leave: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *mut c_void),
    key: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u32, u32, u32),
    modifiers: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u32, u32, u32, u32),
    repeat_info: unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i32),
}

#[repr(C)]
struct BufferListener {
    release: unsafe extern "C" fn(*mut c_void, *mut c_void),
}

#[repr(C)]
struct PointerListener {
    enter: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *mut c_void, i32, i32),
    leave: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *mut c_void),
    motion: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, i32, i32),
    button: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u32, u32, u32),
    axis: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u32, i32),
    frame: unsafe extern "C" fn(*mut c_void, *mut c_void),
    axis_source: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
    axis_stop: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u32),
    axis_discrete: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, i32),
}

#[repr(C)]
struct RelativePointerListener {
    relative_motion: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, u32, i32, i32, i32, i32),
}

// --- Static listeners (function pointers are Sync) ---

static REGISTRY_LISTENER: RegistryListener = RegistryListener {
    global: cb_registry_global,
    global_remove: cb_registry_global_remove,
};

static XDG_WM_BASE_LISTENER: XdgWmBaseListener = XdgWmBaseListener {
    ping: cb_xdg_wm_base_ping,
};

static XDG_SURFACE_LISTENER: XdgSurfaceListener = XdgSurfaceListener {
    configure: cb_xdg_surface_configure,
};

static XDG_TOPLEVEL_LISTENER: XdgToplevelListener = XdgToplevelListener {
    configure: cb_xdg_toplevel_configure,
    close: cb_xdg_toplevel_close,
    configure_bounds: cb_xdg_toplevel_configure_bounds,
    wm_capabilities: cb_xdg_toplevel_wm_capabilities,
};

static SEAT_LISTENER: SeatListener = SeatListener {
    capabilities: cb_seat_capabilities,
    name: cb_seat_name,
};

static KEYBOARD_LISTENER: KeyboardListener = KeyboardListener {
    keymap: cb_keyboard_keymap,
    enter: cb_keyboard_enter,
    leave: cb_keyboard_leave,
    key: cb_keyboard_key,
    modifiers: cb_keyboard_modifiers,
    repeat_info: cb_keyboard_repeat_info,
};

static BUFFER_LISTENER: BufferListener = BufferListener {
    release: cb_buffer_release,
};

static POINTER_LISTENER: PointerListener = PointerListener {
    enter: cb_pointer_enter,
    leave: cb_pointer_leave,
    motion: cb_pointer_motion,
    button: cb_pointer_button,
    axis: cb_pointer_axis,
    frame: cb_pointer_frame,
    axis_source: cb_pointer_axis_source,
    axis_stop: cb_pointer_axis_stop,
    axis_discrete: cb_pointer_axis_discrete,
};

static REL_POINTER_LISTENER: RelativePointerListener = RelativePointerListener {
    relative_motion: cb_relative_motion,
};

// --- Wayland state (passed as data pointer to callbacks) ---

struct WlState {
    fns: WlFns,
    display: *mut c_void,
    compositor: *mut c_void,
    shm: *mut c_void,
    xdg_wm_base: *mut c_void,
    seat: *mut c_void,
    keyboard: *mut c_void,
    pointer: *mut c_void,
    rel_pointer_mgr: *mut c_void,
    rel_pointer: *mut c_void,
    mouse_dx: f32,
    mouse_dy: f32,
    pointer_entered: bool,
    prev_mouse_x: i32,
    prev_mouse_y: i32,
    has_rel_pointer: bool,
    surface: *mut c_void,
    xdg_surface: *mut c_void,
    xdg_toplevel: *mut c_void,
    buffers: [*mut c_void; 2],
    shm_data: *mut u8,
    shm_size: usize,
    buf_released: [bool; 2],
    cur_buf: usize,
    configured: bool,
    should_close: bool,
    pending_width: i32,
    pending_height: i32,
    width: usize,
    height: usize,
    keys: [bool; 256],
    // Interface pointers loaded from libwayland-client.so
    wl_registry_iface: *const WlInterface,
    wl_compositor_iface: *const WlInterface,
    wl_shm_iface: *const WlInterface,
    wl_shm_pool_iface: *const WlInterface,
    wl_surface_iface: *const WlInterface,
    wl_buffer_iface: *const WlInterface,
    wl_seat_iface: *const WlInterface,
    wl_keyboard_iface: *const WlInterface,
    wl_pointer_iface: *const WlInterface,
}

// --- Callback implementations ---

unsafe extern "C" fn cb_registry_global(data: *mut c_void, registry: *mut c_void, name: u32, interface: *const c_char, version: u32) {
    let s = &mut *(data as *mut WlState);
    let iface = CStr::from_ptr(interface);

    if iface == c"wl_compositor" {
        let ver = version.min(4);
        s.compositor = (s.fns.proxy_marshal_flags)(registry, 0, s.wl_compositor_iface, ver, 0,
            name, c"wl_compositor".as_ptr(), ver, ptr::null::<c_void>());
    } else if iface == c"wl_shm" {
        let ver = version.min(1);
        s.shm = (s.fns.proxy_marshal_flags)(registry, 0, s.wl_shm_iface, ver, 0,
            name, c"wl_shm".as_ptr(), ver, ptr::null::<c_void>());
    } else if iface == c"xdg_wm_base" {
        let ver = version.min(2);
        s.xdg_wm_base = (s.fns.proxy_marshal_flags)(registry, 0, &XDG_WM_BASE_INTERFACE, ver, 0,
            name, c"xdg_wm_base".as_ptr(), ver, ptr::null::<c_void>());
        (s.fns.proxy_add_listener)(s.xdg_wm_base, &XDG_WM_BASE_LISTENER as *const _ as *const c_void, data);
    } else if iface == c"zwp_relative_pointer_manager_v1" {
        s.rel_pointer_mgr = (s.fns.proxy_marshal_flags)(registry, 0, &ZWP_REL_PTR_MGR_INTERFACE, 1, 0,
            name, c"zwp_relative_pointer_manager_v1".as_ptr(), 1u32, ptr::null::<c_void>());
    } else if iface == c"wl_seat" {
        let ver = version.min(5);
        s.seat = (s.fns.proxy_marshal_flags)(registry, 0, s.wl_seat_iface, ver, 0,
            name, c"wl_seat".as_ptr(), ver, ptr::null::<c_void>());
        (s.fns.proxy_add_listener)(s.seat, &SEAT_LISTENER as *const _ as *const c_void, data);
    }
}

unsafe extern "C" fn cb_registry_global_remove(_data: *mut c_void, _registry: *mut c_void, _name: u32) {}

unsafe extern "C" fn cb_xdg_wm_base_ping(data: *mut c_void, xdg_wm_base: *mut c_void, serial: u32) {
    let s = &*(data as *mut WlState);
    (s.fns.proxy_marshal_flags)(xdg_wm_base, 3, ptr::null(), (s.fns.proxy_get_version)(xdg_wm_base), 0, serial);
}

unsafe extern "C" fn cb_xdg_surface_configure(data: *mut c_void, xdg_surface: *mut c_void, serial: u32) {
    let s = &mut *(data as *mut WlState);
    // ack_configure (opcode 4)
    (s.fns.proxy_marshal_flags)(xdg_surface, 4, ptr::null(), (s.fns.proxy_get_version)(xdg_surface), 0, serial);
    s.configured = true;
}

unsafe extern "C" fn cb_xdg_toplevel_configure(data: *mut c_void, _toplevel: *mut c_void, width: i32, height: i32, _states: *mut WlArray) {
    let s = &mut *(data as *mut WlState);
    if width > 0 && height > 0 {
        s.pending_width = width;
        s.pending_height = height;
    }
}

unsafe extern "C" fn cb_xdg_toplevel_close(data: *mut c_void, _toplevel: *mut c_void) {
    let s = &mut *(data as *mut WlState);
    s.should_close = true;
}

unsafe extern "C" fn cb_xdg_toplevel_configure_bounds(_data: *mut c_void, _toplevel: *mut c_void, _w: i32, _h: i32) {}
unsafe extern "C" fn cb_xdg_toplevel_wm_capabilities(_data: *mut c_void, _toplevel: *mut c_void, _caps: *mut WlArray) {}

unsafe extern "C" fn cb_seat_capabilities(data: *mut c_void, _seat: *mut c_void, caps: u32) {
    let s = &mut *(data as *mut WlState);
    if caps & WL_SEAT_CAPABILITY_POINTER != 0 && s.pointer.is_null() {
        // wl_seat.get_pointer (opcode 0)
        s.pointer = (s.fns.proxy_marshal_flags)(s.seat, 0, s.wl_pointer_iface,
            (s.fns.proxy_get_version)(s.seat), 0, ptr::null::<c_void>());
        (s.fns.proxy_add_listener)(s.pointer, &POINTER_LISTENER as *const _ as *const c_void, data);
    }
    if caps & WL_SEAT_CAPABILITY_KEYBOARD != 0 && s.keyboard.is_null() {
        // wl_seat.get_keyboard (opcode 1)
        s.keyboard = (s.fns.proxy_marshal_flags)(s.seat, 1, s.wl_keyboard_iface,
            (s.fns.proxy_get_version)(s.seat), 0, ptr::null::<c_void>());
        (s.fns.proxy_add_listener)(s.keyboard, &KEYBOARD_LISTENER as *const _ as *const c_void, data);
    }
}

unsafe extern "C" fn cb_seat_name(_data: *mut c_void, _seat: *mut c_void, _name: *const c_char) {}

unsafe extern "C" fn cb_keyboard_keymap(_data: *mut c_void, _kb: *mut c_void, _format: u32, fd: i32, _size: u32) {
    close(fd);
}

unsafe extern "C" fn cb_keyboard_enter(_data: *mut c_void, _kb: *mut c_void, _serial: u32, _surface: *mut c_void, _keys: *mut WlArray) {}
unsafe extern "C" fn cb_keyboard_leave(_data: *mut c_void, _kb: *mut c_void, _serial: u32, _surface: *mut c_void) {}

unsafe extern "C" fn cb_keyboard_key(data: *mut c_void, _kb: *mut c_void, _serial: u32, _time: u32, key: u32, state: u32) {
    let s = &mut *(data as *mut WlState);
    if (key as usize) < s.keys.len() {
        s.keys[key as usize] = state != 0;
    }
}

unsafe extern "C" fn cb_keyboard_modifiers(_data: *mut c_void, _kb: *mut c_void, _serial: u32, _dep: u32, _lat: u32, _lock: u32, _group: u32) {}
unsafe extern "C" fn cb_keyboard_repeat_info(_data: *mut c_void, _kb: *mut c_void, _rate: i32, _delay: i32) {}

// --- Pointer callbacks ---

unsafe extern "C" fn cb_pointer_enter(data: *mut c_void, pointer: *mut c_void, serial: u32, _surface: *mut c_void, sx: i32, sy: i32) {
    let s = &mut *(data as *mut WlState);
    s.pointer_entered = true;
    s.prev_mouse_x = sx;
    s.prev_mouse_y = sy;
    // Hide cursor: wl_pointer.set_cursor (opcode 0): "u?oii"
    (s.fns.proxy_marshal_flags)(pointer, 0, ptr::null(), (s.fns.proxy_get_version)(pointer), 0,
        serial, ptr::null::<c_void>(), 0i32, 0i32);
}

unsafe extern "C" fn cb_pointer_leave(data: *mut c_void, _pointer: *mut c_void, _serial: u32, _surface: *mut c_void) {
    let s = &mut *(data as *mut WlState);
    s.pointer_entered = false;
}

unsafe extern "C" fn cb_pointer_motion(data: *mut c_void, _pointer: *mut c_void, _time: u32, sx: i32, sy: i32) {
    let s = &mut *(data as *mut WlState);
    // Fallback: compute delta from absolute coords (only if no relative pointer)
    if !s.has_rel_pointer && s.pointer_entered {
        s.mouse_dx += (sx - s.prev_mouse_x) as f32 / 256.0;
        s.mouse_dy += (sy - s.prev_mouse_y) as f32 / 256.0;
    }
    s.prev_mouse_x = sx;
    s.prev_mouse_y = sy;
}

unsafe extern "C" fn cb_pointer_button(_data: *mut c_void, _pointer: *mut c_void, _serial: u32, _time: u32, _button: u32, _state: u32) {}
unsafe extern "C" fn cb_pointer_axis(_data: *mut c_void, _pointer: *mut c_void, _time: u32, _axis: u32, _value: i32) {}
unsafe extern "C" fn cb_pointer_frame(_data: *mut c_void, _pointer: *mut c_void) {}
unsafe extern "C" fn cb_pointer_axis_source(_data: *mut c_void, _pointer: *mut c_void, _axis_source: u32) {}
unsafe extern "C" fn cb_pointer_axis_stop(_data: *mut c_void, _pointer: *mut c_void, _time: u32, _axis: u32) {}
unsafe extern "C" fn cb_pointer_axis_discrete(_data: *mut c_void, _pointer: *mut c_void, _axis: u32, _discrete: i32) {}

unsafe extern "C" fn cb_relative_motion(data: *mut c_void, _rel_pointer: *mut c_void, _utime_hi: u32, _utime_lo: u32, dx: i32, dy: i32, _dx_unaccel: i32, _dy_unaccel: i32) {
    let s = &mut *(data as *mut WlState);
    s.mouse_dx += dx as f32 / 256.0;
    s.mouse_dy += dy as f32 / 256.0;
}

unsafe extern "C" fn cb_buffer_release(data: *mut c_void, buffer: *mut c_void) {
    let s = &mut *(data as *mut WlState);
    for i in 0..2 {
        if s.buffers[i] == buffer {
            s.buf_released[i] = true;
        }
    }
}

// --- Helper: load symbol from dlopen handle ---

unsafe fn load_sym<T>(handle: *mut c_void, name: &CStr) -> T {
    let sym = dlsym(handle, name.as_ptr());
    if sym.is_null() {
        panic!("Failed to load symbol: {}", name.to_str().unwrap_or("?"));
    }
    std::mem::transmute_copy(&sym)
}

unsafe fn load_iface(handle: *mut c_void, name: &CStr) -> *const WlInterface {
    let sym = dlsym(handle, name.as_ptr());
    if sym.is_null() {
        panic!("Failed to load interface: {}", name.to_str().unwrap_or("?"));
    }
    sym as *const WlInterface
}

// --- Create SHM buffer pair ---

unsafe fn create_shm_buffers(s: &mut WlState, width: usize, height: usize) {
    let stride = width * 4;
    let buf_size = stride * height;
    let pool_size = buf_size * 2;

    if !s.shm_data.is_null() {
        munmap(s.shm_data as *mut c_void, s.shm_size);
    }
    for i in 0..2 {
        if !s.buffers[i].is_null() {
            // wl_buffer.destroy (opcode 0)
            (s.fns.proxy_marshal_flags)(s.buffers[i], 0, ptr::null(), (s.fns.proxy_get_version)(s.buffers[i]), 1);
            s.buffers[i] = ptr::null_mut();
        }
    }

    let fd = memfd_create(c"clauding-shm".as_ptr(), MFD_CLOEXEC);
    if fd < 0 { panic!("memfd_create failed"); }
    if ftruncate(fd, pool_size as i64) < 0 { panic!("ftruncate failed"); }

    let data = mmap(ptr::null_mut(), pool_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if data == usize::MAX as *mut c_void { panic!("mmap failed"); }

    // wl_shm.create_pool (opcode 0): signature "nhi" (new_id, fd, int32)
    let pool = (s.fns.proxy_marshal_flags)(s.shm, 0, s.wl_shm_pool_iface,
        (s.fns.proxy_get_version)(s.shm), 0,
        ptr::null::<c_void>(), fd, pool_size as i32);

    // wl_shm_pool.create_buffer (opcode 0): signature "niiiiu" (new_id, offset, width, height, stride, format)
    for i in 0..2u32 {
        let offset = (i as usize * buf_size) as i32;
        s.buffers[i as usize] = (s.fns.proxy_marshal_flags)(pool, 0, s.wl_buffer_iface,
            (s.fns.proxy_get_version)(pool), 0,
            ptr::null::<c_void>(), offset, width as i32, height as i32, stride as i32, WL_SHM_FORMAT_XRGB8888);
        (s.fns.proxy_add_listener)(s.buffers[i as usize], &BUFFER_LISTENER as *const _ as *const c_void,
            s as *mut WlState as *mut c_void);
        s.buf_released[i as usize] = true;
    }

    // wl_shm_pool.destroy (opcode 1)
    (s.fns.proxy_marshal_flags)(pool, 1, ptr::null(), (s.fns.proxy_get_version)(pool), 1);

    close(fd);
    s.shm_data = data as *mut u8;
    s.shm_size = pool_size;
    s.width = width;
    s.height = height;
}

// --- Public API ---

pub struct WaylandWindow {
    state: Box<WlState>,
}

impl WaylandWindow {
    pub fn new() -> Self {
        unsafe {
            let lib = dlopen(c"libwayland-client.so.0".as_ptr(), RTLD_LAZY);
            if lib.is_null() { panic!("Failed to load libwayland-client.so.0"); }

            let fns = WlFns {
                display_connect:        load_sym(lib, c"wl_display_connect"),
                display_disconnect:     load_sym(lib, c"wl_display_disconnect"),
                display_roundtrip:      load_sym(lib, c"wl_display_roundtrip"),
                display_flush:          load_sym(lib, c"wl_display_flush"),
                display_get_fd:         load_sym(lib, c"wl_display_get_fd"),
                display_prepare_read:   load_sym(lib, c"wl_display_prepare_read"),
                display_read_events:    load_sym(lib, c"wl_display_read_events"),
                display_cancel_read:    load_sym(lib, c"wl_display_cancel_read"),
                display_dispatch_pending:load_sym(lib, c"wl_display_dispatch_pending"),
                proxy_marshal_flags:    load_sym(lib, c"wl_proxy_marshal_flags"),
                proxy_add_listener:     load_sym(lib, c"wl_proxy_add_listener"),
                proxy_get_version:      load_sym(lib, c"wl_proxy_get_version"),
                proxy_destroy:          load_sym(lib, c"wl_proxy_destroy"),
            };

            let display = (fns.display_connect)(ptr::null());
            if display.is_null() { panic!("Failed to connect to Wayland display"); }

            let mut state = Box::new(WlState {
                fns,
                display,
                compositor: ptr::null_mut(),
                shm: ptr::null_mut(),
                xdg_wm_base: ptr::null_mut(),
                seat: ptr::null_mut(),
                keyboard: ptr::null_mut(),
                pointer: ptr::null_mut(),
                rel_pointer_mgr: ptr::null_mut(),
                rel_pointer: ptr::null_mut(),
                mouse_dx: 0.0,
                mouse_dy: 0.0,
                pointer_entered: false,
                prev_mouse_x: 0,
                prev_mouse_y: 0,
                has_rel_pointer: false,
                surface: ptr::null_mut(),
                xdg_surface: ptr::null_mut(),
                xdg_toplevel: ptr::null_mut(),
                buffers: [ptr::null_mut(); 2],
                shm_data: ptr::null_mut(),
                shm_size: 0,
                buf_released: [true; 2],
                cur_buf: 0,
                configured: false,
                should_close: false,
                pending_width: 0,
                pending_height: 0,
                width: 0,
                height: 0,
                keys: [false; 256],
                wl_registry_iface: load_iface(lib, c"wl_registry_interface"),
                wl_compositor_iface: load_iface(lib, c"wl_compositor_interface"),
                wl_shm_iface: load_iface(lib, c"wl_shm_interface"),
                wl_shm_pool_iface: load_iface(lib, c"wl_shm_pool_interface"),
                wl_surface_iface: load_iface(lib, c"wl_surface_interface"),
                wl_buffer_iface: load_iface(lib, c"wl_buffer_interface"),
                wl_seat_iface: load_iface(lib, c"wl_seat_interface"),
                wl_keyboard_iface: load_iface(lib, c"wl_keyboard_interface"),
                wl_pointer_iface: load_iface(lib, c"wl_pointer_interface"),
            });

            let data_ptr = &mut *state as *mut WlState as *mut c_void;

            // wl_display.get_registry (opcode 1)
            let registry = (state.fns.proxy_marshal_flags)(display, 1, state.wl_registry_iface,
                (state.fns.proxy_get_version)(display), 0, ptr::null::<c_void>());
            (state.fns.proxy_add_listener)(registry, &REGISTRY_LISTENER as *const _ as *const c_void, data_ptr);

            (state.fns.display_roundtrip)(display);
            (state.fns.display_roundtrip)(display);

            if state.compositor.is_null() { panic!("No wl_compositor"); }
            if state.shm.is_null() { panic!("No wl_shm"); }
            if state.xdg_wm_base.is_null() { panic!("No xdg_wm_base"); }

            // wl_compositor.create_surface (opcode 0): signature "n"
            state.surface = (state.fns.proxy_marshal_flags)(state.compositor, 0, state.wl_surface_iface,
                (state.fns.proxy_get_version)(state.compositor), 0, ptr::null::<c_void>());

            // xdg_wm_base.get_xdg_surface (opcode 2): signature "no"
            state.xdg_surface = (state.fns.proxy_marshal_flags)(state.xdg_wm_base, 2, &XDG_SURFACE_INTERFACE,
                (state.fns.proxy_get_version)(state.xdg_wm_base), 0,
                ptr::null::<c_void>(), state.surface);
            (state.fns.proxy_add_listener)(state.xdg_surface, &XDG_SURFACE_LISTENER as *const _ as *const c_void, data_ptr);

            // xdg_surface.get_toplevel (opcode 1): signature "n"
            state.xdg_toplevel = (state.fns.proxy_marshal_flags)(state.xdg_surface, 1, &XDG_TOPLEVEL_INTERFACE,
                (state.fns.proxy_get_version)(state.xdg_surface), 0, ptr::null::<c_void>());
            (state.fns.proxy_add_listener)(state.xdg_toplevel, &XDG_TOPLEVEL_LISTENER as *const _ as *const c_void, data_ptr);

            // xdg_toplevel.set_title (opcode 2): signature "s"
            (state.fns.proxy_marshal_flags)(state.xdg_toplevel, 2, ptr::null(),
                (state.fns.proxy_get_version)(state.xdg_toplevel), 0, c"Clauding".as_ptr());

            // xdg_toplevel.set_app_id (opcode 3): signature "s"
            (state.fns.proxy_marshal_flags)(state.xdg_toplevel, 3, ptr::null(),
                (state.fns.proxy_get_version)(state.xdg_toplevel), 0, c"clauding".as_ptr());

            // xdg_toplevel.set_fullscreen (opcode 11): signature "?o"
            (state.fns.proxy_marshal_flags)(state.xdg_toplevel, 11, ptr::null(),
                (state.fns.proxy_get_version)(state.xdg_toplevel), 0, ptr::null::<c_void>());

            // Initial empty commit to indicate we're ready
            // wl_surface.commit (opcode 6)
            (state.fns.proxy_marshal_flags)(state.surface, 6, ptr::null(),
                (state.fns.proxy_get_version)(state.surface), 0);

            // Create relative pointer if available (after globals are bound)
            if !state.pointer.is_null() && !state.rel_pointer_mgr.is_null() {
                // zwp_relative_pointer_manager_v1.get_relative_pointer (opcode 1): "no"
                state.rel_pointer = (state.fns.proxy_marshal_flags)(
                    state.rel_pointer_mgr, 1, &ZWP_REL_PTR_INTERFACE,
                    (state.fns.proxy_get_version)(state.rel_pointer_mgr), 0,
                    ptr::null::<c_void>(), state.pointer);
                (state.fns.proxy_add_listener)(state.rel_pointer,
                    &REL_POINTER_LISTENER as *const _ as *const c_void, data_ptr);
                state.has_rel_pointer = true;
                eprintln!("Relative pointer: enabled");
            } else if !state.pointer.is_null() {
                eprintln!("Relative pointer: fallback (no zwp_relative_pointer_manager_v1)");
            }

            // Wait for initial configure
            while !state.configured {
                (state.fns.display_roundtrip)(display);
            }

            // Use pending size or fallback
            let w = if state.pending_width > 0 { state.pending_width as usize } else { crate::state::DEFAULT_WIDTH };
            let h = if state.pending_height > 0 { state.pending_height as usize } else { crate::state::DEFAULT_HEIGHT };

            create_shm_buffers(&mut state, w, h);

            WaylandWindow { state }
        }
    }

    pub fn width(&self) -> usize { self.state.width }
    pub fn height(&self) -> usize { self.state.height }
    pub fn poll_events(&mut self, keys: &mut [bool; 256], should_quit: &mut bool, mouse_dx: &mut f32, mouse_dy: &mut f32) {
        unsafe {
            let s = &mut *self.state;
            let fd = (s.fns.display_get_fd)(s.display);

            (s.fns.display_flush)(s.display);

            if (s.fns.display_prepare_read)(s.display) == 0 {
                let mut pfd = PollFd { fd, events: POLLIN, revents: 0 };
                let ret = poll(&mut pfd, 1, 0);
                if ret > 0 && pfd.revents & POLLIN != 0 {
                    (s.fns.display_read_events)(s.display);
                } else {
                    (s.fns.display_cancel_read)(s.display);
                }
            }

            (s.fns.display_dispatch_pending)(s.display);

            // Handle resize
            if s.pending_width > 0 && s.pending_height > 0 {
                let nw = s.pending_width as usize;
                let nh = s.pending_height as usize;
                if nw != s.width || nh != s.height {
                    create_shm_buffers(s, nw, nh);
                }
                s.pending_width = 0;
                s.pending_height = 0;
            }

            *keys = s.keys;
            *should_quit = s.should_close;
            *mouse_dx = s.mouse_dx;
            *mouse_dy = s.mouse_dy;
            s.mouse_dx = 0.0;
            s.mouse_dy = 0.0;
        }
    }

    pub fn present(&mut self, pixels: &[u32]) {
        unsafe {
            let s = &mut *self.state;
            if s.width == 0 || s.height == 0 { return; }

            let buf_idx = s.cur_buf;
            if !s.buf_released[buf_idx] {
                return; // skip frame, buffer still in use by compositor
            }

            let buf_size = s.width * s.height * 4;
            let offset = buf_idx * buf_size;
            let dst = s.shm_data.add(offset) as *mut u32;
            let count = s.width * s.height;
            let src_count = pixels.len().min(count);
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), dst, src_count);

            s.buf_released[buf_idx] = false;

            // wl_surface.attach (opcode 1): signature "oii"
            (s.fns.proxy_marshal_flags)(s.surface, 1, ptr::null(),
                (s.fns.proxy_get_version)(s.surface), 0,
                s.buffers[buf_idx], 0i32, 0i32);

            // wl_surface.damage_buffer (opcode 9): signature "iiii"
            (s.fns.proxy_marshal_flags)(s.surface, 9, ptr::null(),
                (s.fns.proxy_get_version)(s.surface), 0,
                0i32, 0i32, s.width as i32, s.height as i32);

            // wl_surface.commit (opcode 6)
            (s.fns.proxy_marshal_flags)(s.surface, 6, ptr::null(),
                (s.fns.proxy_get_version)(s.surface), 0);

            (s.fns.display_flush)(s.display);

            s.cur_buf = 1 - buf_idx;
        }
    }
}

impl Drop for WaylandWindow {
    fn drop(&mut self) {
        unsafe {
            let s = &mut *self.state;
            if !s.shm_data.is_null() {
                munmap(s.shm_data as *mut c_void, s.shm_size);
            }
            (s.fns.display_disconnect)(s.display);
        }
    }
}
