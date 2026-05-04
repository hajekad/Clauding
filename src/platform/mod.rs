//! Platform abstraction — window creation, event polling, buffer presentation.
//! Currently Linux/Wayland only.

#[cfg(target_os = "linux")]
pub mod wayland;
#[cfg(target_os = "linux")]
pub use wayland::WaylandWindow as PlatformWindow;
