// Surface material properties for physics interactions

#[derive(Clone, Copy)]
pub struct SurfaceMaterial {
    pub static_friction: f32,
    pub dynamic_friction: f32,
    pub restitution: f32,       // bounciness 0..1
    pub rolling_resistance: f32, // tire rolling friction
}

// Per-surface material constants
pub const MAT_ASPHALT: SurfaceMaterial = SurfaceMaterial {
    static_friction: 0.9,
    dynamic_friction: 0.7,
    restitution: 0.1,
    rolling_resistance: 0.015,
};

pub const MAT_CONCRETE: SurfaceMaterial = SurfaceMaterial {
    static_friction: 0.85,
    dynamic_friction: 0.65,
    restitution: 0.15,
    rolling_resistance: 0.018,
};

pub const MAT_GRASS: SurfaceMaterial = SurfaceMaterial {
    static_friction: 0.6,
    dynamic_friction: 0.4,
    restitution: 0.05,
    rolling_resistance: 0.08,
};

pub const MAT_GRAVEL: SurfaceMaterial = SurfaceMaterial {
    static_friction: 0.5,
    dynamic_friction: 0.35,
    restitution: 0.05,
    rolling_resistance: 0.06,
};

pub const MAT_DIRT: SurfaceMaterial = SurfaceMaterial {
    static_friction: 0.55,
    dynamic_friction: 0.4,
    restitution: 0.05,
    rolling_resistance: 0.07,
};

pub const MAT_WATER: SurfaceMaterial = SurfaceMaterial {
    static_friction: 0.1,
    dynamic_friction: 0.05,
    restitution: 0.0,
    rolling_resistance: 0.3,
};

pub const MAT_METAL: SurfaceMaterial = SurfaceMaterial {
    static_friction: 0.5,
    dynamic_friction: 0.4,
    restitution: 0.3,
    rolling_resistance: 0.01,
};

pub const MAT_ICE: SurfaceMaterial = SurfaceMaterial {
    static_friction: 0.08,
    dynamic_friction: 0.04,
    restitution: 0.02,
    rolling_resistance: 0.005,
};

pub const MAT_WET_ROAD: SurfaceMaterial = SurfaceMaterial {
    static_friction: 0.45,
    dynamic_friction: 0.3,
    restitution: 0.1,
    rolling_resistance: 0.02,
};

/// Combine two surface materials for a contact pair.
/// Uses geometric mean for friction, max for restitution.
pub fn combine_materials(a: &SurfaceMaterial, b: &SurfaceMaterial) -> SurfaceMaterial {
    SurfaceMaterial {
        static_friction: (a.static_friction * b.static_friction).sqrt(),
        dynamic_friction: (a.dynamic_friction * b.dynamic_friction).sqrt(),
        restitution: a.restitution.max(b.restitution),
        rolling_resistance: (a.rolling_resistance + b.rolling_resistance) * 0.5,
    }
}

/// Get surface material from the game's Surface enum
pub fn material_for_surface(surface: crate::state::Surface) -> &'static SurfaceMaterial {
    match surface {
        crate::state::Surface::CarRoad => &MAT_ASPHALT,
        crate::state::Surface::Sidewalk => &MAT_CONCRETE,
        crate::state::Surface::FieldRoad => &MAT_GRAVEL,
        crate::state::Surface::Terrain => &MAT_GRASS,
    }
}
