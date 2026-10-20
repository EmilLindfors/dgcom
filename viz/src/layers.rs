//! The water column of a 3D run: which layer's current the water shows.
//!
//! `,` and `.` step the current shown by the water's colouring, the arrows and the
//! close-up through the depth mean and the σ-layers, from the surface layer down
//! ([`ShownLayer`]). The profiles over the depth at a point are the pinned point's
//! ([`crate::inspect`]).

use bevy::prelude::*;

use crate::actions::{Act, Action};
use crate::field::ShownLayer;

/// The σ-levels of a 3D run: layer centres from the bed up, for the legend.
#[derive(Resource, Clone, Debug)]
pub struct Levels {
    pub sigma: Vec<f32>,
    /// Water depth at rest at the farm (m)
    pub depth: f32,
}

pub struct LayersPlugin;

impl Plugin for LayersPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, keys);
    }
}

fn keys(mut acts: MessageReader<Act>, levels: Res<Levels>, mut shown: ResMut<ShownLayer>) {
    let top = levels.sigma.len() - 1;
    for &Act(action) in acts.read() {
        // Downwards: depth mean, surface layer, ..., bed layer
        *shown = match (action, *shown) {
            (Action::LayerDown, ShownLayer::DepthMean) => ShownLayer::Level(top),
            (Action::LayerDown, ShownLayer::Level(l)) => ShownLayer::Level(l.saturating_sub(1)),
            (Action::LayerUp, ShownLayer::Level(l)) if l >= top => ShownLayer::DepthMean,
            (Action::LayerUp, ShownLayer::Level(l)) => ShownLayer::Level(l + 1),
            (_, shown) => shown,
        };
    }
}
