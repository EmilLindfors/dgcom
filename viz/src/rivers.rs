//! A run's rivers on the map: a marker where each one's water enters.
//!
//! A case's snapshot file lists its rivers (`river=name,x,y`: the centre of the
//! element each one enters, the largest first), and the number keys frame them after
//! the run's stations. Each is marked by a post from the bed to above the surface
//! and a cross on the water, both sized to the camera's distance so that they read
//! from the whole domain down to a mouth's close-up. With the water coloured by
//! salinity (`--colour salinity`, or C), the river's fresh water is the dark plume
//! around the marker; `,` and `.` step through the layers, and a pinned point's
//! profiles show the salinity over the depth ([`crate::inspect`]).

use bevy::prelude::*;

use crate::camera::OrbitCamera;
use crate::field::{Field, Frame, Nodes, Probe};
use crate::screen::{RunResources, Viewing};

/// The rivers of the run, where their water enters.
#[derive(Resource, Default)]
pub struct RiverMarks(pub Vec<RiverMark>);

pub struct RiverMark {
    /// Mesh coordinates (m)
    pub at: [f64; 2],
    /// The point in the mesh, if it is in it
    pub probe: Option<Probe>,
}

pub struct RiversPlugin;

impl Plugin for RiversPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            draw.run_if(resource_exists::<RiverMarks>).in_set(Viewing),
        )
        .run_resource::<RiverMarks>();
    }
}

/// Fresh water's colour: pale cyan, apart from the haline map's indigo to yellow.
const COLOUR: LinearRgba = LinearRgba::rgb(0.55, 0.95, 1.0);

fn draw(
    marks: Res<RiverMarks>,
    field: Res<Field>,
    nodes: Res<Nodes>,
    frame: Res<Frame>,
    camera: Query<&OrbitCamera>,
    mut gizmos: Gizmos,
) {
    let Ok(orbit) = camera.single() else { return };
    let size = 0.02 * orbit.distance;
    for mark in &marks.0 {
        let Some(probe) = &mark.probe else { continue };
        let bed = probe.eval(&nodes.bed);
        let eta = if field.eta.is_empty() {
            0.0
        } else {
            probe.eval(&field.eta)
        };
        let surface = frame.world(mark.at, eta.max(bed));
        gizmos.line(frame.world(mark.at, bed), surface + Vec3::Y * size, COLOUR);
        gizmos.line(surface - Vec3::X * size, surface + Vec3::X * size, COLOUR);
        gizmos.line(surface - Vec3::Z * size, surface + Vec3::Z * size, COLOUR);
    }
}
