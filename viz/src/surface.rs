//! The bed and the water as meshes of the DG nodes themselves.
//!
//! Each element of order p is drawn as p² quads between its GLL nodes (as `dg_rs::io`'s
//! VTK writer does), with the elements' own nodes, so jumps between elements show as
//! they are. Normals are those of the element polynomial ([`Nodes::normals`]), so the
//! lighting follows the solution rather than the triangulation.
//!
//! Along the mesh boundary both meshes hang walls: the bed's reach down to a flat base
//! and the water's from the surface to the bed, so the domain reads as a cut-out block.
//! With an elevation model around the domain ([`crate::terrain`]) the land and the sea
//! bed carry on beyond the mesh instead: where they do, the bed's walls rise to the
//! land at the boundary (the coast's cliffs) and the water has none. Where the
//! elevation model ends at the mesh boundary (Frøya's covers just the domain), the
//! domain keeps its cut-out edge.
//!
//! The water's heights, normals and colours are rewritten whenever the shown field
//! changes. Keys: C colours the water by current speed or by surface elevation, and
//! in a run with waves by the significant wave height, the peak period or the power
//! the waves lose to whitecapping and breaking (on a log scale); T cycles the water
//! through translucent, opaque and hidden; - and = make the translucent water clearer
//! or denser, down to a tint over the bed.
//!
//! In the photo view ([`crate::photo`]) both meshes carry what its shader needs too:
//! the water the current (UV 0, world X and Z, averaged over the nodes elements share
//! so the waves riding it have no seams at element edges) and in a run with waves the
//! significant wave height and the share of their energy the waves lose to breaking
//! per period (UV 1, averaged so too), the bed the depth of the water over it (UV 1).

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;

use crate::actions::{Act, Action};
use crate::colormap::{CIVIDIS, DIVERGING, ICE, Lut, SEABED, THERMAL, VIRIDIS};
use crate::field::{Field, Frame, Nodes};
use crate::photo::{Photo, PhotoPart};
use crate::playback::Playback;
use crate::solver::{H_DRY, Waves};
use crate::terrain::{GroundColours, Terrain};

pub struct SurfacePlugin;

impl Plugin for SurfacePlugin {
    fn build(&self, app: &mut App) {
        // Translucent, unless the app chose (opaque under a wave colouring)
        if !app.world().contains_resource::<SurfaceStyle>() {
            app.insert_resource(SurfaceStyle::Translucent);
        }
        app.init_resource::<WaterOpacity>()
            .add_systems(Startup, spawn)
            .add_systems(
                Update,
                (keys, style, update_water.after(crate::field::interpolate)),
            );
    }
}

/// What the water's colour shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColourBy {
    Speed,
    Elevation,
    /// Significant wave height
    WaveHeight,
    /// Peak period
    WavePeriod,
    /// Whitecapping and depth-induced breaking, on a log scale
    WaveDissipation,
}

impl ColourBy {
    /// The colouring C steps to: the waves' only in a run that has them.
    pub fn next(self, waves: bool) -> Self {
        match self {
            Self::Speed => Self::Elevation,
            Self::Elevation if waves => Self::WaveHeight,
            Self::Elevation => Self::Speed,
            Self::WaveHeight => Self::WavePeriod,
            Self::WavePeriod => Self::WaveDissipation,
            Self::WaveDissipation => Self::Speed,
        }
    }

    pub fn is_waves(self) -> bool {
        matches!(
            self,
            Self::WaveHeight | Self::WavePeriod | Self::WaveDissipation
        )
    }

    /// The colouring a `--colour` value names.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "speed" => Self::Speed,
            "elevation" | "eta" => Self::Elevation,
            "hs" | "height" => Self::WaveHeight,
            "tp" | "period" => Self::WavePeriod,
            "dissipation" | "breaking" => Self::WaveDissipation,
            _ => return None,
        })
    }
}

/// Decades the dissipation's log scale spans below its top.
pub const DISSIPATION_DECADES: f32 = 4.0;

#[derive(Resource)]
pub struct Colouring {
    pub by: ColourBy,
    /// Top of the speed scale (m/s)
    pub speed_max: f32,
    /// Half-width of the elevation scale (m)
    pub eta_max: f32,
    /// Tops of the wave height (m) and peak period (s) scales
    pub hs_max: f32,
    pub tp_max: f32,
    /// Top of the dissipation's log scale (W/m², a power of ten)
    pub dissipation_max: f32,
    /// Whether the speed scale was set by hand (else it follows the flow)
    fixed_speed: bool,
    luts: [Lut; 5],
}

impl Colouring {
    pub fn new(speed_max: Option<f32>, by: ColourBy) -> Self {
        Self {
            by,
            speed_max: speed_max.unwrap_or(0.05),
            eta_max: 0.1,
            hs_max: 1.0,
            tp_max: 10.0,
            dissipation_max: 100.0,
            fixed_speed: speed_max.is_some(),
            luts: [VIRIDIS, DIVERGING, THERMAL, CIVIDIS, ICE].map(Lut::new),
        }
    }

    /// Where the node `g` falls on the scale shown, 0 to 1 (speeds in `speed`).
    fn position(&self, g: usize, speed: &[f32], field: &Field) -> f32 {
        let waves = field.waves.as_ref();
        match (self.by, waves) {
            (ColourBy::Elevation, _) => 0.5 + 0.5 * field.eta[g] / self.eta_max,
            (ColourBy::WaveHeight, Some(w)) => w.hs[g] / self.hs_max,
            (ColourBy::WavePeriod, Some(w)) => w.tp[g] / self.tp_max,
            (ColourBy::WaveDissipation, Some(w)) => {
                let power = w.whitecapping[g] + w.breaking[g];
                1.0 + (power.max(1e-30) / self.dissipation_max).log10() / DISSIPATION_DECADES
            }
            // The speed, also for a wave colouring of a field without waves
            _ => speed[g] / self.speed_max,
        }
    }

    /// The colour of node `g`.
    fn at(&self, g: usize, speed: &[f32], field: &Field) -> [f32; 4] {
        let by = if self.by.is_waves() && field.waves.is_none() {
            ColourBy::Speed
        } else {
            self.by
        };
        self.luts[by as usize].at(self.position(g, speed, field))
    }

    /// Fits the scales to the shown field in round steps: speed to its 98th
    /// percentile, η to its largest magnitude, H_s and T_p to their 99th percentiles
    /// and the dissipation's top to the power of ten over its 99.5th. A scale steps up
    /// as soon as the field exceeds it and down only once the field fits a step lower
    /// with a quarter to spare, so it does not flicker at a step's edge.
    fn fit(&mut self, speed: &[f32], eta: &[f32], waves: Option<&Waves>) {
        fn follow(scale: &mut f32, x: f32, floor: f32) {
            if x > *scale || nice_ceil(1.25 * x) < *scale {
                *scale = nice_ceil(x).max(floor);
            }
        }
        let stride = (speed.len() / 20_000).max(1);
        let percentile = |field: &[f32], p: f32, of: &dyn Fn(usize) -> f32| {
            let mut sample: Vec<f32> = (0..field.len()).step_by(stride).map(of).collect();
            if sample.is_empty() {
                return None;
            }
            let at = (p * (sample.len() - 1) as f32) as usize;
            Some(*sample.select_nth_unstable_by(at, f32::total_cmp).1)
        };
        if !self.fixed_speed
            && let Some(p98) = percentile(speed, 0.98, &|g| speed[g])
        {
            follow(&mut self.speed_max, p98, 1e-3);
        }
        let eta_abs = eta
            .iter()
            .step_by(stride)
            .fold(0.0_f32, |m, e| m.max(e.abs()));
        follow(&mut self.eta_max, eta_abs, 1e-2);
        let Some(w) = waves else { return };
        if let Some(p99) = percentile(&w.hs, 0.99, &|g| w.hs[g]) {
            follow(&mut self.hs_max, p99, 0.1);
        }
        if let Some(p99) = percentile(&w.tp, 0.99, &|g| w.tp[g]) {
            follow(&mut self.tp_max, p99, 1.0);
        }
        let power = |g: usize| w.whitecapping[g] + w.breaking[g];
        if let Some(top) = percentile(&w.hs, 0.995, &power) {
            // A decade up at once, down once the field fits a decade lower with a
            // quarter to spare
            let top = top.max(1e-2);
            let decade = 10f32.powf(top.log10().ceil());
            if decade > self.dissipation_max || 12.5 * top < self.dissipation_max {
                self.dissipation_max = decade;
            }
        }
    }

    pub fn stops(&self) -> &'static [[u8; 3]] {
        match self.by {
            ColourBy::Speed => VIRIDIS,
            ColourBy::Elevation => DIVERGING,
            ColourBy::WaveHeight => THERMAL,
            ColourBy::WavePeriod => CIVIDIS,
            ColourBy::WaveDissipation => ICE,
        }
    }

    /// The scale's title and its labels at the low end, the middle and the top.
    pub fn legend(&self) -> (&'static str, [String; 3]) {
        let linear = |lo: f32, hi: f32| {
            std::array::from_fn(|i| {
                let v = lo + (hi - lo) * i as f32 / 2.0;
                format!("{}", (v * 1000.0).round() / 1000.0)
            })
        };
        match self.by {
            ColourBy::Speed => ("current speed (m/s)", linear(0.0, self.speed_max)),
            ColourBy::Elevation => (
                "surface elevation η (m)",
                linear(-self.eta_max, self.eta_max),
            ),
            ColourBy::WaveHeight => ("significant wave height H_s (m)", linear(0.0, self.hs_max)),
            ColourBy::WavePeriod => ("peak period T_p (s)", linear(0.0, self.tp_max)),
            ColourBy::WaveDissipation => (
                "waves' losses (W/m², log)",
                std::array::from_fn(|i| {
                    let decades = DISSIPATION_DECADES * (1.0 - i as f32 / 2.0);
                    let v = self.dissipation_max / 10f32.powf(decades);
                    if v >= 1.0 {
                        format!("{v:.0}")
                    } else {
                        format!("{v}")
                    }
                }),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_cycles_through_the_waves_only_when_the_run_has_them() {
        let mut by = ColourBy::Speed;
        let mut seen = Vec::new();
        for _ in 0..5 {
            by = by.next(true);
            seen.push(by);
        }
        assert_eq!(
            seen,
            [
                ColourBy::Elevation,
                ColourBy::WaveHeight,
                ColourBy::WavePeriod,
                ColourBy::WaveDissipation,
                ColourBy::Speed
            ]
        );
        assert_eq!(ColourBy::Elevation.next(false), ColourBy::Speed);
        assert_eq!(ColourBy::parse("hs"), Some(ColourBy::WaveHeight));
        assert_eq!(ColourBy::parse("waves"), None);
    }

    #[test]
    fn the_dissipation_scale_follows_in_decades() {
        let mut colouring = Colouring::new(None, ColourBy::WaveDissipation);
        let n = 1000;
        let waves = |power: f32| Waves {
            hs: vec![1.0; n],
            tp: vec![8.0; n],
            whitecapping: vec![power; n],
            breaking: vec![0.0; n],
            ..Waves::default()
        };
        let still = vec![0.0; n];
        colouring.fit(&still, &still, Some(&waves(3.0)));
        assert_eq!(colouring.dissipation_max, 10.0);
        assert_eq!(colouring.legend().1, ["0.001", "0.1", "10"]);
        // Up a decade at once, down once a decade lower fits
        colouring.fit(&still, &still, Some(&waves(30.0)));
        assert_eq!(colouring.dissipation_max, 100.0);
        colouring.fit(&still, &still, Some(&waves(9.0)));
        assert_eq!(colouring.dissipation_max, 100.0);
        colouring.fit(&still, &still, Some(&waves(0.9)));
        assert_eq!(colouring.dissipation_max, 1.0);
        assert_eq!(colouring.hs_max, 1.0);
        assert_eq!(colouring.tp_max, 10.0);
    }

    #[test]
    fn breaking_shares_the_energy_per_period() {
        // H_s 1 m holds ρg/16 ≈ 628 J/m²; losing 62.8 W/m² over a 10 s period is all of it
        let energy = 1025.0 * 9.81 / 16.0;
        assert!((breaking_share(0.1 * energy, 10.0, 1.0) - 1.0).abs() < 1e-5);
        assert_eq!(breaking_share(5.0, 8.0, 0.0), 0.0);
        assert_eq!(breaking_share(1e6, 8.0, 0.1), 4.0);
    }
}

/// The smallest of 1, 2, 2.5, 5 × 10ⁿ at or above `x`.
pub(crate) fn nice_ceil(x: f32) -> f32 {
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }
    let decade = 10f32.powf(x.log10().floor());
    [1.0, 2.0, 2.5, 5.0, 10.0]
        .iter()
        .map(|m| m * decade)
        .find(|&v| v >= x * (1.0 - 1e-6))
        .unwrap_or(10.0 * decade)
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceStyle {
    Translucent,
    Opaque,
    Hidden,
}

/// Opacity of the translucent water (`--water-alpha`; - and = step it). Low enough by
/// default that the bed reads through it, high enough that the colouring still does.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct WaterOpacity(pub f32);

impl Default for WaterOpacity {
    fn default() -> Self {
        Self(0.4)
    }
}

impl WaterOpacity {
    /// Steps of the - and = keys, and the range they keep to.
    const STEP: f32 = 0.1;
    const MIN: f32 = 0.05;
    const MAX: f32 = 0.95;

    fn step(&mut self, up: bool) {
        let next = self.0 + if up { Self::STEP } else { -Self::STEP };
        // Round to the step so repeated presses land on 0.1, 0.2, ...
        self.0 = ((next / Self::STEP).round() * Self::STEP).clamp(Self::MIN, Self::MAX);
    }
}

/// Depths (m, positive down) at the two ends of the bed's colour map, for the legend.
#[derive(Resource, Clone, Copy, Debug)]
pub struct BedScale {
    pub shallow: f32,
    pub deep: f32,
}

/// The water's mesh and material, and its walls: the node each wall column hangs from.
#[derive(Resource)]
struct Water {
    mesh: Handle<Mesh>,
    material: Handle<StandardMaterial>,
    wall_nodes: Vec<usize>,
    bed_mesh: Handle<Mesh>,
    /// The place of every node, shared by the elements' nodes there: `shared[g]` < `n_shared`
    shared: Vec<u32>,
    n_shared: usize,
}

/// Numbers the nodes' places, the nodes of neighbouring elements on their common face
/// sharing one: returns the number of each node's place and how many there are.
fn shared_places(nodes: &Nodes) -> (Vec<u32>, usize) {
    // Within a centimetre: a place's nodes are the same point mapped from each element
    let mut places = std::collections::HashMap::new();
    let shared = nodes
        .xz
        .iter()
        .map(|p| {
            let key = ((p.x * 100.0).round() as i64, (p.y * 100.0).round() as i64);
            let next = places.len() as u32;
            *places.entry(key).or_insert(next)
        })
        .collect();
    (shared, places.len())
}

#[derive(Component)]
struct WaterSurface;

/// A mesh's vertex data.
#[derive(Default)]
struct Sheet {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colours: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl Sheet {
    /// The nodes as a surface at heights `z` (m), with the element's p² quads.
    fn surface(nodes: &Nodes, z: &[f32], vz: f32, colour: impl Fn(usize) -> [f32; 4]) -> Self {
        let mut normals = vec![[0.0; 3]; nodes.len()];
        nodes.normals(z, vz, &mut normals);
        let n1 = nodes.n_1d as u32;
        let mut indices = Vec::with_capacity(nodes.n_elements * 6 * (nodes.n_1d - 1).pow(2));
        for k in 0..nodes.n_elements as u32 {
            let base = k * nodes.n_nodes as u32;
            for j in 0..n1 - 1 {
                for i in 0..n1 - 1 {
                    let v0 = base + j * n1 + i;
                    let (v1, v2, v3) = (v0 + 1, v0 + n1 + 1, v0 + n1);
                    indices.extend([v0, v1, v2, v0, v2, v3]);
                }
            }
        }
        Self {
            positions: (0..nodes.len())
                .map(|g| [nodes.xz[g].x, z[g] * vz, nodes.xz[g].y])
                .collect(),
            normals,
            colours: (0..nodes.len()).map(colour).collect(),
            indices,
        }
    }

    /// Walls down from each boundary node's vertex to height `bottom(node)` (m), two
    /// vertices per column (top, bottom), on the boundary faces `(element, face)` that
    /// `on` takes; returns the node of every column.
    fn walls(
        &mut self,
        nodes: &Nodes,
        vz: f32,
        on: impl Fn(usize, usize) -> bool,
        bottom: impl Fn(usize) -> f32,
        bottom_colour: impl Fn(usize) -> [f32; 4],
    ) -> Vec<usize> {
        let n = nodes.n_nodes;
        let mut columns = Vec::new();
        for &(k, face) in nodes.boundary.iter().filter(|&&(k, f)| on(k, f)) {
            let column_nodes: Vec<usize> =
                nodes.face_nodes[face].iter().map(|&i| k * n + i).collect();
            // Outward: along the face turned away from the element's centre.
            let centre = (0..n).map(|i| nodes.xz[k * n + i]).sum::<Vec2>() / n as f32;
            let (first, last) = (
                nodes.xz[column_nodes[0]],
                nodes.xz[*column_nodes.last().unwrap()],
            );
            let mut out = (last - first).perp().normalize_or_zero();
            if out.dot(first - centre) < 0.0 {
                out = -out;
            }
            let normal = [out.x, 0.0, out.y];
            for (c, &g) in column_nodes.iter().enumerate() {
                let top = self.positions.len() as u32;
                self.positions.push(self.positions[g]);
                self.positions
                    .push([nodes.xz[g].x, bottom(g) * vz, nodes.xz[g].y]);
                self.normals.extend([normal, normal]);
                self.colours.extend([self.colours[g], bottom_colour(g)]);
                if c > 0 {
                    let (t0, b0, t1, b1) = (top - 2, top - 1, top, top + 1);
                    self.indices.extend([t0, b0, b1, t0, b1, t1]);
                }
            }
            columns.extend(column_nodes);
        }
        columns
    }

    fn mesh(self) -> Mesh {
        let n = self.positions.len();
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colours)
        // The photo view's current (water) and depth (bed)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32; 2]; n])
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, vec![[0.0f32; 2]; n])
        .with_inserted_indices(Indices::U32(self.indices))
    }
}

/// Colour of the water where the walls meet the bed.
const WATER_DEEP: [f32; 4] = [0.004, 0.02, 0.035, 0.9];

pub(crate) fn spawn(
    mut commands: Commands,
    nodes: Res<Nodes>,
    frame: Res<Frame>,
    opacity: Res<WaterOpacity>,
    terrain: Option<Res<Terrain>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let vz = frame.vz;
    let deepest = nodes.bed.iter().copied().fold(0.0_f32, f32::min);
    let shallowest = nodes
        .bed
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .min(0.0);
    let base = 1.1 * deepest;

    commands.insert_resource(BedScale {
        shallow: -shallowest,
        deep: -deepest,
    });
    let seabed = Lut::new(SEABED);
    let bed_colour =
        |g: usize| seabed.at((shallowest - nodes.bed[g]) / (shallowest - deepest).max(1e-6));
    let mut bed = Sheet::surface(&nodes, &nodes.bed, vz, bed_colour);
    let earth = Color::srgb(0.16, 0.14, 0.12).to_linear();
    // Whether the terrain carries on beyond a boundary face
    let beyond = |k: usize, face: usize| {
        terrain.as_ref().is_some_and(|terrain| {
            let ends = &nodes.face_nodes[face];
            let n = nodes.n_nodes;
            let mid = 0.5 * (nodes.xz[k * n + ends[0]] + nodes.xz[k * n + ends[ends.len() - 1]]);
            terrain.covers(mid)
        })
    };
    let open = |k: usize, face: usize| !beyond(k, face);
    bed.walls(
        &nodes,
        vz,
        open,
        |_| base,
        |_| [earth.red, earth.green, earth.blue, 1.0],
    );
    if let Some(terrain) = &terrain {
        let ground = GroundColours::new(BedScale {
            shallow: -shallowest,
            deep: -deepest,
        });
        let land = |g: usize| terrain.coast(g).unwrap_or(nodes.bed[g]);
        bed.walls(&nodes, vz, beyond, land, |g| ground.at(land(g)));
    }
    let bed_mesh = meshes.add(bed.mesh());
    commands.spawn((
        Name::new("Bed"),
        PhotoPart::Bed,
        Mesh3d(bed_mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            perceptual_roughness: 0.95,
            reflectance: 0.2,
            double_sided: true,
            cull_mode: None,
            ..default()
        })),
    ));

    // At rest until the first snapshot arrives.
    let still: Vec<f32> = nodes.bed.iter().map(|&b| b.max(0.0)).collect();
    let mut water = Sheet::surface(&nodes, &still, vz, |_| [0.1, 0.3, 0.45, 1.0]);
    let wall_nodes = water.walls(&nodes, vz, open, |g| nodes.bed[g], |_| WATER_DEEP);
    let material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, opacity.0),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.3,
        reflectance: 0.35,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    let mesh = meshes.add(water.mesh());
    commands.spawn((
        Name::new("Water"),
        WaterSurface,
        PhotoPart::Sea,
        Mesh3d(mesh.clone()),
        MeshMaterial3d(material.clone()),
        // The surface moves: its bounding box from rest would cull it wrongly.
        NoFrustumCulling,
    ));
    let (shared, n_shared) = shared_places(&nodes);
    commands.insert_resource(Water {
        mesh,
        material,
        wall_nodes,
        bed_mesh,
        shared,
        n_shared,
    });
}

fn keys(
    mut acts: MessageReader<Act>,
    field: Res<Field>,
    mut colouring: ResMut<Colouring>,
    mut style: ResMut<SurfaceStyle>,
    mut opacity: ResMut<WaterOpacity>,
) {
    for &Act(action) in acts.read() {
        let next = match action {
            Action::NextColour => colouring.by.next(field.waves.is_some()),
            Action::Colour(by) => by,
            Action::WaterStyle => {
                *style = match *style {
                    SurfaceStyle::Translucent => SurfaceStyle::Opaque,
                    SurfaceStyle::Opaque => SurfaceStyle::Hidden,
                    SurfaceStyle::Hidden => SurfaceStyle::Translucent,
                };
                continue;
            }
            Action::Clearer | Action::Denser => {
                opacity.step(action == Action::Denser);
                // Stepping the opacity shows the translucent water
                if *style != SurfaceStyle::Translucent {
                    *style = SurfaceStyle::Translucent;
                }
                continue;
            }
            _ => continue,
        };
        // The waves' maps read on opaque water, not mixed with the bed's colours
        match (colouring.by.is_waves(), next.is_waves(), *style) {
            (false, true, SurfaceStyle::Translucent) => *style = SurfaceStyle::Opaque,
            (true, false, SurfaceStyle::Opaque) => *style = SurfaceStyle::Translucent,
            _ => {}
        }
        colouring.by = next;
    }
}

fn style(
    style: Res<SurfaceStyle>,
    opacity: Res<WaterOpacity>,
    water: Option<Res<Water>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut surface: Query<&mut Visibility, With<WaterSurface>>,
) {
    let Some(water) = water else { return };
    if !(style.is_changed() || opacity.is_changed()) {
        return;
    }
    for mut visibility in &mut surface {
        *visibility = if *style == SurfaceStyle::Hidden {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
    }
    if let Some(mut material) = materials.get_mut(&water.material) {
        let (alpha, mode) = match *style {
            SurfaceStyle::Opaque => (1.0, AlphaMode::Opaque),
            _ => (opacity.0, AlphaMode::Blend),
        };
        material.base_color.set_alpha(alpha);
        material.alpha_mode = mode;
    }
}

/// Per-node buffers of [`update_water`], kept between frames.
#[derive(Default)]
struct WaterScratch {
    heights: Vec<f32>,
    speed: Vec<f32>,
    normals: Vec<[f32; 3]>,
    /// Sums of the current (world X, Z) and node counts at each shared place
    current: Vec<[f32; 3]>,
    /// Sums of H_s and of the breaking's share at each shared place
    sea: Vec<[f32; 2]>,
}

/// The share of the waves' energy depth-induced breaking takes per period: its power
/// `breaking` (W/m²) times the period `tp` (s) over the energy `ρ g H_s²/16`, 0 for no
/// waves.
pub fn breaking_share(breaking: f32, tp: f32, hs: f32) -> f32 {
    const RHO_G: f32 = 1025.0 * 9.81;
    let energy = RHO_G * hs * hs / 16.0;
    if energy > 1e-3 {
        (breaking * tp / energy).clamp(0.0, 4.0)
    } else {
        0.0
    }
}

#[allow(clippy::too_many_arguments)] // a Bevy system's parameters are its queries
fn update_water(
    playback: Res<Playback>,
    field: Res<Field>,
    nodes: Res<Nodes>,
    frame: Res<Frame>,
    water: Option<Res<Water>>,
    photo: Res<Photo>,
    mut colouring: ResMut<Colouring>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut scratch: Local<WaterScratch>,
) {
    let Some(water) = water else { return };
    if field.t.is_none() || !(playback.changed || colouring.is_changed() || photo.is_changed()) {
        return;
    }
    let WaterScratch {
        heights,
        speed,
        normals,
        current,
        sea,
    } = &mut *scratch;
    // Dry nodes sink just under the bed, out of sight.
    heights.clear();
    heights.extend(
        field
            .eta
            .iter()
            .zip(&nodes.bed)
            .map(|(&eta, &b)| if eta - b > H_DRY { eta } else { b - 0.5 }),
    );
    speed.clear();
    speed.extend(field.u.iter().zip(&field.v).map(|(u, v)| u.hypot(*v)));
    normals.resize(nodes.len(), [0.0; 3]);
    nodes.normals(heights, frame.vz, normals);
    // Refitting the scales is part of drawing this field: no change detection, or the
    // water would be rewritten every frame.
    colouring
        .bypass_change_detection()
        .fit(speed, &field.eta, field.waves.as_ref());

    let Some(mut mesh) = meshes.get_mut(&water.mesh) else {
        return;
    };
    let n = nodes.len();
    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for g in 0..n {
            positions[g][1] = heights[g] * frame.vz;
        }
        for (c, &g) in water.wall_nodes.iter().enumerate() {
            positions[n + 2 * c][1] = heights[g] * frame.vz;
        }
    }
    if let Some(VertexAttributeValues::Float32x3(out)) = mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
    {
        out[..n].copy_from_slice(normals);
    }
    if let Some(VertexAttributeValues::Float32x4(colours)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_COLOR)
    {
        for (g, colour) in colours.iter_mut().enumerate().take(n) {
            *colour = colouring.at(g, speed, &field);
        }
        for (c, &g) in water.wall_nodes.iter().enumerate() {
            colours[n + 2 * c] = colours[g];
        }
    }
    if !photo.on {
        return;
    }
    // The current at each place, the mean of its nodes': mesh (u, v) is world (u, −v)
    current.clear();
    current.resize(water.n_shared, [0.0; 3]);
    for g in 0..n {
        let c = &mut current[water.shared[g] as usize];
        c[0] += field.u[g];
        c[1] -= field.v[g];
        c[2] += 1.0;
    }
    let at = |g: usize| {
        let [x, z, count] = current[water.shared[g] as usize];
        [x / count, z / count]
    };
    if let Some(VertexAttributeValues::Float32x2(uv)) = mesh.attribute_mut(Mesh::ATTRIBUTE_UV_0) {
        for (g, uv) in uv.iter_mut().enumerate().take(n) {
            *uv = at(g);
        }
        for (c, &g) in water.wall_nodes.iter().enumerate() {
            uv[n + 2 * c] = at(g);
            uv[n + 2 * c + 1] = at(g);
        }
    }
    // The model's H_s and its breaking's share per period, averaged at each place as
    // the current is; 0 without waves
    sea.clear();
    sea.resize(water.n_shared, [0.0; 2]);
    if let Some(w) = &field.waves {
        for g in 0..n {
            let s = &mut sea[water.shared[g] as usize];
            s[0] += w.hs[g];
            s[1] += breaking_share(w.breaking[g], w.tp[g], w.hs[g]);
        }
    }
    let sea_at = |g: usize| {
        let place = water.shared[g] as usize;
        let count = current[place][2];
        [sea[place][0] / count, sea[place][1] / count]
    };
    if let Some(VertexAttributeValues::Float32x2(uv)) = mesh.attribute_mut(Mesh::ATTRIBUTE_UV_1) {
        for (g, uv) in uv.iter_mut().enumerate().take(n) {
            *uv = sea_at(g);
        }
        for (c, &g) in water.wall_nodes.iter().enumerate() {
            uv[n + 2 * c] = sea_at(g);
            uv[n + 2 * c + 1] = sea_at(g);
        }
    }
    // The depth of the water over the bed, 0 where it is dry
    drop(mesh);
    let Some(mut bed) = meshes.get_mut(&water.bed_mesh) else {
        return;
    };
    if let Some(VertexAttributeValues::Float32x2(uv)) = bed.attribute_mut(Mesh::ATTRIBUTE_UV_1) {
        for (g, uv) in uv.iter_mut().enumerate().take(n) {
            let depth = field.eta[g] - nodes.bed[g];
            uv[0] = if depth > H_DRY { depth } else { 0.0 };
        }
    }
}
