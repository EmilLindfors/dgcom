//! The pycnocline of a 3D run as a sheet in the water.
//!
//! In every water column (every DG node) the sheet sits at the height of the
//! strongest stratification, so its shape is the pycnocline's: doming over banks,
//! deepening in the basins, heaving with the tide. N cycles what "strongest" means:
//!
//! - density: the buoyancy frequency `N² = −(g/ρ₀) ∂ρ/∂z` (s⁻²), with ρ from the
//!   linear equation of state `froya_real_data` runs with (`LinearEOS::default()`);
//! - salinity: `−∂S/∂z` (per m), the halocline;
//! - temperature: `∂T/∂z` (°C per m), the thermocline;
//! - off.
//!
//! Each is taken between neighbouring layer centres (stable when positive) and its
//! largest value's height refined by the parabola through it and its two neighbours,
//! so the sheet moves smoothly through the layers instead of stepping from one
//! interface to the next. It is coloured by its depth below the surface, yellow
//! shallow to deep blue, on a scale fitted to the first frame it is drawn for (2nd to
//! 98th percentile): the pycnocline is a few metres to tens of metres down, too little
//! to read from its height alone. It is cut out where the stratification is weaker
//! than the 5th percentile of the first frame's (rounded down to a half decade), or
//! the column is shallower than [`MIN_DEPTH`]: there is no pycnocline to show.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::{NoFrustumCulling, RenderLayers};
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use dg_rs::physics::{EquationOfState, LinearEOS};

use crate::actions::{Act, Action};
use crate::colormap::{Lut, PLASMA};
use crate::field::{Frame, Nodes};
use crate::layers::Levels;
use crate::photo::Photo;
use crate::playback::Playback;
use crate::solver::Layers;

/// Columns shallower than this (m) carry no sheet, and the readout no steps.
pub(crate) const MIN_DEPTH: f32 = 2.0;
const G: f32 = 9.81;

/// What the sheet follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SheetShows {
    Density,
    Salinity,
    Temperature,
    Off,
}

impl SheetShows {
    fn next(self) -> Self {
        match self {
            Self::Density => Self::Salinity,
            Self::Salinity => Self::Temperature,
            Self::Temperature => Self::Off,
            Self::Off => Self::Density,
        }
    }

    fn index(self) -> Option<usize> {
        match self {
            Self::Density => Some(0),
            Self::Salinity => Some(1),
            Self::Temperature => Some(2),
            Self::Off => None,
        }
    }

    /// Name, and the unit of its strength, for the legend.
    pub fn describe(self) -> (&'static str, &'static str) {
        match self {
            Self::Density => ("pycnocline, N²", "s⁻²"),
            Self::Salinity => ("halocline, -dS/dz", "/m"),
            Self::Temperature => ("thermocline, dT/dz", "°C/m"),
            Self::Off => ("off", ""),
        }
    }
}

/// The sheet: what it shows, its colour scales and its mesh.
#[derive(Resource)]
pub struct Stratification {
    pub shows: SheetShows,
    /// log₁₀ of the strength at the 5th and 98th percentile of each criterion,
    /// fitted to the first frame drawn: the sheet is cut out below the first
    scales: [Option<[f32; 2]>; 3],
    /// Depth below the surface (m) at the ends of each criterion's colour map,
    /// fitted to the first frame drawn
    depth_scales: [Option<[f32; 2]>; 3],
    /// Whether the run has salinity (a snapshot file may not)
    pub has_salt: bool,
    /// Columns with a sheet in the frame drawn
    pub shown_columns: usize,
    eos: LinearEOS,
    lut: Lut,
    mesh: Handle<Mesh>,
    /// Height (m), depth below the surface (m) and strength of the sheet at every
    /// node, reused
    heights: Vec<f32>,
    depths: Vec<f32>,
    strength: Vec<f32>,
}

impl Stratification {
    pub fn new(shows: SheetShows) -> Self {
        Self {
            shows,
            scales: [None; 3],
            depth_scales: [None; 3],
            has_salt: true,
            shown_columns: 0,
            eos: LinearEOS::default(),
            lut: Lut::new(PLASMA),
            mesh: Handle::default(),
            heights: Vec::new(),
            depths: Vec::new(),
            strength: Vec::new(),
        }
    }

    /// The strength below which the sheet is cut out (not its logarithm).
    pub fn threshold(&self) -> Option<f32> {
        let [lo, _] = self.scales[self.shows.index()?]?;
        Some(10f32.powf(lo))
    }

    /// The depths (m below the surface) at the ends of the colour map shown.
    pub fn depth_scale(&self) -> Option<[f32; 2]> {
        self.depth_scales[self.shows.index()?]
    }
}

/// The strongest stratification in one column: its height (m) and strength.
///
/// `z` are the layer centres from the bed up, `value(l)` the criterion's field there,
/// oriented so that stable stratification is a positive difference upwards
/// (density: `ρ_{l−1} − ρ_l`; salinity likewise; temperature `T_l − T_{l−1}`), and
/// `scale` turns a difference per metre into the strength (`g/ρ₀` for N², else 1).
fn strongest(z: &[f32], value: impl Fn(usize) -> f32, scale: f32) -> Option<(f32, f32)> {
    let nl = z.len();
    if nl < 2 {
        return None;
    }
    // Strength at the interfaces between layer centres l − 1 and l (l = 1..nl)
    let at = |l: usize| scale * value(l) / (z[l] - z[l - 1]).max(1e-6);
    let mid = |l: usize| 0.5 * (z[l] + z[l - 1]);
    let (best, s0) = (1..nl)
        .map(|l| (l, at(l)))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    // No stable step (or NaN)
    if s0.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    if best == 1 || best == nl - 1 {
        return Some((mid(best), s0));
    }
    // The parabola's vertex through the three interfaces, in index space
    let (below, above) = (at(best - 1), at(best + 1));
    let curvature = below - 2.0 * s0 + above;
    let offset = if curvature < 0.0 {
        (0.5 * (below - above) / curvature).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let height = if offset >= 0.0 {
        mid(best) + offset * (mid(best + 1) - mid(best))
    } else {
        mid(best) + offset * (mid(best) - mid(best - 1))
    };
    Some((height, s0 - 0.25 * (below - above) * offset))
}

/// The strongest stratification of one column by `shows` (not `Off`): its height
/// (m) and strength, from the layer centres `z` (bed up) and the column's
/// temperature and salinity there.
pub(crate) fn column_strongest(
    shows: SheetShows,
    z: &[f32],
    temp: &[f32],
    salt: &[f32],
) -> Option<(f32, f32)> {
    let eos = LinearEOS::default();
    let rho = |l: usize| eos.compute_density(temp[l] as f64, salt[l] as f64, 0.0) as f32;
    match shows {
        SheetShows::Density => strongest(z, |l| rho(l - 1) - rho(l), G / eos.rho0 as f32),
        SheetShows::Salinity => strongest(z, |l| salt[l - 1] - salt[l], 1.0),
        SheetShows::Temperature => strongest(z, |l| temp[l] - temp[l - 1], 1.0),
        SheetShows::Off => None,
    }
}

pub struct StratificationPlugin;

impl Plugin for StratificationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn).add_systems(
            Update,
            (keys, draw.after(crate::field::interpolate)).chain(),
        );
    }
}

#[derive(Component)]
struct Sheet;

fn spawn(
    mut commands: Commands,
    nodes: Res<Nodes>,
    mut sheet: ResMut<Stratification>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // The water's topology: each element's p² quads between its own nodes
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
    let n = nodes.len();
    sheet.mesh = meshes.add(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0_f32; 3]; n])
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0_f32, 1.0, 0.0]; n])
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.0_f32; 4]; n])
        .with_inserted_indices(Indices::U32(indices)),
    );
    commands.spawn((
        Name::new("Pycnocline"),
        Sheet,
        RenderLayers::layer(crate::minimap::MAIN_ONLY),
        Mesh3d(sheet.mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            // Cut out where the stratification is weak; opaque elsewhere, so it sorts
            // correctly under the translucent water
            alpha_mode: AlphaMode::Mask(0.5),
            double_sided: true,
            cull_mode: None,
            perceptual_roughness: 0.6,
            ..default()
        })),
        // It moves with the water: a bounding box from its first frame would cull it.
        NoFrustumCulling,
        Visibility::Hidden,
    ));
}

fn keys(mut acts: MessageReader<Act>, mut sheet: ResMut<Stratification>) {
    for _ in acts.read().filter(|a| a.0 == Action::Sheet) {
        let mut next = sheet.shows.next();
        if next == SheetShows::Salinity && !sheet.has_salt {
            next = next.next();
        }
        sheet.shows = next;
    }
}

/// The criterion's stable difference upwards between layers `l − 1` and `l` of
/// column `g` (`[node][level]` fields `a` and `b`, `w` of the way to `b`).
fn difference(
    shows: SheetShows,
    eos: &LinearEOS,
    a: &Layers,
    b: &Layers,
    w: f32,
    g: usize,
    l: usize,
) -> f32 {
    let nl = a.n_levels;
    let at = |field_a: &[f32], field_b: &[f32], l: usize| {
        let (x, y) = (field_a[g * nl + l], field_b[g * nl + l]);
        x + w * (y - x)
    };
    match shows {
        SheetShows::Density => {
            let rho = |l: usize| {
                eos.compute_density(
                    at(&a.temp, &b.temp, l) as f64,
                    at(&a.salt, &b.salt, l) as f64,
                    0.0,
                ) as f32
            };
            rho(l - 1) - rho(l)
        }
        SheetShows::Salinity => at(&a.salt, &b.salt, l - 1) - at(&a.salt, &b.salt, l),
        SheetShows::Temperature => at(&a.temp, &b.temp, l) - at(&a.temp, &b.temp, l - 1),
        SheetShows::Off => 0.0,
    }
}

#[allow(clippy::too_many_arguments)] // a Bevy system's parameters are its queries
fn draw(
    playback: Res<Playback>,
    frame: Res<Frame>,
    nodes: Res<Nodes>,
    levels: Res<Levels>,
    mut sheet: ResMut<Stratification>,
    mut meshes: ResMut<Assets<Mesh>>,
    photo: Res<Photo>,
    mut visibility: Query<&mut Visibility, With<Sheet>>,
) {
    let Ok(mut visible) = visibility.single_mut() else {
        return;
    };
    // A chart of the water column, as the arrows and contours: not in the photo view
    let Some(criterion) = sheet.shows.index().filter(|_| !photo.on) else {
        visible.set_if_neq(Visibility::Hidden);
        return;
    };
    if !(playback.changed || sheet.is_changed() || photo.is_changed()) {
        return;
    }
    let Some((a, b, w)) = playback.bracket() else {
        return;
    };
    let (Some(la), Some(lb)) = (&a.layers, &b.layers) else {
        return;
    };
    let has_salt = !la.salt.is_empty() && !lb.salt.is_empty();
    if sheet.has_salt != has_salt {
        sheet.has_salt = has_salt;
    }
    let shows = sheet.shows;
    if !has_salt && shows != SheetShows::Temperature {
        // Density and salinity need salinity
        sheet.shows = SheetShows::Temperature;
        return;
    }
    let nl = la.n_levels;
    let n = nodes.len();
    let strength_scale = match shows {
        SheetShows::Density => G / sheet.eos.rho0 as f32,
        _ => 1.0,
    };
    let Stratification {
        eos,
        heights,
        depths,
        strength,
        ..
    } = &mut *sheet;
    heights.resize(n, 0.0);
    depths.resize(n, 0.0);
    strength.resize(n, 0.0);
    let mut z = vec![0.0_f32; nl];
    for g in 0..n {
        let eta = a.eta[g] + w * (b.eta[g] - a.eta[g]);
        let depth = eta - nodes.bed[g];
        strength[g] = 0.0;
        heights[g] = eta;
        if depth < MIN_DEPTH {
            continue;
        }
        for (z, &s) in z.iter_mut().zip(&levels.sigma) {
            *z = eta + s * depth;
        }
        if let Some((height, s)) = strongest(
            &z,
            |l| difference(shows, eos, la, lb, w, g, l),
            strength_scale,
        ) {
            heights[g] = height;
            depths[g] = eta - height;
            strength[g] = s;
        }
    }

    // The scales: fitted once per criterion, to this frame's sheet
    let percentiles = |mut values: Vec<f32>, lo: f32, hi: f32| {
        values.sort_by(f32::total_cmp);
        let pick = |q: f32| values[(q * (values.len() - 1) as f32) as usize];
        (!values.is_empty()).then(|| [pick(lo), pick(hi)])
    };
    if sheet.scales[criterion].is_none() {
        let logs = sheet
            .strength
            .iter()
            .filter(|&&s| s > 0.0)
            .map(|s| s.log10())
            .collect();
        let Some([lo, hi]) = percentiles(logs, 0.05, 0.98) else {
            return;
        };
        let lo = (2.0 * lo).floor() / 2.0;
        let hi = ((2.0 * hi).ceil() / 2.0).max(lo + 0.5);
        sheet.bypass_change_detection().scales[criterion] = Some([lo, hi]);
    }
    let [lo, _] = sheet.scales[criterion].expect("fitted above");
    let shown = |s: f32| s > 0.0 && s.log10() >= lo;
    if sheet.depth_scales[criterion].is_none() {
        let depths = (0..n)
            .filter(|&g| shown(sheet.strength[g]))
            .map(|g| sheet.depths[g])
            .collect();
        let Some([shallow, deep]) = percentiles(depths, 0.02, 0.98) else {
            return;
        };
        // Whole metres, or steps of 5 m over a range of more than 20 m
        let step = if deep - shallow > 20.0 { 5.0 } else { 1.0 };
        let shallow = (shallow / step).floor() * step;
        let deep = ((deep / step).ceil() * step).max(shallow + step);
        sheet.bypass_change_detection().depth_scales[criterion] = Some([shallow, deep]);
    }
    let [shallow, deep] = sheet.depth_scales[criterion].expect("fitted above");

    let mut positions = Vec::with_capacity(n);
    let mut colours = Vec::with_capacity(n);
    let mut n_shown = 0;
    for g in 0..n {
        positions.push([nodes.xz[g].x, sheet.heights[g] * frame.vz, nodes.xz[g].y]);
        // Shallow yellow to deep blue
        let mut colour = sheet
            .lut
            .at(1.0 - (sheet.depths[g] - shallow) / (deep - shallow));
        colour[3] = if shown(sheet.strength[g]) {
            n_shown += 1;
            1.0
        } else {
            0.0
        };
        colours.push(colour);
    }
    let mut normals = vec![[0.0_f32; 3]; n];
    nodes.normals(&sheet.heights, frame.vz, &mut normals);
    sheet.bypass_change_detection().shown_columns = n_shown;
    let Some(mut mesh) = meshes.get_mut(&sheet.mesh) else {
        return;
    };
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    visible.set_if_neq(Visibility::Inherited);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tanh pycnocline centred at −12 m: the sheet finds its centre between the
    /// layer centres, to a fraction of a layer.
    #[test]
    fn the_sheet_finds_a_pycnocline_between_layers() {
        let z: Vec<f32> = (0..20).map(|l| -30.0 + 1.5 * (l as f32 + 0.5)).collect();
        let rho = |z: f32| 1025.0 - 2.0 * ((z + 12.0) / 2.0).tanh();
        let (height, strength) =
            strongest(&z, |l| rho(z[l - 1]) - rho(z[l]), G / 1025.0).expect("stratified");
        assert!((height + 12.0).abs() < 0.3, "height {height}");
        // The peak of N² = (g/ρ₀)(2/2) sech²(0) ≈ 9.6e-3 s⁻², sampled over 1.5 m
        assert!((strength - 9.57e-3).abs() < 1e-3, "N² {strength}");
    }

    /// No stable stratification, no sheet.
    #[test]
    fn a_mixed_or_unstable_column_has_no_sheet() {
        let z = [-3.0, -2.0, -1.0];
        assert_eq!(strongest(&z, |_| 0.0, 1.0), None);
        assert_eq!(strongest(&z, |_| -0.1, 1.0), None);
    }
}
