//! The photo view (R): the sea as a camera would see it, in place of the colour maps
//! and arrows, so a run can be set beside a photograph of the place.
//!
//! The model's surface carries wind waves that the model does not resolve, drawn per
//! pixel by the shader (`shaders/water.wesl`) as a sum of [`N_WAVES`] sinusoids:
//!
//! - **Spectrum** ([`SeaState::new`]): the peak where a fully developed sea
//!   (Pierson–Moskowitz, ω_p = 0.877 g/U₁₀) or a fetch-limited one (JONSWAP,
//!   ω_p = 22 (g²/(U₁₀F))^⅓; Hasselmann et al. 1973) puts it, whichever is the
//!   younger; above it the saturation range, equal slope variance per octave of
//!   wavenumber (F(k) ∝ k⁻³), below it the Pierson–Moskowitz roll-off and the
//!   JONSWAP peak enhancement γ = 3.3. The total mean square slope is Cox & Munk's
//!   (1954) for a clean surface, 0.003 + 5.12·10⁻³ U₁₀: the waves get the share of
//!   it from the peak to ≈ 10 cm, the shorter ones are the surface's roughness. So the
//!   sun's glitter is as wide as it is over a real sea in that wind. Directions spread
//!   about the wind, wider for the shorter waves; wavevectors are snapped to a
//!   lattice that repeats every [`TILE`] metres, so the shader can evaluate them on
//!   positions wrapped to a tile, exactly in f32 anywhere on a coast.
//! - **Current**: the waves ride the model's depth-mean current (a vertex attribute,
//!   averaged over the nodes elements share so the pattern has no seams at element
//!   edges) by a two-phase flow map, and steepen against it (wave action conserved to
//!   first order in U/c: slope ∝ 1 − 3 U·k̂/c), so the chop of a tidal stream running
//!   against the wind shows, and whitecaps with it.
//! - **Whitecaps**: on the crests above the height that the wind's whitecap coverage
//!   (Monahan & O'Muircheartaigh 1980, W = 3.84·10⁻⁶ U₁₀^3.41) leaves above it in a
//!   Gaussian sea.
//! - **Optics**: the sky reflected by Fresnel, the sun's glint by a Beckmann
//!   distribution whose width is the slope variance below the pixel (the waves too
//!   short for it fade into it, so the far sea turns to glitter instead of shimmering).
//!   The bed and the ground under the sea are seen through the water: attenuated per
//!   channel down to them and back up (coastal water's attenuation, red first), the
//!   water's colour scattered in, so shallows read green and the deep fjord dark.
//! - **Sky and haze**: a sky dome with the sun where the scene's light is, haze
//!   toward the horizon (on everything else too, as Bevy's distance fog), bloom on
//!   the glints, HDR.
//!
//! - **The model's sea**: a run with waves (a coupled run's snapshot file, whose
//!   frames carry the sea state per node) draws its long waves from the model instead
//!   of the wind ([`SeaState::with_model`]): [`N_MODEL`] waves about the peak period,
//!   mean direction and spread of the model's sea where the camera looks
//!   ([`ModelSea::around`]), a JONSWAP spectrum from 0.7 to [`MODEL_TAIL`] times the
//!   peak wavenumber for an H_s of 1 m, which the shader scales per pixel by the
//!   model's H_s there (UV 1 of the water), so lees and fjords calm down and the
//!   exposed coast stands up. The wind then gives the shorter waves only (its
//!   saturation range beyond that band), the glitter and the whitecaps, whose crest
//!   height follows the local sea. Where the model's depth-induced breaking takes a
//!   large share of the waves' energy per period (UV 1's second value), the surf is
//!   white. The reference sea is rebuilt when the period, the direction or the spread
//!   where the camera looks moves on by more than [`REBUILD`].
//!
//! The waves run in wall-clock time while the playback runs (they are the sea of the
//! instant shown, not of the model's time-lapse); the current they ride is the
//! model's. The wind is `--wind SPEED,FROM` (m/s at 10 m, the compass direction it
//! blows from); W steps its speed, Shift+W veers it 45°. `--fetch KM` is the open
//! water upwind (10 km: a fjord's or a sound's). The shader's (u, v) is the model's
//! depth mean, so the surface current, faster in a stratified run, is understated.

use bevy::asset::embedded_asset;
use bevy::camera::Hdr;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{
    DistanceFog, FogFalloff, Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin,
};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::shader::ShaderRef;

use crate::actions::{Act, Action};
use crate::camera::OrbitCamera;
use crate::field::{Field, Frame, Nodes};
use crate::hud::RunClock;
use crate::playback::Playback;
use crate::screen::{RUN, RunResources, Screen, Viewing};
use crate::surface::SurfaceStyle;

/// Waves the shader sums.
pub const N_WAVES: usize = 32;
/// Of them, the model's sea's (the first ones), when the run has waves.
pub const N_MODEL: usize = 16;
/// The model's waves reach this many times the peak wavenumber; the wind's start there.
pub const MODEL_TAIL: f64 = 6.0;
/// The model's sea where the camera looks moves on this far (relative period, and
/// radians of direction and spread) before its waves are rebuilt.
pub const REBUILD: f64 = 0.12;
/// The wave pattern repeats every `TILE` metres in X and Z.
pub const TILE: f64 = 2048.0;
const G: f64 = 9.81;
/// Period (s) of the flow map's phases.
const FLOW_PERIOD: f32 = 6.0;
/// The shortest wave drawn (m), and the capillary end of the slope spectrum.
const SHORTEST: f64 = 0.1;
const CAPILLARY: f64 = 0.01;
/// The sky's luminance (cd/m², linear sRGB) overhead and at the horizon.
const SKY_ZENITH: Vec3 = Vec3::new(70.0, 170.0, 520.0);
const SKY_HORIZON: Vec3 = Vec3::new(520.0, 610.0, 740.0);
/// How far one sees through the haze (m), by Koschmieder's 3.912 / density.
const VISIBILITY: f32 = 70_000.0;
/// Diffuse attenuation of coastal water (1/m, red, green, blue): pure water's red
/// absorption, and coloured dissolved matter and plankton in the blue.
const ATTENUATION: Vec3 = Vec3::new(0.50, 0.12, 0.17);
/// Irradiance reflectance of deep coastal water: a dark blue-green.
const DEEP: Vec3 = Vec3::new(0.004, 0.022, 0.020);
/// Radius of the sky dome (m), inside the camera's far plane.
const SKY_RADIUS: f32 = 300_000.0;
/// The wind speeds W steps through (m/s).
const WIND_STEPS: [f32; 8] = [0.0, 2.0, 4.0, 6.0, 8.0, 11.0, 14.0, 18.0];

pub struct PhotoPlugin;

impl Plugin for PhotoPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/water.wesl");
        app.add_plugins(MaterialPlugin::<WaterMaterial>::default())
            .add_systems(
                OnEnter(Screen::Viewer),
                setup
                    .after(crate::surface::spawn)
                    .after(crate::terrain::spawn),
            )
            .add_systems(Update, (keys, switch, animate).chain().in_set(Viewing))
            .run_resource::<Photo>()
            .run_resource::<Sun>()
            .run_resource::<PhotoMaterials>()
            .run_resource::<ShownSea>();
    }
}

/// Whether the photo view is on, and its wind.
#[derive(Resource, Clone, Debug)]
pub struct Photo {
    pub on: bool,
    /// Wind speed at 10 m (m/s)
    pub wind_speed: f32,
    /// Compass direction the wind blows from (degrees)
    pub wind_from: f32,
    /// Open water upwind (m)
    pub fetch: f32,
    /// Latitude and longitude of the domain, for the real sun (with the run's clock)
    pub place: Option<[f64; 2]>,
}

impl Photo {
    /// The compass point the wind blows from, `SW`.
    pub fn wind_name(&self) -> &'static str {
        compass(self.wind_from)
    }
}

/// A photographer's view: from an eye `height` m above mean sea level over the mesh
/// point `at`, looking toward the compass `bearing` and `tilt` below the horizon
/// (degrees), turning about the point of the sea it looks at. Heights are exaggerated
/// as the world's are.
pub fn eye_view(frame: &Frame, at: [f64; 2], height: f32, bearing: f32, tilt: f32) -> OrbitCamera {
    let pitch = tilt.clamp(0.5, 89.0).to_radians();
    let rise = (height * frame.vz).max(1.0);
    let reach = rise / pitch.tan();
    let (sin, cos) = bearing.to_radians().sin_cos();
    // Looking along world (sin β, −cos β): the focus lies that way from the eye
    OrbitCamera {
        focus: frame.world(at, 0.0) + reach * Vec3::new(sin, 0.0, -cos),
        yaw: -bearing.to_radians(),
        pitch,
        distance: rise / pitch.sin(),
    }
}

/// [`eye_view`]'s numbers for a view: the eye's mesh point (m), its height (m), and
/// the bearing and tilt (degrees).
pub fn eye_of(view: &OrbitCamera, frame: &Frame) -> ([f64; 2], f32, f32, f32) {
    let eye = view.transform().translation;
    let at = [
        eye.x as f64 + frame.origin[0],
        -eye.z as f64 + frame.origin[1],
    ];
    let bearing = (-view.yaw.to_degrees()).rem_euclid(360.0);
    (at, eye.y / frame.vz, bearing, view.pitch.to_degrees())
}

/// Which of the photo view's materials an entity takes in place of its own.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhotoPart {
    /// A water surface
    Sea,
    /// The bed or the ground, seen through the water over it (the depth in UV 1)
    Bed,
}

/// An entity's own material while the photo view has replaced it.
#[derive(Component)]
struct ChartMaterial(Handle<StandardMaterial>);

#[derive(Component)]
struct SkyDome;

/// One wave: its wavevector in world (X, Z) (rad/m), its slope amplitude a·k, its
/// angular frequency (rad/s) and its phase at time 0 (rad).
#[derive(Clone, Copy, Debug)]
pub struct Wave {
    pub k: [f64; 2],
    pub slope: f64,
    pub omega: f64,
    pub phase: f64,
}

/// The model's sea at a place: its peak period (s), the direction it travels toward
/// in world (X, Z) (rad from +X toward +Z) and its directional spread (rad).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelSea {
    pub tp: f64,
    pub toward: f64,
    pub spread: f64,
}

impl ModelSea {
    /// The variance-weighted mean sea of the field's wet nodes within `radius` (world
    /// m) of `focus` (world X, Z), if any carry waves.
    pub fn around(field: &Field, nodes: &Nodes, focus: Vec2, radius: f32) -> Option<Self> {
        let w = field.waves.as_ref()?;
        let r2 = radius * radius;
        let (mut sum, mut tp, mut spread, mut dir) = (0.0f64, 0.0, 0.0, [0.0f64; 2]);
        for g in 0..nodes.len() {
            let hs = w.hs[g] as f64;
            if hs < 0.01 || nodes.xz[g].distance_squared(focus) > r2 {
                continue;
            }
            let weight = hs * hs;
            sum += weight;
            tp += weight * w.tp[g] as f64;
            spread += weight * w.spread[g] as f64;
            // Mesh (x, y) is world (X, −Z)
            dir[0] += weight * w.dir_x[g] as f64;
            dir[1] -= weight * w.dir_y[g] as f64;
        }
        (sum > 0.0 && tp > 0.0).then(|| Self {
            tp: tp / sum,
            toward: dir[1].atan2(dir[0]),
            spread: spread / sum,
        })
    }

    /// Whether `other` is far enough on to rebuild the waves for.
    pub fn moved(&self, other: &Self) -> bool {
        let turn = (self.toward - other.toward + std::f64::consts::PI)
            .rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        (self.tp / other.tp).ln().abs() > REBUILD
            || turn.abs() > REBUILD
            || (self.spread - other.spread).abs() > REBUILD
    }
}

/// The sea the shader draws: the wind's, or the model's long waves and the wind's
/// short ones.
#[derive(Clone, Debug)]
pub struct SeaState {
    /// The model's waves first (for an H_s of 1 m), then the wind's
    pub waves: Vec<Wave>,
    /// How many of the waves are the model's
    pub n_model: usize,
    /// Slope variance of the waves shorter than [`SHORTEST`]
    pub sub_variance: f64,
    /// Total mean square slope (Cox & Munk)
    #[cfg_attr(not(test), allow(dead_code))] // the tests check the waves add up to it
    pub mss: f64,
    /// Direction the wind blows toward, in world (X, Z) (rad from +X toward +Z)
    pub toward: f64,
    /// Rms height of the wind's drawn waves (m)
    pub rms_height: f64,
    /// Fraction of the sea under whitecaps
    pub whitecaps: f64,
    /// Rms heights above the mean that the whitecap fraction of a Gaussian sea exceeds
    pub crest: f64,
}

impl SeaState {
    /// The sea a wind of `speed` m/s from `from` (compass degrees) raises over `fetch` m.
    pub fn new(speed: f64, from: f64, fetch: f64) -> Self {
        Self::build(speed, from, fetch, None)
    }

    /// The model's sea `sea` and the wind's waves shorter than it.
    pub fn with_model(sea: ModelSea, speed: f64, from: f64, fetch: f64) -> Self {
        Self::build(speed, from, fetch, Some(sea))
    }

    fn build(speed: f64, from: f64, fetch: f64, model: Option<ModelSea>) -> Self {
        let u = speed.max(0.0);
        // The peak, from a wind of at least 1 m/s
        let ue = u.max(1.0);
        let omega_p = (0.877 * G / ue).max(22.0 * (G * G / (ue * fetch.max(100.0))).cbrt());
        let k_p = omega_p * omega_p / G;
        let mss = 0.003 + 5.12e-3 * u;
        // Slope variance per unit ln k, up to a constant
        let shape = |k: f64| {
            let omega = (G * k).sqrt();
            let r = omega_p / omega;
            let width = if omega <= omega_p { 0.07 } else { 0.09 };
            let peak = (-(omega - omega_p).powi(2) / (2.0 * (width * omega_p).powi(2))).exp();
            (-1.25 * r.powi(4)).exp() * 3.3f64.powf(peak)
        };
        let (ln_lo, ln_cap) = ((0.25 * k_p).ln(), (std::f64::consts::TAU / CAPILLARY).ln());
        let steps = 2000;
        let dl = (ln_cap - ln_lo) / steps as f64;
        let total: f64 = (0..steps)
            .map(|i| shape((ln_lo + (i as f64 + 0.5) * dl).exp()) * dl)
            .sum();

        let lattice = std::f64::consts::TAU / TILE;
        let mut rng = SplitMix(0x0005_eed0_f5ea);
        let mut waves = Vec::with_capacity(N_WAVES);
        // The model's waves for an H_s of 1 m, and where the wind's start
        let (n_wind, k_from) = match model {
            Some(sea) => {
                let k_m = (std::f64::consts::TAU / sea.tp.max(0.5)).powi(2) / G;
                waves = model_waves(sea, k_m, lattice, &mut rng);
                (N_WAVES - waves.len(), MODEL_TAIL * k_m)
            }
            None => (N_WAVES, 0.0),
        };
        let n_model = waves.len();

        // The wind's drawn waves, log-spaced from just below its peak (or from the
        // model's) to SHORTEST
        let k_hi = std::f64::consts::TAU / SHORTEST;
        let k_lo = (0.6 * k_p).max(k_from).min(k_hi / 4.0);
        let span = (k_hi / k_lo).ln();
        let toward = {
            let theta = from.to_radians();
            // World X east, Z south: the wind blows toward (−sin θ, cos θ)
            theta.cos().atan2(-theta.sin())
        };
        let mut resolved = 0.0;
        let mut height2 = 0.0;
        for i in 0..n_wind {
            let ln_k =
                k_lo.ln() + span * (i as f64 + 0.5 + 0.6 * (rng.next() - 0.5)) / n_wind as f64;
            let k = ln_k.exp();
            // Spread about the wind, wider for shorter waves
            let width = (0.5 + 0.5 * (k / k_p).ln() / 30f64.ln()).clamp(0.5, 1.0);
            let x = 2.0 * rng.next() - 1.0;
            let theta =
                toward + std::f64::consts::FRAC_PI_2 * width * x.signum() * x.abs().powf(1.5);
            let mut kv = [k * theta.cos(), k * theta.sin()];
            for c in &mut kv {
                *c = (*c / lattice).round() * lattice;
            }
            let k = kv[0].hypot(kv[1]);
            let phase = std::f64::consts::TAU * rng.next();
            if k == 0.0 {
                continue;
            }
            let variance = mss * shape(k) * span / n_wind as f64 / total;
            let slope = (2.0 * variance).sqrt();
            resolved += variance;
            height2 += 0.5 * (slope / k).powi(2);
            waves.push(Wave {
                k: kv,
                slope,
                omega: (G * k).sqrt(),
                phase,
            });
        }
        let whitecaps = if u > 3.0 {
            (3.84e-6 * u.powf(3.41)).min(0.2)
        } else {
            0.0
        };
        let rms_height = height2.sqrt();
        // Below the pixel: the wind's slopes beyond SHORTEST (all it does not draw,
        // without a model; with one, what the model's waves stand in for is not lost)
        let sub_variance = match model {
            None => (mss - resolved).max(0.0),
            Some(_) => {
                let steps = 400;
                let dl = (ln_cap - k_hi.ln()).max(0.0) / steps as f64;
                let above: f64 = (0..steps)
                    .map(|i| shape((k_hi.ln() + (i as f64 + 0.5) * dl).exp()) * dl)
                    .sum();
                mss * above / total
            }
        };
        Self {
            waves,
            n_model,
            sub_variance,
            mss,
            toward,
            rms_height,
            whitecaps,
            crest: upper_quantile(whitecaps.max(1e-9)),
        }
    }
}

/// [`N_MODEL`] waves of a JONSWAP sea of H_s 1 m peaking at wavenumber `k_p` (deep
/// water), from 0.7 to [`MODEL_TAIL`] times it, spread about `sea`'s direction as
/// wide as its spread (its rms angle), on the tile's lattice.
fn model_waves(sea: ModelSea, k_p: f64, lattice: f64, rng: &mut SplitMix) -> Vec<Wave> {
    let omega_p = (G * k_p).sqrt();
    // Height variance per unit ln k: the slope's, `k² F(k) k`, over k²
    let height = |k: f64| {
        let omega = (G * k).sqrt();
        let width = if omega <= omega_p { 0.07 } else { 0.09 };
        let peak = (-(omega - omega_p).powi(2) / (2.0 * (width * omega_p).powi(2))).exp();
        (-1.25 * (omega_p / omega).powi(4)).exp() * 3.3f64.powf(peak) / (k * k)
    };
    let (lo, hi) = ((0.7 * k_p).ln(), (MODEL_TAIL * k_p).ln());
    // |x|^1.5 of a uniform x has an rms of half its reach
    let reach = (2.0 * sea.spread).clamp(0.1, std::f64::consts::FRAC_PI_2);
    let mut waves: Vec<(Wave, f64)> = Vec::with_capacity(N_MODEL);
    for i in 0..N_MODEL {
        let ln_k = lo + (hi - lo) * (i as f64 + 0.5 + 0.6 * (rng.next() - 0.5)) / N_MODEL as f64;
        let x = 2.0 * rng.next() - 1.0;
        let theta = sea.toward + reach * x.signum() * x.abs().powf(1.5);
        let k = ln_k.exp();
        let kv = [k * theta.cos(), k * theta.sin()].map(|c| (c / lattice).round() * lattice);
        let k = kv[0].hypot(kv[1]);
        let phase = std::f64::consts::TAU * rng.next();
        if k == 0.0 {
            continue;
        }
        let wave = Wave {
            k: kv,
            slope: 0.0,
            omega: (G * k).sqrt(),
            phase,
        };
        waves.push((wave, height(k)));
    }
    // Their variances add up to an H_s of 1 m: m₀ = 1/16
    let total: f64 = waves.iter().map(|(_, v)| v).sum();
    waves
        .into_iter()
        .map(|(mut wave, v)| {
            let k = wave.k[0].hypot(wave.k[1]);
            wave.slope = (2.0 * v / total / 16.0).sqrt() * k;
            wave
        })
        .collect()
}

/// z with P(Z > z) = `p` for a standard normal Z, by bisection on erfc.
fn upper_quantile(p: f64) -> f64 {
    let tail = |z: f64| 0.5 * erfc(z / std::f64::consts::SQRT_2);
    let (mut lo, mut hi) = (-8.0, 8.0);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if tail(mid) > p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The complementary error function (Numerical Recipes' erfcc, |error| < 1.2·10⁻⁷).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// A small deterministic generator (SplitMix64), so the waves are the same every run.
struct SplitMix(u64);

impl SplitMix {
    /// Uniform in [0, 1).
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The shader's uniform; see `shaders/water.wesl` for the fields.
#[derive(Clone, Copy, Default, Debug, ShaderType)]
struct WaterParams {
    waves: [Vec4; N_WAVES],
    flow: Vec4,
    sun: Vec4,
    sun_colour: Vec4,
    sky_zenith: Vec4,
    sky_horizon: Vec4,
    attenuation: Vec4,
    deep: Vec4,
    fog: Vec4,
    foam: Vec4,
    misc: Vec4,
}

/// The material of the photo view's sea (kind 0), bed (1) and sky (2).
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct WaterMaterial {
    #[uniform(0)]
    params: WaterParams,
}

impl WaterMaterial {
    fn new(kind: f32) -> Self {
        Self {
            params: WaterParams {
                misc: Vec4::new(kind, TILE as f32, 0.0, 0.0),
                ..default()
            },
        }
    }
}

impl Material for WaterMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://dg_viz/shaders/water.wesl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        if self.params.misc.x == 0.0 {
            AlphaMode::Premultiplied
        } else {
            AlphaMode::Opaque
        }
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // The sky is seen from inside, the water from below its walls
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// The photo view's materials, and the time its waves have run (s).
#[derive(Resource)]
struct PhotoMaterials {
    sea: Handle<WaterMaterial>,
    bed: Handle<WaterMaterial>,
    sky: Handle<WaterMaterial>,
    clock: f64,
    state: SeaState,
    /// The model's sea the waves were built for, if the run has waves
    model: Option<ModelSea>,
    chart_light: Option<ChartLight>,
}

impl PhotoMaterials {
    /// The waves for the photo view's wind and the model's sea, if any.
    fn rebuild(&mut self, photo: &Photo) {
        let wind = (
            photo.wind_speed as f64,
            photo.wind_from as f64,
            photo.fetch as f64,
        );
        self.state = match self.model {
            Some(sea) => SeaState::with_model(sea, wind.0, wind.1, wind.2),
            None => SeaState::new(wind.0, wind.1, wind.2),
        };
    }
}

/// The model's sea the photo view draws, for the status (none without waves).
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct ShownSea(pub Option<ModelSea>);

fn setup(
    mut commands: Commands,
    photo: Res<Photo>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaterMaterial>>,
) {
    let sky = materials.add(WaterMaterial::new(2.0));
    commands.spawn((
        RUN,
        Name::new("Sky"),
        SkyDome,
        Mesh3d(meshes.add(Sphere::new(1.0).mesh().uv(48, 24))),
        MeshMaterial3d(sky.clone()),
        Transform::from_scale(Vec3::splat(SKY_RADIUS)),
        Visibility::Hidden,
        NoFrustumCulling,
    ));
    commands.insert_resource(Sun::default());
    commands.insert_resource(PhotoMaterials {
        sea: materials.add(WaterMaterial::new(0.0)),
        bed: materials.add(WaterMaterial::new(1.0)),
        sky,
        clock: 0.0,
        chart_light: None,
        model: None,
        state: SeaState::new(
            photo.wind_speed as f64,
            photo.wind_from as f64,
            photo.fetch as f64,
        ),
    });
    commands.insert_resource(ShownSea::default());
}

fn keys(mut acts: MessageReader<Act>, mut photo: ResMut<Photo>) {
    for &Act(action) in acts.read() {
        match action {
            Action::Photo => photo.on = !photo.on,
            Action::WindFrom if photo.on => {
                photo.wind_from = (photo.wind_from + 45.0).rem_euclid(360.0);
            }
            Action::Wind if photo.on => {
                photo.wind_speed = WIND_STEPS
                    .iter()
                    .copied()
                    .find(|&s| s > photo.wind_speed + 0.01)
                    .unwrap_or(0.0);
            }
            _ => {}
        }
    }
}

/// An entity the photo view re-materials: its own material, or the photo view's with
/// its own kept aside.
type PartMaterials = (
    Entity,
    &'static PhotoPart,
    Option<&'static MeshMaterial3d<StandardMaterial>>,
    Option<&'static ChartMaterial>,
);

/// Puts the photo view's materials, sky, haze and HDR in place, or takes them away.
#[allow(clippy::too_many_arguments)] // a Bevy system's parameters are its queries
fn switch(
    mut commands: Commands,
    photo: Res<Photo>,
    photo_materials: Option<ResMut<PhotoMaterials>>,
    mut style: ResMut<SurfaceStyle>,
    mut clear: ResMut<ClearColor>,
    mut chart_clear: Local<Option<Color>>,
    parts: Query<PartMaterials>,
    mut sky: Query<&mut Visibility, With<SkyDome>>,
    cameras: Query<Entity, With<OrbitCamera>>,
) {
    let Some(mut photo_materials) = photo_materials else {
        return;
    };
    if !photo.is_changed() {
        return;
    }
    if photo.on {
        photo_materials.rebuild(&photo);
    }
    for (entity, part, standard, chart) in &parts {
        let mut e = commands.entity(entity);
        match (photo.on, standard, chart) {
            (true, Some(standard), None) => {
                let material = match part {
                    PhotoPart::Sea => photo_materials.sea.clone(),
                    PhotoPart::Bed => photo_materials.bed.clone(),
                };
                e.insert((ChartMaterial(standard.0.clone()), MeshMaterial3d(material)))
                    .remove::<MeshMaterial3d<StandardMaterial>>();
            }
            (false, None, Some(chart)) => {
                e.insert(MeshMaterial3d(chart.0.clone()))
                    .remove::<(ChartMaterial, MeshMaterial3d<WaterMaterial>)>();
            }
            _ => {}
        }
    }
    for mut visibility in &mut sky {
        *visibility = if photo.on {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    for camera in &cameras {
        let mut e = commands.entity(camera);
        if photo.on {
            e.insert((
                Hdr,
                Bloom::NATURAL,
                DistanceFog {
                    // Bevy's fog colour is in display units: the haze's luminance at the
                    // default exposure
                    color: Color::linear_rgb(
                        SKY_HORIZON.x / 1000.0,
                        SKY_HORIZON.y / 1000.0,
                        SKY_HORIZON.z / 1000.0,
                    ),
                    falloff: FogFalloff::Exponential {
                        density: 3.912 / VISIBILITY,
                    },
                    ..default()
                },
            ));
        } else {
            e.remove::<(Hdr, Bloom, DistanceFog)>();
        }
    }
    if photo.on {
        chart_clear.get_or_insert(clear.0);
        // The water is the point of the view
        if *style == SurfaceStyle::Hidden {
            *style = SurfaceStyle::Translucent;
        }
    } else if let Some(colour) = chart_clear.take() {
        clear.0 = colour;
    }
}

/// The sun in the photo view: its elevation and azimuth (degrees, compass), and
/// whether it is the real sun of the shown instant and place.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct Sun {
    pub elevation: f32,
    pub azimuth: f32,
    pub real: bool,
}

/// The sun's elevation and azimuth (degrees; compass, clockwise from north) at Unix time
/// `unix` (s) seen from latitude `lat`, longitude `lon` (degrees): the U.S. Naval
/// Observatory's low-precision formulas (≈ 1′ in 1800–2200), no refraction.
pub fn sun_position(unix: f64, lat: f64, lon: f64) -> (f64, f64) {
    // Days since J2000.0, 2000-01-01 12:00 UTC
    let d = (unix - 946_728_000.0) / 86_400.0;
    let g = (357.529 + 0.985_600_28 * d).to_radians();
    let q = 280.459 + 0.985_647_36 * d;
    let ecliptic = (q + 1.915 * g.sin() + 0.020 * (2.0 * g).sin()).to_radians();
    let obliquity = (23.439 - 3.6e-7 * d).to_radians();
    let right_ascension = (obliquity.cos() * ecliptic.sin()).atan2(ecliptic.cos());
    let declination = (obliquity.sin() * ecliptic.sin()).asin();
    let gmst = 18.697_374_558 + 24.065_709_824_419_08 * d;
    let hour_angle = (15.0 * gmst + lon).to_radians() - right_ascension;
    let phi = lat.to_radians();
    let elevation =
        (phi.sin() * declination.sin() + phi.cos() * declination.cos() * hour_angle.cos()).asin();
    let azimuth =
        (-hour_angle.sin()).atan2(declination.tan() * phi.cos() - phi.sin() * hour_angle.cos());
    (
        elevation.to_degrees(),
        azimuth.to_degrees().rem_euclid(360.0),
    )
}

/// Light from a sun `elevation` degrees up, relative to the view's sun 35° up, as a
/// camera exposing for the scene would record it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Daylight {
    /// The sun's illuminance on a surface facing it, per channel (reddened low down)
    pub direct: Vec3,
    /// The sky's luminance
    pub sky: f32,
    /// How low and red the sun is, 0 to 1
    pub warmth: f32,
}

/// [`Daylight`] at `elevation` (degrees): the direct beam through Kasten & Young's
/// (1989) air mass at a clear sky's optical depths (Rayleigh and aerosol, most in the
/// blue), gone below the horizon; the sky's luminance falling with the sun into
/// twilight; and an exposure that follows the light three quarters of the way, so
/// dusk records darker than noon, but not as much darker as it is.
pub fn daylight(elevation: f64) -> Daylight {
    const DEPTH: [f64; 3] = [0.10, 0.17, 0.32];
    const REFERENCE: f64 = 35.0;
    let air_mass = |h: f64| 1.0 / (h.to_radians().sin() + 0.505_72 * (h + 6.079_95).powf(-1.6364));
    // The sun's disk going under the horizon
    let up = (elevation + 0.5).clamp(0.0, 1.0);
    let h = elevation.max(0.0);
    let direct = DEPTH.map(|tau| up * (-tau * (air_mass(h) - air_mass(REFERENCE))).exp());
    let sky = ((elevation + 6.0).to_radians().sin() / (REFERENCE + 6.0).to_radians().sin())
        .clamp(0.02, 1.3)
        .powf(0.8);
    // Horizontal illuminance, sun and sky, against the reference's
    let lux = |direct: f64, sky: f64, h: f64| direct * h.to_radians().sin() + 0.35 * sky;
    let exposure = (lux(direct[1], sky, h) / lux(1.0, 1.0, REFERENCE))
        .powf(0.75)
        .max(0.2);
    let warmth = 1.0 - (elevation / 12.0).clamp(0.0, 1.0);
    Daylight {
        direct: Vec3::from_array(direct.map(|d| (d / exposure) as f32)),
        sky: (sky / exposure) as f32,
        warmth: warmth as f32,
    }
}

/// The compass point of a bearing (degrees), `SW`.
pub fn compass(bearing: f32) -> &'static str {
    const POINTS: [&str; 8] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    POINTS[((bearing.rem_euclid(360.0) + 22.5) / 45.0) as usize % 8]
}

/// The scene's own light, kept while the photo view sets the sun.
#[derive(Clone, Copy, Debug)]
struct ChartLight {
    transform: Transform,
    illuminance: f32,
    colour: Color,
    ambient: f32,
}

/// Runs the waves, sets the sun (the real one when the run has a clock and a place)
/// and keeps the shader's uniform current; the sky follows the camera.
#[allow(clippy::too_many_arguments)] // a Bevy system's parameters are its queries
fn animate(
    time: Res<Time>,
    photo: Res<Photo>,
    playback: Res<Playback>,
    frame: Res<Frame>,
    (field, nodes): (Res<Field>, Res<Nodes>),
    mut shown_sea: ResMut<ShownSea>,
    mut looked: Local<Option<Vec2>>,
    run_clock: Option<Res<RunClock>>,
    photo_materials: Option<ResMut<PhotoMaterials>>,
    mut materials: ResMut<Assets<WaterMaterial>>,
    mut sun_out: ResMut<Sun>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut clear: ResMut<ClearColor>,
    mut lights: Query<(&mut DirectionalLight, &mut Transform), Without<SkyDome>>,
    mut cameras: Query<(&GlobalTransform, &OrbitCamera, Option<&mut DistanceFog>)>,
    mut sky: Query<&mut Transform, (With<SkyDome>, Without<DirectionalLight>)>,
) {
    let Some(mut pm) = photo_materials else {
        return;
    };
    let Ok((mut light, mut light_transform)) = lights.single_mut() else {
        return;
    };
    if !photo.on {
        // Back to the scene's own light
        if let Some(chart) = pm.chart_light.take() {
            *light_transform = chart.transform;
            light.illuminance = chart.illuminance;
            light.color = chart.colour;
            ambient.brightness = chart.ambient;
        }
        return;
    }
    let chart = *pm.chart_light.get_or_insert(ChartLight {
        transform: *light_transform,
        illuminance: light.illuminance,
        colour: light.color,
        ambient: ambient.brightness,
    });
    if !playback.paused {
        pm.clock += time.delta_secs_f64();
    }

    // The sun: the real one, or where the scene's light is
    let (elevation, azimuth, real) = match (&run_clock, photo.place) {
        (Some(clock), Some([lat, lon])) => {
            let (e, a) = sun_position(clock.0.unix(playback.t), lat, lon);
            (e as f32, a as f32, true)
        }
        _ => {
            let to = chart.transform.back();
            (
                to.y.asin().to_degrees(),
                to.x.atan2(-to.z).to_degrees().rem_euclid(360.0),
                false,
            )
        }
    };
    *sun_out = Sun {
        elevation,
        azimuth,
        real,
    };
    let light_of = daylight(elevation as f64);
    let toward = |e: f32| {
        let (e, a) = (e.to_radians(), azimuth.to_radians());
        Vec3::new(a.sin() * e.cos(), e.sin(), -a.cos() * e.cos())
    };
    let sun = toward(elevation);
    // The scene's light never shines up from under the horizon
    *light_transform = Transform::IDENTITY.looking_to(-toward(elevation.max(0.5)), Vec3::Y);
    let peak = light_of.direct.max_element().max(1e-6);
    light.illuminance = chart.illuminance * light_of.direct.max_element();
    let tint = light_of.direct / peak;
    light.color = Color::linear_rgb(tint.x, tint.y, tint.z);
    ambient.brightness = chart.ambient * light_of.sky;

    let zenith = SKY_ZENITH * light_of.sky;
    let horizon = SKY_HORIZON * light_of.sky;
    let haze = Color::linear_rgb(horizon.x / 1000.0, horizon.y / 1000.0, horizon.z / 1000.0);
    clear.0 = haze;
    if let Ok((camera, orbit, fog)) = cameras.single_mut() {
        if let Ok(mut dome) = sky.single_mut() {
            dome.translation = camera.translation();
        }
        if let Some(mut fog) = fog {
            fog.color = haze;
        }
        // The model's sea where the camera looks, within a kilometre or the view's
        // reach, looked for again when the field changes or the view moves on
        let focus = orbit.focus.xz();
        let radius = orbit.distance.max(1000.0);
        let moved = looked.is_none_or(|at| at.distance(focus) > 0.25 * radius);
        if playback.changed || moved || field.is_changed() {
            *looked = Some(focus);
            let sea = ModelSea::around(&field, &nodes, focus, radius);
            let rebuild = match (sea, pm.model) {
                (Some(new), Some(old)) => new.moved(&old),
                (a, b) => a.is_some() != b.is_some(),
            };
            if rebuild {
                pm.model = sea;
                pm.rebuild(&photo);
            }
            shown_sea.0 = pm.model;
        }
    }

    let t = pm.clock;
    let state = &pm.state;
    let mut waves = [Vec4::ZERO; N_WAVES];
    for (out, w) in waves.iter_mut().zip(&state.waves) {
        let phase = (w.phase - w.omega * t).rem_euclid(std::f64::consts::TAU);
        *out = Vec4::new(w.k[0] as f32, w.k[1] as f32, w.slope as f32, phase as f32);
    }
    let common = WaterParams {
        waves,
        flow: Vec4::new(
            (t / FLOW_PERIOD as f64).fract() as f32,
            FLOW_PERIOD,
            state.sub_variance as f32,
            state.n_model as f32,
        ),
        sun: sun.extend(chart.illuminance),
        sun_colour: light_of
            .direct
            .extend(0.5 * (zenith + horizon).dot(Vec3::splat(1.0 / 3.0))),
        sky_zenith: zenith.extend(light_of.warmth),
        sky_horizon: horizon.extend(0.0),
        attenuation: ATTENUATION.extend(0.0),
        deep: DEEP.extend(0.0),
        fog: horizon.extend(3.912 / VISIBILITY),
        foam: Vec4::new(
            state.crest as f32,
            state.rms_height as f32,
            state.whitecaps as f32,
            state.toward as f32,
        ),
        misc: Vec4::ZERO,
    };
    for handle in [&pm.sea, &pm.bed, &pm.sky] {
        if let Some(mut material) = materials.get_mut(handle) {
            let misc = Vec4::new(
                material.params.misc.x,
                TILE as f32,
                t.rem_euclid(3600.0) as f32,
                frame.vz,
            );
            material.params = WaterParams { misc, ..common };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R twice, and again: the chart view comes back each time with its own
    /// materials, light, ambient light, clear colour and camera (no HDR, bloom or fog).
    #[test]
    fn the_chart_view_comes_back_after_the_photo_view() {
        let mut world = World::new();
        let chart_clear = Color::srgb(0.05, 0.065, 0.08);
        let photo = Photo {
            on: false,
            wind_speed: 6.0,
            wind_from: 225.0,
            fetch: 10e3,
            place: None,
        };
        world.insert_resource(Time::<()>::default());
        world.insert_resource(Playback::new(60.0, 60.0, 1 << 20));
        world.insert_resource(Frame {
            origin: [0.0; 2],
            vz: 1.0,
        });
        world.insert_resource(Field::default());
        world.insert_resource(Nodes {
            n_elements: 0,
            n_nodes: 0,
            n_1d: 0,
            xz: Vec::new(),
            bed: Vec::new(),
            rx: Vec::new(),
            ry: Vec::new(),
            sx: Vec::new(),
            sy: Vec::new(),
            dr: Vec::new(),
            ds: Vec::new(),
            face_nodes: Default::default(),
            boundary: Vec::new(),
        });
        world.insert_resource(ShownSea::default());
        world.insert_resource(Sun::default());
        world.insert_resource(GlobalAmbientLight {
            brightness: 350.0,
            ..default()
        });
        world.insert_resource(ClearColor(chart_clear));
        world.insert_resource(SurfaceStyle::Translucent);
        world.init_resource::<Assets<WaterMaterial>>();
        let mut water_materials = world.resource_mut::<Assets<WaterMaterial>>();
        let (sea, bed, sky) = (
            water_materials.add(WaterMaterial::new(0.0)),
            water_materials.add(WaterMaterial::new(1.0)),
            water_materials.add(WaterMaterial::new(2.0)),
        );
        world.insert_resource(PhotoMaterials {
            sea,
            bed,
            sky,
            clock: 0.0,
            state: SeaState::new(6.0, 225.0, 10e3),
            model: None,
            chart_light: None,
        });
        world.insert_resource(photo);
        let own = Handle::<StandardMaterial>::default();
        let water = world
            .spawn((PhotoPart::Sea, MeshMaterial3d(own.clone())))
            .id();
        let dome = world.spawn((SkyDome, Visibility::Hidden)).id();
        let chart_light = Transform::from_xyz(-1.0, 1.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y);
        let light = world
            .spawn((
                DirectionalLight {
                    illuminance: 9_000.0,
                    ..default()
                },
                chart_light,
            ))
            .id();
        let camera = world
            .spawn((
                OrbitCamera {
                    focus: Vec3::ZERO,
                    yaw: 0.0,
                    pitch: 0.5,
                    distance: 1000.0,
                },
                GlobalTransform::default(),
            ))
            .id();
        // Registered once, as in the app: `switch` keeps the chart's clear colour in a `Local`
        let (switch, animate) = (
            world.register_system(switch),
            world.register_system(animate),
        );
        for _ in 0..2 {
            for on in [true, false] {
                world.resource_mut::<Photo>().on = on;
                world.run_system(switch).unwrap();
                world.run_system(animate).unwrap();
                let e = world.entity(water);
                assert_eq!(e.contains::<MeshMaterial3d<WaterMaterial>>(), on);
                assert_eq!(
                    e.get::<MeshMaterial3d<StandardMaterial>>()
                        .map(|m| m.0.clone()),
                    (!on).then(|| own.clone())
                );
                assert_eq!(e.contains::<ChartMaterial>(), on);
                let shown = |v: &Visibility| *v != Visibility::Hidden;
                assert_eq!(world.get::<Visibility>(dome).map(shown), Some(on));
                let c = world.entity(camera);
                assert_eq!(c.contains::<Hdr>(), on);
                assert_eq!(c.contains::<Bloom>(), on);
                assert_eq!(c.contains::<DistanceFog>(), on);
                let clear = world.resource::<ClearColor>().0;
                assert_eq!(clear == chart_clear, !on, "clear colour {clear:?}");
                if !on {
                    let l = world.entity(light);
                    assert_eq!(l.get::<DirectionalLight>().unwrap().illuminance, 9_000.0);
                    assert_eq!(*l.get::<Transform>().unwrap(), chart_light);
                    assert_eq!(world.resource::<GlobalAmbientLight>().brightness, 350.0);
                }
            }
        }
    }

    #[test]
    fn the_slopes_add_up_to_cox_and_munk() {
        for (speed, fetch) in [(0.0, 10e3), (4.0, 2e3), (8.0, 10e3), (15.0, 100e3)] {
            let sea = SeaState::new(speed, 225.0, fetch);
            assert_eq!(sea.waves.len(), N_WAVES);
            let drawn: f64 = sea.waves.iter().map(|w| 0.5 * w.slope * w.slope).sum();
            assert!(
                ((drawn + sea.sub_variance) - sea.mss).abs() < 1e-12,
                "U = {speed}: {drawn} + {} != {}",
                sea.sub_variance,
                sea.mss
            );
            // The drawn waves carry most of the variance, but not the capillaries'
            assert!(
                drawn > 0.3 * sea.mss && sea.sub_variance > 0.1 * sea.mss,
                "U = {speed}"
            );
        }
    }

    /// The model's waves add up to an H_s of 1 m about its peak and direction, and the
    /// wind's start where they end.
    #[test]
    fn the_models_sea_is_drawn_for_an_hs_of_one_metre() {
        let model = ModelSea {
            tp: 10.0,
            toward: 0.5,
            spread: 0.35,
        };
        let sea = SeaState::with_model(model, 12.0, 250.0, 10e3);
        assert_eq!(sea.n_model, N_MODEL);
        assert_eq!(sea.waves.len(), N_WAVES);
        let k = |w: &Wave| w.k[0].hypot(w.k[1]);
        let (ours, wind) = sea.waves.split_at(sea.n_model);
        let m0: f64 = ours.iter().map(|w| 0.5 * (w.slope / k(w)).powi(2)).sum();
        assert!((m0 - 1.0 / 16.0).abs() < 1e-12, "m0 {m0}");
        let k_p = (std::f64::consts::TAU / model.tp).powi(2) / G;
        let lattice = std::f64::consts::TAU / TILE;
        let mut mean = [0.0; 2];
        for w in ours {
            assert!(
                k(w) > 0.5 * k_p && k(w) < 1.1 * MODEL_TAIL * k_p,
                "{}",
                k(w) / k_p
            );
            for (m, c) in mean.iter_mut().zip(w.k) {
                *m += w.slope * w.slope * c / k(w);
            }
        }
        let angle = mean[1].atan2(mean[0]);
        assert!((angle - model.toward).abs() < 0.3, "mean direction {angle}");
        // The wind's are shorter (the lattice may move one by a step)
        let shortest_model = ours.iter().map(k).fold(0.0, f64::max);
        assert!(
            wind.iter()
                .all(|w| k(w) >= 0.9 * MODEL_TAIL * k_p - lattice)
        );
        assert!(wind.iter().map(k).fold(f64::INFINITY, f64::min) >= 0.8 * shortest_model);
        // Glitter below the pixel: the wind's slopes beyond the shortest wave only
        let plain = SeaState::new(12.0, 250.0, 10e3);
        assert!(sea.sub_variance > 0.0 && sea.sub_variance <= plain.sub_variance + 1e-12);
        assert_eq!(plain.n_model, 0);
    }

    #[test]
    fn the_models_sea_is_rebuilt_only_when_it_moves_on() {
        let a = ModelSea {
            tp: 10.0,
            toward: 3.1,
            spread: 0.4,
        };
        assert!(!a.moved(&a));
        // Across ±π the direction turns little
        assert!(!a.moved(&ModelSea { toward: -3.1, ..a }));
        assert!(a.moved(&ModelSea { toward: 2.8, ..a }));
        assert!(a.moved(&ModelSea { tp: 11.5, ..a }));
        assert!(!a.moved(&ModelSea { tp: 10.5, ..a }));
        assert!(a.moved(&ModelSea { spread: 0.6, ..a }));
    }

    #[test]
    fn the_waves_run_downwind_on_the_tile() {
        // From the south-west: toward the north-east, world (+X, −Z)
        let sea = SeaState::new(8.0, 225.0, 10e3);
        let lattice = std::f64::consts::TAU / TILE;
        let mut mean = [0.0; 2];
        for w in &sea.waves {
            for (m, k) in mean.iter_mut().zip(w.k) {
                assert!(((k / lattice) - (k / lattice).round()).abs() < 1e-9);
                *m += w.slope * k / w.k[0].hypot(w.k[1]);
            }
            // Deep-water dispersion
            assert!((w.omega - (G * w.k[0].hypot(w.k[1])).sqrt()).abs() < 1e-12);
        }
        let angle = mean[1].atan2(mean[0]).to_degrees();
        assert!((angle + 45.0).abs() < 25.0, "mean direction {angle}°");
    }

    #[test]
    fn the_peak_is_fetch_limited_in_a_fjord() {
        // 10 m/s over 10 km: JONSWAP's peak (≈ 3 s, ≈ 14 m) is far younger than a fully
        // developed sea's (≈ 7 s, ≈ 74 m)
        let fjord = SeaState::new(10.0, 0.0, 10e3);
        let ocean = SeaState::new(10.0, 0.0, 1e7);
        let longest = |s: &SeaState| {
            s.waves
                .iter()
                .map(|w| std::f64::consts::TAU / w.k[0].hypot(w.k[1]))
                .fold(0.0, f64::max)
        };
        assert!(longest(&fjord) < 30.0, "{}", longest(&fjord));
        assert!(longest(&ocean) > 80.0, "{}", longest(&ocean));
        assert!(ocean.rms_height > 3.0 * fjord.rms_height);
    }

    #[test]
    fn whitecaps_follow_monahan() {
        assert_eq!(SeaState::new(2.0, 0.0, 10e3).whitecaps, 0.0);
        let w = SeaState::new(10.0, 0.0, 10e3).whitecaps;
        assert!((w - 3.84e-6 * 10f64.powf(3.41)).abs() < 1e-12);
        // About 1 % at 10 m/s: the crest is ≈ 2.3 rms heights up
        assert!(
            (upper_quantile(w) - 2.33).abs() < 0.05,
            "{}",
            upper_quantile(w)
        );
        assert!((upper_quantile(0.5)).abs() < 1e-6);
        assert!((erfc(0.0) - 1.0).abs() < 1e-7 && (erfc(1.0) - 0.157_299_2).abs() < 1e-6);
    }

    #[test]
    fn a_photographers_view_comes_back_as_given() {
        let frame = Frame {
            origin: [1000.0, -2000.0],
            vz: 3.0,
        };
        let view = eye_view(&frame, [3500.0, 7200.0], 12.0, 225.0, 4.0);
        let (at, height, bearing, tilt) = eye_of(&view, &frame);
        assert!(
            (at[0] - 3500.0).abs() < 0.05 && (at[1] - 7200.0).abs() < 0.05,
            "{at:?}"
        );
        assert!((height - 12.0).abs() < 1e-3, "{height}");
        assert!((bearing - 225.0).abs() < 1e-3 && (tilt - 4.0).abs() < 1e-4);
        // South-west and down: world (−X, +Z), below the horizon
        let forward = view.transform().forward();
        assert!(
            forward.x < 0.0 && forward.z > 0.0 && forward.y < 0.0,
            "{forward:?}"
        );
    }

    #[test]
    fn the_sun_stands_where_it_should_at_froya() {
        let clock = |s: &str| dg_rs::time::ModelClock::parse(s).unwrap().unix(0.0);
        let (lat, lon) = (63.87, 8.70);
        // Midsummer: at local noon (11:25 UTC) due south, 90° − φ + 23.44° up; at
        // midnight due north, just under the horizon
        let (e, a) = sun_position(clock("2025-06-21T11:25:00Z"), lat, lon);
        assert!((e - (90.0 - lat + 23.44)).abs() < 0.3, "noon elevation {e}");
        assert!((a - 180.0).abs() < 2.0, "noon azimuth {a}");
        let (e, a) = sun_position(clock("2025-06-21T23:25:00Z"), lat, lon);
        assert!(
            (e - (23.44 - (90.0 - lat))).abs() < 0.3,
            "midnight elevation {e}"
        );
        assert!(!(3.0..=357.0).contains(&a), "midnight azimuth {a}");
        // Equinox: rising due east
        let (_, a) = sun_position(clock("2025-03-20T05:40:00Z"), lat, lon);
        assert!((a - 90.0).abs() < 8.0, "equinox morning azimuth {a}");
    }

    #[test]
    fn the_low_sun_is_red_and_the_night_has_none() {
        let reference = daylight(35.0);
        assert!((reference.direct - Vec3::ONE).abs().max_element() < 1e-6);
        assert!((reference.sky - 1.0).abs() < 1e-6 && reference.warmth == 0.0);
        let low = daylight(2.0);
        assert!(
            low.direct.x > 2.0 * low.direct.z && low.warmth > 0.8,
            "{low:?}"
        );
        let night = daylight(-3.0);
        assert_eq!(night.direct, Vec3::ZERO);
        assert!(night.sky > 0.0 && night.sky < reference.sky, "{night:?}");
        assert_eq!(compass(10.0), "N");
        assert_eq!(compass(130.0), "SE");
    }

    #[test]
    fn the_wind_is_named_by_where_it_comes_from() {
        let mut photo = Photo {
            on: true,
            wind_speed: 6.0,
            wind_from: 225.0,
            fetch: 10e3,
            place: None,
        };
        assert_eq!(photo.wind_name(), "SW");
        photo.wind_from = 350.0;
        assert_eq!(photo.wind_name(), "N");
    }
}
