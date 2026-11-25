//! Gate of a nesting parent model's columns on the GPU (P2.7 Phase 4.5k,
//! `notes/gpu.md`): [`Parent3D`] against dgcom's own `Nesting3D::columns`
//! over an `OceanModelColumns`, bit for bit under `--no-fmad`.
//!
//! The open channel (`Shape::CHANNEL`, a 600 m band at both ends) nested in
//! a parent made up in memory: a lon/lat grid at 400 m around it, four
//! hourly snapshots, a row of land along the channel's south side (stencils
//! renormalised over their wet corners) and a land corner at its east end
//! (nodes neither inside a wet cell nor within `max_snap` of a wet point:
//! not covered, so they take the child's own column), the parent's bed 4–16
//! m deep under a child 12–28 m deep (a deep reference below it). Three
//! parents: z-levels cut by the bed, linear in time; s-levels, cubic in time
//! (four snapshots in a stencil); z-levels with tracers alone and no deep
//! reference (the velocity's columns are then written only where the child's
//! own are taken). Each runs one sequence of calls on both sides, with the
//! columns compared after each:
//!
//! 1. every element at one time;
//! 2. the same again (every slot cached);
//! 3. a third of the elements left out, the others at eleven stage times
//!    (two launches of moments), the free surface moved at half the nodes
//!    and the child's u and T changed;
//! 4. the child's u and T changed again, η and times as in 3: every listed
//!    slot cached, so a node the parent does not cover keeps the column of 3;
//! 5. every element at the times of 3; 6. every element at another time.
//!
//! Mutants (the device's inputs): the time 10 min later, the free surface 1
//! cm higher, the deep reference's decay doubled, and call 4 on a device
//! without the cache of 1–3 (compared at its listed slots).

use std::sync::Arc;

use dgcom::boundary::{
    DeepReference, DeviceParent3D, Nesting3D, NestingBand3D, OceanColumnsOptions,
    OceanModelColumns, ParentColumns3D,
};
use dgcom::io::{
    CoordinateProjection, FieldSeries, GeoGrid, LocalProjection, OceanModelReader, ProfileLevels,
    ProfileSeries, TimeInterpolation,
};
use dgcom::mesh::BoundaryTag;
use dgcom::solver::Blocks;
use dgcom::solver::state::Solution3D;
use dgcom::source::SpongeProfile;
use dgcom::time::{ModelClock, StageTime};
use dgcom::vertical::MeanProfile;

use crate::gate::{self, Bar};
use crate::hooks::{Band3D, State3D};
use crate::parent::Parent3D;
use crate::stage3d::{Snapshots, report, report_mutant};
use crate::swe2d::{Gpu, NN, ORDER};
use crate::three_d::Statics3D;
use crate::transport3d::{Fixture, LEVELS, Shape};

/// The parent's first snapshot (Unix s): the model's time 0.
const T0: f64 = 1_750_000_000.0;
/// Columns shallower than this are thin.
const THIN: f64 = 14.0;
/// The parent grid: 400 m apart from (−400, −400) m, 10 × 6 points.
const SPACING: f64 = 400.0;
const NX: usize = 10;
const NY: usize = 6;
/// The z-levels (m).
const DEPTHS: [f64; 7] = [0.0, 2.0, 5.0, 10.0, 15.0, 22.0, 30.0];

/// How the parent's profiles are made.
#[derive(Clone, Copy)]
struct Kind {
    fractions: bool,
    velocity: bool,
    deep: bool,
    time: TimeInterpolation,
}

/// The parent grid point `(i, j)`'s position in the mesh (m).
fn point_xy(i: usize, j: usize) -> (f64, f64) {
    (-SPACING + SPACING * i as f64, -SPACING + SPACING * j as f64)
}

/// Whether grid point `(i, j)` is land: the south row, the east corner.
fn land(i: usize, j: usize) -> bool {
    j == 0 || (i >= 8 && j <= 2)
}

/// The parent of `kind` on a lon/lat grid through `projection`.
fn reader(projection: &LocalProjection, kind: Kind) -> OceanModelReader {
    let to_geo = |(x, y): (f64, f64)| projection.xy_to_geo(x, y);
    let lon: Vec<f64> = (0..NX).map(|i| to_geo(point_xy(i, 0)).1).collect();
    let lat: Vec<f64> = (0..NY).map(|j| to_geo(point_xy(0, j)).0).collect();
    let grid = GeoGrid::regular(lon, lat).expect("a regular grid");
    let m = grid.len();
    let times: Vec<f64> = (0..4).map(|t| T0 + 3600.0 * t as f64).collect();
    let nt = times.len();
    let at = |k: usize| (k % NX, k / NX);
    let depth: Vec<f64> = (0..m)
        .map(|k| {
            let (i, j) = at(k);
            let (x, y) = point_xy(i, j);
            if land(i, j) {
                0.0
            } else {
                10.0 + 6.0 * (1.3e-3 * x).sin() * (2.1e-3 * y + 0.4).cos()
            }
        })
        .collect();
    let n_levels = if kind.fractions { 6 } else { DEPTHS.len() };
    // Every point's levels: z-levels below its bed + 3 m missing; s-levels
    // at fractions that vary from point to point
    let fraction = |k: usize, l: usize| {
        let base = [0.04, 0.18, 0.37, 0.58, 0.79, 0.96][l];
        base + 0.015 * ((k * 7 + l * 3) % 5) as f64 - 0.03
    };
    let position = |k: usize, l: usize| {
        if kind.fractions {
            fraction(k, l) * depth[k]
        } else {
            DEPTHS[l]
        }
    };
    let profiles = |f: &dyn Fn(usize, f64, f64, f64) -> f64| {
        let mut data = Vec::with_capacity(nt * m * n_levels);
        for t in 0..nt {
            for k in 0..m {
                let (i, j) = at(k);
                let (x, y) = point_xy(i, j);
                for l in 0..n_levels {
                    let d = position(k, l);
                    let missing = land(i, j) || (!kind.fractions && d > depth[k] + 3.0);
                    data.push(if missing {
                        f32::NAN
                    } else {
                        f(t, x, y, d) as f32
                    });
                }
            }
        }
        let levels = if kind.fractions {
            ProfileLevels::Fraction(
                (0..m)
                    .flat_map(|k| (0..6).map(move |l| (k, l)))
                    .map(|(k, l)| {
                        if land(at(k).0, at(k).1) {
                            f64::NAN
                        } else {
                            fraction(k, l)
                        }
                    })
                    .collect(),
            )
        } else {
            ProfileLevels::Depth(DEPTHS.to_vec())
        };
        ProfileSeries::new(m, levels, data)
    };
    let u = profiles(&|t, x, y, d| {
        0.3 - 0.01 * d + 0.05 * (1.7e-3 * x + 0.3 * t as f64).sin() + 0.02 * (2e-3 * y).cos()
    });
    let v = profiles(&|t, x, _, d| 0.1 + 0.004 * d - 0.03 * (1.1e-3 * x - 0.5 * t as f64).cos());
    let temp = profiles(&|t, x, y, d| {
        12.0 - 0.2 * d + 0.4 * t as f64 + 0.3 * (9e-4 * x).sin() * (1.5e-3 * y).cos()
    });
    let salt = profiles(&|t, x, _, d| 33.0 + 0.05 * d - 0.02 * t as f64 + 0.1 * (6e-4 * x).cos());
    let ssh = FieldSeries::new(
        m,
        (0..nt * m)
            .map(|n| {
                if land(at(n % m).0, at(n % m).1) {
                    f32::NAN
                } else {
                    0.1
                }
            })
            .collect(),
    );
    OceanModelReader::new(grid, times)
        .expect("a parent")
        .with_ssh(ssh)
        .with_depth(depth)
        .with_profiles(kind.velocity.then_some((u, v)), Some(temp), Some(salt))
}

/// The deep reference: 13 °C at the surface, 0.08 °C cooler a metre down,
/// salter with depth.
fn deep_reference(decay: f64) -> DeepReference {
    let n = 41;
    let z = |i: usize| -40.0 + i as f64;
    DeepReference {
        profile: MeanProfile::new(
            -40.0,
            (0..n).map(|i| 13.0 + 0.08 * z(i)).collect(),
            (0..n).map(|i| 33.2 - 0.03 * z(i)).collect(),
        ),
        decay,
    }
}

/// The nesting of the channel in `kind`'s parent (the deep reference's
/// decay `decay`).
fn nesting(fx: &Fixture, kind: Kind, decay: f64) -> Nesting3D {
    let projection = LocalProjection::new(63.7, 8.6);
    let options = OceanColumnsOptions {
        time_interpolation: kind.time,
        max_snap: 250.0,
    };
    let mut columns = OceanModelColumns::new(
        Arc::new(reader(&projection, kind)),
        &fx.mesh,
        &fx.ops,
        projection,
        ModelClock::new(T0),
        options,
    );
    if kind.deep {
        columns = columns.with_deep_reference(deep_reference(decay));
    }
    let band = NestingBand3D {
        width: 600.0,
        profile: SpongeProfile::default(),
        velocity_timescale: Some(1800.0),
        tracer_timescale: Some(3600.0),
    };
    Nesting3D::new(
        Arc::new(columns) as Arc<dyn ParentColumns3D>,
        &fx.mesh,
        &fx.ops,
        LEVELS,
        &[BoundaryTag::Open],
        &band,
    )
    .expect("a nested channel")
}

/// One call of the sequence: the state, the times, the elements.
struct Call<'a> {
    label: &'static str,
    state: &'a Solution3D,
    time: StageTime<'a>,
    elements: Blocks<'a>,
}

/// The CPU's columns of `call` (`Nesting3D::columns`).
fn cpu_columns(fx: &Fixture, nesting: &Nesting3D, call: &Call) -> Snapshots {
    let columns = nesting.columns(
        call.state,
        &fx.bathymetry,
        &fx.sigma,
        call.time,
        call.elements,
        THIN,
    );
    named(call.label, columns.values().map(<[f64]>::to_vec))
}

fn named(label: &str, values: [Vec<f64>; 4]) -> Snapshots {
    ["u", "v", "T", "S"]
        .iter()
        .zip(values)
        .map(|(name, v)| (format!("{label} {name}"), v))
        .collect()
}

/// The parent of `nesting` on the device, with its band.
struct Device<'a> {
    nesting: &'a Nesting3D,
    parent: Parent3D,
    band: Band3D,
}

impl<'a> Device<'a> {
    fn new(
        gpu: &Gpu,
        statics: &Statics3D,
        nesting: &'a Nesting3D,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let Some(DeviceParent3D::Model(model)) = nesting.device_form() else {
            return Err("the parent model has no device form".into());
        };
        Ok(Self {
            nesting,
            parent: Parent3D::new(gpu, statics, nesting, model, THIN)?,
            band: Band3D::new(gpu, statics, nesting.slots())?,
        })
    }

    /// The device's columns of `call`.
    fn columns(
        &mut self,
        gpu: &Gpu,
        statics: &Statics3D,
        call: &Call,
    ) -> Result<Snapshots, Box<dyn std::error::Error>> {
        let Some(DeviceParent3D::Model(model)) = self.nesting.device_form() else {
            unreachable!("checked in new");
        };
        let state = State3D::upload(gpu, call.state)?;
        self.parent.evaluate(
            gpu,
            statics,
            model,
            &state,
            call.time,
            call.elements,
            &mut self.band,
        )?;
        let mut values: [Vec<f64>; 4] = Default::default();
        for (to, from) in values.iter_mut().zip(&self.band.values) {
            *to = from.to_host_vec(&gpu.stream)?;
        }
        Ok(named(call.label, values))
    }
}

/// `snapshots` at the slots of the elements `listed` only.
fn at_slots(snapshots: &Snapshots, nesting: &Nesting3D, listed: &[bool]) -> Snapshots {
    let nodes = nesting.slots().nodes;
    snapshots
        .iter()
        .map(|(name, values)| {
            let kept = nodes
                .iter()
                .enumerate()
                .filter(|&(_, &node)| listed[node / NN])
                .flat_map(|(slot, _)| values[slot * LEVELS..(slot + 1) * LEVELS].iter().copied())
                .collect();
            (name.clone(), kept)
        })
        .collect()
}

/// One parent's gate; whether all is the same to the bit (and, with
/// `mutants`, every mutant caught).
fn gate(
    gpu: &Arc<Gpu>,
    label: &str,
    kind: Kind,
    mutants: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let fx = Fixture::new(Shape::CHANNEL);
    let ne = fx.n_elements();
    let statics = fx.statics(gpu)?;
    let nest = nesting(&fx, kind, 3.0);
    // The states: A, then B (η moved at half the nodes, u and T changed),
    // then C (u and T changed again)
    let a = fx.state.clone();
    let mut b = a.clone();
    for (idx, eta) in b.eta.data.iter_mut().enumerate() {
        if idx % 2 == 0 {
            *eta += 0.07 * ((idx % 13) as f64 / 13.0 - 0.4);
        }
    }
    let nudge = |s: &mut Solution3D, by: f64| {
        s.u.iter_mut().for_each(|u| *u += by);
        s.temp.iter_mut().for_each(|t| *t -= 3.0 * by);
    };
    nudge(&mut b, 0.013);
    let mut c = b.clone();
    nudge(&mut c, 0.021);
    // Eleven stage times; a third of the elements left out
    let times: Vec<f64> = (0..ne).map(|k| 2000.0 + 450.0 * (k % 11) as f64).collect();
    let list: Vec<u32> = (0..ne as u32).filter(|k| k % 3 != 0).collect();
    let listed: Vec<bool> = (0..ne).map(|k| k % 3 != 0).collect();
    let all = Blocks::All(ne);
    let some = Blocks::list(ne, &list);
    let calls = [
        Call {
            label: "1 one time",
            state: &a,
            time: StageTime::Uniform(1800.0),
            elements: all,
        },
        Call {
            label: "2 cached",
            state: &a,
            time: StageTime::Uniform(1800.0),
            elements: all,
        },
        Call {
            label: "3 per element",
            state: &b,
            time: StageTime::PerElement(&times),
            elements: some,
        },
        Call {
            label: "4 cached",
            state: &c,
            time: StageTime::PerElement(&times),
            elements: some,
        },
        Call {
            label: "5 every element",
            state: &c,
            time: StageTime::PerElement(&times),
            elements: all,
        },
        Call {
            label: "6 another time",
            state: &c,
            time: StageTime::Uniform(9000.0),
            elements: all,
        },
    ];
    let slots = nest.slots();
    let uncovered = {
        let Some(DeviceParent3D::Model(model)) = nest.device_form() else {
            unreachable!("a parent model");
        };
        slots
            .nodes
            .iter()
            .zip(slots.positions)
            .filter(|&(&node, &p)| model.node(node, p).is_none())
            .count()
    };
    let thin = slots
        .nodes
        .iter()
        .filter(|&&n| fx.state.eta.data[n] - fx.bathymetry.data[n] < THIN)
        .count();
    println!(
        "{label}: the channel's {ne} P{ORDER} elements × {LEVELS} levels, {} slots ({uncovered} \
         the parent does not cover, {thin} thin)",
        slots.nodes.len()
    );
    let mut device = Device::new(gpu, &statics, &nest)?;
    println!(
        "  the parent on the device: {:.2} MB",
        device.parent.bytes as f64 / 1e6
    );
    let mut same = true;
    let mut cpu_by_call = Vec::new();
    for call in &calls {
        let cpu = cpu_columns(&fx, &nest, call);
        same &= report(&cpu, &device.columns(gpu, &statics, call)?);
        cpu_by_call.push(cpu);
    }
    if mutants {
        let cpu_first = &cpu_by_call[0];
        let mut later = Device::new(gpu, &statics, &nest)?;
        let call = Call {
            time: StageTime::Uniform(2400.0),
            ..calls[0]
        };
        same &= report_mutant(
            "the time 10 min later",
            cpu_first,
            &later.columns(gpu, &statics, &call)?,
        );
        let mut higher = a.clone();
        higher.eta.data.iter_mut().for_each(|e| *e += 0.01);
        let call = Call {
            state: &higher,
            ..calls[0]
        };
        same &= report_mutant(
            "the free surface 1 cm higher",
            cpu_first,
            &Device::new(gpu, &statics, &nest)?.columns(gpu, &statics, &call)?,
        );
        if kind.deep {
            let doubled = nesting(&fx, kind, 6.0);
            same &= report_mutant(
                "the deep reference's decay doubled",
                cpu_first,
                &Device::new(gpu, &statics, &doubled)?.columns(gpu, &statics, &calls[0])?,
            );
        }
        let fresh = Device::new(gpu, &statics, &nest)?.columns(gpu, &statics, &calls[3])?;
        same &= report_mutant(
            "call 4 without the cache, at its listed slots",
            &at_slots(&cpu_by_call[3], &nest, &listed),
            &at_slots(&fresh, &nest, &listed),
        );
    }
    println!("  {}", gate::verdict(label, same));
    Ok(same)
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!("== A nesting parent model's columns (Phase 4.5k) ==");
    gate::start("parent3d", Bar::MIRRORED);
    let parents = [
        (
            "z-levels, linear in time, a deep reference",
            Kind {
                fractions: false,
                velocity: true,
                deep: true,
                time: TimeInterpolation::Linear,
            },
            true,
        ),
        (
            "s-levels, cubic in time, a deep reference",
            Kind {
                fractions: true,
                velocity: true,
                deep: true,
                time: TimeInterpolation::Cubic,
            },
            false,
        ),
        (
            "z-levels, tracers alone",
            Kind {
                fractions: false,
                velocity: false,
                deep: false,
                time: TimeInterpolation::Linear,
            },
            false,
        ),
    ];
    let mut failed = Vec::new();
    for (label, kind, mutants) in parents {
        if !gate(gpu, label, kind, mutants)? {
            failed.push(label);
        }
    }
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!("parent3d: not bit for bit: {failed:?}").into())
    }
}
