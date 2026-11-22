//! TODO P4.5 gates: temperature and salinity stay physical in the thin and
//! dry columns of a tidal flat, as in `examples/tidal_flat_3d.rs`, whose
//! experiment these run (a closed basin rocked with a tide's period, so that
//! its flat floods to about a metre and falls dry in every cycle; a case
//! file's 3D model with the runs' thin depth of 1 m and 20 levels).
//!
//! In the 72 h nested Frøya run of 2026-10-19 the thin columns reached 3.6
//! and 29.0 °C beside 5.3–15.3 °C in the others, and −2.00 and 40.00 °C (the
//! limiter's bounds) in the June 2026 runs. The step was the tracers'
//! central value on the subcell interfaces of the shore elements: a node
//! that loses the share `c` of its water at the mean of its own value and
//! its neighbour's keeps `C + c/(2(1 − c))` times its difference from the
//! neighbour, without bound as it falls dry (`notes/3d-numerics.md`). On
//! that code this flat's thin columns are at 2.7–12.1 °C beside 8.2–11.1 °C
//! with the weather, at 10.0–12.0 °C beside a start of 8.25–11.0 °C without
//! it, and at −2070 and 751 °C without it and without the limiter.

#[path = "../examples/common/tidal_flat.rs"]
mod tidal_flat;

use dgcom::time::SspScheme;
use tidal_flat::{Config, Outcome};

fn run(config: &Config) -> Outcome {
    let outcome = tidal_flat::run(config, 0.0, |_| {}, |_, _, _| {});
    assert!(
        outcome.error.is_none(),
        "the run failed: {:?}",
        outcome.error
    );
    assert_eq!(outcome.clips, 0, "negative depths clipped");
    outcome
}

/// The flat does what the gate is for: nodes fall dry and wet again,
/// columns are thin with water and cross the thin depth, and film cells
/// lose more than their water in a stage.
fn assert_regime(outcome: &Outcome, steps_per_cycle: usize, cycles: usize) {
    let [dry, thin, crossings] = outcome.seen;
    assert!(
        dry > steps_per_cycle * cycles && thin > 50 * steps_per_cycle * cycles,
        "test regime: {dry} dry and {thin} thin node-steps"
    );
    assert!(
        crossings >= 50 * cycles,
        "test regime: {crossings} crossings of the thin depth"
    );
    assert!(
        outcome.merged[0] > 0,
        "test regime: no cell lost more than its water in a stage"
    );
}

/// The gate. Twelve hours: four tides and two days of weather, with a
/// shortwave of 800 W/m² at noon over one low water and a loss of 150 W/m²
/// and more through the night over the next, evaporation by day and rain
/// by night.
///
/// **The range the physics allows a thin or a dry column**: the range the
/// columns that are not thin have had up to then. A thin column takes no
/// surface flux (the vertical terms skip it), so its water is water that
/// came from the other columns, carried by a transport in which every new
/// value is a mean, with weights that are not negative, of values the
/// water held before; and a dry node keeps its value, or takes that of the
/// water passing its cell. The other columns' own range moves with the
/// weather (here to 9.0–11.2 °C from 8.25–11.0 °C).
///
/// Measured: no thin or dry column outside that range by any amount (5.5 °C
/// and 1.14 in S before), the volume kept to 3.3e-13.
#[test]
fn thin_and_dry_columns_stay_in_range_on_a_sunlit_flat() {
    let config = Config::default();
    let outcome = run(&config);
    assert_regime(&outcome, 180, 4);
    let [t, s] = outcome.thin_excess;
    let (at, node, level, depth) = outcome.thin_excess_at;
    assert!(
        t < 1e-9,
        "a thin or dry column's temperature left the range the others have had by {t:.3e} °C \
         (at {:.2} h, node {node}, level {level}, {depth:.2e} m of water)",
        at / 3600.0
    );
    assert!(
        s < 1e-9,
        "a thin or dry column's salinity left the range the others have had by {s:.3e}"
    );
    // The weather acted on the others: their range is not the start's
    let others = |j: usize| {
        let mut range = outcome.ranges.0[tidal_flat::SHALLOW][j];
        range.merge(outcome.ranges.0[tidal_flat::DEEP][j]);
        range
    };
    assert!(
        others(0).outside(outcome.initial[0]) > 0.05,
        "test regime: the weather did not move the temperature"
    );
    assert!(
        outcome.drift[0] < 1e-12,
        "volume drifted by {:.3e}",
        outcome.drift[0]
    );
}

/// One tide of the gate above, for the fast tier (the others are slow
/// gates, `.config/nextest.toml`): the flat floods and falls dry once under
/// the sun, and the thin and dry columns stay in the range the others have
/// had (1.09 °C and 0.19 in S outside it before).
#[test]
fn thin_and_dry_columns_stay_in_range_through_one_tide() {
    let config = Config {
        hours: 3.0,
        ..Config::default()
    };
    let outcome = run(&config);
    assert_regime(&outcome, 180, 1);
    let [t, s] = outcome.thin_excess;
    assert!(
        t < 1e-9 && s < 1e-9,
        "thin or dry columns left the range the others have had by {t:.3e} °C, {s:.3e}"
    );
}

/// Without the weather the basin's heat and salt are its start's: the
/// inventories hold, and the thin and dry columns stay in the range the
/// others have had, with both 3D schemes, with the vertical advection
/// explicit, and on local time steps of the 3D stages. Each thin column's
/// values after a step lie within what the water within four elements of
/// it held before the step (the reach of four stages). Six hours, two
/// tides.
///
/// Measured (SSP-RK(4,3) at 60 s, SSP-RK3, explicit, local steps): no value
/// outside either range (before: 0.91, 0.70, 2.58 and 0.30 °C outside the
/// others' range, and 4000–5300 values a run outside their surroundings');
/// inventories 3.1e-13, 1.5e-12, 3.1e-13, 3.1e-13 (6.2e-11, 4.4e-12,
/// 7.9e-12, 8.0e-12).
#[test]
fn a_flats_tracers_are_bounded_and_conserved_without_the_weather() {
    let base = Config {
        heat: false,
        hours: 6.0,
        local_check: true,
        ..Config::default()
    };
    let variants = [
        ("SSP-RK(4,3)", base.clone()),
        (
            "SSP-RK3",
            Config {
                scheme: SspScheme::Rk3,
                cfl: 2.0,
                ..base.clone()
            },
        ),
        (
            "explicit vertical advection",
            Config {
                implicit_vertical: false,
                ..base.clone()
            },
        ),
        (
            "local time steps",
            Config {
                levels_lts: 2,
                ..base.clone()
            },
        ),
    ];
    for (name, config) in variants {
        let outcome = run(&config);
        assert_regime(&outcome, 180, 2);
        let [t, s] = outcome.thin_excess;
        assert!(
            t < 1e-9 && s < 1e-9,
            "{name}: thin or dry columns left the range the others have had by {t:.3e} °C, {s:.3e}"
        );
        let (local, count) = outcome.local_excess;
        assert!(
            count == 0,
            "{name}: {count} values of thin columns outside what the water around them held \
             before the step, by up to {local:.3e} °C"
        );
        let [volume, heat, salt] = outcome.drift;
        assert!(volume < 1e-12, "{name}: volume drifted by {volume:.3e}");
        assert!(
            heat < 5e-11 && salt < 5e-11,
            "{name}: inventories drifted by {heat:.3e} (T) and {salt:.3e} (S)"
        );
    }
}

/// The option that gives the thin columns the weather's fluxes
/// (`Hydrostatic3D::with_thin_column_fluxes`, off by default), with a bed
/// that holds heat as 0.1 m of water: the flat warms as a flat does, by
/// several degrees over the water beside it at the sunny low water, and
/// cools under it at night.
///
/// **The range the physics allows** then: no thin column beyond the
/// temperatures at which the weather's flux turns (31.7 °C at noon,
/// `Q_sw = Q_ns`, and 5.0 °C at night), or beyond what the other columns
/// have had; and after every step each thin column within what the water
/// within four elements held before it, widened by what the step's flux
/// does to 0.1 m of water.
///
/// Measured: thin columns 8.5–17.5 °C beside 8.2–12.3 °C; no value outside
/// the step's range.
#[test]
fn a_sunlit_flat_warms_with_its_bed_and_no_faster() {
    let config = Config {
        thin_bed: Some(0.1),
        local_check: true,
        ..Config::default()
    };
    let outcome = run(&config);
    assert_regime(&outcome, 180, 4);
    let range = |kind: usize| outcome.ranges.0[kind][0];
    let (thin, shallow, deep) = (
        range(tidal_flat::THIN),
        range(tidal_flat::SHALLOW),
        range(tidal_flat::DEEP),
    );
    let others = (shallow.low.min(deep.low), shallow.high.max(deep.high));
    // Where the weather's flux turns: noon's shortwave against the loss,
    // and the night's loss alone
    let noon = 10.0 + (config.shortwave - config.cooling) / config.feedback;
    let night = 10.0 - config.cooling / config.feedback;
    assert!(
        thin.high > others.1 + 3.0 && thin.low < others.0 + 1.0,
        "test regime: the flat did not warm and cool: thin columns at {:.2}–{:.2} °C beside          {:.2}–{:.2} °C",
        thin.low,
        thin.high,
        others.0,
        others.1
    );
    assert!(
        thin.high <= noon.max(others.1) && thin.low >= night.min(others.0),
        "thin columns at {:.2}–{:.2} °C, beyond where the weather's flux turns ({night:.1} and          {noon:.1} °C)",
        thin.low,
        thin.high
    );
    let (local, count) = outcome.local_excess;
    assert!(
        count == 0,
        "{count} values of thin columns outside what the water around them held before the          step and the step's flux, by up to {local:.3e} °C"
    );
    assert!(outcome.drift[0] < 1e-12, "volume drifted");
}

/// A uniform tracer stays uniform on the flat, with the weather off (its
/// fluxes would warm it) and at steps up to four minutes, where whole
/// elements of film are flushed within a stage.
///
/// Measured 7.3e-11 °C at 60 s and 9.3e-11 at 240 s (a 9 µm film: the
/// round-off of the pass's balance of a node over its water; 1.4e-10 and
/// 1.2e-10 before), 1.8e-13 and 2.4e-13 of the inventories.
#[test]
fn a_uniform_tracer_stays_uniform_on_the_flat() {
    for (dt_max, cfl) in [(60.0, 4.0), (240.0, 16.0)] {
        let config = Config {
            heat: false,
            hours: 6.0,
            uniform: Some((12.3, 33.1)),
            dt_max,
            cfl,
            ..Config::default()
        };
        let outcome = run(&config);
        assert_regime(&outcome, (10_800.0 / dt_max) as usize, 2);
        let mut drift = 0.0_f64;
        for ranges in &outcome.ranges.0 {
            for (range, uniform) in ranges.iter().zip([12.3, 33.1]) {
                if !range.is_empty() {
                    drift = drift.max(uniform - range.low).max(range.high - uniform);
                }
            }
        }
        assert!(
            drift < 5e-9,
            "steps to {dt_max} s: a uniform tracer drifted by {drift:.3e}"
        );
        let [_, heat, salt] = outcome.drift;
        assert!(
            heat < 5e-11 && salt < 5e-11,
            "steps to {dt_max} s: inventories drifted by {heat:.3e} and {salt:.3e}"
        );
    }
}
