//! A sunlit tidal flat in 3D: do temperature and salinity stay physical
//! where columns are thin or dry (TODO P4.5)?
//!
//! A closed basin 2.4 km long, 6 m deep at one end, with a flat between
//! −0.9 and −0.1 m and a shore above it. A force along the basin rocks the
//! water with the period of a tide (3 h by default), so the flat floods to
//! about a metre and falls dry in every cycle; the weather's day is two
//! tides long, with noon at one low water and midnight at the next: a
//! shortwave of 800 W/m² at noon, a non-solar loss of 150 W/m² that grows
//! with the surface temperature, evaporation by day and rain by night. The
//! model is a case file's 3D run (`dgcom::case`): the `WetDry` 2D module,
//! 20 levels, a thin depth of 1 m, GLS k-ε, the horizontal Kuzmin limiter
//! about the initial profile, implicit vertical advection, SSP-RK(4,3).
//!
//! Every report line gives the range of T and S since the last one in four
//! kinds of column: dry (under a micrometre of water), thin (under the thin
//! depth), shallow (under `deep`, 3 m) and deep. The basin is closed, so the
//! volume is conserved, and without the weather (`heat=0`) the inventories
//! of T and S are too.
//!
//! ```text
//! cargo run --release --no-default-features --features parallel,simd --example tidal_flat_3d
//! cargo run --release ... --example tidal_flat_3d -- heat=0 uniform=1 hours=6
//! ```

#[path = "common/tidal_flat.rs"]
mod tidal_flat;

use dgcom::physics::JerlovWater;
use dgcom::time::SspScheme;
use tidal_flat::{Config, KINDS, Range};

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::args()
        .skip(1)
        .find_map(|a| {
            a.strip_prefix(name)
                .and_then(|v| v.strip_prefix('='))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(default)
}

fn main() {
    let d = Config::default();
    let jerlov: String = arg("jerlov", "III".to_string());
    let c = Config {
        length: arg("length", d.length),
        width: arg("width", d.width),
        nx: arg("nx", d.nx),
        ny: arg("ny", d.ny),
        order: arg("order", d.order),
        levels: arg("levels", d.levels),
        thin_depth: arg("thin", d.thin_depth),
        tide_period: 3600.0 * arg("tide_hours", d.tide_period / 3600.0),
        tide: arg("tide", d.tide),
        day: 3600.0 * arg("day_hours", d.day / 3600.0),
        hours: arg("hours", d.hours),
        heat: arg("heat", 1) != 0,
        shortwave: arg("shortwave", d.shortwave),
        cooling: arg("cooling", d.cooling),
        feedback: arg("feedback", d.feedback),
        freshwater: arg("freshwater", d.freshwater),
        jerlov: jerlov.parse().expect("jerlov: I, IA, IB, II or III"),
        uniform: (arg("uniform", 0) != 0).then_some((12.3, 33.1)),
        generic_mixing: arg("mixing", "k-epsilon".to_string()) == "generic",
        limiter: arg("limiter", 1) != 0,
        scheme: if arg("rk", 43) == 3 {
            SspScheme::Rk3
        } else {
            SspScheme::Rk43
        },
        cfl: arg("cfl", d.cfl),
        dt_max: arg("dt_max", d.dt_max),
        levels_lts: arg("lts", d.levels_lts),
        implicit_vertical: arg("implicit", 1) != 0,
        deep: arg("deep", d.deep),
        local_check: arg("local", 1) != 0,
    };
    let _: JerlovWater = c.jerlov;
    let report = 60.0 * arg("report_minutes", 30.0);
    println!(
        "tidal_flat_3d: {:.1} km × {:.0} m, P{} on {} × {} elements, {} levels, thin depth {} m; \
         tide {} m in {:.1} h, day {:.1} h; heat {}",
        c.length / 1e3,
        c.width,
        c.order,
        c.nx,
        c.ny,
        c.levels,
        c.thin_depth,
        c.tide,
        c.tide_period / 3600.0,
        c.day / 3600.0,
        if c.heat { "on" } else { "off" },
    );
    println!(
        "{:>6} {:>6} {:>11} | {:^13} | {:^13} | {:^13} | {:^13} | SW, Q_ns W/m²",
        "hours", "steps", "dt s", "T dry", "T thin", "T shallow", "T deep"
    );
    let pair = |r: Range| {
        if r.is_empty() {
            format!("{:^13}", "-")
        } else {
            format!("{:>6.2} {:>6.2}", r.low, r.high)
        }
    };
    // `trace=x`: the steps in which a value leaves the range every column
    // had before the step by more than x (°C), up to `trace_count` of them,
    // with the node's element at that level before and after
    let trace: f64 = arg("trace", 0.0);
    let mut traces_left: usize = arg("trace_count", 8);
    let mut previous: Option<dgcom::solver::state::Solution3D> = None;
    let mut each = |s: &dgcom::solver::state::Solution3D, t: f64, physics: &tidal_flat::Physics| {
        if trace <= 0.0 {
            return;
        }
        if let Some(prev) = previous.as_ref().filter(|_| traces_left > 0) {
            let (nn, nl) = (s.n_nodes, s.n_levels);
            let mut before = Range::default();
            prev.temp.iter().for_each(|&x| before.add(x));
            let worst = s
                .temp
                .iter()
                .enumerate()
                .map(|(j, &x)| (j, (before.low - x).max(x - before.high)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("values");
            if worst.1 > trace {
                traces_left -= 1;
                let (idx, l) = (worst.0 / nl, worst.0 % nl);
                let (k, i) = (idx / nn, idx % nn);
                let [x, y] = tidal_flat::position(physics, idx);
                println!(
                    "  trace t = {:.0} s ({:.3} h): T {:.3} at node {i} of element {k} ({x:.0}, {y:.0}) m, \
                     level {l}; every column before the step {:.3}–{:.3}",
                    t,
                    t / 3600.0,
                    s.temp[worst.0],
                    before.low,
                    before.high
                );
                println!("    node: depth before → after (m) | T before → after at level {l} | T column after (bed first)");
                for j in 0..nn {
                    let n = k * nn + j;
                    let bed = physics.bathymetry.data[n];
                    let column: Vec<String> = (0..nl)
                        .step_by((nl / 5).max(1))
                        .map(|m| format!("{:.2}", s.temp[n * nl + m]))
                        .collect();
                    println!(
                        "    {j}{} {:>10.3e} → {:>10.3e} | {:>8.3} → {:>8.3} | {}",
                        if j == i { "*" } else { " " },
                        prev.eta.data[n] - bed,
                        s.eta.data[n] - bed,
                        prev.temp[n * nl + l],
                        s.temp[n * nl + l],
                        column.join(" ")
                    );
                }
            }
        }
        previous = Some(s.clone());
    };
    let outcome = tidal_flat::run(
        &c,
        report,
        |p| {
            let (sw, ns, _) = c.weather(p.t, 10.0);
            println!(
                "{:>6.2} {:>6} {:>5.1} {:>5.1} | {} | {} | {} | {} | {:>4.0} {:>5.0}",
                p.t / 3600.0,
                p.steps,
                p.dt.0,
                p.dt.1,
                pair(p.ranges.0[0][0]),
                pair(p.ranges.0[1][0]),
                pair(p.ranges.0[2][0]),
                pair(p.ranges.0[3][0]),
                sw,
                ns
            );
        },
        &mut each,
    );
    println!(
        "\n{} steps of {:.1}–{:.1} s in {:.1} s ({:.2} ms per step); {} negative-depth clips",
        outcome.steps,
        outcome.dt.0,
        outcome.dt.1,
        outcome.wall,
        1e3 * outcome.wall / outcome.steps.max(1) as f64,
        outcome.clips
    );
    println!(
        "columns seen (node-steps): {} dry, {} thin with water, {} crossings of the thin depth",
        outcome.seen[0], outcome.seen[1], outcome.seen[2]
    );
    println!(
        "initial: T {:.3}–{:.3} °C, S {:.3}–{:.3}",
        outcome.initial[0].low, outcome.initial[0].high, outcome.initial[1].low, outcome.initial[1].high
    );
    for (kind, ranges) in KINDS.iter().zip(&outcome.ranges.0) {
        if ranges[0].is_empty() {
            continue;
        }
        println!(
            "{kind:>8}: T {:>8.3} … {:>8.3} °C, S {:>8.3} … {:>8.3}",
            ranges[0].low, ranges[0].high, ranges[1].low, ranges[1].high
        );
    }
    let (t, idx, level, depth) = outcome.thin_excess_at;
    println!(
        "thin and dry columns outside the range the others have had: T by {:.3e} °C (at {:.2} h, \
         node {idx}, level {level}, {depth:.2e} m of water), S by {:.3e}",
        outcome.thin_excess[0],
        t / 3600.0,
        outcome.thin_excess[1]
    );
    if c.local_check {
        println!(
            "thin columns outside what the water within four elements held before the step: T by \
             {:.3e} °C at most, {} values by more than 1e-9 °C; dry nodes changed by {:.3e} °C",
            outcome.local_excess.0, outcome.local_excess.1, outcome.dry_change
        );
    }
    println!(
        "drift: volume {:.2e}, T inventory {:.2e}, S inventory {:.2e} (relative; the last two \
         count the weather's fluxes)",
        outcome.drift[0], outcome.drift[1], outcome.drift[2]
    );
    if let Some(error) = &outcome.error {
        println!("FAILED: {error}");
        std::process::exit(1);
    }
}
