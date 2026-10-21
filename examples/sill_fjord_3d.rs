//! A deep fjord behind a sill against Stigebrandt's two-layer theory
//! (TODO P4.6).
//!
//! A straight fjord, `length` long and `width` (B) wide, `depth` deep, with a
//! sill of `sill` m at its mouth, opens into a wider outer basin of the same
//! depth whose three outer sides are open (characteristic OBC, still water)
//! with a band relaxing the tracers to the sea's (S = 33, T uniform). A river
//! of `qf` m³/s of fresh water enters at the head. GLS k-ε mixes, the
//! f-plane rotates, no wind and no tide: the brackish layer's entrainment is
//! the outflow's own shear.
//!
//! In a strongly stratified ("N") fjord the brackish layer is controlled at
//! the mouth, where the fjord widens into the sea: its flow is critical
//! there, `F₁² = u₁²/(g′h₁) = 1`, with `g′ = βg(s₂ − s₁)` (Stommel & Farmer
//! 1953; Stigebrandt 1981, Estuar. Coast. Shelf Sci. 13, 197–211; Geyer &
//! Ralston 2011, Treatise on Estuarine and Coastal Science 2, eqs 23–29).
//! With Knudsen's relations `Q₁ = P Q_f`, `P = s₂/(s₂ − s₁)` the mixing
//! factor, this fixes the upper layer at the mouth:
//!
//! ```text
//!     u₁M = (Q_f β g s₂ / B)^{1/3}           (independent of the mixing)
//!     h₁M = (P² Q_f² / (β g s₂ B²))^{1/3}
//! ```
//!
//! and inside the fjord, where the layer is thicker and slower, the flow is
//! subcritical (`F₁² < 1`).
//!
//! The example averages the exchange through sections along the fjord over
//! the last `average` hours. Per section it reports Knudsen's in- and
//! outflow (each node and level by the sign of its mean transport) and
//! their closure `Q_out/(P Q_f)` (1 in a steady state); then the brackish
//! layer as one layer, which the theory needs and a continuous profile
//! does not define. The layer is the levels of the width-mean profile from
//! the surface to the halocline's foot, fresher than `s₂ − delta`, and its
//! thickness and speed are the kinematic equivalents `h₁ = (∫u dz)²/∫u² dz`,
//! `u₁ = ∫u² dz/∫u dz` (one layer with its transport and momentum flux),
//! `s₁` its transport-weighted salinity. The outflow below the halocline,
//! slow and nearly sea water, belongs to Knudsen's outflow but not to the
//! brackish layer: counted in, it made the "layer" 22 m thick at 0.08 m/s
//! where the halocline is 2 m deep. `F₁²` with the interface at `S = s₂/2`
//! instead shows how much the reduction matters. At the mouth: `u₁` and
//! `h₁` against the predictions, with the layer's own `P`.
//!
//! ```text
//! cargo run --release --no-default-features --features parallel,simd --example sill_fjord_3d -- \
//!     [qf=500] [width=500] [length=16000] [depth=100] [sill=30] [basin=6000] [dx=250] \
//!     [order=2] [levels=30] [theta_s=6] [hc=5] [hours=48] [average=12] [f=1.2e-4] \
//!     [cfl=4] [dt_max=120] [band=1500] [delta=1] [report=2] [profiles=0]
//! ```
//!
//! `profiles=N` prints the top N levels of every section's mean profile.

#[path = "common/sill_fjord.rs"]
mod sill_fjord;

use std::time::Instant;

use sill_fjord::{Config, S_SEA};

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
    let c = Config {
        qf: arg("qf", d.qf),
        width: arg("width", d.width),
        length: arg("length", d.length),
        depth: arg("depth", d.depth),
        sill: arg("sill", d.sill),
        basin: arg("basin", d.basin),
        dx: arg("dx", d.dx),
        order: arg("order", d.order),
        levels: arg("levels", d.levels),
        theta_s: arg("theta_s", d.theta_s),
        hc: arg("hc", d.hc),
        hours: arg("hours", d.hours),
        average: arg("average", d.average),
        f: arg("f", d.f),
        cfl: arg("cfl", d.cfl),
        dt_max: arg("dt_max", d.dt_max),
        band: arg("band", d.band),
        delta: arg("delta", d.delta),
        report: arg("report", d.report),
    };
    let shown: usize = arg("profiles", 0);
    let beta_g = c.beta_g();
    let sigma = c.sigma();
    let top = -sigma.sigma_w()[c.levels - 1];
    println!(
        "sill_fjord_3d: fjord {:.0} km × {:.0} m × {:.0} m, sill {:.0} m at {:.1} km, basin \
         {:.0} km; Q_f {:.0} m³/s; P{} on {:.0} m, {} levels (top layer {:.2} m at {:.0} m, \
         {:.2} m over the sill), f {:.1e}, GLS k-ε",
        c.length / 1e3,
        c.width,
        c.depth,
        c.sill,
        c.crest() / 1e3,
        c.basin / 1e3,
        c.qf,
        c.order,
        c.dx,
        c.levels,
        top * c.depth,
        c.depth,
        top * c.sill,
        c.f,
    );
    println!(
        "  Theory at the mouth (s₂ = {S_SEA}): u₁M = (Q_f βg s₂/B)^⅓ = {:.3} m/s, h₁M = {:.2}·P^⅔ m",
        c.mouth_speed(S_SEA),
        c.mouth_thickness(S_SEA, 1.0)
    );
    println!(
        "{:>6} {:>9} {:>7} {:>7} {:>7} {:>6} {:>8} {:>6}",
        "hours", "Q_o/PQ_f", "s₁", "h₁ m", "u₁ m/s", "F₁²", "S min", "dt s"
    );
    let start = Instant::now();
    let outcome = sill_fjord::run(&c, |p| {
        println!(
            "{:>6.1} {:>9.3} {:>7.2} {:>7.2} {:>7.3} {:>6.2} {:>8.3} {:>6.1}",
            p.t / 3600.0,
            p.exchange.knudsen(c.qf),
            p.brackish.s1,
            p.brackish.h1,
            p.brackish.u1,
            p.brackish.froude2(p.exchange.s_in, beta_g),
            p.salt_min,
            p.dt
        );
    });
    if !outcome.finite {
        println!("  the run blew up");
        return;
    }
    println!(
        "{} elements, {} steps ({:.1} s mean), {:.0} s wall; S in [{:.4}, {:.4}]",
        outcome.n_elements,
        outcome.steps,
        c.hours * 3600.0 / outcome.steps as f64,
        start.elapsed().as_secs_f64(),
        outcome.salt_range.0,
        outcome.salt_range.1
    );
    let (t_min, [x_min, y_min], h_min, l_min) = outcome.salt_min_at;
    println!(
        "  lowest S at {:.2} h, ({:.0}, {:.0}) m, {h_min:.1} m deep, level {l_min} (bed 0)",
        t_min / 3600.0,
        x_min,
        y_min
    );

    println!(
        "\nThe mean exchange over hours {:.0}–{:.0}: Knudsen's in- and outflow (by the sign of \
         the transport), and the brackish layer above S = s₂ − {} (its kinematic \
         equivalents; F₁² also with the interface at S = s₂/2)",
        c.hours - c.average,
        c.hours,
        c.delta
    );
    println!(
        "{:>7} {:>6} {:>7} {:>7} {:>6} {:>9} | {:>6} {:>6} {:>6} {:>8} {:>6} {:>8}",
        "x km",
        "bed m",
        "Q_o/Q_f",
        "Q_i/Q_f",
        "s_i",
        "Q_o/PQ_f",
        "h₁ m",
        "u₁ m/s",
        "s₁",
        "Q₁/P₁Q_f",
        "F₁²",
        "F₁²(s₂/2)"
    );
    for section in &outcome.sections {
        let e = &section.exchange;
        let b = section.brackish(c.delta);
        let half = section.brackish(0.5 * e.s_in);
        println!(
            "{:>7.2} {:>6.1} {:>7.2} {:>7.2} {:>6.2} {:>9.3} | {:>6.2} {:>6.3} {:>6.2} {:>8.3} {:>6.2} {:>8.2}",
            section.x / 1e3,
            -section.bed,
            e.q_out / c.qf,
            e.q_in / c.qf,
            e.s_in,
            e.knudsen(c.qf),
            b.h1,
            b.u1,
            b.s1,
            b.u1 * b.h1 * c.width / (b.mixing_factor(e.s_in) * c.qf),
            b.froude2(e.s_in, beta_g),
            half.froude2(e.s_in, beta_g),
        );
    }
    for section in outcome.sections.iter().filter(|_| shown > 0) {
        println!(
            "\nMean profile at x = {:.2} km (the top {shown} levels): depth m, H_z m, u m/s, S",
            section.x / 1e3
        );
        for level in section.profile.iter().take(shown) {
            println!(
                "  {:7.2} {:6.3} {:+8.4} {:7.3}",
                level.depth, level.hz, level.u, level.salt
            );
        }
    }
    let mouth = outcome.mouth();
    let b = mouth.brackish(c.delta);
    let s2 = mouth.exchange.s_in;
    let p = b.mixing_factor(s2);
    let (u_theory, h_theory) = (c.mouth_speed(s2), c.mouth_thickness(s2, p));
    println!(
        "\nAt the mouth (x = {:.2} km), the brackish layer's P = {p:.2}, s₂ = {s2:.2}:\n  \
         u₁: model {:.3} m/s, theory (Q_f βg s₂/B)^⅓ {u_theory:.3} ({:.2}×)\n  \
         h₁: model {:.2} m, theory (P² Q_f²/(βg s₂ B²))^⅓ {h_theory:.2} ({:.2}×)\n  \
         F₁²: {:.2}",
        mouth.x / 1e3,
        b.u1,
        b.u1 / u_theory,
        b.h1,
        b.h1 / h_theory,
        b.froude2(s2, beta_g)
    );
}
