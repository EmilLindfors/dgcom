//! TODO P4.6 gate: a river's brackish layer is hydraulically controlled at
//! the mouth of a deep fjord behind a sill (Stigebrandt 1981), as in
//! `examples/sill_fjord_3d.rs`, whose experiment this runs on a shorter
//! fjord (4 km) and basin (3 km, a 1 km relaxation band) with 20 levels, for
//! 12 h (the exchange is steady from ≈ 8 h; averaged over hours 9–12).

#[path = "../examples/common/sill_fjord.rs"]
mod sill_fjord;

use sill_fjord::{Config, S_SEA};

/// The brackish layer leaves the fjord at the critical speed
/// `(Q_f βg s₂/B)^⅓`, `F₁² = 1` at the mouth and subcritical inside, with the
/// thickness Knudsen's relations and its own mixing factor give; Knudsen's
/// relations hold at every section.
///
/// Measured (2026-10-21): at the mouth `u₁` 0.99× the theory's, `h₁` 1.13×,
/// `F₁²` 0.97; inside the fjord `F₁²` 0.57–0.62; Knudsen's closure
/// 0.98–0.99. The full experiment (16 km fjord, 6 km basin, 30 levels, 48 h)
/// gives 1.00×, 1.09×, 1.04 at the mouth, 0.96×, 1.15–1.18×, 0.90–0.93 with
/// 45 levels or 125 m elements, and follows `Q_f^⅓` from 250 to 1000 m³/s
/// to 3 % (`notes/3d-validation.md`).
#[test]
fn a_rivers_brackish_layer_is_critical_at_the_fjord_mouth() {
    let config = Config {
        length: 4e3,
        basin: 3e3,
        band: 1e3,
        levels: 20,
        hours: 12.0,
        average: 3.0,
        report: 0.0,
        ..Config::default()
    };
    let outcome = sill_fjord::run(&config, |_| {});
    assert!(outcome.finite, "the run blew up");
    // The horizontal Kuzmin limiter's relaxed bounds let S overshoot the
    // sea's by ≈ 1e-4
    let (s_low, s_high) = outcome.salt_range;
    assert!(
        s_low >= 0.0 && s_high <= S_SEA + 1e-3,
        "salinity left [0, {S_SEA}]: [{s_low}, {s_high}]"
    );

    let beta_g = config.beta_g();
    let mouth = outcome.mouth();
    for section in &outcome.sections {
        let closure = section.exchange.knudsen(config.qf);
        assert!(
            (0.95..1.02).contains(&closure),
            "x = {:.2} km: Knudsen's Q_out/(P Q_f) = {closure:.3}, not steady",
            section.x / 1e3
        );
        // Inside the fjord, landward of the sill's flank: subcritical
        if section.x <= config.crest() - 1000.0 {
            let b = section.brackish(config.delta);
            let fr2 = b.froude2(section.exchange.s_in, beta_g);
            assert!(
                fr2 < 0.75,
                "x = {:.2} km: F₁² = {fr2:.2} inside the fjord",
                section.x / 1e3
            );
        }
    }
    let b = mouth.brackish(config.delta);
    let s2 = mouth.exchange.s_in;
    let fr2 = b.froude2(s2, beta_g);
    assert!(
        (0.8..1.2).contains(&fr2),
        "F₁² = {fr2:.2} at the mouth: not critical"
    );
    let speed = b.u1 / config.mouth_speed(s2);
    assert!(
        (0.9..1.1).contains(&speed),
        "u₁ at the mouth {:.3} m/s, {speed:.2}× Stigebrandt's",
        b.u1
    );
    let thickness = b.h1 / config.mouth_thickness(s2, b.mixing_factor(s2));
    assert!(
        (0.9..1.3).contains(&thickness),
        "h₁ at the mouth {:.2} m, {thickness:.2}× Stigebrandt's",
        b.h1
    );
}
