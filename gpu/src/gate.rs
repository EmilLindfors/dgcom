//! How a gate holds the device to the CPU (P2.7, the rule of 2026-11-25 in
//! `notes/gpu.md`).
//!
//! Until 2026-11-25 every gate asked for the CPU's bits, in a build without
//! multiply–add contraction (`--no-fmad`), and printed "differs" in the
//! build that runs (with contraction), which nothing judged: there `all`
//! stopped at its first run. Now a gate has a [`Bar`]:
//!
//! - **Bit for bit**, where the kernels still mirror the CPU's loops
//!   (`Bar::mirrors`) and the build does not contract (found by a probe
//!   kernel, [`probe`]): any difference fails, as before.
//! - **Within a tolerance** otherwise (a build with contraction, or kernels
//!   that have left the CPU's order): every value within `Bar::tolerance` of
//!   its field's largest magnitude, but for a share `Bar::outliers` of a
//!   field's values where a threshold decides over several steps (a merged
//!   cell, a limited layer, a wet/dry node).
//!
//! A mutant is caught if the gate fails it by the same rule, so a tolerance
//! too loose to tell a mutant from round-off shows as `NOT CAUGHT`. The
//! mutants differ in thousands of values by 1e-6 to 1 of a field's largest;
//! contraction moves a right-hand side by 1e-13 to 2e-11 (`notes/gpu.md`).

use std::error::Error;
use std::sync::Mutex;

use cuda_core::DeviceBuffer;

use crate::swe2d::Gpu;

/// What a gate asks of the device's values against the CPU's.
#[derive(Clone, Copy, Debug)]
pub struct Bar {
    /// How far a value may be from the CPU's, over its field's largest
    /// magnitude
    pub tolerance: f64,
    /// The share of a field's values that may be farther
    pub outliers: f64,
    /// Whether the kernels mirror the CPU's order of operations, so that a
    /// build without contraction must give its bits
    pub mirrors: bool,
}

impl Bar {
    /// One evaluation of kernels that mirror the CPU: its bits without
    /// contraction, 1e-9 with (contraction moves these by ≤ 2e-11).
    pub const MIRRORED: Self = Self {
        tolerance: 1e-9,
        outliers: 0.0,
        mirrors: true,
    };

    pub const fn within(tolerance: f64) -> Self {
        Self {
            tolerance,
            outliers: 0.0,
            mirrors: true,
        }
    }

    pub const fn with_outliers(self, outliers: f64) -> Self {
        Self { outliers, ..self }
    }
}

struct State {
    /// Whether the build contracts a multiply and an add (`None`: not
    /// probed, taken as contracting)
    fused: Option<bool>,
    /// The gate that runs, and its bar
    name: &'static str,
    bar: Bar,
    /// The gates' verdicts so far
    held: usize,
    failed: Vec<String>,
}

static STATE: Mutex<State> = Mutex::new(State {
    fused: None,
    name: "",
    bar: Bar::MIRRORED,
    held: 0,
    failed: Vec::new(),
});

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Finds whether this build contracts `a·b + c` into one rounding, on the
/// device: `a·b + c` with `c = −round(a·b)` is the product's rounding error
/// when fused and zero when not. `DGCOM_GATE=tolerance` asks every gate for
/// its tolerance in either build.
pub fn probe(gpu: &Gpu) -> Result<bool, Box<dyn Error>> {
    let (a, b) = (1.0 + f64::EPSILON * 1024.0, 1.0 + f64::EPSILON * 4096.0);
    let c = -(a * b);
    let st = &gpu.stream;
    let (da, db, dc) = (
        DeviceBuffer::from_host(st, &[a])?,
        DeviceBuffer::from_host(st, &[b])?,
        DeviceBuffer::from_host(st, &[c])?,
    );
    let mut out = DeviceBuffer::<f64>::zeroed(st, 1)?;
    crate::model3d::multiply_add(gpu, &da, &db, &dc, &mut out)?;
    let value = out.to_host_vec(st)?[0];
    let fused = value != 0.0;
    assert!(
        value == 0.0 || value == a.mul_add(b, c),
        "the probe's a·b + c is neither fused nor separate: {value:e}"
    );
    state().fused = Some(fused);
    Ok(fused)
}

/// Whether the build contracts multiply–adds (true until [`probe`] has run).
pub fn fused() -> bool {
    state().fused.unwrap_or(true)
}

/// Starts gate `name` with its `bar`; prints how it judges.
pub fn start(name: &'static str, bar: Bar) {
    let mut s = state();
    s.name = name;
    s.bar = bar;
    drop(s);
    scope(None);
    println!("  ({})", describe());
}

/// The bar of the gate that runs, without printing (a part of a gate with
/// another bar).
pub fn set(bar: Bar) {
    state().bar = bar;
}

/// Whether the gate that runs asks for the CPU's bits.
pub fn exact() -> bool {
    let s = state();
    s.bar.mirrors
        && s.fused == Some(false)
        && std::env::var("DGCOM_GATE").as_deref() != Ok("tolerance")
}

/// The tolerance of the gate that runs, over a field's largest magnitude.
pub fn tolerance() -> f64 {
    state().bar.tolerance
}

fn describe() -> String {
    if exact() {
        "judged bit for bit: no multiply–add contraction, kernels in the CPU's order".into()
    } else {
        let bar = state().bar;
        let why = if !bar.mirrors {
            "the kernels have left the CPU's order"
        } else if std::env::var("DGCOM_GATE").as_deref() == Ok("tolerance") {
            "DGCOM_GATE=tolerance"
        } else {
            "a build with multiply–add contraction"
        };
        if bar.outliers > 0.0 {
            format!(
                "judged within {:.0e} of each field's largest value, but for {:.2} % of its \
                 values: {why}",
                bar.tolerance,
                100.0 * bar.outliers
            )
        } else {
            format!(
                "judged within {:.0e} of each field's largest value: {why}",
                bar.tolerance
            )
        }
    }
}

/// Whether a field holds: `identical` of its `len` values are the CPU's
/// bits, `beyond` of them farther than the tolerance.
pub fn holds(identical: usize, beyond: usize, len: usize) -> bool {
    if identical == len {
        return true;
    }
    if exact() {
        return false;
    }
    beyond as f64 <= state().bar.outliers * len as f64
}

/// The verdict's words for a gate's fixture, recorded: `same` is whether
/// every field held (and every mutant was caught).
pub fn verdict(label: &str, same: bool) -> String {
    let exact = exact();
    let mut s = state();
    if same {
        s.held += 1;
        if exact {
            "bit for bit".into()
        } else {
            format!("within {:.0e}", s.bar.tolerance)
        }
    } else {
        let name = s.name;
        s.failed.push(format!("{name}: {label}"));
        if exact {
            "NOT bit for bit".into()
        } else {
            format!("NOT within {:.0e}", s.bar.tolerance)
        }
    }
}

/// Records a failure that is not a comparison's (a count, a run that
/// stopped).
pub fn fail(what: &str) {
    let mut s = state();
    let name = s.name;
    s.failed.push(format!("{name}: {what}"));
}

/// The gates' verdicts: how many fixtures held, and the failures.
pub fn summary() -> (usize, Vec<String>) {
    let s = state();
    (s.held, s.failed.clone())
}

/// The share of a field's values the running gate lets be farther than its
/// tolerance.
pub fn outliers() -> f64 {
    state().bar.outliers
}

/// A mutant's verdict: `beyond` is whether the running gate fails it (a
/// field it does not hold), `differs` whether any value differs at all.
/// Bit for bit, any difference catches it. Within a tolerance, a mutant
/// whose effect is under the tolerance is not caught; where the kernels
/// still mirror the CPU that is no failure of this build (the build without
/// contraction catches it by its bits, and says so here), where they have
/// left the CPU's order it is one: the mutant must be made to show.
pub fn mutant(beyond: bool, differs: bool) -> (bool, &'static str) {
    if beyond {
        (true, ", caught")
    } else if !exact() && differs && state().bar.mirrors {
        (
            true,
            ": under the tolerance (caught bit for bit, in a build without contraction)",
        )
    } else {
        (false, ": NOT CAUGHT")
    }
}

/// Whether a small field of counts (cells merged, substeps, levels) holds:
/// its bits, or each count within 5 % where the gate judges by a tolerance
/// (a merged cell more or fewer where a threshold decides).
pub fn counts_hold(cpu: &[f64], device: &[f64]) -> bool {
    let identical = cpu
        .iter()
        .zip(device)
        .all(|(c, g)| c.to_bits() == g.to_bits());
    identical
        || (!exact()
            && cpu
                .iter()
                .zip(device)
                .all(|(c, g)| (c - g).abs() <= 0.05 * c.abs().max(1.0)))
}

/// Within a tolerance, the fields the running gate judges: those whose name
/// starts with `prefix` (`None`: all). Bit for bit, every field is judged.
pub fn scope(prefix: Option<&'static str>) {
    *SCOPE.lock().unwrap_or_else(|e| e.into_inner()) = prefix;
}

/// Whether the running gate judges the field `name` ([`scope`]).
pub fn judges(name: &str) -> bool {
    exact()
        || SCOPE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none_or(|prefix| name.starts_with(prefix))
}

static SCOPE: Mutex<Option<&'static str>> = Mutex::new(None);
