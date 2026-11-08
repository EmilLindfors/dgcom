//! `dgcom`: run a case file.
//!
//! ```text
//! dgcom run   <case.toml> [--set key.path=value]...
//! dgcom check <case.toml> [--set key.path=value]...
//! ```
//!
//! `run` builds the case and runs it, writing to the case's output
//! directory. `check` reads and validates the case, builds its mesh and bed,
//! and prints the case with every default filled in, without running it.
//! `--set` overrides a key of the case file, e.g. `--set numerics.cfl=0.5`.

use std::path::PathBuf;
use std::process::ExitCode;

use dgcom::case::{CaseFile, Domain, Progress};

const USAGE: &str = "\
dgcom: the Discontinuous Galerkin Coastal Ocean Model

usage:
  dgcom run   <case.toml> [--set key.path=value]...   run a case
  dgcom check <case.toml> [--set key.path=value]...   validate a case and print it resolved

A case file is TOML (or JSON, by its .json extension); see
cases/tidal_channel/case.toml and the `dgcom::case` documentation.";

fn main() -> ExitCode {
    match cli(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cli(args: Vec<String>) -> Result<(), String> {
    let mut args = args.into_iter();
    let command = args.next().ok_or(USAGE)?;
    if matches!(command.as_str(), "-h" | "--help" | "help") {
        println!("{USAGE}");
        return Ok(());
    }
    let mut path = None;
    let mut overrides = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--set" => overrides.push(args.next().ok_or("--set needs key.path=value")?),
            s if s.starts_with("--set=") => overrides.push(s["--set=".len()..].to_string()),
            s if s.starts_with('-') => return Err(format!("unknown option {s}\n\n{USAGE}")),
            _ if path.is_none() => path = Some(PathBuf::from(arg)),
            _ => return Err(format!("unexpected argument {arg}\n\n{USAGE}")),
        }
    }
    let path = path.ok_or_else(|| format!("{command} needs a case file\n\n{USAGE}"))?;
    let file = CaseFile::read(&path, &overrides).map_err(|e| e.to_string())?;
    match command.as_str() {
        "check" => check(&file),
        "run" => run(&file),
        other => Err(format!("unknown command {other}\n\n{USAGE}")),
    }
}

/// The domain's summary and what building it did, one line each.
fn describe(file: &CaseFile, domain: &Domain) -> Vec<String> {
    let bed = &domain.bathymetry;
    let mut lines = vec![format!(
        "{} elements of order {} ({} nodes), bed {:.2} to {:.2} m{}",
        domain.mesh.n_elements,
        domain.ops.order,
        domain.mesh.n_elements * domain.ops.n_nodes,
        bed.min(),
        bed.max(),
        match file.case.projection_center() {
            Some(c) => format!(", placed about {:.4}°N {:.4}°E", c.lat, c.lon),
            None => String::new(),
        }
    )];
    lines.extend(domain.notes.iter().cloned());
    lines
}

fn check(file: &CaseFile) -> Result<(), String> {
    let domain = file.build_domain().map_err(|e| e.to_string())?;
    for line in describe(file, &domain) {
        eprintln!("  {line}");
    }
    eprintln!("ok: output to {}", file.output_dir().display());
    print!("{}", file.case.resolved().to_toml());
    Ok(())
}

fn run(file: &CaseFile) -> Result<(), String> {
    let case = &file.case;
    println!(
        "dgcom: {} for {} h, output every {} min to {}",
        case.run.name,
        case.run.duration_hours,
        case.output.interval_minutes,
        file.output_dir().display()
    );
    let mut report = |p: &Progress| {
        println!(
            "  {:7.2} h / {:.2} h   η {:+.3} to {:+.3} m   |u| ≤ {:.3} m/s",
            p.time_s / 3600.0,
            p.end_s / 3600.0,
            p.eta_range.0,
            p.eta_range.1,
            p.max_speed
        );
    };
    let domain = file.build_domain().map_err(|e| e.to_string())?;
    for line in describe(file, &domain) {
        println!("  {line}");
    }
    println!(
        "  {:?}, CFL {:.3}",
        case.numerics.integrator,
        case.numerics.cfl_at(case.mesh.order)
    );
    let done = file
        .run_on(&domain, &mut report)
        .map_err(|e| e.to_string())?;
    let (v0, v1) = done.volume_m3;
    println!(
        "done: {} steps (dt {:.3}–{:.3} s) in {:.1} s; volume {:.6e} → {:.6e} m³",
        done.n_steps, done.dt_range_s.0, done.dt_range_s.1, done.wall_time_s, v0, v1
    );
    for s in &done.stations {
        let Some(g) = &s.gauge else { continue };
        println!(
            "gauge {} after {} h (gauge_skill.csv, gauge_constituents.csv):",
            s.name,
            case.spinup_hours()
        );
        for (what, score) in [
            ("observations", &g.vs_observations),
            ("tidal prediction", &g.vs_prediction),
        ] {
            match score {
                Ok(v) => println!(
                    "  vs {what}: centred RMSE {:.1} cm, bias {:+.1} cm, correlation {:.4} ({} samples)",
                    100.0 * v.centred_rmse,
                    100.0 * v.bias,
                    v.correlation,
                    v.n
                ),
                Err(e) => println!("  vs {what}: not compared ({e})"),
            }
        }
        match &g.model_fit {
            Ok(_) => println!(
                "  reference constants: root sum of squares of |ΔZ| {:.1} cm over {}",
                100.0 * g.rss_difference(),
                g.constituents
                    .iter()
                    .map(|c| c.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Err(e) => println!("  reference constants: not fitted ({e})"),
        }
    }
    Ok(())
}
