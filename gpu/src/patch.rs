//! The patch gate (P8.14, `docs/patches.md`): a cut-out of a case's mesh
//! nested in its whole run's snapshots (`[mesh.patch] whole_run`, dgcom's
//! `RunParent`), on the CPU and on the GPU through the case runner, compared
//! as the case gate compares a case ([`case::compare`]).
//!
//! The whole run's state at the patch's open faces and across its band is
//! a parent's series per node (`ParentSeries2D`), which the device holds and
//! evaluates as a parent model's (`DeviceExternal2D::Parent`,
//! `DeviceSource2D::NestingRelaxation`); a 3D patch's band columns (u, v, T,
//! S on the σ-levels) are evaluated on the host (`RunParent` has no
//! `DeviceParent3D`) and uploaded.
//!
//! The tidal channel (`cases/tidal_channel`), 2D over 3 h with frames every
//! 5 minutes and 3D over 1.5 h with frames every 2: the whole run on the CPU,
//! then the 6 km around the channel's middle nested in it. Mutant (the
//! device's input): the GPU's patch nested in another whole run, its M2 1 cm
//! higher at the mouth.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use dgcom::case::CaseFile;

use crate::case;
use crate::gate::{self, Bar};
use crate::swe2d::Gpu;

/// The tidal channel's case files, 2D and 3D.
const CASES: [(&str, &str); 2] = [
    (
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cases/tidal_channel/case.toml"
        ),
        "2D",
    ),
    (
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../cases/tidal_channel/three_d.toml"
        ),
        "3D",
    ),
];

/// The case's tide with M2 `amplitude` m at the mouth.
fn tide(amplitude: f64) -> String {
    format!(
        "boundary.0.tide.constituents=[{{ name = \"M2\", amplitude_m = {amplitude:?}, phase_deg \
         = 0.0 }}, {{ name = \"S2\", amplitude_m = 0.35, phase_deg = 30.0 }}]"
    )
}

/// The whole run of `path` with `overrides` on the CPU into `directory`
/// (under the case's own folder): its snapshot file.
fn whole_run(
    path: &Path,
    overrides: &[String],
    directory: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let mut all = overrides.to_vec();
    all.push("compute.device=\"cpu\"".into());
    all.push(format!("output.directory={directory:?}"));
    let file = CaseFile::read(path, &all)?;
    file.run(&mut |_| {})?;
    let snapshot = file
        .output_dir()
        .join(format!("{}.dgsnap", file.case.run.name));
    Ok(std::path::absolute(snapshot)?)
}

pub fn run(_gpu: &Arc<Gpu>) -> Result<(), Box<dyn Error>> {
    println!("== a patch nested in its whole run (P8.14) ==");
    // Whole runs, as the case gate's
    gate::start("patch", Bar::within(1e-7));
    let mut all = true;
    for (path, label) in CASES {
        let path = Path::new(path);
        let window = if label == "2D" {
            ["run.duration_hours=3.0", "output.interval_minutes=5.0"]
        } else {
            ["run.duration_hours=1.5", "output.interval_minutes=2.0"]
        }
        .map(String::from);
        let with = |amplitude: f64| {
            let mut o = window.to_vec();
            o.push(tide(amplitude));
            o
        };
        let whole = whole_run(path, &with(1.0), "output/gpu_gate/patch_whole")?;
        let other = whole_run(path, &with(1.01), "output/gpu_gate/patch_whole_m2")?;
        let nested_in =
            |snapshot: &Path| format!("mesh.patch.whole_run={:?}", snapshot.display().to_string());
        let mut patch = window.to_vec();
        patch.extend([
            "mesh.patch.element=60".to_string(),
            "mesh.patch.radius_km=6.0".to_string(),
            nested_in(&whole),
        ]);

        println!("{label}, the patch nested in its whole run:");
        let (same, _) = case::compare(path, &patch, &[])?;
        println!("{label}, mutant: the GPU's patch nested in the whole run with M2 1 cm higher:");
        let (held, differs) = case::compare(path, &patch, &[nested_in(&other)])?;
        let (caught, words) = gate::mutant(!held, differs);
        println!("  mutant{words}");
        let verdict = gate::verdict(&format!("{label} patch"), same && caught);
        println!("  the GPU's patch is the CPU's {verdict}");
        all &= same && caught;
    }
    if all {
        Ok(())
    } else {
        Err("a patch nested in its whole run differs on the GPU".into())
    }
}
