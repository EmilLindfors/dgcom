//! The build's provenance: the git commit the crate is built from and
//! whether its sources differ from it, for `dgcom::provenance`.
//!
//! Without git (a packaged crate, no repository) neither is set and the
//! provenance is the version alone.

use std::path::Path;
use std::process::Command;

/// The sources whose changes make a build differ from its commit.
const SOURCES: &[&str] = &["src", "Cargo.toml", "Cargo.lock", "build.rs"];

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    for source in SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
    let Some(git_dir) = git(&["rev-parse", "--git-dir"]) else {
        return;
    };
    // A commit moves HEAD's ref (or, packed, packed-refs); staging writes the index
    let git_dir = Path::new(&git_dir);
    for file in ["HEAD", "index", "packed-refs"] {
        println!("cargo:rerun-if-changed={}", git_dir.join(file).display());
    }
    if let Some(head) = git(&["symbolic-ref", "-q", "HEAD"]) {
        println!("cargo:rerun-if-changed={}", git_dir.join(head).display());
    }
    let Some(commit) = git(&["rev-parse", "--short=12", "HEAD"]) else {
        return;
    };
    println!("cargo:rustc-env=DGCOM_GIT_COMMIT={commit}");
    let mut status = vec!["status", "--porcelain", "--"];
    status.extend(SOURCES);
    if let Some(changes) = git(&status) {
        let modified = if changes.is_empty() { "false" } else { "true" };
        println!("cargo:rustc-env=DGCOM_GIT_MODIFIED={modified}");
    }
}
