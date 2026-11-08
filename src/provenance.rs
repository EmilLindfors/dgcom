//! Which build made a run: the crate's version and, when it was built from a
//! git checkout, the commit and whether the sources differed from it
//! (`build.rs`).
//!
//! A case run writes [`describe`] into its `case.resolved.toml` and its
//! snapshot and NetCDF metadata, so an output names the code that made it.

/// The crate's version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The git commit the build was made from (12 hex digits), if it was made
/// from a git checkout
pub const COMMIT: Option<&str> = option_env!("DGCOM_GIT_COMMIT");

/// Whether the build's sources (`src`, `Cargo.toml`, `Cargo.lock`,
/// `build.rs`) differed from [`COMMIT`]; `None` without git
pub const MODIFIED: Option<&str> = option_env!("DGCOM_GIT_MODIFIED");

/// The build in one line: `dgcom 0.1.0 (commit 45388bf4e2a1)`, with
/// `, modified` when the sources differed from the commit, or
/// `dgcom 0.1.0` without git.
pub fn describe() -> String {
    match COMMIT {
        Some(commit) => {
            let modified = if MODIFIED == Some("true") {
                ", modified"
            } else {
                ""
            };
            format!("dgcom {VERSION} (commit {commit}{modified})")
        }
        None => format!("dgcom {VERSION}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_names_its_version_and_commit() {
        let line = describe();
        assert!(line.starts_with(&format!("dgcom {VERSION}")), "{line}");
        if let Some(commit) = COMMIT {
            assert_eq!(commit.len(), 12);
            assert!(commit.chars().all(|c| c.is_ascii_hexdigit()), "{commit}");
            assert!(line.contains(commit), "{line}");
            assert!(matches!(MODIFIED, Some("true" | "false")));
        }
    }
}
