//! Time series of gridded 3D fields (profiles), and their sampling at the
//! depths of another model's layers.
//!
//! [`ProfileSeries`] stores one profile per grid point and snapshot, strided
//! `[time][point][level]` in `f32` (missing values, below the bed or on land,
//! are NaN), with the levels ordered from the surface down. Where the levels
//! lie ([`ProfileLevels`]):
//!
//! - **z-levels** (NorKyst-800's `*_zdepth` files): fixed depths below the
//!   surface, the same at every point;
//! - **s-levels** (ROMS history files): a fixed fraction of the water column
//!   at every point. With either ROMS `Vtransform` the depth below the free
//!   surface of a level is `(ζ − z)/(ζ + h) = −S/h` (1) or `−S` (2) of the
//!   column, independent of `ζ`.
//!
//! [`ProfileSeries::sample`] interpolates each column linearly between its
//! valid levels, holding the shallowest value up to the surface and the
//! deepest valid value down to any depth below it (NorKyst's z-levels end
//! at 300 m in the files, and a parent's bed need not match the child's), then
//! combines the columns and snapshots with the given weights. Its arithmetic
//! at each depth is [`profile_sample_at`], over any [`Profiles`]: a device's
//! copy of the points it samples runs the same (P2.7).

use crate::io::{Stencil, TimeStencil};

/// Where the levels of a [`ProfileSeries`] lie, from the surface down.
#[derive(Clone, Debug, PartialEq)]
pub enum ProfileLevels {
    /// Depths below the surface (m, positive down, increasing), the same at
    /// every point.
    Depth(Vec<f64>),
    /// Fractions of the water column below the surface (0 at the surface, 1
    /// at the bed, increasing), per point: `[point][level]`, NaN on land.
    Fraction(Vec<f64>),
}

/// A 3D field per snapshot, strided `[time][point][level]`, levels from the
/// surface down; NaN marks missing values (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileSeries {
    n_points: usize,
    n_levels: usize,
    levels: ProfileLevels,
    data: Vec<f32>,
}

impl ProfileSeries {
    /// Profiles at `n_points` points on `levels`, from values strided
    /// `[time][point][level]`.
    pub fn new(n_points: usize, levels: ProfileLevels, data: Vec<f32>) -> Self {
        let n_levels = match &levels {
            ProfileLevels::Depth(d) => {
                assert!(
                    d.windows(2).all(|w| w[1] > w[0]),
                    "ProfileSeries: depths must increase from the surface down"
                );
                d.len()
            }
            ProfileLevels::Fraction(f) => {
                assert!(
                    n_points > 0 && f.len().is_multiple_of(n_points),
                    "ProfileSeries: {} fractions for {n_points} points",
                    f.len()
                );
                f.len() / n_points
            }
        };
        assert!(
            n_levels > 0 && data.len().is_multiple_of(n_points * n_levels),
            "ProfileSeries: {} values for {n_points} points × {n_levels} levels per snapshot",
            data.len()
        );
        Self {
            n_points,
            n_levels,
            levels,
            data,
        }
    }

    /// Number of grid points.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Number of levels.
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Number of snapshots.
    pub fn n_times(&self) -> usize {
        self.data.len() / (self.n_points * self.n_levels)
    }

    /// Where the levels lie.
    pub fn levels(&self) -> &ProfileLevels {
        &self.levels
    }

    /// The profile of point `k` at snapshot `t`, from the surface down.
    pub fn column(&self, t: usize, k: usize) -> &[f32] {
        let start = (t * self.n_points + k) * self.n_levels;
        &self.data[start..start + self.n_levels]
    }

    /// The value of point `k`'s profile at snapshot `t` at `depth` (m below
    /// the surface) in a water column `column_depth` deep, interpolated
    /// linearly between the valid levels and held constant beyond them;
    /// `None` without any valid level ([`profile_value_at`]).
    pub fn value_at(&self, t: usize, k: usize, depth: f64, column_depth: f64) -> Option<f64> {
        profile_value_at(self, t, k, depth, column_depth)
    }

    /// The profile at the depths `depths` (m below the surface) of a water
    /// column `column_depth` deep, from the snapshots of `time` and the points
    /// of `space` into `out` ([`profile_sample_at`] at each depth). Points
    /// without any valid level are left out and the remaining weights
    /// renormalised; `false` (and `out` untouched) if none is left.
    pub fn sample(
        &self,
        time: &TimeStencil,
        space: &Stencil,
        depths: &[f64],
        column_depth: f64,
        out: &mut [f64],
    ) -> bool {
        debug_assert_eq!(depths.len(), out.len());
        if !profile_covers(self, time, &space.idx, &space.w) {
            return false;
        }
        for (o, &depth) in out.iter_mut().zip(depths) {
            *o = profile_sample_at(self, time, &space.idx, &space.w, depth, column_depth);
        }
        true
    }
}

/// Profiles as the sampling reads them ([`profile_value_at`],
/// [`profile_sample_at`]): a [`ProfileSeries`], or a device's copy of the
/// points it samples (P2.7), so that both run the same arithmetic.
pub trait Profiles {
    /// Levels per profile.
    fn n_levels(&self) -> usize;
    /// Value of point `k`'s level `l` at snapshot `t` (NaN where missing).
    fn value(&self, t: usize, k: usize, l: usize) -> f32;
    /// Where point `k`'s level `l` lies: a depth (m) or, with
    /// [`Self::fractions`], a fraction of the water column.
    fn position(&self, k: usize, l: usize) -> f64;
    /// Whether the levels are fractions of the water column (s-levels).
    fn fractions(&self) -> bool;
}

impl Profiles for ProfileSeries {
    #[inline]
    fn n_levels(&self) -> usize {
        self.n_levels
    }

    #[inline]
    fn value(&self, t: usize, k: usize, l: usize) -> f32 {
        self.data[(t * self.n_points + k) * self.n_levels + l]
    }

    #[inline]
    fn position(&self, k: usize, l: usize) -> f64 {
        match &self.levels {
            ProfileLevels::Depth(d) => d[l],
            ProfileLevels::Fraction(f) => f[k * self.n_levels + l],
        }
    }

    #[inline]
    fn fractions(&self) -> bool {
        matches!(self.levels, ProfileLevels::Fraction(_))
    }
}

/// The value of point `k`'s profile at snapshot `t` at `depth` (m below the
/// surface) in a water column `column_depth` deep, interpolated linearly
/// between the valid levels and held constant beyond them; `None` without
/// any valid level.
#[inline]
pub fn profile_value_at<P: Profiles + ?Sized>(
    profiles: &P,
    t: usize,
    k: usize,
    depth: f64,
    column_depth: f64,
) -> Option<f64> {
    let target = if profiles.fractions() {
        depth / column_depth.max(f64::MIN_POSITIVE)
    } else {
        depth
    };
    let mut above: Option<(f64, f64)> = None;
    for l in 0..profiles.n_levels() {
        let (z, value) = (profiles.position(k, l), profiles.value(t, k, l) as f64);
        if !value.is_finite() || !z.is_finite() {
            continue;
        }
        if z >= target {
            return Some(match above {
                Some((z0, v0)) if z > z0 => v0 + (value - v0) * (target - z0) / (z - z0),
                _ => value,
            });
        }
        above = Some((z, value));
    }
    above.map(|(_, v)| v)
}

/// Whether point `k` has a valid level at snapshot `t`.
#[inline]
fn profile_has_level<P: Profiles + ?Sized>(profiles: &P, t: usize, k: usize) -> bool {
    (0..profiles.n_levels()).any(|l| profiles.value(t, k, l).is_finite())
}

/// Whether the points `idx` of weights `w` leave a point to sample at the
/// snapshots of `time`: one of weight other than zero with a valid level at
/// the stencil's first snapshot ([`profile_sample_at`]'s points).
#[inline]
pub fn profile_covers<P: Profiles + ?Sized>(
    profiles: &P,
    time: &TimeStencil,
    idx: &[u32],
    w: &[f64],
) -> bool {
    (0..idx.len()).any(|i| w[i] != 0.0 && profile_has_level(profiles, time.idx[0], idx[i] as usize))
}

/// The profile at `depth` (m below the surface) of a water column
/// `column_depth` deep, from the snapshots of `time` and the points `idx`
/// with weights `w`: points of weight zero or without a valid level at the
/// stencil's first snapshot are left out and the remaining weights
/// renormalised. NaN if none is left ([`profile_covers`]), or if a point
/// valid at the first snapshot has no valid level at another.
#[inline]
pub fn profile_sample_at<P: Profiles + ?Sized>(
    profiles: &P,
    time: &TimeStencil,
    idx: &[u32],
    w: &[f64],
    depth: f64,
    column_depth: f64,
) -> f64 {
    let mut total = 0.0;
    let mut sum = 0.0;
    let mut first = true;
    for i in 0..idx.len() {
        let (k, wk) = (idx[i] as usize, w[i]);
        if wk == 0.0 || !profile_has_level(profiles, time.idx[0], k) {
            continue;
        }
        total += wk;
        let value: f64 = (0..time.len)
            .map(|j| {
                time.w[j]
                    * profile_value_at(profiles, time.idx[j], k, depth, column_depth)
                        .unwrap_or(f64::NAN)
            })
            .sum();
        if first {
            sum = wk * value;
        } else {
            sum += wk * value;
        }
        first = false;
    }
    if first { f64::NAN } else { sum / total }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single() -> TimeStencil {
        TimeStencil {
            idx: [0, 0, 0, 0],
            w: [1.0, 0.0, 0.0, 0.0],
            len: 1,
        }
    }

    /// z-levels: linear between valid levels, the shallowest value up to the
    /// surface, the deepest valid one below it (a column cut by its bed).
    #[test]
    fn z_levels_are_interpolated_and_held_beyond_the_valid_range() {
        let levels = ProfileLevels::Depth(vec![5.0, 10.0, 50.0]);
        let data = vec![1.0, 0.5, 0.0, 2.0, 1.0, f32::NAN];
        let series = ProfileSeries::new(2, levels, data);
        assert_eq!(series.value_at(0, 0, 0.0, 60.0), Some(1.0));
        assert_eq!(series.value_at(0, 0, 7.5, 60.0), Some(0.75));
        assert_eq!(series.value_at(0, 0, 30.0, 60.0), Some(0.25));
        assert_eq!(series.value_at(0, 0, 80.0, 60.0), Some(0.0));
        assert_eq!(series.value_at(0, 1, 30.0, 60.0), Some(1.0));
    }

    /// s-levels: positions are fractions of the column, so the same profile
    /// stretches over any depth.
    #[test]
    fn s_levels_follow_the_column() {
        let levels = ProfileLevels::Fraction(vec![0.25, 0.75]);
        let series = ProfileSeries::new(1, levels, vec![10.0, 6.0]);
        for depth in [8.0, 100.0] {
            assert_eq!(series.value_at(0, 0, 0.5 * depth, depth), Some(8.0));
            assert_eq!(series.value_at(0, 0, 0.9 * depth, depth), Some(6.0));
        }
    }

    /// Columns and snapshots combine with their weights; a column with no
    /// valid level is left out and the rest renormalised.
    #[test]
    fn samples_combine_columns_and_snapshots() {
        let levels = ProfileLevels::Depth(vec![0.0, 10.0]);
        // Two snapshots × three points × two levels; point 2 is land
        let nan = f32::NAN;
        let data = vec![
            1.0, 3.0, 5.0, 7.0, nan, nan, // t = 0
            2.0, 4.0, 6.0, 8.0, nan, nan, // t = 1
        ];
        let series = ProfileSeries::new(3, levels, data);
        let time = TimeStencil {
            idx: [0, 1, 0, 0],
            w: [0.5, 0.5, 0.0, 0.0],
            len: 2,
        };
        let space = Stencil {
            idx: [0, 1, 2, 0],
            w: [0.25, 0.25, 0.5, 0.0],
        };
        let mut out = [0.0; 2];
        assert!(series.sample(&time, &space, &[0.0, 5.0], 10.0, &mut out));
        // Points 0 and 1 half each: surface (1.5 + 5.5)/2, 5 m (2.5 + 6.5)/2
        assert!((out[0] - 3.5).abs() < 1e-12 && (out[1] - 4.5).abs() < 1e-12);
        let land = Stencil::point(2);
        assert!(!series.sample(&single(), &land, &[0.0, 5.0], 10.0, &mut out));
    }
}
