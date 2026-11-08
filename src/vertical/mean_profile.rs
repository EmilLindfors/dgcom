//! A domain's mean temperature and salinity profile.
//!
//! [`MeanProfile`] is a horizontally uniform stratification on a 1 m grid in
//! `z`: the reference for the pressure gradient (Mellor et al. 1998), the
//! tracer limiter and the open faces, and the water below a parent model's
//! bed, in a 3D run started from a parent's columns.
//!
//! At each height it is the mean of the columns that reach it, interpolated
//! between their layer centres, then made statically stable: adjacent
//! heights where the density decreases downwards are pooled into their
//! weighted mean (pool-adjacent-violators on `density(T, S)`; for a density
//! linear in T and S a pool's density is its mean's).
//!
//! # Why
//!
//! The columns that reach a height change with depth (below 100 m only the
//! deep holes), and their mean alone was inverted in places: as a reference,
//! and as the state, its overturning drove a bed jet of 0.26 m/s within ten
//! minutes in a 120 m hole at Mausund (TODO P1.3). Taking every column at
//! every height instead (extended below its bed) is stable but puts the
//! shallow columns' bottom water in the deep holes: 10.1 °C at 100 m where
//! the holes have 8.3 °C.

use crate::solver::state::Solution3D;
use crate::vertical::SigmaGrid;

/// A horizontally uniform T, S profile (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct MeanProfile {
    /// Height of the lowest grid point (m)
    z0: f64,
    temp: Vec<f64>,
    salt: Vec<f64>,
}

impl MeanProfile {
    /// The stable mean of the columns of `state` at the nodes `use_column`
    /// picks, each at rest at mean sea level over its bed `bed[node]` (only
    /// wet nodes, `bed < 0`, count). `None` if no column is picked.
    pub fn of(
        state: &Solution3D,
        bed: &[f64],
        sigma: &SigmaGrid,
        use_column: impl Fn(usize) -> bool,
        density: impl Fn(f64, f64) -> f64,
    ) -> Option<Self> {
        let nl = state.n_levels;
        let deepest = bed.iter().copied().fold(0.0, f64::min);
        let n = (-deepest).ceil() as usize + 1;
        let z0 = -((n - 1) as f64);
        let (mut sum_t, mut sum_s, mut count) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for (idx, &b) in bed.iter().enumerate() {
            if b >= 0.0 || !use_column(idx) {
                continue;
            }
            // The layer centres at mean sea level, from the bed up
            let z = |l: usize| sigma.sigma_rho()[l] * -b;
            let (t, sal) = (
                &state.temp[idx * nl..(idx + 1) * nl],
                &state.salt[idx * nl..(idx + 1) * nl],
            );
            let mut l = 0;
            for i in 0..n {
                let zi = z0 + i as f64;
                // Within the column: its bed to the surface
                if zi < b {
                    continue;
                }
                while l + 1 < nl && z(l + 1) <= zi {
                    l += 1;
                }
                let (value_t, value_s) = if zi <= z(0) {
                    (t[0], sal[0])
                } else if l + 1 >= nl {
                    (t[nl - 1], sal[nl - 1])
                } else {
                    let w = (zi - z(l)) / (z(l + 1) - z(l));
                    (
                        t[l] + w * (t[l + 1] - t[l]),
                        sal[l] + w * (sal[l + 1] - sal[l]),
                    )
                };
                sum_t[i] += value_t;
                sum_s[i] += value_s;
                count[i] += 1.0;
            }
        }
        if count.iter().all(|&c| c == 0.0) {
            return None;
        }
        // Pools from the surface down: (Σ T, Σ S, weight, heights)
        let mut pools: Vec<(f64, f64, f64, usize)> = Vec::new();
        for i in (0..n).rev() {
            if count[i] == 0.0 {
                // Below every column: the pool above extends
                if let Some(last) = pools.last_mut() {
                    last.3 += 1;
                }
                continue;
            }
            pools.push((sum_t[i], sum_s[i], count[i], 1));
            while pools.len() >= 2 {
                let rho = |p: &(f64, f64, f64, usize)| density(p.0 / p.2, p.1 / p.2);
                let (lower, upper) = (pools[pools.len() - 1], pools[pools.len() - 2]);
                if rho(&lower) >= rho(&upper) {
                    break;
                }
                pools.pop();
                let merged = pools.last_mut().expect("two pools");
                *merged = (
                    merged.0 + lower.0,
                    merged.1 + lower.1,
                    merged.2 + lower.2,
                    merged.3 + lower.3,
                );
            }
        }
        // Back to heights, from the bottom up
        let (mut temp, mut salt) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for &(t, s, w, len) in pools.iter().rev() {
            temp.extend(std::iter::repeat_n(t / w, len));
            salt.extend(std::iter::repeat_n(s / w, len));
        }
        // Heights above the top layer centre of every column take the top
        let (t_top, s_top) = (*temp.last()?, *salt.last()?);
        temp.resize(n, t_top);
        salt.resize(n, s_top);
        Some(Self { z0, temp, salt })
    }

    /// T and S at height `z` (m), linear between grid points and held beyond
    /// them.
    pub fn at(&self, z: f64) -> (f64, f64) {
        let x = (z - self.z0).clamp(0.0, (self.temp.len() - 1) as f64);
        let i = (x.floor() as usize).min(self.temp.len().saturating_sub(2));
        let w = (x - i as f64).clamp(0.0, 1.0);
        let j = (i + 1).min(self.temp.len() - 1);
        (
            self.temp[i] + w * (self.temp[j] - self.temp[i]),
            self.salt[i] + w * (self.salt[j] - self.salt[i]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertical::UniformStretching;

    /// Two columns, 10 m and 20 m deep, of `temp(z)` and `salt(z)` at their
    /// layer centres at rest.
    fn columns(
        temp: impl Fn(f64) -> f64,
        salt: impl Fn(f64) -> f64,
    ) -> (Solution3D, Vec<f64>, SigmaGrid) {
        let sigma = SigmaGrid::new(10, UniformStretching);
        let bed = vec![-10.0, -20.0];
        let mut state = Solution3D::new(1, 2, 10);
        for (idx, &b) in bed.iter().enumerate() {
            for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                let z = s * -b;
                state.temp[idx * 10 + l] = temp(z);
                state.salt[idx * 10 + l] = salt(z);
            }
        }
        (state, bed, sigma)
    }

    /// A stable linear stratification is its own mean wherever every column
    /// that reaches a height has layer centres around it (between −1 and
    /// −9 m both columns, between −10 and −19 m the deep one); each column
    /// holds its end layers' values beyond them, and the profile is held
    /// above the surface and below the deepest bed.
    #[test]
    fn a_stable_linear_profile_is_its_own_mean() {
        let (state, bed, sigma) = columns(|z| 10.0 + 0.1 * z, |z| 33.0 - 0.05 * z);
        let density = |t: f64, s: f64| 1025.0 - 0.2 * (t - 10.0) + 0.8 * (s - 35.0);
        let mean = MeanProfile::of(&state, &bed, &sigma, |_| true, density).unwrap();
        for z in [-2.0, -3.5, -6.0, -12.0, -15.5] {
            let (t, s) = mean.at(z);
            assert!((t - (10.0 + 0.1 * z)).abs() < 1e-12, "T at {z}: {t}");
            assert!((s - (33.0 - 0.05 * z)).abs() < 1e-12, "S at {z}: {s}");
        }
        assert_eq!(mean.at(1.0), mean.at(0.0));
        assert_eq!(mean.at(-30.0), mean.at(-20.0));
    }

    /// An inverted profile (denser on top) is pooled into one stable layer
    /// of its mean.
    #[test]
    fn an_inversion_is_pooled_into_its_mean() {
        let (state, bed, sigma) = columns(|z| 10.0 - 0.1 * z, |_| 35.0);
        let density = |t: f64, _s: f64| 1025.0 - 0.2 * (t - 10.0);
        let mean = MeanProfile::of(&state, &bed, &sigma, |_| true, density).unwrap();
        let (top, bottom) = (mean.at(-1.0).0, mean.at(-19.0).0);
        assert!((top - bottom).abs() < 1e-12, "{top} against {bottom}");
        assert!(
            MeanProfile::of(&state, &bed, &sigma, |_| false, density).is_none(),
            "no columns, no profile"
        );
    }
}
