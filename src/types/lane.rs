//! A number of the pointwise physics: one `f64`, or one per SIMD lane.
//!
//! The 2D split-form kernel's pointwise physics (the two-point flux, the
//! hydrostatic HLL flux with its wave speeds and wall pressure, the subcells'
//! limited reconstruction) is written once, generic over [`Lane`]. The scalar
//! kernel and the GPU kernels (`gpu/`, `notes/gpu.md`) instantiate it with
//! `f64`, the batched kernel (`simd`) with `f64x8`, one element per lane.
//!
//! A branch of that physics is a select: both sides are evaluated and one is
//! kept per lane, with [`Lane::any`] to skip work no lane needs. So every
//! instantiation performs the same operations in the same order, and gives
//! the same bits.

use std::ops::{Add, BitAnd, BitOr, Div, Mul, Neg, Sub};

/// A value of the pointwise physics: `f64`, or `f64x8` (`simd` feature).
///
/// Arithmetic is IEEE in every lane, with no fused multiply-add, and `sqrt`
/// and division are correctly rounded: an `f64x8` lane and an `f64` give the
/// same result for the same operations.
pub trait Lane:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// The lanes where a comparison holds.
    type Mask: Copy + BitAnd<Output = Self::Mask> + BitOr<Output = Self::Mask>;

    /// `x` in every lane, of the same kind as `self` (its SIMD level).
    fn constant(self, x: f64) -> Self;

    /// Correctly rounded square root.
    fn sqrt(self) -> Self;

    /// Absolute value.
    fn abs(self) -> Self;

    /// `self < other`, lane by lane.
    fn less_than(self, other: Self) -> Self::Mask;

    /// `self <= other`, lane by lane.
    fn at_most(self, other: Self) -> Self::Mask;

    /// `self > other`, lane by lane.
    fn greater_than(self, other: Self) -> Self::Mask;

    /// `self >= other`, lane by lane.
    fn at_least(self, other: Self) -> Self::Mask;

    /// `if_true` in the lanes of `mask`, `if_false` in the others.
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    /// Whether any lane of `mask` holds.
    fn any(mask: Self::Mask) -> bool;

    /// A mask that holds in no lane, of the same kind as `self`.
    fn none(self) -> Self::Mask;

    /// `self` where `self > other`, else `other`: `f64::max` but for a NaN
    /// `other` and a tie of +0 with −0 (which give `other`).
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::select(self.greater_than(other), self, other)
    }

    /// `self` where `self < other`, else `other`: `f64::min` but for a NaN
    /// `other` and a tie of +0 with −0 (which give `other`).
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::select(self.less_than(other), self, other)
    }
}

impl Lane for f64 {
    type Mask = bool;

    #[inline(always)]
    fn constant(self, x: f64) -> Self {
        x
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline(always)]
    fn less_than(self, other: Self) -> bool {
        self < other
    }

    #[inline(always)]
    fn at_most(self, other: Self) -> bool {
        self <= other
    }

    #[inline(always)]
    fn greater_than(self, other: Self) -> bool {
        self > other
    }

    #[inline(always)]
    fn at_least(self, other: Self) -> bool {
        self >= other
    }

    #[inline(always)]
    fn select(mask: bool, if_true: Self, if_false: Self) -> Self {
        if mask { if_true } else { if_false }
    }

    #[inline(always)]
    fn any(mask: bool) -> bool {
        mask
    }

    #[inline(always)]
    fn none(self) -> bool {
        false
    }
}

#[cfg(feature = "simd")]
mod simd {
    use fearless_simd::{Simd, SimdMask, f64x8, mask64x8};

    use super::Lane;

    impl<S: Simd> Lane for f64x8<S> {
        type Mask = mask64x8<S>;

        #[inline(always)]
        fn constant(self, x: f64) -> Self {
            self.simd.splat_f64x8(x)
        }

        #[inline(always)]
        fn sqrt(self) -> Self {
            self.simd.sqrt_f64x8(self)
        }

        #[inline(always)]
        fn abs(self) -> Self {
            self.simd.abs_f64x8(self)
        }

        #[inline(always)]
        fn less_than(self, other: Self) -> mask64x8<S> {
            self.simd.simd_lt_f64x8(self, other)
        }

        #[inline(always)]
        fn at_most(self, other: Self) -> mask64x8<S> {
            self.simd.simd_le_f64x8(self, other)
        }

        #[inline(always)]
        fn greater_than(self, other: Self) -> mask64x8<S> {
            self.simd.simd_gt_f64x8(self, other)
        }

        #[inline(always)]
        fn at_least(self, other: Self) -> mask64x8<S> {
            self.simd.simd_ge_f64x8(self, other)
        }

        #[inline(always)]
        fn select(mask: mask64x8<S>, if_true: Self, if_false: Self) -> Self {
            if_true.simd.select_f64x8(mask, if_true, if_false)
        }

        #[inline(always)]
        fn any(mask: mask64x8<S>) -> bool {
            SimdMask::any_true(mask)
        }

        #[inline(always)]
        fn none(self) -> mask64x8<S> {
            self.simd.splat_mask64x8(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Lane;

    #[test]
    fn f64_max_and_min_keep_the_other_value_on_ties_and_nan() {
        assert_eq!(Lane::max(1.0, 2.0), 2.0);
        assert_eq!(Lane::min(1.0, 2.0), 1.0);
        // A NaN self gives the other value, as f64::max does
        assert_eq!(Lane::max(f64::NAN, 0.0), 0.0);
        assert_eq!(Lane::min(f64::NAN, 0.0), 0.0);
        // A tie of −0 with +0 gives the other: +0 here
        assert!(Lane::max(-0.0_f64, 0.0).is_sign_positive());
        assert_eq!(<f64 as Lane>::select(true, 1.0, 2.0), 1.0);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn f64x8_lanes_compute_what_f64_does() {
        use fearless_simd::{Level, dispatch, f64x8, prelude::*};
        let a = [1.5, -2.0, 0.0, 3.25, f64::NAN, 1e-300, -7.0, 4.0];
        let b = [2.5, -3.0, -0.0, 3.25, 0.0, 1e300, 7.0, -4.0];
        let level = Level::new();
        dispatch!(level, simd => {
            let (va, vb) = (f64x8::from_slice(simd, &a), f64x8::from_slice(simd, &b));
            let max = Lane::max(va, vb);
            let min = Lane::min(va, vb);
            let quotient = va / vb;
            let root = Lane::sqrt(Lane::abs(va));
            for i in 0..8 {
                assert_eq!(max.as_slice()[i].to_bits(), Lane::max(a[i], b[i]).to_bits());
                assert_eq!(min.as_slice()[i].to_bits(), Lane::min(a[i], b[i]).to_bits());
                assert_eq!(quotient.as_slice()[i].to_bits(), (a[i] / b[i]).to_bits());
                assert_eq!(root.as_slice()[i].to_bits(), a[i].abs().sqrt().to_bits());
            }
        });
    }
}
