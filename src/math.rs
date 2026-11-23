//! Elementary functions in plain IEEE arithmetic: the same bits on every
//! CPU and on the GPU.
//!
//! `f64::exp`, `ln`, `powf` and `hypot` call the platform's C library
//! (MSVC's on Windows, glibc's on Linux), which round differently from each
//! other and from CUDA's libdevice, to which the GPU kernels' calls go (and
//! cuda-oxide sends the `libm` crate's there too). These are FreeBSD's
//! (fdlibm's) algorithms as the `libm` crate ports them: additions,
//! multiplications, divisions, `sqrt` (all correctly rounded on both) and
//! bit manipulation, without fused multiply–adds. So the CPU and a GPU kernel
//! built without contraction (`--no-fmad`) give the same bits, on Windows
//! and Linux alike.
//!
//! The 3D model's column physics takes them (the GLS closure, the log-layer
//! bottom drag, the shortwave absorption, the friction velocities, the cage
//! speeds), which the GPU runs too (`notes/gpu.md`, Phase 4.4a). Each is
//! within 1 ulp of the exact result. They are slower than the platform's:
//! `pow` ≈ 3× MSVC's.
//!
//! The FreeBSD sources carry this notice:
//!
//! ```text
//! Copyright (C) 1993, 2004 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ```

// fdlibm's code as it writes it: its constants (more digits than an f64
// holds; some are `std`'s constants), its NaNs (`(x − x)/0`) and its
// declared-then-set temporaries, so that it reads as its sources do
#![allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::eq_op,
    clippy::needless_late_init
)]

#[inline]
fn high_word(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline]
fn with_high_word(x: f64, high: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0x0000_0000_ffff_ffff) | ((high as u64) << 32))
}

#[inline]
fn with_low_word(x: f64, low: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff_0000_0000) | low as u64)
}

/// `x · 2ⁿ` (musl's `scalbn`).
fn scalbn(x: f64, mut n: i32) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0_0000_0000_0000); // 2^1023
    let x1p53 = f64::from_bits(0x4340_0000_0000_0000); // 2^53
    let x1p_1022 = f64::from_bits(0x0010_0000_0000_0000); // 2^-1022
    let mut y = x;
    if n > 1023 {
        y *= x1p1023;
        n -= 1023;
        if n > 1023 {
            y *= x1p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        // A final n below −53 avoids double rounding in the subnormal range
        y *= x1p_1022 * x1p53;
        n += 1022 - 53;
        if n < -1022 {
            y *= x1p_1022 * x1p53;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

/// `eˣ` (FreeBSD's `e_exp.c`): `x = k ln 2 + r` with `|r| ≤ ½ ln 2`, `eʳ`
/// from a degree-5 Remez fit of `r (eʳ + 1)/(eʳ − 1)`, scaled by `2ᵏ`.
pub fn exp(mut x: f64) -> f64 {
    const HALF: [f64; 2] = [0.5, -0.5];
    const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
    const INV_LN2: f64 = 1.442_695_040_888_963_387_00;
    const P1: f64 = 1.666_666_666_666_660_190_37e-1;
    const P2: f64 = -2.777_777_777_701_559_338_42e-3;
    const P3: f64 = 6.613_756_321_437_934_361_17e-5;
    const P4: f64 = -1.653_390_220_546_525_153_90e-6;
    const P5: f64 = 4.138_136_797_057_238_460_39e-8;
    let x1p1023 = f64::from_bits(0x7fe0_0000_0000_0000);

    let mut hx = high_word(x);
    let sign = (hx >> 31) as usize;
    hx &= 0x7fff_ffff;

    // Overflow, underflow, NaN
    if hx >= 0x4086_232b {
        if x.is_nan() {
            return x;
        }
        if x > 709.782_712_893_383_973_096 {
            return x * x1p1023;
        }
        if x < -745.133_219_101_941_108_42 {
            return 0.0;
        }
    }

    // Argument reduction
    let (hi, lo, k);
    if hx > 0x3fd6_2e42 {
        // |x| > ½ ln 2
        k = if hx >= 0x3ff0_a2b2 {
            // |x| ≥ 1.5 ln 2
            (INV_LN2 * x + HALF[sign]) as i32
        } else {
            1 - sign as i32 - sign as i32
        };
        hi = x - k as f64 * LN2_HI; // k ln2_hi is exact here
        lo = k as f64 * LN2_LO;
        x = hi - lo;
    } else if hx > 0x3e30_0000 {
        // |x| > 2⁻²⁸
        k = 0;
        hi = x;
        lo = 0.0;
    } else {
        return 1.0 + x;
    }

    let xx = x * x;
    let c = x - xx * (P1 + xx * (P2 + xx * (P3 + xx * (P4 + xx * P5))));
    let y = 1.0 + (x * c / (2.0 - c) - lo + hi);
    if k == 0 { y } else { scalbn(y, k) }
}

/// The natural logarithm (FreeBSD's `e_log.c`): `x = 2ᵏ (1 + f)` with
/// `√2/2 < 1 + f < √2`, `ln(1 + f)` from `s = f/(2 + f)` and a degree-14
/// Remez fit.
pub fn ln(mut x: f64) -> f64 {
    const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
    const LG1: f64 = 6.666_666_666_666_735_130e-1;
    const LG2: f64 = 3.999_999_999_940_941_908e-1;
    const LG3: f64 = 2.857_142_874_366_239_149e-1;
    const LG4: f64 = 2.222_219_843_214_978_396e-1;
    const LG5: f64 = 1.818_357_216_161_805_012e-1;
    const LG6: f64 = 1.531_383_769_920_937_332e-1;
    const LG7: f64 = 1.479_819_860_511_658_591e-1;
    let x1p54 = f64::from_bits(0x4350_0000_0000_0000);

    let mut ui = x.to_bits();
    let mut hx = (ui >> 32) as u32;
    let mut k: i32 = 0;

    if hx < 0x0010_0000 || (hx >> 31) != 0 {
        if ui << 1 == 0 {
            return -1.0 / (x * x); // ln(±0) = −∞
        }
        if hx >> 31 != 0 {
            return (x - x) / 0.0; // ln(−x) = NaN
        }
        // Subnormal: scale up
        k -= 54;
        x *= x1p54;
        ui = x.to_bits();
        hx = (ui >> 32) as u32;
    } else if hx >= 0x7ff0_0000 {
        return x;
    } else if hx == 0x3ff0_0000 && ui << 32 == 0 {
        return 0.0;
    }

    // x into [√2/2, √2]
    hx += 0x3ff0_0000 - 0x3fe6_a09e;
    k += (hx >> 20) as i32 - 0x3ff;
    hx = (hx & 0x000f_ffff) + 0x3fe6_a09e;
    ui = ((hx as u64) << 32) | (ui & 0xffff_ffff);
    x = f64::from_bits(ui);

    let f = x - 1.0;
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    let dk = k as f64;
    s * (hfsq + r) + dk * LN2_LO - hfsq + f + dk * LN2_HI
}

/// `√(x² + y²)` without undue overflow or underflow (musl's `hypot`): the
/// squares in two parts each (Dekker's split), so the sum is exact enough
/// for the correctly rounded `sqrt`.
pub fn hypot(x: f64, y: f64) -> f64 {
    // x² as hi + lo, exactly
    fn square(x: f64) -> (f64, f64) {
        const SPLIT: f64 = 134_217_728.0 + 1.0; // 2^27 + 1
        let xc = x * SPLIT;
        let xh = x - xc + xc;
        let xl = x - xh;
        let hi = x * x;
        let lo = xh * xh - hi + 2.0 * xh * xl + xl * xl;
        (hi, lo)
    }
    let x1p700 = f64::from_bits(0x6bb0_0000_0000_0000);
    let x1p_700 = f64::from_bits(0x1430_0000_0000_0000);

    // |x| ≥ |y|
    let (mut ux, mut uy) = (x.to_bits() & (u64::MAX >> 1), y.to_bits() & (u64::MAX >> 1));
    if ux < uy {
        std::mem::swap(&mut ux, &mut uy);
    }
    let (ex, ey) = ((ux >> 52) as i64, (uy >> 52) as i64);
    let (mut x, mut y) = (f64::from_bits(ux), f64::from_bits(uy));
    // hypot(∞, NaN) = ∞
    if ey == 0x7ff {
        return y;
    }
    if ex == 0x7ff || uy == 0 {
        return x;
    }
    if ex - ey > 64 {
        return x + y;
    }
    // Scaled so that the split neither overflows nor underflows
    let mut z = 1.0;
    if ex > 0x3ff + 510 {
        z = x1p700;
        x *= x1p_700;
        y *= x1p_700;
    } else if ey < 0x3ff - 450 {
        z = x1p_700;
        x *= x1p700;
        y *= x1p700;
    }
    let (hx, lx) = square(x);
    let (hy, ly) = square(y);
    z * (ly + lx + hy + hx).sqrt()
}

/// `xʸ` (FreeBSD's `e_pow.c`): `log₂ x` in two parts, `y log₂ x = n + y'`
/// in simulated extra precision, `2ⁿ e^(y' ln 2)`. `pow(x, 0.5)` is `√x`,
/// `pow(x, 2)` is `x·x`, `pow(x, ±1)` is `x` or `1/x`, and integer powers of
/// integers are exact where representable.
pub fn pow(x: f64, y: f64) -> f64 {
    const BP: [f64; 2] = [1.0, 1.5];
    const DP_H: [f64; 2] = [0.0, 5.849_624_872_207_641_601_56e-1];
    const DP_L: [f64; 2] = [0.0, 1.350_039_202_129_748_971_28e-8];
    const TWO53: f64 = 9_007_199_254_740_992.0;
    const HUGE: f64 = 1.0e300;
    const TINY: f64 = 1.0e-300;
    const L1: f64 = 5.999_999_999_999_946_487_25e-1;
    const L2: f64 = 4.285_714_285_785_501_842_52e-1;
    const L3: f64 = 3.333_333_298_183_774_329_18e-1;
    const L4: f64 = 2.727_281_238_085_340_064_89e-1;
    const L5: f64 = 2.306_607_457_755_617_540_67e-1;
    const L6: f64 = 2.069_750_178_003_384_177_84e-1;
    const P1: f64 = 1.666_666_666_666_660_190_37e-1;
    const P2: f64 = -2.777_777_777_701_559_338_42e-3;
    const P3: f64 = 6.613_756_321_437_934_361_17e-5;
    const P4: f64 = -1.653_390_220_546_525_153_90e-6;
    const P5: f64 = 4.138_136_797_057_238_460_39e-8;
    const LG2: f64 = 6.931_471_805_599_452_862_27e-1;
    const LG2_H: f64 = 6.931_471_824_645_996_093_75e-1;
    const LG2_L: f64 = -1.904_654_299_957_768_045_25e-9;
    const OVT: f64 = 8.008_566_259_537_294_437_2e-17;
    const CP: f64 = 9.617_966_939_259_755_543_29e-1;
    const CP_H: f64 = 9.617_967_009_544_372_558_59e-1;
    const CP_L: f64 = -7.028_461_650_952_758_265_16e-9;
    const IVLN2: f64 = 1.442_695_040_888_963_387_00;
    const IVLN2_H: f64 = 1.442_695_021_629_333_496_09;
    const IVLN2_L: f64 = 1.925_962_991_126_617_468_87e-8;

    let t1: f64;
    let t2: f64;
    let (hx, lx) = ((x.to_bits() >> 32) as i32, x.to_bits() as u32);
    let (hy, ly) = ((y.to_bits() >> 32) as i32, y.to_bits() as u32);
    let mut ix = hx & 0x7fff_ffff;
    let iy = hy & 0x7fff_ffff;

    // x⁰ = 1, even for NaN
    if ((iy as u32) | ly) == 0 {
        return 1.0;
    }
    // 1ʸ = 1, even for NaN
    if hx == 0x3ff0_0000 && lx == 0 {
        return 1.0;
    }
    // NaN if either is
    if ix > 0x7ff0_0000
        || (ix == 0x7ff0_0000 && lx != 0)
        || iy > 0x7ff0_0000
        || (iy == 0x7ff0_0000 && ly != 0)
    {
        return x + y;
    }

    // y an odd integer (1), an even one (2), or not an integer (0), for x < 0
    let mut yisint: i32 = 0;
    let mut k: i32;
    let mut j: i32;
    if hx < 0 {
        if iy >= 0x4340_0000 {
            yisint = 2;
        } else if iy >= 0x3ff0_0000 {
            k = (iy >> 20) - 0x3ff;
            if k > 20 {
                j = (ly >> (52 - k)) as i32;
                if (j << (52 - k)) == (ly as i32) {
                    yisint = 2 - (j & 1);
                }
            } else if ly == 0 {
                j = iy >> (20 - k);
                if (j << (20 - k)) == iy {
                    yisint = 2 - (j & 1);
                }
            }
        }
    }

    if ly == 0 {
        // y = ±∞
        if iy == 0x7ff0_0000 {
            return if ((ix - 0x3ff0_0000) | (lx as i32)) == 0 {
                1.0 // (−1)^±∞
            } else if ix >= 0x3ff0_0000 {
                if hy >= 0 { y } else { 0.0 }
            } else if hy >= 0 {
                0.0
            } else {
                -y
            };
        }
        // y = ±1
        if iy == 0x3ff0_0000 {
            return if hy >= 0 { x } else { 1.0 / x };
        }
        // y = 2
        if hy == 0x4000_0000 {
            return x * x;
        }
        // y = ½, x ≥ +0
        if hy == 0x3fe0_0000 && hx >= 0 {
            return x.sqrt();
        }
    }

    let mut ax = x.abs();
    if lx == 0 && (ix == 0x7ff0_0000 || ix == 0 || ix == 0x3ff0_0000) {
        // x = ±0, ±∞, ±1
        let mut z = ax;
        if hy < 0 {
            z = 1.0 / z;
        }
        if hx < 0 {
            if ((ix - 0x3ff0_0000) | yisint) == 0 {
                z = (z - z) / (z - z); // (−1)^non-integer is NaN
            } else if yisint == 1 {
                z = -z;
            }
        }
        return z;
    }

    let mut s = 1.0; // the result's sign
    if hx < 0 {
        if yisint == 0 {
            return (x - x) / (x - x); // (x < 0)^non-integer is NaN
        }
        if yisint == 1 {
            s = -1.0;
        }
    }

    if iy > 0x41e0_0000 {
        // |y| > 2³¹
        if iy > 0x43f0_0000 {
            // |y| > 2⁶⁴: over- or underflow
            if ix <= 0x3fef_ffff {
                return if hy < 0 { HUGE * HUGE } else { TINY * TINY };
            }
            if ix >= 0x3ff0_0000 {
                return if hy > 0 { HUGE * HUGE } else { TINY * TINY };
            }
        }
        // Over- or underflow if x is not close to 1
        if ix < 0x3fef_ffff {
            return if hy < 0 {
                s * HUGE * HUGE
            } else {
                s * TINY * TINY
            };
        }
        if ix > 0x3ff0_0000 {
            return if hy > 0 {
                s * HUGE * HUGE
            } else {
                s * TINY * TINY
            };
        }
        // |1 − x| ≤ 2⁻²⁰: ln x = t − t²/2 + t³/3 − t⁴/4
        let t = ax - 1.0;
        let w = (t * t) * (0.5 - t * (0.333_333_333_333_333_333_333_3 - t * 0.25));
        let u = IVLN2_H * t;
        let v = t * IVLN2_L - w * IVLN2;
        t1 = with_low_word(u + v, 0);
        t2 = v - (t1 - u);
    } else {
        let mut n: i32 = 0;
        if ix < 0x0010_0000 {
            // Subnormal x
            ax *= TWO53;
            n -= 53;
            ix = high_word(ax) as i32;
        }
        n += (ix >> 20) - 0x3ff;
        j = ix & 0x000f_ffff;
        // The interval
        let k: usize;
        ix = j | 0x3ff0_0000;
        if j <= 0x3988e {
            k = 0; // |x| < √(3/2)
        } else if j < 0xbb67a {
            k = 1; // |x| < √3
        } else {
            k = 0;
            n += 1;
            ix -= 0x0010_0000;
        }
        ax = with_high_word(ax, ix as u32);

        // ss = s_h + s_l = (x − 1)/(x + 1) or (x − 1.5)/(x + 1.5)
        let u = ax - BP[k];
        let v = 1.0 / (ax + BP[k]);
        let ss = u * v;
        let s_h = with_low_word(ss, 0);
        // t_h = ax + bp[k], high part
        let t_h = with_high_word(
            0.0,
            (((ix as u32) >> 1) | 0x2000_0000) + 0x0008_0000 + ((k as u32) << 18),
        );
        let t_l = ax - (t_h - BP[k]);
        let s_l = v * ((u - s_h * t_h) - s_h * t_l);

        // ln(ax)
        let s2 = ss * ss;
        let mut r = s2 * s2 * (L1 + s2 * (L2 + s2 * (L3 + s2 * (L4 + s2 * (L5 + s2 * L6)))));
        r += s_l * (s_h + ss);
        let s2 = s_h * s_h;
        let t_h = with_low_word(3.0 + s2 + r, 0);
        let t_l = r - ((t_h - 3.0) - s2);
        // u + v = ss (1 + …)
        let u = s_h * t_h;
        let v = s_l * t_h + t_l * ss;
        // 2/(3 ln 2) (ss + …)
        let p_h = with_low_word(u + v, 0);
        let p_l = v - (p_h - u);
        let z_h = CP_H * p_h;
        let z_l = CP_L * p_h + p_l * CP + DP_L[k];
        // log₂(ax) = n + dp_h + z_h + z_l
        let t = n as f64;
        t1 = with_low_word(((z_h + z_l) + DP_H[k]) + t, 0);
        t2 = z_l - (((t1 - t) - DP_H[k]) - z_h);
    }

    // (y1 + y2)(t1 + t2)
    let y1 = with_low_word(y, 0);
    let p_l = (y - y1) * t1 + y * t2;
    let mut p_h = y1 * t1;
    let z = p_l + p_h;
    let mut j = (z.to_bits() >> 32) as i32;
    let i = z.to_bits() as i32;
    if j >= 0x4090_0000 {
        // z ≥ 1024
        if ((j - 0x4090_0000) | i) != 0 || p_l + OVT > z - p_h {
            return s * HUGE * HUGE;
        }
    } else if (j & 0x7fff_ffff) >= 0x4090_cc00 {
        // z ≤ −1075
        if (((j as u32).wrapping_sub(0xc090_cc00)) | (i as u32)) != 0 || p_l <= z - p_h {
            return s * TINY * TINY;
        }
    }

    // 2^(p_h + p_l)
    let i = j & 0x7fff_ffff;
    k = (i >> 20) - 0x3ff;
    let mut n: i32 = 0;
    if i > 0x3fe0_0000 {
        // |z| > ½: n = [z + ½]
        n = j + (0x0010_0000 >> (k + 1));
        k = ((n & 0x7fff_ffff) >> 20) - 0x3ff;
        let t = with_high_word(0.0, (n & !(0x000f_ffff >> k)) as u32);
        n = ((n & 0x000f_ffff) | 0x0010_0000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let t = with_low_word(p_l + p_h, 0);
    let u = t * LG2_H;
    let v = (p_l - (t - p_h)) * LG2 + t * LG2_L;
    let mut z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let t1 = z - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let r = (z * t1) / (t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    j = high_word(z) as i32;
    j += n << 20;
    if (j >> 20) <= 0 {
        z = scalbn(z, n); // subnormal result
    } else {
        z = with_high_word(z, j as u32);
    }
    s * z
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in ulps between `a` and `b` (finite, same sign).
    fn ulps(a: f64, b: f64) -> u64 {
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
    }

    /// Inputs spread over the ranges the column physics takes and beyond.
    fn samples(n: usize, low: f64, high: f64) -> impl Iterator<Item = f64> {
        (0..n).map(move |i| low + (high - low) * ((i as f64 * 0.618_033_988_749_895) % 1.0))
    }

    /// Each within 1 ulp of the platform's (itself within 1 ulp of the
    /// exact result), over decades of arguments.
    #[test]
    fn within_an_ulp_of_the_platform() {
        for x in samples(20_000, -700.0, 700.0) {
            assert!(
                ulps(exp(x), x.exp()) <= 1,
                "exp({x}) {} {}",
                exp(x),
                x.exp()
            );
        }
        for e in samples(20_000, -300.0, 300.0) {
            let x = 10f64.powf(e);
            assert!(ulps(ln(x), x.ln()) <= 1, "ln({x}) {} {}", ln(x), x.ln());
        }
        for (e, y) in samples(20_000, -12.0, 8.0).zip(samples(20_000, -4.0, 4.0).skip(7)) {
            let x = 10f64.powf(e);
            let (a, b) = (pow(x, y), x.powf(y));
            assert!(ulps(a, b) <= 1, "pow({x}, {y}) {a} {b}");
        }
        for (a, b) in samples(20_000, -1e3, 1e3).zip(samples(20_000, -1e-2, 1e-2).skip(3)) {
            assert!(ulps(hypot(a, b), a.hypot(b)) <= 1, "hypot({a}, {b})");
        }
    }

    #[test]
    fn exact_cases_and_specials() {
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(pow(7.3, 1.0), 7.3);
        assert_eq!(pow(7.3, -1.0), 1.0 / 7.3);
        assert_eq!(pow(7.3, 2.0), 7.3 * 7.3);
        assert_eq!(pow(7.3, 0.5), 7.3f64.sqrt());
        assert_eq!(pow(3.0, 5.0), 243.0);
        assert_eq!(pow(2.0, -3.0), 0.125);
        assert_eq!(pow(-2.0, 3.0), -8.0);
        assert!(pow(-2.0, 0.5).is_nan());
        assert_eq!(pow(0.0, -1.5), f64::INFINITY);
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(hypot(-3.0, 0.0), 3.0);
        assert!(ulps(hypot(1e300, 1e300), 1e300f64.hypot(1e300)) <= 1);
        assert!(ulps(hypot(1e-300, 3e-301), 1e-300f64.hypot(3e-301)) <= 1);
        assert_eq!(exp(-800.0), 0.0);
        assert_eq!(exp(800.0), f64::INFINITY);
        assert_eq!(ln(0.0), f64::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        assert_eq!(ln(f64::INFINITY), f64::INFINITY);
        assert!((ln(5e-324) - 5e-324f64.ln()).abs() < 1e-12);
        assert_eq!(scalbn(1.0, -1074), 5e-324);
    }
}
