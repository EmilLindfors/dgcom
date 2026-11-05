//! Element-block loops of the 3D kernels, parallel with the `parallel`
//! feature, and the per-thread scratch they work in.
//!
//! The 3D kernels write each element's `[node][level]` block of their output
//! fields from shared inputs. [`for_each_block`] runs such a kernel over every
//! block, on rayon's pool with the `parallel` feature and in order without
//! it. Every block is computed by the same code from the same inputs either
//! way, so the result does not depend on the feature or the thread count, bit
//! for bit; reductions across blocks ([`max_over_blocks`]) use only
//! exact operations.
//!
//! A kernel's per-element buffers come from [`Pooled`], a cache of this
//! thread's scratch: rayon's workers persist, so after the first call nothing
//! is allocated.

use std::any::Any;
use std::cell::RefCell;

use super::disjoint::DisjointChunks;

/// Scratch values cached per thread (of any type; a handful at most).
const POOL_CAPACITY: usize = 16;

thread_local! {
    static POOL: RefCell<Vec<Box<dyn Any>>> = const { RefCell::new(Vec::new()) };
}

/// A scratch value taken from this thread's cache, returned to it on drop.
///
/// Take it once per loop or rayon job, not per element: a thread-local
/// lookup per element costs more than many kernels' work.
pub(crate) struct Pooled<T: 'static>(Option<Box<T>>);

impl<T: 'static> Pooled<T> {
    /// A cached `T` for which `fits` holds, or else `make()`.
    pub(crate) fn take(fits: impl Fn(&T) -> bool, make: impl FnOnce() -> T) -> Self {
        let cached = POOL
            .try_with(|pool| {
                let mut pool = pool.try_borrow_mut().ok()?;
                let i = pool
                    .iter()
                    .position(|v| v.downcast_ref::<T>().is_some_and(&fits))?;
                pool.swap_remove(i).downcast::<T>().ok()
            })
            .ok()
            .flatten();
        Self(Some(cached.unwrap_or_else(|| Box::new(make()))))
    }
}

impl<T: 'static> std::ops::Deref for Pooled<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0.as_ref().expect("present until drop")
    }
}

impl<T: 'static> std::ops::DerefMut for Pooled<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.as_mut().expect("present until drop")
    }
}

impl<T: 'static> Drop for Pooled<T> {
    fn drop(&mut self) {
        let Some(value) = self.0.take() else {
            return;
        };
        // `try_with`: the thread-local may already be gone at thread exit
        let _ = POOL.try_with(|pool| {
            if let Ok(mut pool) = pool.try_borrow_mut()
                && pool.len() < POOL_CAPACITY
            {
                pool.push(value);
            }
        });
    }
}

/// The blocks a loop visits: all `n` of them (`0..n`), or an increasing list
/// of distinct ones among `n` (an element subset of local time stepping).
/// A `usize` converts to all.
#[derive(Clone, Copy, Debug)]
pub enum Blocks<'a> {
    All(usize),
    List { n: usize, list: &'a [u32] },
}

impl<'a> Blocks<'a> {
    /// `list` among `n` blocks (increasing; checked in debug builds).
    pub fn list(n: usize, list: &'a [u32]) -> Self {
        debug_assert!(
            list.windows(2).all(|w| w[0] < w[1]),
            "blocks not increasing"
        );
        debug_assert!(
            list.last().is_none_or(|&k| (k as usize) < n),
            "block out of range"
        );
        Self::List { n, list }
    }

    /// Whether block `k` is visited.
    pub fn contains(&self, k: usize) -> bool {
        match *self {
            Self::All(n) => k < n,
            Self::List { list, .. } => list.binary_search(&(k as u32)).is_ok(),
        }
    }

    /// The number of blocks the outputs are split into.
    pub fn n(&self) -> usize {
        match *self {
            Self::All(n) | Self::List { n, .. } => n,
        }
    }

    /// The number of blocks visited.
    pub fn len(&self) -> usize {
        match *self {
            Self::All(n) => n,
            Self::List { list, .. } => list.len(),
        }
    }

    /// Whether no block is visited.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `i`-th block visited.
    #[inline]
    pub fn get(&self, i: usize) -> usize {
        match *self {
            Self::All(_) => i,
            Self::List { list, .. } => list[i] as usize,
        }
    }

    /// Whether every block is visited.
    pub fn is_all(&self) -> bool {
        matches!(self, Self::All(_))
    }

    /// The visited blocks, in order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }
}

impl From<usize> for Blocks<'_> {
    fn from(n: usize) -> Self {
        Self::All(n)
    }
}

/// An element list grown by the elements around it, for the first pass of
/// a kernel whose second pass on the list reads the first's results there.
/// Every element, if the list is.
#[derive(Debug, Default)]
pub(crate) struct Neighbourhood {
    list: Vec<u32>,
    mark: Vec<bool>,
}

impl Neighbourhood {
    /// `blocks` (elements of `mesh`) and their face neighbours, in order.
    pub(crate) fn faces<'a>(
        &'a mut self,
        mesh: &crate::mesh::Mesh2D,
        blocks: Blocks,
    ) -> Blocks<'a> {
        self.grow(mesh.n_elements, blocks, |k, add| {
            for face in 0..4 {
                if let Some(nb) = mesh.neighbor(crate::types::ElementIndex::new(k), face) {
                    add(nb.element);
                }
            }
        })
    }

    /// `blocks` and every element sharing a vertex with one, in order.
    pub(crate) fn vertex_patches<'a>(
        &'a mut self,
        mesh: &crate::mesh::Mesh2D,
        blocks: Blocks,
    ) -> Blocks<'a> {
        self.grow(mesh.n_elements, blocks, |k, add| {
            for vertex in mesh.element_vertex_indices(crate::types::ElementIndex::new(k)) {
                mesh.elements_at_vertex(vertex).iter().for_each(|&e| add(e));
            }
        })
    }

    fn grow<'a>(
        &'a mut self,
        n: usize,
        blocks: Blocks,
        around: impl Fn(usize, &mut dyn FnMut(usize)),
    ) -> Blocks<'a> {
        let Blocks::List { list, .. } = blocks else {
            return Blocks::All(n);
        };
        self.mark.resize(n, false);
        self.list.clear();
        let (out, mark) = (&mut self.list, &mut self.mark);
        let mut add = |e: usize| {
            if !std::mem::replace(&mut mark[e], true) {
                out.push(e as u32);
            }
        };
        for &k in list {
            add(k as usize);
            around(k as usize, &mut add);
        }
        for &e in out.iter() {
            mark[e as usize] = false;
        }
        out.sort_unstable();
        Blocks::List { n, list: out }
    }
}

/// Run `f(scratch, block, chunks)` for every `block` of `blocks` (all of
/// `0..n_blocks` for a `usize`), with `chunks[i]` the `block`-th of the `n`
/// equal chunks of `outputs[i]` (in parallel with the `parallel` feature).
/// `init` makes a worker's scratch, once per loop or rayon job.
pub(crate) fn for_each_block<'b, T, S, const N: usize>(
    n_blocks: impl Into<Blocks<'b>>,
    outputs: [&mut [T]; N],
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize, [&mut [T]; N]) + Sync + Send,
) where
    T: Send,
{
    max_over_blocks(n_blocks, outputs, init, |scratch, block, chunks| {
        f(scratch, block, chunks);
        f64::NEG_INFINITY
    });
}

/// [`for_each_block`] for a kernel that returns a value per block: their
/// largest (`−∞` for none; `max` is exact, so this too is independent of the
/// thread count).
pub(crate) fn max_over_blocks<'b, T, S, const N: usize>(
    n_blocks: impl Into<Blocks<'b>>,
    outputs: [&mut [T]; N],
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize, [&mut [T]; N]) -> f64 + Sync + Send,
) -> f64
where
    T: Send,
{
    reduce_blocks(n_blocks, outputs, init, f, || f64::NEG_INFINITY, f64::max)
}

/// [`for_each_block`] for a kernel that returns a value per block, combined
/// with `combine` from `identity()`. The grouping of the combinations
/// depends on the thread count: `combine` must be exact (and associative and
/// commutative) for the result not to, e.g. a maximum or an integer count.
pub(crate) fn reduce_blocks<'b, T, S, R, const N: usize>(
    blocks: impl Into<Blocks<'b>>,
    outputs: [&mut [T]; N],
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize, [&mut [T]; N]) -> R + Sync + Send,
    identity: impl Fn() -> R + Sync + Send,
    combine: impl Fn(R, R) -> R + Sync + Send,
) -> R
where
    T: Send,
    R: Send,
{
    let blocks = blocks.into();
    let n_blocks = blocks.n();
    let chunks = outputs.map(|out| {
        assert!(
            out.len().is_multiple_of(n_blocks.max(1)),
            "output of length {} in {n_blocks} blocks",
            out.len()
        );
        let chunk = out.len() / n_blocks.max(1);
        (chunk > 0).then(|| DisjointChunks::new(out, chunk))
    });
    // SAFETY: every block index is visited at most once (a range, or a list
    // of distinct blocks), so no chunk is handed out twice. An empty output
    // has no chunks: an empty slice
    let block_of = |block: usize| {
        std::array::from_fn(|i| match &chunks[i] {
            Some(c) => unsafe { c.chunk(block) },
            None => &mut [][..],
        })
    };
    let visit = |scratch: &mut S, i: usize| {
        let block = blocks.get(i);
        f(scratch, block, block_of(block))
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..blocks.len())
            .into_par_iter()
            .map_init(init, visit)
            .reduce(identity, combine)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = init();
        (0..blocks.len()).fold(identity(), |r, i| combine(r, visit(&mut scratch, i)))
    }
}

/// Values per job of the streaming loops ([`update_values`],
/// [`update_with`]): large enough that dispatch is negligible next to the
/// memory traffic.
const STREAM_CHUNK: usize = 1 << 14;

/// `f(x)` for every value of `out` (in parallel with the `parallel`
/// feature; elementwise, so the result is the serial one).
pub(crate) fn update_values(out: &mut [f64], f: impl Fn(&mut f64) + Sync + Send) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(STREAM_CHUNK)
            .for_each(|chunk| chunk.iter_mut().for_each(&f));
    }
    #[cfg(not(feature = "parallel"))]
    out.iter_mut().for_each(f);
}

/// `out ← a·base + Σᵢ cᵢ·xᵢ` (`base` is `out` itself when `None`), in one
/// streaming pass (as [`update_values`]), every value evaluated as
/// `((base·a) + c₁x₁) + c₂x₂ …` with `·a` left out for `a = 1`: the bits of
/// a `copy`, `scale`, `axpy` sequence ([`crate::time::Integrable::combine`]).
pub(crate) fn combine_values(
    out: &mut [f64],
    base: Option<&[f64]>,
    a: f64,
    terms: &[(f64, &[f64])],
) {
    if let Some(base) = base {
        assert_eq!(out.len(), base.len(), "lengths of the combined fields");
    }
    for (_, x) in terms {
        assert_eq!(out.len(), x.len(), "lengths of the combined fields");
    }
    let chunk = |offset: usize, out: &mut [f64]| {
        let range = offset..offset + out.len();
        if let Some(base) = base {
            out.copy_from_slice(&base[range.clone()]);
        }
        if a != 1.0 {
            out.iter_mut().for_each(|v| *v *= a);
        }
        // One sweep per term over a cache-sized chunk: the chunk stays in
        // cache, so the memory traffic is that of a single pass
        for &(c, x) in terms {
            for (v, &x) in out.iter_mut().zip(&x[range.clone()]) {
                *v += c * x;
            }
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(STREAM_CHUNK)
            .enumerate()
            .for_each(|(i, out)| chunk(i * STREAM_CHUNK, out));
    }
    #[cfg(not(feature = "parallel"))]
    for (i, out) in out.chunks_mut(STREAM_CHUNK).enumerate() {
        chunk(i * STREAM_CHUNK, out);
    }
}

/// `f(x, y)` for every value `x` of `out` and `y` of `input` at the same
/// index (as [`update_values`]).
pub(crate) fn update_with(out: &mut [f64], input: &[f64], f: impl Fn(&mut f64, f64) + Sync + Send) {
    assert_eq!(out.len(), input.len(), "lengths of the streamed fields");
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(STREAM_CHUNK)
            .zip(input.par_chunks(STREAM_CHUNK))
            .for_each(|(out, input)| {
                for (x, &y) in out.iter_mut().zip(input) {
                    f(x, y);
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for (x, &y) in out.iter_mut().zip(input) {
        f(x, y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_block_is_written_once_from_its_own_chunks() {
        let (mut a, mut b) = (vec![0.0; 12], vec![0.0; 6]);
        let largest = max_over_blocks(
            3,
            [&mut a[..], &mut b[..]],
            || 0usize,
            |calls, block, [a, b]| {
                *calls += 1;
                assert_eq!((a.len(), b.len()), (4, 2));
                a.fill(block as f64);
                b.fill(10.0 * block as f64);
                block as f64
            },
        );
        assert_eq!(largest, 2.0);
        assert_eq!(a, [0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2.]);
        assert_eq!(b, [0., 0., 10., 10., 20., 20.]);
    }

    #[test]
    fn pooled_scratch_is_reused_on_the_same_thread() {
        let first = Pooled::take(|v: &Vec<f64>| v.len() == 7, || vec![1.0; 7]);
        let address = first.as_ptr();
        drop(first);
        let again = Pooled::take(|v: &Vec<f64>| v.len() == 7, || vec![2.0; 7]);
        assert_eq!(again.as_ptr(), address);
        assert_eq!(again[0], 1.0);
        // Another size is not taken from the cache
        let other = Pooled::take(|v: &Vec<f64>| v.len() == 3, || vec![3.0; 3]);
        assert_eq!(other[0], 3.0);
    }
}
