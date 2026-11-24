//! Element lists on the device (P2.7 Phase 4.5d, `notes/gpu.md`): the
//! elements one pass of the 3D stages' local time stepping acts on.
//!
//! dgcom's `ModeSplitIntegrator` hands its model each pass's elements as a
//! host list (`Blocks`): a stage's elements whose rate is at least the
//! finest substep's depth, the elements whose substeps end there, a level's
//! elements at a substep's end. A coarse step's lists are few and come pass
//! after pass, step after step while the levels hold, so [`Lists`] keeps each
//! list on the device by its content: uploaded the first time it comes,
//! found again after.
//!
//! A kernel restricted to a list takes it in one of two forms:
//!
//! - `listed`, the elements themselves: a kernel with a thread per value
//!   launches over the listed elements' values only ([`listed_index`]);
//! - `member`, a flag per element: a kernel over some other collection (the
//!   elements with a merged group, the flushed elements, the columns with an
//!   implicit part) skips the members' complement ([`is_member`]).
//!
//! An empty buffer stands for every element in both.

use cuda_core::DeviceBuffer;
use dgcom::solver::Blocks;

use crate::swe2d::Gpu;

/// Thread `t`'s value among the values of the elements `listed` (empty:
/// every element), `per` values an element.
#[inline]
pub fn listed_index(t: usize, listed: &[u32], per: usize) -> usize {
    if listed.is_empty() {
        t
    } else {
        listed[t / per] as usize * per + t % per
    }
}

/// Whether element `k` is a member of `member` (empty: every element is).
#[inline]
pub fn is_member(member: &[u32], k: usize) -> bool {
    member.is_empty() || member[k] != 0
}

/// One list on the device: its elements and every element's membership.
pub struct DeviceList {
    listed: DeviceBuffer<u32>,
    member: DeviceBuffer<u32>,
    len: usize,
}

/// The elements a device operation acts on: every element, or a list's.
#[derive(Clone, Copy, Default)]
pub struct On<'a> {
    list: Option<&'a DeviceList>,
}

impl<'a> On<'a> {
    /// Every element.
    pub const ALL: Self = Self { list: None };

    /// Whether it is every element.
    pub fn is_all(&self) -> bool {
        self.list.is_none()
    }

    /// How many elements of `n_elements`.
    pub fn count(&self, n_elements: usize) -> usize {
        self.list.map_or(n_elements, |list| list.len)
    }

    /// The elements, or `none` (empty) for every element.
    pub fn listed<'b>(&self, none: &'b DeviceBuffer<u32>) -> &'b DeviceBuffer<u32>
    where
        'a: 'b,
    {
        self.list.map_or(none, |list| &list.listed)
    }

    /// The membership flags, or `none` (empty) for every element.
    pub fn member<'b>(&self, none: &'b DeviceBuffer<u32>) -> &'b DeviceBuffer<u32>
    where
        'a: 'b,
    {
        self.list.map_or(none, |list| &list.member)
    }
}

/// A list kept by [`Lists`], with the host's copy it is found by.
struct Entry {
    host: Vec<u32>,
    list: DeviceList,
    used: u64,
}

/// Which list [`Lists::upload`] found or made: [`Lists::on`] reads it.
#[derive(Clone, Copy, Debug)]
pub struct ListId(Option<usize>);

/// The lists a run's passes act on, on the device, by content.
pub struct Lists {
    n_elements: usize,
    entries: Vec<Entry>,
    clock: u64,
}

/// The lists kept: a coarse step on L levels with an s-stage scheme has
/// `(s + 1)(L + 1)` lists and its substeps' ends a level's each.
const MAX_LISTS: usize = 96;

impl Lists {
    /// No lists yet, on a mesh of `n_elements`.
    pub fn new(n_elements: usize) -> Self {
        Self {
            n_elements,
            entries: Vec::new(),
            clock: 0,
        }
    }

    /// The device's list of `elements` (uploaded unless kept: the copy waits
    /// for the queued kernels), to be read with [`Self::on`]. The least
    /// recently used list goes when more are kept than [`MAX_LISTS`], never
    /// the one used last.
    pub fn upload(
        &mut self,
        gpu: &Gpu,
        elements: Blocks<'_>,
    ) -> Result<ListId, Box<dyn std::error::Error>> {
        let list = match elements {
            Blocks::All(_) => return Ok(ListId(None)),
            Blocks::List { list, .. } if list.len() == self.n_elements => {
                return Ok(ListId(None));
            }
            Blocks::List { list, .. } => list,
        };
        self.clock += 1;
        if let Some(i) = self.entries.iter().position(|e| e.host == list) {
            self.entries[i].used = self.clock;
            return Ok(ListId(Some(i)));
        }
        let st = &gpu.stream;
        let mut member = vec![0u32; self.n_elements];
        for &k in list {
            member[k as usize] = 1;
        }
        let entry = Entry {
            host: list.to_vec(),
            list: DeviceList {
                // An empty list's elements are never read (no thread), its
                // flags all zero: no element
                listed: if list.is_empty() {
                    DeviceBuffer::zeroed(st, 0)?
                } else {
                    DeviceBuffer::from_host(st, list)?
                },
                member: DeviceBuffer::from_host(st, &member)?,
                len: list.len(),
            },
            used: self.clock,
        };
        if self.entries.len() < MAX_LISTS {
            self.entries.push(entry);
            return Ok(ListId(Some(self.entries.len() - 1)));
        }
        let oldest = (0..self.entries.len())
            .min_by_key(|&i| self.entries[i].used)
            .expect("a list is kept");
        self.entries[oldest] = entry;
        Ok(ListId(Some(oldest)))
    }

    /// The elements of a list [`Self::upload`] gave.
    pub fn on(&self, id: ListId) -> On<'_> {
        On {
            list: id.0.map(|i| &self.entries[i].list),
        }
    }
}
