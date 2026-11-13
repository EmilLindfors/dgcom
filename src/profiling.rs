//! Wall-clock phase timers for profiling a run (`DG_PROFILE=1`), and the
//! process's peak memory.
//!
//! [`scope`] times a closure under a name when the environment variable
//! `DG_PROFILE` is set, and runs it untimed otherwise (one relaxed atomic load).
//! [`report`] lists the totals, [`reset`] clears them. [`peak_memory`] is
//! what the process has held at most; which allocation holds it is the heap
//! profile's to say (the `dhat-heap` feature, `notes/performance.md`).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

static TIMES: Mutex<Vec<(&'static str, Duration, u64)>> = Mutex::new(Vec::new());
/// 0: not read yet, 1: off, 2: on.
static ENABLED: AtomicU8 = AtomicU8::new(0);

/// Whether `DG_PROFILE` is set.
#[inline]
pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        0 => {
            let on = std::env::var_os("DG_PROFILE").is_some();
            ENABLED.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
        v => v == 2,
    }
}

/// Run `f`, adding its wall time to `name`'s total when profiling.
#[inline]
pub fn scope<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let start = Instant::now();
    let r = f();
    add(name, start.elapsed());
    r
}

/// The start of a timed section ([`stop`]), `None` when not profiling.
#[inline]
pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Add the time since `start` to `name`'s total (nothing for `None`).
#[inline]
pub fn stop(name: &'static str, start: Option<Instant>) {
    if let Some(start) = start {
        add(name, start.elapsed());
    }
}

/// Add `elapsed` to `name`'s total.
pub fn add(name: &'static str, elapsed: Duration) {
    let mut times = TIMES.lock().unwrap_or_else(|e| e.into_inner());
    match times.iter_mut().find(|(n, _, _)| *n == name) {
        Some((_, t, count)) => {
            *t += elapsed;
            *count += 1;
        }
        None => times.push((name, elapsed, 1)),
    }
}

/// Clear the totals.
pub fn reset() {
    TIMES.lock().unwrap_or_else(|e| e.into_inner()).clear();
}

/// The totals, one line per name in order of first use: total seconds, calls,
/// milliseconds per call.
pub fn report() -> String {
    let times = TIMES.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = String::new();
    for (name, t, count) in times.iter() {
        out.push_str(&format!(
            "  {name:<32} {:>9.3} s {count:>9} calls {:>10.3} ms/call\n",
            t.as_secs_f64(),
            1e3 * t.as_secs_f64() / *count as f64
        ));
    }
    out
}

/// The most memory the process has held so far (bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeakMemory {
    /// In physical memory at once (the peak working set on Windows, the peak
    /// resident set `VmHWM` on Linux): the pages the process has written or
    /// read, so a buffer allocated zeroed and never touched is not in it.
    pub resident: u64,
    /// Committed at once (Windows' peak commit charge: every allocation,
    /// touched or not, which the page file must be able to back). `None`
    /// where the system keeps no such peak.
    pub committed: Option<u64>,
}

impl PeakMemory {
    /// `16.7 GB in memory (19.9 GB committed)`.
    pub fn describe(&self) -> String {
        let gb = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        match self.committed {
            Some(committed) => format!(
                "{:.1} GB in memory ({:.1} GB committed)",
                gb(self.resident),
                gb(committed)
            ),
            None => format!("{:.1} GB in memory", gb(self.resident)),
        }
    }
}

/// The process's [`PeakMemory`], `None` where it cannot be read (other
/// systems than Windows and Linux).
pub fn peak_memory() -> Option<PeakMemory> {
    system::peak_memory()
}

#[cfg(windows)]
mod system {
    use super::PeakMemory;
    use std::ffi::c_void;

    /// `PROCESS_MEMORY_COUNTERS` (psapi.h).
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    pub fn peak_memory() -> Option<PeakMemory> {
        let mut counters = ProcessMemoryCounters::default();
        let cb = size_of::<ProcessMemoryCounters>() as u32;
        counters.cb = cb;
        // SAFETY: the pseudo-handle of this process needs no closing, and
        // `counters` is a `PROCESS_MEMORY_COUNTERS` of the size passed
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, cb) };
        (ok != 0).then_some(PeakMemory {
            resident: counters.peak_working_set_size as u64,
            committed: Some(counters.peak_pagefile_usage as u64),
        })
    }
}

#[cfg(target_os = "linux")]
mod system {
    use super::PeakMemory;

    pub fn peak_memory() -> Option<PeakMemory> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        Some(PeakMemory {
            resident: super::status_kb(&status, "VmHWM")? * 1024,
            committed: None,
        })
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod system {
    pub fn peak_memory() -> Option<super::PeakMemory> {
        None
    }
}

/// The value of `key` in a `/proc/<pid>/status` text (kB).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn status_kb(status: &str, key: &str) -> Option<u64> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix(':'))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The peak is at least what a touched allocation holds, and reads as
    /// gigabytes.
    #[test]
    fn peak_memory_counts_a_touched_allocation() {
        let Some(before) = peak_memory() else {
            return;
        };
        let held = vec![1u8; 64 << 20];
        let after = peak_memory().expect("read once already");
        assert!(after.resident >= before.resident);
        assert!(after.resident >= held.len() as u64, "{after:?}");
        if let Some(committed) = after.committed {
            assert!(committed >= held.len() as u64, "{after:?}");
        }
        assert_eq!(held[held.len() - 1], 1);

        let peak = PeakMemory {
            resident: 17 << 30,
            committed: Some(43 << 29),
        };
        assert_eq!(peak.describe(), "17.0 GB in memory (21.5 GB committed)");
        let peak = PeakMemory {
            committed: None,
            ..peak
        };
        assert_eq!(peak.describe(), "17.0 GB in memory");
    }

    #[test]
    fn a_status_value_is_read_by_its_key() {
        let status = "Name:	dgcom
VmPeak:	  204800 kB
VmHWM:	   51200 kB
";
        assert_eq!(status_kb(status, "VmHWM"), Some(51200));
        assert_eq!(status_kb(status, "VmPeak"), Some(204800));
        assert_eq!(status_kb(status, "VmRSS"), None);
    }
}
