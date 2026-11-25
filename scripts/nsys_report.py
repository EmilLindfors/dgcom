"""Per-kernel and per-gap times of a run from Nsight Systems' sqlite export.

    uv run python nsys_report.py <file.sqlite> [skip_seconds] [top]

Kernels: launches, total, mean and median duration, share of the kernel time.
Gaps: the device idle between one kernel's end and the next one's start, by the
kernel that follows (what the device waited for: the host's work before that
launch, a synchronous copy, a launch's own overhead). Memory copies: counts and
bytes by kind. `skip_seconds` leaves out the start (reading the case, building
the bed, the first step's allocations).
"""

import sqlite3
import sys
from collections import defaultdict
from statistics import median


def main():
    path = sys.argv[1]
    skip = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = db.execute(
        "SELECT start, end, shortName, gridX, blockX FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    if not rows:
        print("no kernels")
        return
    t0 = rows[0][0] + skip * 1e9
    rows = [r for r in rows if r[0] >= t0]
    span = (rows[-1][1] - rows[0][0]) / 1e9
    by = defaultdict(list)
    threads = defaultdict(int)
    for start, end, name, gx, bx in rows:
        by[names[name]].append(end - start)
        threads[names[name]] += gx * bx
    total = sum(sum(v) for v in by.values()) / 1e9
    # The device busy: the union of the kernels' intervals
    busy, cursor = 0, rows[0][0]
    gaps = defaultdict(lambda: [0, 0])
    for start, end, name, _, _ in rows:
        if start > cursor:
            g = gaps[names[name]]
            g[0] += start - cursor
            g[1] += 1
            cursor = start
        if end > cursor:
            busy += end - cursor
            cursor = end
    busy /= 1e9
    print(f"{path}: {len(rows)} launches over {span:.2f} s; kernels' sum {total:.2f} s, "
          f"device busy {busy:.2f} s ({100 * busy / span:.1f} %), idle {span - busy:.2f} s")
    print(f"{'kernel':<28} {'launches':>9} {'total s':>9} {'share':>6} {'mean µs':>9} "
          f"{'median µs':>10} {'threads/launch':>15}")
    for name, v in sorted(by.items(), key=lambda kv: -sum(kv[1]))[:top]:
        s = sum(v) / 1e9
        print(f"{name:<28} {len(v):>9} {s:>9.3f} {100 * s / total:>5.1f}% {sum(v) / len(v) / 1e3:>9.1f} "
              f"{median(v) / 1e3:>10.1f} {threads[name] / len(v):>15.0f}")
    idle = sum(g[0] for g in gaps.values()) / 1e9
    print(f"\ngaps (device idle before a kernel starts): {idle:.2f} s in all")
    print(f"{'before kernel':<28} {'gaps':>9} {'total s':>9} {'share':>6} {'mean µs':>9}")
    for name, (ns, n) in sorted(gaps.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"{name:<28} {n:>9} {ns / 1e9:>9.3f} {100 * ns / 1e9 / max(idle, 1e-12):>5.1f}% {ns / n / 1e3:>9.1f}")
    try:
        copies = db.execute(
            "SELECT copyKind, COUNT(*), SUM(bytes), SUM(end - start) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE start >= ? GROUP BY copyKind", (t0,)
        ).fetchall()
        kinds = {1: "host to device", 2: "device to host", 8: "device to device"}
        print("\nmemory copies:")
        for kind, n, nbytes, ns in copies:
            print(f"  {kinds.get(kind, kind)}: {n} copies, {nbytes / 1e6:.1f} MB, {ns / 1e9:.3f} s")
    except sqlite3.Error as e:
        print(f"(no memcpy table: {e})")


if __name__ == "__main__":
    main()
