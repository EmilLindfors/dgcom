"""The allocation sites of a heap profile (dhat-heap.json), by size.

A `dgcom` built with `--features dhat-heap` writes `dhat-heap.json` to a
run's output directory (`notes/performance.md`, "Heap profile"). This lists
its sites by their bytes at the run's peak, by their bytes at the run's last
output ("at the end": what the run holds), and by the blocks they allocated
in all (a site that allocates in every step shows here).

    uv run python scripts/dhat_top.py <dhat-heap.json> [frames] [top] [unit_bytes]

`frames` (2) is how many of the crate's own frames name a site, `top` (30)
how many sites each list shows. With `unit_bytes` (say elements x nodes x 8,
one 2D field) each site's bytes are also given in that unit, which reads a
site as so many fields: n_levels of them is one 3D field.

On Windows run the profiled binary from its build directory: the profiler
finds its symbols in `dgcom.pdb` beside the exe, and without it every site is
nameless.
"""

import json
import re
import sys
from collections import defaultdict

# Frames of the allocator, the standard library and the thread pool
SKIP = re.compile(
    r"\b(alloc::|core::|std::|dhat::|__rust|rayon|hashbrown|<alloc|<core|<std"
    r"|RtlUserThreadStart|BaseThreadInitThunk|__scrt|mainCRTStartup|invoke_main)"
)


def site_name(frame):
    """`function [file:line]` of a frame of the crate's, None of another's."""
    m = re.match(r"0x[0-9a-fA-F]+: (.*?)(?: \((.*)\))?$", frame)
    if not m:
        return frame
    symbol, where = m.group(1), m.group(2) or ""
    symbol = re.sub(r"::h[0-9a-f]{16}$", "", symbol)
    for _ in range(2):
        symbol = re.sub(r"<[^<>]*>", "<>", symbol)
    where = re.sub(r"^.*[\\/]src[\\/]", "src/", where).replace("\\", "/")
    where = re.sub(r":\d+$", "", where)
    if "dgcom" in symbol or where.startswith("src/"):
        return f"{symbol.split('::')[-1]} [{where}]"
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    unit = float(sys.argv[4]) if len(sys.argv) > 4 else None

    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    frames = profile["ftbl"]

    def key(point):
        names = []
        for i in point["fs"]:
            if SKIP.search(frames[i].split(" (")[0]):
                continue
            name = site_name(frames[i])
            if name and (not names or names[-1] != name):
                names.append(name)
            if len(names) == n_frames:
                break
        if names:
            return " <- ".join(names)
        return "(not the crate's) " + (frames[point["fs"][0]][:80] if point["fs"] else "no symbols")

    # Per site: bytes at the peak, bytes at the end, bytes and blocks in all
    sites = defaultdict(lambda: [0, 0, 0, 0])
    for point in profile["pps"]:
        site = sites[key(point)]
        site[0] += point.get("gb", 0)
        site[1] += point.get("eb", 0)
        site[2] += point["tb"]
        site[3] += point["tbk"]

    gb = 1024.0**3
    total = [sum(site[i] for site in sites.values()) for i in range(4)]
    print(path)
    print(
        f"at the peak {total[0] / gb:.3f} GB, at the end {total[1] / gb:.3f} GB, "
        f"allocated in all {total[2] / gb:.3f} GB in {total[3]} blocks"
    )

    def show(title, column, line):
        print(f"\n== {title}")
        for name, site in sorted(sites.items(), key=lambda kv: -kv[1][column])[:top]:
            if site[column] == 0:
                break
            print(f"{line(site)}  {name}")

    def held(column):
        def line(site):
            text = f"{site[column] / gb:8.3f} GB {100.0 * site[column] / max(total[column], 1):5.1f} %"
            if unit:
                text += f" {site[column] / unit:8.1f} u"
            return text

        return line

    show("bytes at the peak", 0, held(0))
    show("bytes at the end", 1, held(1))
    show("blocks allocated in all", 3, lambda site: f"{site[3]:10d} blocks {site[2] / gb:9.3f} GB")


if __name__ == "__main__":
    main()
