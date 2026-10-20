#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""A run's snapshot file against the solver's own station records, read by a decoder
written from the format spec in src/io/snapshot.rs (not from the viewer's or the
library's reader): what the viewer replays is what the model computed.

    uv run scripts/dgsnap_check.py <run dir with froya.dgsnap, station_*.txt, currents_*.txt>

For the first station of the header (`station=NAME,X,Y`): the node the run sampled
(the nearest node of an element whose every node is at least 3 m deep, as
`froya_real_data` and the viewer's trace choose it), then per frame η against
`station_<name>.txt` and ū, v̄ against `currents_<name>.txt` at the same Unix time.
Also checks the node order (r fastest) by the bed's continuity between elements, and
reports the range of a 3D run's temperature in wet and dry nodes.

A coupled wave run's file (format version 3, surface fields `wave_*`) is checked
against the run's `wave_points.txt` too: H_s, T_p and the direction the waves come
from at each of its points, against the snapshot's fields evaluated there by the
element polynomial, at the record's hours. The run samples H_s by the wave mesh's
polynomial, the snapshot holds it interpolated onto the circulation's nodes: they
must agree to a few per cent (H_s is judged). T_p and the direction are the run's
at the nearest wave node, so they are reported, not judged.
"""
import os
import sys

import numpy as np

MIN_DEPTH = 3.0


class Snap:
    def __init__(self, path):
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic[:6] != b"DGSNAP":
                sys.exit(f"{path}: not a snapshot file")
            self.version = magic[7]
            hlen = int(np.frombuffer(f.read(8), "<u8")[0])
            h = f.read(hlen)
        o = 0

        def take(fmt, n=1):
            nonlocal o
            a = np.frombuffer(h, dtype=np.dtype("<" + fmt), count=n, offset=o)
            o += a.nbytes
            return a

        def text():
            nonlocal o
            n = int(take("u8")[0])
            o += n
            return h[o - n:o].decode()

        self.order = int(take("u4")[0])
        nv = int(take("u8")[0])
        self.verts = take("f8", 2 * nv).reshape(nv, 2)
        ne = int(take("u8")[0])
        self.quads = take("u4", 4 * ne).reshape(ne, 4)
        take("u4", 4 * int(take("u8")[0]))  # periodic pairs
        take("u4", 4 * int(take("u8")[0]))  # boundary faces
        self.np_ = (self.order + 1) ** 2
        self.n_nodes = ne * self.np_
        self.bed = take("f8", self.n_nodes).reshape(ne, self.np_)
        self.clock = float(take("f8")[0])
        self.meta = [line.split("=", 1) for line in text().splitlines() if "=" in line]
        self.n_levels, self.fields = 0, []
        if self.version >= 2:
            self.n_levels = int(take("u4")[0])
            if self.n_levels:
                take("f8", 2 * self.n_levels + 1)  # sigma_w, sigma_rho
                text()  # stretching
                self.fields = [text() for _ in range(int(take("u4")[0]))]
        self.surface_fields = []
        if self.version >= 3:
            self.surface_fields = [text() for _ in range(int(take("u4")[0]))]
        assert o == hlen, (o, hlen)
        nn, nl, ns = self.n_nodes, self.n_levels, len(self.surface_fields)
        self.frame_bytes = 8 + 4 * nn * (3 + ns) + 4 * nn * nl * len(self.fields)
        self.n_frames = (os.path.getsize(path) - 16 - hlen) // self.frame_bytes
        self.mm = np.memmap(path, np.uint8, "r", 16 + hlen, (self.n_frames * self.frame_bytes,))

    def _f32(self, start, n):
        return np.frombuffer(self.mm[start:start + 4 * n].tobytes(), "<f4")

    def time(self, k):
        b = self.mm[k * self.frame_bytes:k * self.frame_bytes + 8].tobytes()
        return float(np.frombuffer(b, "<f8")[0])

    def surface(self, k, which):
        i = "euv".index(which[0])
        return self._f32(k * self.frame_bytes + 8 + 4 * self.n_nodes * i, self.n_nodes)

    def sea(self, k, name):
        """Surface field `name` of frame k, one value per node."""
        j = 3 + self.surface_fields.index(name)
        return self._f32(k * self.frame_bytes + 8 + 4 * self.n_nodes * j, self.n_nodes)

    def field(self, k, name):
        nn, nl = self.n_nodes, self.n_levels
        j = self.fields.index(name)
        start = k * self.frame_bytes + 8 + 4 * nn * (3 + len(self.surface_fields)) + 4 * nn * nl * j
        return self._f32(start, nn * nl).reshape(nn, nl)


def gll(n):
    from numpy.polynomial import legendre
    c = np.zeros(n + 1)
    c[n] = 1
    inner = np.sort(legendre.legroots(legendre.legder(c))) if n > 1 else []
    return np.concatenate([[-1.0], inner, [1.0]])


def node_xy(s, r_fastest=True):
    g = gll(s.order)
    r, t = np.meshgrid(g, g, indexing="xy" if r_fastest else "ij")
    r, t = r.ravel(), t.ravel()
    shape = np.stack([(1 - r) * (1 - t), (1 + r) * (1 - t), (1 + r) * (1 + t), (1 - r) * (1 + t)], 1) / 4
    return np.einsum("pk,ekd->epd", shape, s.verts[s.quads])


def bed_jump(s, r_fastest):
    """Largest difference of the bed between elements' nodes at the same point."""
    _, inv = np.unique(np.round(node_xy(s, r_fastest).reshape(-1, 2), 3), axis=0, return_inverse=True)
    inv = inv.ravel()
    lo = np.full(inv.max() + 1, np.inf)
    hi = np.full(inv.max() + 1, -np.inf)
    np.minimum.at(lo, inv, s.bed.ravel())
    np.maximum.at(hi, inv, s.bed.ravel())
    return float(np.max(hi - lo))


def local_xy(lat, lon, ref_lat, ref_lon):
    """io::LocalProjection: metres east and north of the reference on the WGS84
    ellipsoid's radii of curvature there."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = 2 * f - f * f
    phi = np.radians(ref_lat)
    w = 1 - e2 * np.sin(phi) ** 2
    rho, nu = a * (1 - e2) / w ** 1.5, a / np.sqrt(w)
    return (lon - ref_lon) * nu * np.cos(phi) * np.pi / 180, (lat - ref_lat) * rho * np.pi / 180


def probe(s, x, y):
    """(element, weights over its nodes) of the point, by the inverse bilinear map
    and the GLL Lagrange polynomials (r fastest), or None outside the mesh."""
    v = s.verts[s.quads]
    centre = v.mean(axis=1)
    for e in np.argsort(np.hypot(centre[:, 0] - x, centre[:, 1] - y))[:16]:
        q = v[e]
        r = t = 0.0
        for _ in range(30):
            sh = np.array([(1 - r) * (1 - t), (1 + r) * (1 - t), (1 + r) * (1 + t), (1 - r) * (1 + t)]) / 4
            dr = np.array([-(1 - t), 1 - t, 1 + t, -(1 + t)]) / 4
            dt = np.array([-(1 - r), -(1 + r), 1 + r, 1 - r]) / 4
            res = sh @ q - [x, y]
            J = np.stack([dr @ q, dt @ q], 1)
            r, t = np.array([r, t]) - np.linalg.solve(J, res)
        if abs(r) <= 1 + 1e-9 and abs(t) <= 1 + 1e-9:
            g = gll(s.order)
            def lagrange(z):
                return np.array([np.prod([(z - g[m]) / (g[j] - g[m]) for m in range(len(g)) if m != j])
                                 for j in range(len(g))])
            return e, np.outer(lagrange(t), lagrange(r)).ravel()
    return None


def check_waves(s, run):
    """The snapshot's sea state against the run's `wave_points.txt`."""
    path = os.path.join(run, "wave_points.txt")
    if not os.path.exists(path):
        print(f"  no {path}")
        return True
    header = open(path, encoding="utf-8").read().splitlines()[1]
    points = [p.strip().replace("°E", "").replace("°N", "").split()
              for p in header.split(":", 1)[1].split(";") if p.strip()]
    projection = dict(s.meta).get("projection", "").split(",")
    ref_lat, ref_lon = float(projection[1]), float(projection[2])
    rec = np.loadtxt(path, comments="#", ndmin=2)
    times = np.array([s.time(k) for k in range(s.n_frames)])
    ok = True
    for j, (lon, lat) in enumerate(points):
        x, y = local_xy(float(lat), float(lon), ref_lat, ref_lon)
        found = probe(s, x, y)
        if found is None:
            print(f"  {lon}E {lat}N: outside the mesh")
            continue
        e, w = found
        nodes = slice(e * s.np_, (e + 1) * s.np_)
        hs_d, tp_d, dir_d, n = [], [], [], 0
        for row in rec:
            k = np.flatnonzero(np.abs(times - row[0]) < 1e-3)
            if not len(k):
                continue
            k, n = k[0], n + 1
            at = {name: float(w @ s.sea(k, name)[nodes]) for name in s.surface_fields}
            hs, tp, dx, dy = row[1 + 4 * j], row[3 + 4 * j], at["wave_dir_x"], at["wave_dir_y"]
            hs_d.append(at["wave_hs"] / hs - 1)
            tp_d.append(at["wave_tp"] / tp - 1)
            came = (np.degrees(np.arctan2(dx, dy)) + 180) % 360
            dir_d.append((came - row[4 + 4 * j] + 180) % 360 - 180)
        hs_d, tp_d, dir_d = map(np.abs, map(np.array, (hs_d, tp_d, dir_d)))
        # Interpolated onto the circulation's nodes and back to the point: close, not
        # equal where the field bends within a wave element (a lee)
        good = n > 0 and np.median(hs_d) < 0.05
        ok &= good
        print(f"  {lon}E {lat}N, {n} hours: H_s {100 * np.median(hs_d):.1f} % (median), "
              f"{100 * hs_d.max():.1f} % (worst); T_p {100 * np.median(tp_d):.1f} %, "
              f"{100 * tp_d.max():.1f} %; direction {np.median(dir_d):.1f}, {dir_d.max():.1f} deg "
              f"({'ok' if good else 'MISMATCH'})")
    return ok


def main(run):
    s = Snap(os.path.join(run, "froya.dgsnap"))
    print(f"{s.n_frames} frames, P{s.order}, {len(s.quads)} elements, {s.n_levels} levels {s.fields}")
    jumps = bed_jump(s, True), bed_jump(s, False)
    print(f"node order: bed jump {jumps[0]:.1e} m with r fastest, {jumps[1]:.1e} m with s fastest")
    ok = jumps[0] < 1e-6

    stations = [v.split(",") for k, v in s.meta if k == "station"]
    if not stations:
        sys.exit("no station in the header")
    name, x, y = stations[0][0], float(stations[0][1]), float(stations[0][2])
    xy = node_xy(s)
    d = np.hypot(xy[..., 0] - x, xy[..., 1] - y)
    d[~(-s.bed >= MIN_DEPTH).all(axis=1)] = np.inf
    e, p = np.unravel_index(np.argmin(d), d.shape)
    node = e * s.np_ + p
    print(f"{name}: node {node}, {d[e, p]:.0f} m from the station, {-s.bed[e, p]:.1f} m deep")

    slug = name.lower().replace(" ", "_")
    t = np.array([s.time(k) for k in range(s.n_frames)]) + s.clock
    for kind, columns in (("station", "e"), ("currents", "uv")):
        path = os.path.join(run, f"{kind}_{slug}.txt")
        if not os.path.exists(path):
            print(f"  no {path}")
            continue
        rec = np.loadtxt(path, comments="#", ndmin=2)
        worst, matched = 0.0, 0
        for k in range(s.n_frames):
            i = np.flatnonzero(np.abs(rec[:, 0] - t[k]) < 1e-3)
            if len(i):
                matched += 1
                for c, which in enumerate(columns):
                    worst = max(worst, abs(float(s.surface(k, which)[node]) - rec[i[0], 1 + c]))
        # f32 rounding of values of order 1
        good = matched == s.n_frames and worst < 1e-5
        ok &= good
        print(f"  {kind}: {matched} of {s.n_frames} frames at the record's times, "
              f"worst difference {worst:.1e} ({'ok' if good else 'MISMATCH'})")

    if "wave_hs" in s.surface_fields:
        print(f"sea state: {s.surface_fields}")
        ok &= check_waves(s, run)

    if "temp" in s.fields:
        lo_wet = lo_dry = np.inf
        hi_wet = hi_dry = -np.inf
        for k in range(s.n_frames):
            temp = s.field(k, "temp")
            wet = s.surface(k, "eta") - s.bed.ravel() > 0.01
            if wet.any():
                lo_wet, hi_wet = min(lo_wet, temp[wet].min()), max(hi_wet, temp[wet].max())
            if (~wet).any():
                lo_dry, hi_dry = min(lo_dry, temp[~wet].min()), max(hi_dry, temp[~wet].max())
        print(f"temperature over the run: wet nodes {lo_wet:.2f}..{hi_wet:.2f} °C, "
              f"dry nodes {lo_dry:.2f}..{hi_dry:.2f} °C")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
