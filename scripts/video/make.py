"""A video of a saved run, made by the viewer offscreen: a title card, one shot along a
camera path with a clock and the water level at a tide gauge drawn over it, an end
card with the run's own scores, crossfaded, H.264 at 1080p30.

    uv run --with pillow --with numpy --with imageio-ffmpeg python \
        scripts/video/make.py scripts/video/bodo.toml RUN_DIR [OUT_DIR] [--no-build] [--window on]
        [--no-record]

The video's file (`bodo.toml`, `chiloe.toml`) holds its text, the camera path, the
viewer's options and the gauge. RUN_DIR is a `dgcom run` output folder: its `.dgsnap`
is replayed, its `stations.csv` gives the model's water level at the gauge, and its
`gauge_skill.csv` and `gauge_daily.csv` the end card's numbers. The viewer records
`OUT_DIR/frames/<name>/frame_NNNNN.png` [OUT_DIR: output/video] with `--window off`
(no display needed; `--window on` to watch), then the frames go to ffmpeg as raw RGB
through a pipe with the overlays drawn on, into `OUT_DIR/<name>.mp4` (`--no-record`
assembles the frames already there again). ffmpeg is
`$FFMPEG`, else imageio-ffmpeg's, else `ffmpeg` on the PATH; the fonts DejaVu Sans
(`$FONT_DIR`).

Each frame's model time comes from the camera path, as `viz/src/record.rs` sets it:
linear between keys.
"""
import csv
import datetime as dt
import math
import os
import subprocess
import sys
import tomllib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 30
W, H = 1920, 1080
FADE = 0.6
TITLE_S, END_S = 4.0, 5.5
BG = (11, 14, 18)
MODEL_C = (94, 200, 255)
GAUGE_C = (255, 196, 92)
TEXT = (240, 244, 248)
SOFT = (201, 212, 223)
DIM = (143, 163, 181)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONT_DIR = os.environ.get("FONT_DIR", "/usr/share/fonts/truetype/dejavu")


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def ffmpeg():
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


# --- inputs ---------------------------------------------------------------------------


def camera_keys(path):
    """(video s, model h) of each key, as `record::parse` reads them."""
    keys = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].split()
        if line:
            keys.append((float(line[0]), float(line[1])))
    return keys


def model_hour(keys, t):
    """Model time (h) at video time t: linear between keys, the end keys beyond them."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, m0), (t1, m1) in zip(keys, keys[1:]):
        if t <= t1:
            return m0 + (t - t0) / (t1 - t0) * (m1 - m0)
    return keys[-1][1]


def model_series(path, station):
    hours, eta = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["station"] == station:
                hours.append(float(row["time_s"]) / 3600)
                eta.append(float(row["eta_m"]))
    if not hours:
        sys.exit(f"no station {station} in {path}")
    return np.array(hours), np.array(eta)


def gauge_series(path, start, h0, h1):
    """A gauge file (`# …` header, then `ISO-time level_m` lines) over hours h0..h1."""
    hours, level = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        stamp, value = line.split()[:2]
        h = (dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")) - start).total_seconds() / 3600
        if h0 <= h <= h1 and math.isfinite(float(value)):
            hours.append(h)
            level.append(float(value))
    return np.array(hours), np.array(level)


def scores(run, station):
    """The station's day-by-day score (RMS of the daily centred RMSEs), its number of
    days and its correlation over the run, from the run's files."""
    with open(os.path.join(run, "gauge_daily.csv"), encoding="utf-8") as f:
        days = [float(r["centred_rmse_m"]) for r in csv.DictReader(f) if r["station"] == station]
    with open(os.path.join(run, "gauge_skill.csv"), encoding="utf-8") as f:
        obs = next(r for r in csv.DictReader(f)
                   if r["station"] == station and r["reference"] == "observations")
    return math.sqrt(sum(d * d for d in days) / len(days)), len(days), float(obs["correlation"])


def m2_ratio(run, station):
    """The model's M2 amplitude over the gauge's (`gauge_constituents.csv`), or None."""
    with open(os.path.join(run, "gauge_constituents.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["station"] == station and r["constituent"] == "M2":
                return float(r["amplitude_ratio"])
    return None


def spring_and_neap(ranges):
    """Days whose observed range is the largest (spring tide) or smallest (neap tide)
    within three days either side, at least five days from the last one labelled. A
    first or last day is the extreme of a week the record cuts short, so it is labelled
    only if it is also in the top (bottom) third of the record's ranges: Bodø's 15 June
    (2.48 m, new moon that day) is a spring tide, Chiloé's 14 November (3.68 m, two days
    after neap and growing) is not."""
    days = sorted(ranges)
    low, high = min(ranges.values()), max(ranges.values())
    picked = []
    for kind, better, far in (("spring tide", max, lambda r: r >= high - (high - low) / 3),
                              ("neap tide", min, lambda r: r <= low + (high - low) / 3)):
        for k in days:
            near = [ranges[j] for j in days if abs(j - k) <= 3]
            inside = days[0] < k < days[-1]
            if (ranges[k] == better(near) and (inside or far(ranges[k]))
                    and all(abs(k - j) >= 5 for j, kd in picked if kd == kind)):
                picked.append((k, kind))
    return picked


# --- drawing --------------------------------------------------------------------------


class Trace:
    """The strip along the bottom with the water level at the gauge over the shot's
    hours: all of it dim, the part up to the shown time bright, a marker at the shown
    time. It starts right of the viewer's colour legend (`--ui legend`, bottom left)."""

    X0, X1 = 450, 1580          # the plot's left and right (px)
    Y0, Y1 = 968, 1036          # its top and bottom

    def __init__(self, cfg, model, gauge, start, h0, h1):
        self.h0, self.h1 = h0, h1
        mh, me = model
        gh, gl = gauge
        inside = (mh >= h0) & (mh <= h1)
        mh, me = mh[inside], me[inside]
        self.lim = max(np.abs(me).max(), np.abs(gl).max() if len(gl) else 0) * 1.05
        self.model = (mh, me)
        self.base = Image.new("RGBA", (W, H))
        self.bright = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(self.base)
        d.rounded_rectangle((372, 930, 1900, 1072), radius=12, fill=(8, 11, 15, 185))
        # Days, labelled every other one
        first = math.ceil(h0 / 24)
        for day in range(first, int(h1 // 24) + 1):
            x = self.x(24 * day)
            d.line((x, self.Y0, x, self.Y1), fill=(255, 255, 255, 26), width=1)
            if (day - first) % 2 == 0:
                when = start + dt.timedelta(days=day)
                d.text((x, self.Y1 + 5), f"{when.day} {when:%b}", font=font(14),
                       fill=DIM + (255,), anchor="mt")
        # Metres: a round step, one or two a side
        step = next(s for s in (0.5, 1, 2, 5, 10) if self.lim / s < 2)
        n = int(self.lim // step)
        for k in range(-n, n + 1):
            m = k * step
            y = self.y(m)
            d.line((self.X0, y, self.X1, y), fill=(255, 255, 255, 36 if k else 64), width=1)
            d.text((self.X0 - 10, y), f"{m:+g} m" if k else "0", font=font(13),
                   fill=DIM + (255,), anchor="rm")
        # Title and legend, right of the plot
        lx = self.X1 + 30
        d.text((lx, 942), cfg["title"], font=font(18, True), fill=TEXT + (255,))
        d.line((lx, 984, lx + 30, 984), fill=MODEL_C + (255,), width=4)
        d.text((lx + 42, 984), "model", font=font(16), fill=SOFT + (255,), anchor="lm")
        d.ellipse((lx + 11, 1010 - 4, lx + 19, 1010 + 4), fill=GAUGE_C + (255,))
        d.text((lx + 42, 1010), cfg["gauge_label"], font=font(16), fill=SOFT + (255,),
               anchor="lm")
        # The two layers of the curves: the gauge's samples as beads under the model's line
        for layer, alpha in ((self.base, 64), (self.bright, 255)):
            dd = ImageDraw.Draw(layer)
            for h, level in zip(gh, gl):
                x, y = self.x(h), self.y(level)
                dd.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=GAUGE_C + (alpha,))
            dd.line([(self.x(h), self.y(e)) for h, e in zip(mh, me)], fill=MODEL_C + (alpha,),
                    width=2, joint="curve")
        # Spring and neap tides, from the observed range of each whole day
        days = {}
        for h, level in zip(gh, gl):
            days.setdefault(int(h // 24), []).append(level)
        self.ranges = {k: max(v) - min(v) for k, v in days.items()
                       if len(v) >= 0.8 * max(len(u) for u in days.values())}
        for k, label in spring_and_neap(self.ranges):
            x = min(max(self.x(24 * k + 12), self.X0 + 40), self.X1 - 40)
            d.text((x, self.Y0 - 6), label, font=font(14), fill=DIM + (255,), anchor="mb")

    def x(self, h):
        return self.X0 + (self.X1 - self.X0) * (h - self.h0) / (self.h1 - self.h0)

    def y(self, m):
        return (self.Y0 + self.Y1) / 2 - (self.Y1 - self.Y0) / 2 * m / self.lim

    def draw(self, img, h):
        img.alpha_composite(self.base)
        xn = int(round(self.x(h)))
        if xn > 0:
            img.alpha_composite(self.bright.crop((0, 0, xn + 3, H)), (0, 0))
        d = ImageDraw.Draw(img)
        mh, me = self.model
        e = float(np.interp(h, mh, me))
        d.line((xn, self.Y0 - 2, xn, self.Y1 + 2), fill=(255, 255, 255, 140), width=2)
        d.ellipse((xn - 5, self.y(e) - 5, xn + 5, self.y(e) + 5), fill=MODEL_C + (255,),
                  outline=(255, 255, 255, 255), width=2)


def clock(img, place, start, h):
    when = start + dt.timedelta(hours=h)
    d = ImageDraw.Draw(img)
    width = max(520, int(font(30, True).getlength(place)) + 50)
    d.rounded_rectangle((40, 36, 40 + width, 150), radius=14, fill=(8, 11, 15, 170))
    d.text((64, 52), place, font=font(30, True), fill=TEXT + (255,))
    d.text((64, 96), f"{when:%A} {when.day} {when:%B %Y}", font=font(22), fill=SOFT + (255,))
    d.text((16 + width, 96), f"{when:%H:%M} UTC", font=font(22), fill=SOFT + (255,), anchor="ra")


def card(lines, alpha):
    """A text card: (text, size, bold, colour, y) per line, centred, or (…, y, x, anchor)
    for a table's cell; faded in by `alpha` (0..1)."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for text, size, bold, colour, y, *at in lines:
        x, anchor = at if at else (W / 2, "mm")
        c = tuple(int(BG[i] + (colour[i] - BG[i]) * alpha) for i in range(3))
        d.text((x, y), text, font=font(size, bold), fill=c, anchor=anchor)
    return img


def ramp(t, d, rise=0.7, tail=0.6):
    """A card's text: in over `rise` s, out over the last `tail` s."""
    return max(0.0, min(1.0, t / rise, (d - t) / tail))


def title_card(lines):
    sizes = [(64, True, TEXT), (32, False, SOFT)] + [(24, False, DIM)] * (len(lines) - 2)
    ys = [470, 560] + [640 + 36 * i for i in range(len(lines) - 2)]
    return [(t, *s, y) for t, s, y in zip(lines, sizes, ys)]


def end_card(cfg, run, strip_range):
    end = cfg["end"]
    predicted = end.get("predicted_cm", {})
    lines = [(end["heading"], 40, True, TEXT, 420)]
    stations = end["stations"]
    if len(stations) == 1:
        s = stations[0]
        score, n, r = scores(run, s)
        lines += [
            (f"A typical day's error: {100 * score:.1f} cm, on a tide of up to {strip_range:.1f} m",
             32, False, SOFT, 500),
            (f"(RMS of {n} daily centred RMS errors"
             + (f"; pre-registered prediction: {predicted[s]:.1f} cm)" if s in predicted else ")"),
             22, False, DIM, 548),
            (f"Correlation over the run: {r:.3f}", 26, False, SOFT, 610),
        ]
    else:
        # A table: the gauge, its typical day's error (and days), M2's amplitude, r
        cols = [(560, "lm"), (1010, "mm"), (1260, "mm"), (1440, "mm")]
        head = ["", "a typical day's error", "M2 amplitude", "correlation"]
        lines += [(t, 20, False, DIM, 490, *c) for t, c in zip(head, cols) if t]
        y = 536
        for s in stations:
            score, n, r = scores(run, s)
            ratio = m2_ratio(run, s)
            row = [s, f"{100 * score:.1f} cm ({n} days)",
                   f"{100 * (ratio - 1):+.0f} %" if ratio else "", f"{r:.3f}"]
            lines += [(t, 28, i == 0, SOFT, y, *c) for i, (t, c) in enumerate(zip(row, cols))]
            y += 46
        lines.append(("the RMS of each day's centred RMS error against the gauge; M2: the "
                      "model's harmonic amplitude against the gauge's", 20, False, DIM, y + 14))
    lines.append((end["footer"], 22, False, DIM, 760))
    return lines


# --- the video ------------------------------------------------------------------------


def record(cfg, here, snap, frames, window, build):
    viz = os.path.join(REPO, "viz")
    if build:
        subprocess.run(["cargo", "build", "--release"], cwd=viz, check=True)
    for f in os.listdir(frames):
        if f.startswith("frame_") and f.endswith(".png"):
            os.remove(os.path.join(frames, f))
    cmd = [os.path.join(viz, "target", "release", "dg-viz"), "--replay", snap,
           "--window", window, "--minimap", "off", "--record", frames,
           "--camera", os.path.join(here, cfg["camera"]), "--ui", "legend"]
    if "terrain" in cfg:
        cmd += ["--terrain", os.path.join(REPO, cfg["terrain"])]
    cmd += [str(a) for a in cfg.get("viewer", [])]
    env = dict(os.environ, RUST_LOG="error")
    out = subprocess.run(cmd, cwd=viz, env=env, check=True, capture_output=True, text=True)
    print([line for line in out.stdout.splitlines() if "frames (model" in line][-1])


def assemble(cfg, here, run, frames, out):
    keys = camera_keys(os.path.join(here, cfg["camera"]))
    files = sorted(f for f in os.listdir(frames) if f.startswith("frame_") and f.endswith(".png"))
    if not files:
        sys.exit(f"no frame_*.png in {frames}")
    start = dt.datetime.fromisoformat(cfg["start"].replace("Z", "+00:00"))
    strip = cfg["strip"]
    h0, h1 = keys[0][1], keys[-1][1]
    model = model_series(os.path.join(run, "stations.csv"), strip["station"])
    gauge = gauge_series(os.path.join(REPO, strip["gauge"]), start, h0, h1)
    trace = Trace(strip, model, gauge, start, h0, h1)
    title = title_card(cfg["title"])
    end = end_card(cfg, run, max(trace.ranges.values()))

    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def put(img):
        enc.stdin.write(np.asarray(img.convert("RGB")).tobytes())

    def shot(i):
        img = Image.open(os.path.join(frames, files[i])).convert("RGBA")
        if img.size != (W, H):
            img = img.resize((W, H))
        h = model_hour(keys, i / FPS)
        clock(img, cfg["place"], start, h)
        trace.draw(img, h)
        return img.convert("RGB")

    n_fade = int(FADE * FPS)
    n_title = int(TITLE_S * FPS)
    for k in range(n_title):
        img = card(title, ramp(k / FPS, TITLE_S + FADE))
        if k >= n_title - n_fade:
            img = Image.blend(img, shot(0), (k - (n_title - n_fade) + 1) / (n_fade + 1))
        put(img)
    for i in range(len(files)):
        img = shot(i)
        if i >= len(files) - n_fade:
            img = Image.blend(img, card(end, 0.0), (i - (len(files) - n_fade) + 1) / (n_fade + 1))
        put(img)
        if (i + 1) % 300 == 0:
            print(f"  {i + 1} of {len(files)} frames", flush=True)
    for k in range(int(END_S * FPS)):
        put(card(end, ramp(k / FPS, END_S)))
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"wrote {out}: {(n_title + len(files) + int(END_S * FPS)) / FPS:.1f} s")


def main(argv):
    flags = {"--no-build": False, "--no-record": False}
    window = "off"
    args = []
    it = iter(argv)
    for a in it:
        if a in flags:
            flags[a] = True
        elif a == "--window":
            window = next(it)
        else:
            args.append(a)
    if len(args) not in (2, 3):
        sys.exit(__doc__)
    config, run = os.path.abspath(args[0]), os.path.abspath(args[1])
    out_dir = os.path.abspath(args[2] if len(args) == 3 else os.path.join(REPO, "output", "video"))
    with open(config, "rb") as f:
        cfg = tomllib.load(f)
    here = os.path.dirname(config)
    snaps = sorted(f for f in os.listdir(run) if f.endswith(".dgsnap"))
    if not snaps:
        sys.exit(f"no .dgsnap in {run}")
    frames = os.path.join(out_dir, "frames", cfg["name"])
    os.makedirs(frames, exist_ok=True)
    if not flags["--no-record"]:
        record(cfg, here, os.path.join(run, snaps[0]), frames, window, not flags["--no-build"])
    assemble(cfg, here, run, frames, os.path.join(out_dir, cfg["name"] + ".mp4"))


if __name__ == "__main__":
    main(sys.argv[1:])
