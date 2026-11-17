"""Assemble the Bodø video from the frames `scripts/bodo_video.sh` recorded: a title
card, the 15-day shot with its clock and the tide at the Bodø gauge drawn over it, an
end card, crossfaded, H.264 at 1080p30.

    uv run --with pillow --with numpy --with imageio-ffmpeg python \
        scripts/bodo_video/assemble.py FRAMES CAMERA STATIONS_CSV GAUGE_TXT SCORES_CSV OUT.mp4 \
        [PREDICTED_CM]

FRAMES holds the viewer's `frame_NNNNN.png`; CAMERA is the path file they were recorded
along (each frame's model time comes from it, as `viz/src/record.rs` sets it); the
model's water level at the gauge is the run's `stations.csv`, the observations
Kartverket's file (`# … columns: datetime water_level(m)`), and the end card's numbers
come from the run's `gauge_daily.csv` and `gauge_skill.csv` (SCORES_CSV is the
latter, the former beside it), with the pre-registered prediction of the daily score
if PREDICTED_CM is given. Frames go to ffmpeg as raw RGB through a pipe, so no
captioned copies are written. ffmpeg is `$FFMPEG`, else imageio-ffmpeg's, else
`ffmpeg` on the PATH. Fonts: DejaVu Sans (`$FONT_DIR`).
"""
import csv
import datetime as dt
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAMES, CAMERA, STATIONS, GAUGE, SCORES, OUT = sys.argv[1:7]
PREDICTED_CM = float(sys.argv[7]) if len(sys.argv) > 7 else None
FPS = 30
W, H = 1920, 1080
FADE = 0.6
TITLE_S, END_S = 4.0, 5.0
START = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
STATION = "Bodø"
BG = (11, 14, 18)
MODEL_C = (94, 200, 255)
GAUGE_C = (255, 196, 92)
TEXT = (240, 244, 248)
SOFT = (201, 212, 223)
DIM = (143, 163, 181)

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


def model_series(path):
    hours, eta = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["station"] == STATION:
                hours.append(float(row["time_s"]) / 3600)
                eta.append(float(row["eta_m"]))
    return np.array(hours), np.array(eta)


def gauge_series(path, h_end):
    hours, level = [], []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        stamp, value = line.split()[:2]
        h = (dt.datetime.fromisoformat(stamp.replace("Z", "+00:00")) - START).total_seconds() / 3600
        if 0 <= h <= h_end:
            hours.append(h)
            level.append(float(value))
    return np.array(hours), np.array(level)


def scores(skill_path):
    """Score 4 (RMS of the daily centred RMSEs) and the correlation, from the run's files."""
    daily = os.path.join(os.path.dirname(skill_path), "gauge_daily.csv")
    with open(daily, encoding="utf-8") as f:
        days = [float(r["centred_rmse_m"]) for r in csv.DictReader(f) if r["station"] == STATION]
    with open(skill_path, encoding="utf-8") as f:
        obs = next(r for r in csv.DictReader(f)
                   if r["station"] == STATION and r["reference"] == "observations")
    return math.sqrt(sum(d * d for d in days) / len(days)), len(days), float(obs["correlation"])


# --- drawing --------------------------------------------------------------------------


class Trace:
    """The strip along the bottom with the 15 days of water level at the gauge: the
    whole fortnight dim, the part up to the shown time bright, a marker at the shown
    time. It starts right of the viewer's colour legend (`--ui legend`, bottom left)."""

    X0, X1 = 450, 1610          # the plot's left and right (px)
    Y0, Y1 = 968, 1036          # its top and bottom

    def __init__(self, model, gauge, h_end):
        self.h_end = h_end
        mh, me = model
        gh, gl = gauge
        self.lim = max(np.abs(me).max(), np.abs(gl).max()) * 1.05
        self.model = (mh, me)
        self.base = Image.new("RGBA", (W, H))
        self.bright = Image.new("RGBA", (W, H))
        d = ImageDraw.Draw(self.base)
        d.rounded_rectangle((372, 930, 1900, 1072), radius=12, fill=(8, 11, 15, 185))
        # Days
        for day in range(0, int(h_end // 24) + 1):
            x = self.x(24 * day)
            d.line((x, self.Y0, x, self.Y1), fill=(255, 255, 255, 26), width=1)
            if day % 2 == 0:
                d.text((x, self.Y1 + 5), f"{day + 1} Jun", font=font(14), fill=DIM + (255,),
                       anchor="mt")
        # Metres
        for m in (-1, 0, 1):
            y = self.y(m)
            d.line((self.X0, y, self.X1, y), fill=(255, 255, 255, 36 if m else 64), width=1)
            d.text((self.X0 - 10, y), f"{m:+d} m" if m else "0", font=font(13), fill=DIM + (255,),
                   anchor="rm")
        # Title and legend, right of the plot
        lx = self.X1 + 30
        d.text((lx, 942), "Water level at Bodø", font=font(18, True), fill=TEXT + (255,))
        d.line((lx, 984, lx + 30, 984), fill=MODEL_C + (255,), width=4)
        d.text((lx + 42, 984), "model", font=font(16), fill=SOFT + (255,), anchor="lm")
        d.ellipse((lx + 11, 1010 - 4, lx + 19, 1010 + 4), fill=GAUGE_C + (255,))
        d.text((lx + 42, 1010), "tide gauge (Kartverket)", font=font(16), fill=SOFT + (255,),
               anchor="lm")
        # The two layers of the curves
        for layer, alpha in ((self.base, 64), (self.bright, 255)):
            dd = ImageDraw.Draw(layer)
            # The gauge's hours as beads under the model's line
            for h, level in zip(gh, gl):
                x, y = self.x(h), self.y(level)
                dd.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=GAUGE_C + (alpha,))
            pts = [(self.x(h), self.y(e)) for h, e in zip(mh, me) if h <= h_end]
            dd.line(pts, fill=MODEL_C + (alpha,), width=2, joint="curve")
        # Spring and neap tides: the days of the largest and smallest observed range
        days = {}
        for h, level in zip(gh, gl):
            days.setdefault(int(h // 24), []).append(level)
        ranges = {k: max(v) - min(v) for k, v in days.items() if len(v) >= 20}
        neap = min(ranges, key=ranges.get)
        springs = [max((k for k in ranges if k < neap), key=ranges.get, default=None),
                   max((k for k in ranges if k > neap), key=ranges.get, default=None)]
        for k, label in [(neap, "neap tide")] + [(k, "spring tide") for k in springs if k is not None]:
            x = min(max(self.x(24 * k + 12), self.X0 + 40), self.X1 - 40)
            d.text((x, self.Y0 - 6), label, font=font(14), fill=DIM + (255,), anchor="mb")
        self.ranges = ranges

    def x(self, h):
        return self.X0 + (self.X1 - self.X0) * h / self.h_end

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


def clock(img, h):
    when = START + dt.timedelta(hours=h)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((40, 36, 560, 150), radius=14, fill=(8, 11, 15, 170))
    d.text((64, 52), "Bodø, northern Norway", font=font(30, True), fill=TEXT + (255,))
    stamp = f"{when:%A} {when.day} {when:%B %Y}"
    d.text((64, 96), stamp, font=font(22), fill=SOFT + (255,))
    d.text((536, 96), f"{when:%H:%M} UTC", font=font(22), fill=SOFT + (255,), anchor="ra")


def card(lines, alpha):
    """A text card: (text, size, bold, colour, y) per line, faded by `alpha` (0..1)."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for text, size, bold, colour, y in lines:
        c = tuple(int(BG[i] + (colour[i] - BG[i]) * alpha) for i in range(3))
        d.text((W / 2, y), text, font=font(size, bold), fill=c, anchor="mm")
    return img


def ramp(t, d, rise=0.7, tail=0.6):
    """A card's text: in over `rise` s, out over the last `tail` s."""
    return max(0.0, min(1.0, t / rise, (d - t) / tail))


# --- the video ------------------------------------------------------------------------


def main():
    keys = camera_keys(CAMERA)
    files = sorted(f for f in os.listdir(FRAMES) if f.startswith("frame_") and f.endswith(".png"))
    if not files:
        sys.exit(f"no frame_*.png in {FRAMES}")
    h_end = keys[-1][1]
    model = model_series(STATIONS)
    gauge = gauge_series(GAUGE, h_end)
    trace = Trace(model, gauge, h_end)
    score4, n_days, r = scores(SCORES)
    spring = max(trace.ranges.values())

    title = [
        ("Fifteen days of tide at Bodø", 64, True, TEXT, 470),
        ("1–16 June 2026: from spring tide to neap tide and back", 32, False, SOFT, 560),
        ("dgcom, a discontinuous Galerkin coastal ocean model in Rust,", 24, False, DIM, 640),
        ("nested in MET Norway's NorKyst-800 · colour: current speed", 24, False, DIM, 676),
    ]
    end = [
        ("Checked against Kartverket's tide gauge at Bodø", 40, True, TEXT, 440),
        (f"A typical day's error: {100 * score4:.1f} cm, on a tide of up to {spring:.1f} m", 32, False,
         SOFT, 520),
        (f"(RMS of {n_days} daily centred RMS errors"
         + (f"; pre-registered prediction: {PREDICTED_CM:.1f} cm)" if PREDICTED_CM else ")"), 22,
         False, DIM, 568),
        (f"Correlation over the fortnight: {r:.3f}", 26, False, SOFT, 630),
        ("dgcom · discontinuous Galerkin coastal ocean model, written in Rust", 22, False, DIM,
         740),
    ]

    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def put(img):
        enc.stdin.write(np.asarray(img.convert("RGB")).tobytes())

    def shot(i):
        img = Image.open(os.path.join(FRAMES, files[i])).convert("RGBA")
        if img.size != (W, H):
            img = img.resize((W, H))
        h = model_hour(keys, i / FPS)
        clock(img, h)
        trace.draw(img, h)
        return img.convert("RGB")

    n_fade = int(FADE * FPS)
    n_title = int(TITLE_S * FPS)
    for k in range(n_title):
        t = k / FPS
        img = card(title, ramp(t, TITLE_S + FADE))
        if k >= n_title - n_fade:
            img = Image.blend(img, shot(0), (k - (n_title - n_fade) + 1) / (n_fade + 1))
        put(img)
    for i in range(len(files)):
        img = shot(i)
        if i >= len(files) - n_fade:
            a = (i - (len(files) - n_fade) + 1) / (n_fade + 1)
            img = Image.blend(img, card(end, 0.0), a)
        put(img)
        if (i + 1) % 300 == 0:
            print(f"  {i + 1} of {len(files)} frames", flush=True)
    for k in range(int(END_S * FPS)):
        put(card(end, ramp(k / FPS, END_S)))
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"wrote {OUT}: {(n_title + len(files) + int(END_S * FPS)) / FPS:.1f} s")


main()
