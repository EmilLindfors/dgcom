"""Assemble the Frøya showcase video from the shots `scripts/froya_video.sh` recorded:
a title card, four captioned shots with crossfades and an end card, H.264 at 1080p30.

    uv run python scripts/froya_video/assemble.py <frames dir with 1/ 2/ 3/ 4/> <out.mp4>

The captions go through drawtext's textfile=, so they need no escaping. Fonts: Segoe UI
(Windows)."""
import os
import subprocess
import sys

V, OUT = sys.argv[1], sys.argv[2]
FPS = 30
FADE = 0.6
FONT = "C\\:/Windows/Fonts/segoeui.ttf"
BOLD = "C\\:/Windows/Fonts/segoeuib.ttf"
LIGHT = "C\\:/Windows/Fonts/segoeuisl.ttf"
BG = "0x0b0e12"

shots = [
    ("1", "Tidal currents around Frøya, Hitra and Smøla",
     "3D model (20 layers) nested in MET Norway's NorKyst-800, driven by MET weather"),
    ("2", "Currents through a salmon farm",
     "Kattholmen, Frøya: the tide turning around the cages"),
    ("3", "A storm reaching the coast",
     "Wave height on 9 October 2025: a spectral wave model coupled to the tide"),
    ("4", "The same storm at sea level",
     "The model's waves and currents, rendered at the farm"),
]
title = ("Simulating the Norwegian coast",
         "Tides, 3D currents and waves around Frøya, mid-Norway",
         "A discontinuous Galerkin ocean model, written in Rust")
end = ("dgcom",
       "Discontinuous Galerkin coastal ocean model, written in Rust",
       "Nested in NorKyst-800 · checked against Kartverket's tide gauges")
TITLE_S, END_S = 3.5, 4.0

texts = os.path.join(V, "texts")
os.makedirs(texts, exist_ok=True)


def textfile(name, s):
    path = os.path.join(texts, name + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    return path.replace("\\", "/").replace(":", "\\:")


def fade_alpha(d, start=0.3, rise=0.7, tail=0.9):
    return (f"if(lt(t,{start}),0,if(lt(t,{start + rise}),(t-{start})/{rise},"
            f"if(lt(t,{d - tail}),1,max(0,({d - FADE / 2}-t)/{tail - FADE / 2}))))")


def card(lines, d, sizes, fonts, ys, label):
    f = f"color=c={BG}:s=1920x1080:r={FPS}:d={d}"
    for i, (s, size, font, y) in enumerate(zip(lines, sizes, fonts, ys)):
        tf = textfile(f"{label}_{i}", s)
        a = fade_alpha(d, start=0.2 + 0.25 * i, rise=0.6, tail=0.8)
        colour = "white" if i == 0 else ("0xc9d4df" if i == 1 else "0x8fa3b5")
        f += (f",drawtext=fontfile='{font}':textfile='{tf}':fontsize={size}:"
              f"fontcolor={colour}:x=(w-text_w)/2:y={y}:alpha='{a}'")
    return f + ",format=yuv420p,setsar=1"


def frames(d):
    return len([n for n in os.listdir(os.path.join(V, d)) if n.endswith(".png")])


inputs, chains, durations = [], [], []
chains.append(card(title, TITLE_S, [76, 38, 30], [BOLD, FONT, LIGHT], [400, 510, 570], "title") + "[c0]")
durations.append(TITLE_S)
for k, (d, head, sub) in enumerate(shots, start=1):
    n = frames(d)
    dur = n / FPS
    inputs += ["-framerate", str(FPS), "-i", os.path.join(V, d, "frame_%05d.png")]
    a = fade_alpha(dur, start=0.5, rise=0.6, tail=1.0)
    th, ts = textfile(f"s{d}_head", head), textfile(f"s{d}_sub", sub)
    chains.append(
        f"[{k - 1}:v]format=yuv420p,setsar=1,"
        f"drawtext=fontfile='{BOLD}':textfile='{th}':fontsize=46:fontcolor=white:"
        f"x=64:y=52:box=1:boxcolor=black@0.42:boxborderw=14:alpha='{a}',"
        f"drawtext=fontfile='{FONT}':textfile='{ts}':fontsize=28:fontcolor=0xe6edf3:"
        f"x=64:y=132:box=1:boxcolor=black@0.42:boxborderw=12:alpha='{a}'[c{k}]")
    durations.append(dur)
chains.append(card(end, END_S, [84, 36, 28], [BOLD, FONT, LIGHT], [400, 510, 570], "end") + f"[c{len(shots) + 1}]")
durations.append(END_S)

# Crossfade the pieces in order
last, length = "c0", durations[0]
for k in range(1, len(durations)):
    offset = length - FADE
    out = f"x{k}"
    chains.append(f"[{last}][c{k}]xfade=transition=fade:duration={FADE}:offset={offset:.3f}[{out}]")
    last, length = out, offset + durations[k]

out = OUT
cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(chains),
       "-map", f"[{last}]", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
       "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", str(FPS),
       "-movflags", "+faststart", out]
print(f"{length:.1f} s")
sys.exit(subprocess.call(cmd))
