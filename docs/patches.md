# Developing on patches

A change to dgcom (a scheme, a switch, a kernel) is judged on the runs it
is for: Frøya's 3D rivers case, the coastline mesh, Chiloé. The whole
domain takes from minutes on a card to hours on the CPU. Most changes act
locally: a shore element, a river mouth, a farm, a cliff. A **patch** runs
the case on a cut-out of its mesh (`[mesh.patch]`, `dgcom::case::Patch`).
Nested in the whole run's own output, it replays that run inside the cut.
So one whole run gives a reference, and every variant after it costs a
patch's run.

The measurements below are from 2026-11-28 (`notes/case-files.md`, P8.14),
on the laptop's CPU with 24 threads.

## The loop

1. **The reference.** Run the whole case once over the window you care about,
   with snapshots often enough for the cut: every 1–2 minutes, or every
   step's landing. A restart a little before the window starts it there:

   ```bash
   dgcom run cases/froya/rivers_3d.toml --resume <restart at 46 h> \
       --set run.duration_hours=47 --set output.interval_minutes=2 \
       --set output.directory='"output/ref"' \
       --set "restart.changes=['output.interval_minutes']"
   ```

   An hour of Frøya's 3D case at 2-minute frames is 0.9 GB of snapshot.
   The steps land on the frames, so the reference steps at most 120 s.

2. **The patch, nested in the reference.** `whole_run` is the reference's
   snapshot. Every open face of the patch takes the whole run's state there:
   at a face the cut made, that is the dropped neighbour's value at the same
   point. A band inside the faces (`band_km`, default 1; `band_minutes`,
   default 30) relaxes to the whole run's state. Use the same restart and
   output interval:

   ```bash
   dgcom run cases/froya/rivers_3d.toml --resume <restart at 46 h> \
       --set run.duration_hours=47 --set output.interval_minutes=2 \
       --set "mesh.patch.center={lat=63.869331,lon=8.665231}" --set mesh.patch.radius_km=5 \
       --set mesh.patch.whole_run='"output/ref/froya_rivers_3d.dgsnap"' \
       --set output.directory='"output/patch_a"' \
       --set "restart.changes=['output.interval_minutes']"
   ```

   A patch can also start from rest at time 0, if the whole run did. Paths
   in a case are relative to its file (`whole_run` and `output.directory`
   both land under `cases/froya/` here); `--resume` is relative to the
   working directory.

3. **The change, on the same patch.** Rebuild with the change, or set the
   switch, and run step 2 into `output/patch_b`. Compare the two patch
   runs node by node; they share the mesh and the boundary data, so the
   patch's own error cancels:

   ```bash
   cargo run --release --example snapshot_compare_3d -- \
       output/patch_a/froya_rivers_3d.dgsnap output/patch_b/froya_rivers_3d.dgsnap \
       points=@output/radical/tide/points.txt
   ```

4. **Before merging**, judge the change on the whole run as before. A patch
   screens changes; it does not replace the scored runs.

`patch_effect_3d` checks whether a patch sees a change as the whole run
does. Give it the change on both: the whole runs A and B, and the patch
runs a and b, both patches nested in A. It prints, at each point, the
whole run's effect, the patch's, and their difference.

## How close a patch is

At Frøya's Mausund gauge, 5 km around it (322 of 9,848 elements), from the
46 h restart, over the hour:

| | nested in NorKyst at the cut | nested in the whole run |
|---|---|---|
| Mausund η (largest difference) | 52 mm | 0.22 mm |
| Kattholmen, 0.6 km from the centre: η, ū (RMS) | 1.9 mm, 3.0 mm/s | 0.07 mm, 0.02 mm/s |
| Oløya_N, 0.9 km inside the cut: η, ū (RMS) | 139 mm, 65 mm/s | 3.1 mm, 2.8 mm/s |
| stepping, wall time (whole run: 42.0 s, 49.3 s) | 3.3 s, 11.5 s | 3.0 s, 8.2 s |

- **Keep the points of interest away from the cut.** Within about a
  kilometre of it, a point carries the boundary condition's error. The
  characteristic condition is not the interior's Riemann flux, and turbulence
  does not cross the cut. The band's width hardly changes this: 0, 0.5, 1
  and 2 km gave 3.0–3.2 mm at Oløya_N. A 10-minute band time takes its T
  error from 45 to 15 m°C.
- **Snapshot often enough for what crosses the cut.** Between frames the
  state at the cut is linear in time. A tide's ramped front sweeping the
  3D test channel in 15 minutes was 20 mm off at 5-minute frames and 0.9 mm
  at 1-minute frames (`a_3d_patch_nested_in_its_whole_run_replays_it`).
- **A patch reproduces a change's local effect, not its remote one.** Raising
  the bed roughness from 3 to 10 mm, the whole run's change at Kattholmen
  against the 5 km patch's:

  | | whole run | 5 km patch |
  |---|---|---|
  | ū | 0.053 mm/s | 0.047 mm/s (0.014 off) |
  | surface u | 0.171 mm/s | 0.162 mm/s (0.018 off) |
  | T | 0.200 m°C | 0.181 m°C (0.030 off) |
  | η | 0.53 mm | 0.16 mm |

  η is the whole domain's tide, and the patch's boundary holds it at the
  reference's. A 10 km patch gets 0.505 of the 0.526 mm. Size a patch for
  the reach of what the change moves.
- The patch's own error against the whole run (0.45 mm/s in surface u at
  Kattholmen) is larger than that effect. Yet the effect comes out right,
  because the error is the same in both patch runs. That is why step 3
  compares patch with patch.

## Patches for optimizing

The patch keeps the whole run's elements and bed, so its kernels do the
whole run's work on fewer elements. Its fixed costs per step stay, about
46 ms on the Frøya case (a fit over patch sizes). Per element-step, against
the whole run's 142 µs:

| radius | elements | µs per element-step |
|---|---|---|
| 3 km | 116 | 546 |
| 5 km | 322 | 311 |
| 10 km | 1,278 | 206 |
| 15 km | 2,739 | 174 |
| 20 km | 4,316 | 164 |

- **Timing:** use a patch of a few thousand elements that holds the mix of
  elements the whole run has (shores, deep water, rivers). Compare builds on
  the same patch, alternating runs as `notes/performance.md` describes for the
  whole runs. `DG_PROFILE=1` prints the phase timers on a patch as on the
  whole run.
- **Finding a hot spot or a blow-up:** a patch of a few hundred elements is
  enough. Use `element = K` with the index the run reports, and `faces =
  "wall"` for a closed basin at rest.
- **The build:** a patch rebuilds the whole bed first, 4.7 s of the
  Mausund patch's 8.2 s, of which 2.5 s is the 3D bed's smoothing of the
  whole mesh. A cache is parked (`notes/case-files.md`).
- **GPU:** `dgcom-gpu run … --set compute.device=cuda` runs a patch nested
  in its whole run. Its open faces and its 2D band take the device's parent
  kernels. A 3D patch's band columns are evaluated on the host and uploaded:
  at most 8–9 % of the stepping, on the 5 and 20 km patches. The results are
  the CPU's bit for bit in a build without multiply–add contraction
  (`dgcom-gpu patch`; the Mausund patch's hour too). On the laptop's
  RTX 4060 the card pays from about a thousand elements up:

  | patch | elements | CPU (24 threads) | RTX 4060 |
  |---|---|---|---|
  | 3 km | 116 | 1.9 s | 2.5 s |
  | 5 km | 322 | 3.0 s | 3.1 s |
  | 10 km | 1,278 | 7.9 s | 6.6 s |
  | 15 km | 2,739 | 14.3 s | 10.4 s |
  | 20 km | 4,316 | 21.2 s | 16.2 s |
  | whole | 9,848 | 41.3 s | 24.4 s |

  That is the stepping of the hour (46 → 47 h), the FMA build. The patches
  take 30 steps of 120 s, landing on their 2-minute frames. The whole run
  takes 19 steps of up to 200 s with 10-minute frames.

## Checks

- `tests/case_patch_test.rs` checks the cut against the whole bed, seeding
  from a 2D and a 3D restart bit for bit, and the replay in 2D and 3D. The
  2D replay is 0.66 mm in η against 427 mm with the case's tide at the cut;
  the 3D one is 0.86 mm against 492 mm walled.
- A mutant that holds the frames instead of interpolating them fails both
  replays.
- `dgcom-gpu patch` (the GPU crate, `gpu/`) runs the tidal channel's middle
  6 km nested in its whole run on the CPU and on the GPU, 2D and 3D. The two
  agree bit for bit without multiply–add contraction, and within 1e-7 with
  it. Its mutant nests the GPU's patch in a whole run with M2 1 cm higher.
