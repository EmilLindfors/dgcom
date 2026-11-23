#!/usr/bin/env bash
# Gates and times dgcom-gpu on this machine's GPU against its CPU (P2.7;
# gpu/README.md, "Timing a card"). From any directory, with README.md's
# toolchain installed:
#
#   gpu/time.sh                                  # gates, timing, profile
#   REPEATS=5 SKIP_GATES=1 gpu/time.sh
#   SIZES="800x80 1600x160 3200x320" gpu/time.sh
#   FORCE=1 gpu/time.sh                          # time on a busy GPU anyway
#
# 1. The gates, built with --no-fmad, against the SIMD batch: the RHS and
#    Simulation runs and the nested runs (`all`), the drying beach (`beach`) and
#    cases/tidal_channel through the case runner (`case`), also with local
#    time stepping (`case_lts`) and forced by a tidal atlas (`case_atlas`).
#    On any card each must be the CPU's bit for bit.
# 2. The timing, built with FMA: the refined tidal channel at each size, on
#    the CPU and on the GPU, REPEATS times; and the 3D transport, momentum
#    terms, stage hooks, vertical step and implicit vertical advection
#    (`transport3d`, `momentum3d`, `stage3d`, `limiter3d`, `vertical3d`,
#    `implicit3d`) at 96,000 P2 elements × 20 levels.
# 3. Nsight Systems' kernel times on the first size, if nsys is found.
#
# Everything goes to output/gpu_timing/<gpu>_<date>/, which git ignores:
# each command's log, and summary.md with the numbers for notes/gpu.md.
set -euo pipefail
export LC_ALL=C

SIZES=${SIZES:-"800x80 1600x160"} # channel meshes, <nx>x<ny> elements (the example is 40x4)
REPEATS=${REPEATS:-3}
SKIP_GATES=${SKIP_GATES:-0}
SKIP_PROFILE=${SKIP_PROFILE:-0}
FORCE=${FORCE:-0}

gpu_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(dirname "$gpu_dir")
case_file="$repo/cases/tidal_channel/case.toml"
atlas_file="$repo/cases/tidal_channel/atlas.toml"
exe="$gpu_dir/target/release/dgcom-gpu"
case $(uname -s) in MINGW* | MSYS*) exe+=".exe" ;; esac

# README.md's environment, where the shell has not set it.
if [[ -z ${CUDA_TOOLKIT_PATH:-} && -d /usr/local/cuda ]]; then
    export CUDA_TOOLKIT_PATH=/usr/local/cuda
fi
if ! command -v nvcc >/dev/null && [[ -n ${CUDA_TOOLKIT_PATH:-} ]]; then
    export PATH="$CUDA_TOOLKIT_PATH/bin:$PATH"
fi

query() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits -i 0 | head -1 | tr -d '\r' | sed 's/^ *//; s/ *$//'; }
gpu_name=$(query name)
driver=$(query driver_version)
arch="sm_$(query compute_cap | tr -d .)"
# The GPU is shared: time only on an idle one.
busiest=0
for _ in 1 2 3 4 5; do
    u=$(query utilization.gpu)
    ((u > busiest)) && busiest=$u
    sleep 0.4
done
if ((busiest > 10 && FORCE == 0)); then
    nvidia-smi
    echo "the GPU is $busiest % busy: stop what runs on it, or set FORCE=1 (the times will be off)" >&2
    exit 1
fi
cpu=$(awk -F': *' '/^model name/ { print $2; exit }' /proc/cpuinfo | sed 's/ *$//')
threads=$(nproc)
commit="$(git -C "$repo" rev-parse --abbrev-ref HEAD) @ $(git -C "$repo" rev-parse --short HEAD)"
oxide=$(cd "$gpu_dir" && cargo oxide --version 2>/dev/null | head -1 || true)

tag=$(printf '%s' "$gpu_name" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_*//; s/_*$//')
out="$repo/output/gpu_timing/${tag}_$(date +%Y-%m-%d_%H%M)"
mkdir -p "$out"
summary="$out/summary.md"
: >"$summary"
say() { printf '%s\n' "$*" >>"$summary"; }
echo "dgcom-gpu on $gpu_name ($arch, driver $driver), CPU $cpu ($threads threads), $commit"
echo "logs in $out"

# Runs a command in gpu/, its output to the screen and to <name>.log; sets
# $code to its exit code.
run_logged() {
    local name=$1
    shift
    echo "== $name"
    set +e
    (cd "$gpu_dir" && "$@") 2>&1 | tee "$out/$name.log"
    code=${PIPESTATUS[0]}
    set -e
}
build() {
    local name=$1
    shift
    run_logged "$name" cargo oxide build "$@"
    if ((code != 0)); then
        echo "cargo oxide build $* failed: $out/$name.log" >&2
        exit 1
    fi
}
hours_for() { awk -v nx="$1" 'BEGIN { print 0.25 * 800 / nx }'; } # about 1,150 steps at any size

say "# dgcom-gpu on $gpu_name, $(date +%Y-%m-%d)"
say ""
say "- GPU: $gpu_name ($arch), driver $driver; at most $busiest % busy before the runs"
say "- CPU: $cpu, $threads threads (the SIMD batch on all of them)"
say "- Commit: $commit; $oxide"
say ""

# 1. The gates: every node, step and record bit for bit.
if ((SKIP_GATES == 0)); then
    build build_nofmad --no-fmad --features simd
    say "## Gates (--no-fmad, against the SIMD batch)"
    say ""
    say "| gate | result |"
    say "|---|---|"
    for gate in all beach case case_lts case_atlas; do
        if [[ $gate == case ]]; then
            run_logged "gate_$gate" "$exe" case "$case_file"
        elif [[ $gate == case_lts ]]; then
            run_logged "gate_$gate" "$exe" case "$case_file" --set numerics.local_time_stepping_levels=3
        elif [[ $gate == case_atlas ]]; then
            run_logged "gate_$gate" "$exe" case "$atlas_file"
        else
            run_logged "gate_$gate" "$exe" "$gate"
        fi
        log="$out/gate_$gate.log"
        compared=$({ grep -oE '[0-9]+ of [0-9]+ nodes bit-identical' "$log" || true; } | wc -l)
        partial=$({ grep -oE '[0-9]+ of [0-9]+ nodes bit-identical' "$log" || true; } | awk '$1 != $3' | wc -l)
        result="bit for bit"
        if ((code != 0 || partial > 0)); then result="**differs** (gate_$gate.log)"; fi
        if [[ $gate == case* ]] && ! grep -q 'bit for bit' "$log"; then result="**differs** (gate_$gate.log)"; fi
        # The 3D transport's fixtures (in `all`) print their own verdict
        if grep -q 'NOT bit for bit\|NOT CAUGHT' "$log"; then result="**differs** (gate_$gate.log)"; fi
        if [[ $gate == case ]]; then
            what="steps, volume, clips and station records"
        elif [[ $gate == case_lts ]]; then
            what="the same with local time stepping (3 levels)"
        elif [[ $gate == case_atlas ]]; then
            what="the same forced by a tidal atlas node by node"
        else
            what="every node of $compared state$( ((compared == 1)) || echo s)"
        fi
        say "| \`dgcom-gpu $gate\` | $result: $what |"
    done
    say ""
fi

# 2. The timing, with FMA: each run's wall time over its steps.
build build_fma --features simd
rows=()
for size in $SIZES; do
    nx=${size%x*}
    ny=${size#*x}
    for ((i = 1; i <= REPEATS; i++)); do
        run_logged "time_${size}_$i" "$exe" case "$case_file" \
            --set "mesh.rectangle.nx=$nx" --set "mesh.rectangle.ny=$ny" \
            --set "run.duration_hours=$(hours_for "$nx")"
        line=$(grep -oE '[0-9]+ steps on the CPU \([0-9.]+ s\) and [0-9]+ on the GPU \([0-9.]+ s\)' \
            "$out/time_${size}_$i.log" | head -1 || true)
        if [[ -z $line ]]; then
            echo "no timing line in time_${size}_$i.log" >&2
            continue
        fi
        read -r cpu_steps cpu_s gpu_steps gpu_s < <(printf '%s\n' "$line" | tr -d '()' |
            awk '{ print $1, $6, $9, $13 }')
        rows+=("$(awk -v size="$size" -v el=$((nx * ny)) -v run="$i" -v cs="$cpu_steps" -v ct="$cpu_s" \
            -v gs="$gpu_steps" -v gt="$gpu_s" 'BEGIN {
                c = 1000 * ct / cs; g = 1000 * gt / gs
                printf "| %s | %d | %d | %d | %.2f | %.2f | %.2f |", size, el, run, gs, c, g, c / g }')")
    done
done
say "## Time per step (FMA build; each run's wall time over its steps, its start-up included)"
say ""
say "| mesh | elements | run | steps | CPU ms/step | GPU ms/step | CPU / GPU |"
say "|---|---|---|---|---|---|---|"
for row in "${rows[@]}"; do say "$row"; done
say ""
say "A build's first GPU run also compiles its PTX for $arch (the driver caches it): judge by the later runs."
say ""

# 2b. The 3D stages' transport, momentum terms and hooks (Phase 4), with FMA:
# per element and level at 96,000 P2 elements × 20 levels.
parts="transport3d momentum3d stage3d limiter3d vertical3d implicit3d forcing3d pass3d"
for part in $parts; do
    run_logged "time_$part" "$exe" "$part"
done
say "## The 3D transport, momentum terms and stage hooks (FMA build; 96,000 P2 elements × 20 levels, per element and level)"
say ""
say '```'
for part in $parts; do
    grep -E '^timing, |per element and level|ns \|' "$out/time_$part.log" >>"$summary" || say "(no timing lines: time_$part.log)"
done
say '```'
say ""

# 3. The kernels, from Nsight Systems.
nsys=$(command -v nsys || true)
if [[ -z $nsys ]]; then
    nsys=$(ls -d /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys \
        "/c/Program Files/NVIDIA Corporation/Nsight Systems "*/target-windows-x64/nsys.exe 2>/dev/null |
        sort | tail -1 || true)
fi
if ((SKIP_PROFILE != 0)); then
    :
elif [[ -z $nsys ]]; then
    say "Nsight Systems (nsys) not found: no kernel times."
else
    size=${SIZES%% *}
    nx=${size%x*}
    ny=${size#*x}
    run_logged "profile_$size" "$nsys" profile --trace=cuda --force-overwrite=true -o "$out/profile_$size" \
        "$exe" run "$case_file" --set compute.device=cuda \
        --set "mesh.rectangle.nx=$nx" --set "mesh.rectangle.ny=$ny" \
        --set "run.duration_hours=$(hours_for "$nx")" --set output.directory=output/gpu_timing
    steps=$(grep -oE 'done: [0-9]+ steps' "$out/profile_$size.log" | head -1 | awk '{ print $2 }' || true)
    if [[ -z $steps ]]; then
        steps=$(grep -oE '[0-9]+ on the GPU' "$out/time_${size}_1.log" 2>/dev/null | head -1 | awk '{ print $1 }' || true)
    fi
    run_logged "kernels_$size" "$nsys" stats --report cuda_gpu_kern_sum "$out/profile_$size.nsys-rep"
    run_logged "kernels_${size}_csv" "$nsys" stats --report cuda_gpu_kern_sum --format csv \
        --force-export=true -o "$out/kernels_$size" "$out/profile_$size.nsys-rep"
    csv="$out/kernels_${size}_cuda_gpu_kern_sum.csv"
    if [[ -f $csv && -n $steps ]]; then
        say "## GPU kernels per step ($size, Nsight Systems, $steps steps)"
        say ""
        say "| kernel | ms per step | share |"
        say "|---|---|---|"
        # Columns: Time (%), Total Time (ns), Instances, Avg, Med, Min, Max, StdDev, Name.
        awk -F, -v steps="$steps" 'NR > 1 {
                gsub(/"/, ""); name = $9; for (i = 10; i <= NF; i++) name = name "," $i
                sub(/\(.*$/, "", name); ns[NR] = $2; names[NR] = name; total += $2; n = NR }
            END {
                for (i = 2; i <= n; i++) printf "| `%s` | %.3f | %.0f %% |\n", names[i], ns[i] / 1e6 / steps, 100 * ns[i] / total
                printf "| all | %.2f | |\n", total / 1e6 / steps }' "$csv" >>"$summary"
    else
        say "Nsight Systems ran (exit $code); its kernel table is in kernels_$size.log."
    fi
fi

echo
cat "$summary"
echo
echo "summary: $summary"
