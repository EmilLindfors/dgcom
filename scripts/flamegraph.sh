#!/bin/bash
# Flamegraph profiling script for dgcom examples
#
# Prerequisites:
#   cargo install flamegraph
#   # On Linux, may need: sudo apt install linux-perf
#   # Or on Ubuntu: sudo apt install linux-tools-common linux-tools-$(uname -r)
#
# Usage:
#   ./scripts/flamegraph.sh froya_real_data          # Profile specific example
#   ./scripts/flamegraph.sh advection_1d             # Profile advection example
#   ./scripts/flamegraph.sh froya_real_data netcdf   # With features
#
# Output:
#   output/flamegraph/<example>_<timestamp>.svg

set -euo pipefail

EXAMPLE="${1:-froya_real_data}"
FEATURES="${2:-}"

# Create output directory
OUTPUT_DIR="output/flamegraph"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/${EXAMPLE}_${TIMESTAMP}.svg"

echo "========================================"
echo "  Flamegraph Profiling"
echo "========================================"
echo "  Example:  $EXAMPLE"
echo "  Features: ${FEATURES:-none}"
echo "  Output:   $OUTPUT_FILE"
echo "========================================"
echo

# Build feature flags
FEATURE_FLAGS=""
if [ -n "$FEATURES" ]; then
    FEATURE_FLAGS="--features $FEATURES"
fi

# Check if flamegraph is installed
if ! command -v cargo-flamegraph &> /dev/null && ! cargo flamegraph --help &> /dev/null 2>&1; then
    echo "Error: flamegraph not found. Install with:"
    echo "  cargo install flamegraph"
    echo
    echo "On Linux, you also need perf:"
    echo "  sudo apt install linux-perf"
    echo "  # or: sudo apt install linux-tools-common linux-tools-\$(uname -r)"
    exit 1
fi

# Check perf permissions on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PERF_PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
    if [ "$PERF_PARANOID" -gt 1 ]; then
        echo "Warning: perf_event_paranoid is $PERF_PARANOID (restricts profiling)"
        echo "For user-level profiling, run:"
        echo "  sudo sysctl kernel.perf_event_paranoid=1"
        echo
        echo "Attempting to continue (may require sudo)..."
        echo
    fi
fi

# Run flamegraph with the profiling profile
echo "Starting profiling..."

# Find perf binary (WSL2 workaround)
PERF_PATH=""
if [ -x "/usr/lib/linux-tools/6.8.0-90-generic/perf" ]; then
    PERF_PATH="/usr/lib/linux-tools/6.8.0-90-generic"
elif [ -d "/usr/lib/linux-tools" ]; then
    PERF_PATH=$(ls -d /usr/lib/linux-tools/*/  2>/dev/null | head -1)
fi

if [ -n "$PERF_PATH" ]; then
    echo "  Using perf from: $PERF_PATH"
    export PATH="$PERF_PATH:$PATH"
fi

echo "  cargo flamegraph --profile profiling --example $EXAMPLE $FEATURE_FLAGS -o $OUTPUT_FILE"
echo

cargo flamegraph \
    --profile profiling \
    --example "$EXAMPLE" \
    $FEATURE_FLAGS \
    -o "$OUTPUT_FILE"

echo
echo "========================================"
echo "  Profiling complete!"
echo "========================================"
echo "  Output: $OUTPUT_FILE"
echo
echo "View the flamegraph:"
echo "  firefox $OUTPUT_FILE"
echo "  # or: open $OUTPUT_FILE  (macOS)"
echo "  # or: xdg-open $OUTPUT_FILE  (Linux)"
