#!/bin/bash
# Quick performance stats for dgcom examples
#
# Runs the example with perf stat to get CPU counters like:
# - Instructions, cycles, IPC
# - Cache hits/misses
# - Branch predictions
#
# Usage:
#   ./scripts/perf-stat.sh froya_real_data
#   ./scripts/perf-stat.sh froya_real_data netcdf

set -euo pipefail

EXAMPLE="${1:-froya_real_data}"
FEATURES="${2:-}"

echo "========================================"
echo "  Performance Statistics"
echo "========================================"
echo "  Example:  $EXAMPLE"
echo "  Features: ${FEATURES:-none}"
echo "========================================"
echo

# Build feature flags
FEATURE_FLAGS=""
if [ -n "$FEATURES" ]; then
    FEATURE_FLAGS="--features $FEATURES"
fi

# Build the example first (don't include compile time in stats)
echo "Building..."
cargo build --profile profiling --example "$EXAMPLE" $FEATURE_FLAGS

BINARY="target/profiling/examples/$EXAMPLE"

if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    exit 1
fi

echo
echo "Running with perf stat..."
echo

perf stat -d "$BINARY"
