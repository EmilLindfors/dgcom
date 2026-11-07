#!/bin/bash
# Samply profiling script for dgcom examples
#
# Samply is a modern sampling profiler with an excellent web UI.
# It's often easier to use than perf/flamegraph on Linux.
#
# Prerequisites:
#   cargo install samply
#
# Usage:
#   ./scripts/samply.sh froya_real_data
#   ./scripts/samply.sh froya_real_data netcdf

set -euo pipefail

EXAMPLE="${1:-froya_real_data}"
FEATURES="${2:-}"

echo "========================================"
echo "  Samply Profiling"
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

# Check if samply is installed
if ! command -v samply &> /dev/null; then
    echo "Error: samply not found. Install with:"
    echo "  cargo install samply"
    exit 1
fi

# Build the example first
echo "Building..."
cargo build --profile profiling --example "$EXAMPLE" $FEATURE_FLAGS

BINARY="target/profiling/examples/$EXAMPLE"

if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    exit 1
fi

echo
echo "Starting samply (will open browser UI)..."
echo

samply record "$BINARY"
