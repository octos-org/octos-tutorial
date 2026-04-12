#!/usr/bin/env bash
# setup.sh — Verify prerequisites and install dependencies
set -eo pipefail

echo "=== Octos Tutorial Setup ==="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi
echo "Python: $(python3 --version)"

# Check dora
if command -v dora &>/dev/null; then
    echo "Dora:   $(dora --version)"
else
    echo "Dora:   not found — installing via pip..."
    pip install dora-rs
fi

# Install Python deps
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify imports
echo ""
echo "Verifying imports..."
python3 -c "import dora; print('  dora-rs: ok')" 2>&1 || echo "  dora-rs: FAILED"
python3 -c "import pyarrow; print('  pyarrow: ok')" 2>&1 || echo "  pyarrow: FAILED"
python3 -c "import numpy; print('  numpy: ok')" 2>&1 || echo "  numpy: FAILED"

# Optional deps
echo ""
echo "Optional dependencies:"
python3 -c "import mujoco; print('  mujoco: ok')" 2>&1 || echo "  mujoco: not installed (needed for 05-slam-nav-sim)"
python3 -c "import rerun; print('  rerun: ok')" 2>&1 || echo "  rerun: not installed (needed for 05-slam-nav-sim)"
command -v ollama &>/dev/null && echo "  ollama: ok" || echo "  ollama: not installed (needed for 03-llm-agent)"

echo ""
echo "=== Setup complete ==="
echo "Run: cd 01-pipeline-basics && dora up && dora start dataflow.yaml --attach"
