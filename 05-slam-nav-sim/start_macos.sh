#!/usr/bin/env bash
# start_macos.sh — Launch SLAM nav sim on macOS (all-Python, no C++ required)
#
# Usage:
#   ./start_macos.sh              # Start the simulation
#   ./start_macos.sh stop         # Stop running dataflow
#
# Prerequisites:
#   pip install dora-rs mujoco rerun-sdk pyarrow numpy
#   dora CLI installed (~/.dora/bin/dora)
#   dora-mujoco cloned locally

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.dora/bin:$HOME/.cargo/bin:$PATH"

cd "$SCRIPT_DIR"

# Handle stop command
if [ "${1:-}" = "stop" ]; then
    echo "Stopping dora dataflow..."
    dora destroy
    exit 0
fi

# Check required Python packages
python3 -c "import dora, mujoco, rerun, pyarrow, numpy" 2>/dev/null || {
    echo "ERROR: Missing Python packages. Install with:"
    echo "  pip install dora-rs mujoco rerun-sdk pyarrow numpy"
    exit 1
}

# Check dora CLI
command -v dora >/dev/null 2>&1 || {
    echo "ERROR: dora CLI not found. Install with:"
    echo "  cargo install dora-cli"
    echo "  or: pip install dora-rs"
    exit 1
}

# Check MuJoCo sim path
DORA_MUJOCO_PATH="${DORA_MUJOCO_PATH:-}"
if [ -z "$DORA_MUJOCO_PATH" ]; then
    # Try common locations
    for candidate in \
        "$HOME/Public/dora-moveit2/dora-mujoco" \
        "$HOME/Public/github_dorarobotics/dora-moveit2/dora-mujoco" \
        "$HOME/dora-moveit2/dora-mujoco"; do
        if [ -f "$candidate/dora_mujoco/main.py" ]; then
            DORA_MUJOCO_PATH="$candidate"
            break
        fi
    done
fi

if [ -z "$DORA_MUJOCO_PATH" ] || [ ! -f "$DORA_MUJOCO_PATH/dora_mujoco/main.py" ]; then
    echo "ERROR: dora-mujoco not found."
    echo "  Clone it: git clone https://github.com/dorarobotics/dora-moveit2.git"
    echo "  Then set: export DORA_MUJOCO_PATH=/path/to/dora-moveit2/dora-mujoco"
    exit 1
fi
export DORA_MUJOCO_PATH

# Check MuJoCo model
MODEL_NAME="${MODEL_NAME:-}"
if [ -z "$MODEL_NAME" ]; then
    # Try common locations for the warehouse model
    for candidate in \
        "$SCRIPT_DIR/models/hunter_se_warehouse.xml" \
        "$HOME/Public/octos_inspection/models/hunter_se_warehouse.xml" \
        "$HOME/Public/github_dorarobotics/octos_inspection/models/hunter_se_warehouse.xml"; do
        if [ -f "$candidate" ]; then
            MODEL_NAME="$candidate"
            break
        fi
    done
fi

if [ -z "$MODEL_NAME" ] || [ ! -f "$MODEL_NAME" ]; then
    echo "ERROR: MuJoCo model not found."
    echo "  Set: export MODEL_NAME=/path/to/hunter_se_warehouse.xml"
    exit 1
fi
export MODEL_NAME

# Ensure Waypoints.txt exists
if [ ! -f Waypoints.txt ]; then
    for candidate in \
        "$HOME/Public/octos_inspection/Waypoints.txt" \
        "$HOME/Public/github_dorarobotics/octos_inspection/Waypoints.txt"; do
        if [ -f "$candidate" ]; then
            ln -sf "$candidate" Waypoints.txt
            break
        fi
    done
fi

if [ ! -f Waypoints.txt ]; then
    echo "ERROR: Waypoints.txt not found."
    echo "  Copy or symlink from octos_inspection."
    exit 1
fi

# macOS: use CGL for MuJoCo rendering
if [ "$(uname)" = "Darwin" ]; then
    export MUJOCO_GL="${MUJOCO_GL:-cgl}"
fi

echo "============================================"
echo "  SLAM Navigation Simulation (All-Python)"
echo "============================================"
echo "  mujoco:     $DORA_MUJOCO_PATH"
echo "  model:      $MODEL_NAME"
echo "  platform:   $(uname -s) $(uname -m)"
echo "============================================"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

dora up
dora start dataflow_nav_sim_py.yaml --attach
