#!/usr/bin/env bash
# start.sh — Launch the SLAM navigation visual simulation example
#
# Usage:
#   ./start.sh              # Manual mode (robot follows full waypoint path)
#   ./start.sh stop         # Stop running dataflow
#
# Prerequisites:
#   - conda env: robobrain2
#   - dora-rs CLI in ~/.cargo/bin
#   - dora-nav C++ nodes built at /home/demo/Public/dora-nav
#   - dora-mujoco at /home/demo/Public/dora-moveit2/dora-mujoco
#   - MuJoCo model at /home/demo/Public/octos_inspection/models/

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environment setup
export PATH="/home/demo/.cargo/bin:$PATH"
# Conda activate needs PS1 set (nounset would fail otherwise)
export PS1="${PS1:-}"
source /home/demo/anaconda3/bin/activate robobrain2
set -u

# Display for MuJoCo GL and Rerun
export DISPLAY="${DISPLAY:-:0}"

# Paths
export DORA_NAV_PATH="${DORA_NAV_PATH:-/home/demo/Public/dora-nav}"
export DORA_MUJOCO_PATH="${DORA_MUJOCO_PATH:-/home/demo/Public/dora-moveit2/dora-mujoco}"
export MODEL_NAME="${MODEL_NAME:-/home/demo/Public/octos_inspection/models/hunter_se_warehouse.xml}"
export MAP_PCD_FILE="${MAP_PCD_FILE:-/home/demo/Public/dora-nav/data/map.pcd}"

cd "$SCRIPT_DIR"

# Ensure Waypoints.txt symlink exists
if [ ! -f Waypoints.txt ]; then
    ln -sf /home/demo/Public/octos_inspection/Waypoints.txt Waypoints.txt
fi

# Handle stop command
if [ "${1:-}" = "stop" ]; then
    echo "Stopping dora dataflow..."
    dora destroy
    exit 0
fi

echo "============================================"
echo "  SLAM Navigation Visual Simulation"
echo "============================================"
echo "  dora-nav:   $DORA_NAV_PATH"
echo "  mujoco:     $DORA_MUJOCO_PATH"
echo "  model:      $MODEL_NAME"
echo "  display:    $DISPLAY"
echo "============================================"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

dora up
dora start dataflow_nav_sim.yaml --attach
