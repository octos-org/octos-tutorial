#!/usr/bin/env bash
# start_octos_serve.sh — Run octos serve + dora dataflow together
#
# Architecture (Approach B):
#   octos serve (agent + LLM)
#     ↕ MCP stdio
#   mcp_dora_bridge.py
#     ↕ Unix socket
#   dora dataflow (nav-bridge + sim + planning)
#
# Usage:
#   ./start_octos_serve.sh              # Start both
#   ./start_octos_serve.sh stop         # Stop both
#
# Prerequisites:
#   - octos CLI installed (cargo install --path crates/octos-cli)
#   - dora CLI installed
#   - Ollama running with a model

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.dora/bin:$HOME/.cargo/bin:$PATH"

cd "$SCRIPT_DIR"

if [ "${1:-}" = "stop" ]; then
    echo "Stopping..."
    dora destroy 2>/dev/null || true
    pkill -f "octos serve" 2>/dev/null || true
    rm -f /tmp/octos_dora_bridge.sock
    echo "Done"
    exit 0
fi

echo "============================================"
echo "  Octos Serve + Dora Dataflow (Approach B)"
echo "============================================"
echo ""

# 1. Start dora dataflow in background
echo "Starting dora dataflow..."
export MUJOCO_GL="${MUJOCO_GL:-cgl}"
dora up
dora start dataflow_nav_sim_py.yaml &
DORA_PID=$!
sleep 5

# 2. Start octos serve with MCP bridge config
echo ""
echo "Starting octos serve..."
echo "  Config: octos-config/config.json"
echo "  MCP bridge: octos-config/mcp_dora_bridge.py"
echo ""
echo "Open http://localhost:3141 for the web dashboard"
echo "Or use: octos chat (with config.json in current dir)"
echo ""

OCTOS_CONFIG="octos-config/config.json" octos serve

# Cleanup on exit
dora destroy 2>/dev/null || true
