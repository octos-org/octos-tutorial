#!/usr/bin/env python3
"""MCP stdio server that bridges octos tool calls to a running dora dataflow.

octos spawns this as an MCP server. When octos calls a tool (e.g. navigate_to),
this server forwards the call to the dora dataflow's nav-bridge node via a
shared socket, and returns the result.

Protocol: MCP JSON-RPC over stdio (stdin/stdout).

Requires: a dora dataflow already running with nav-bridge listening on a
Unix socket at /tmp/octos_dora_bridge.sock.
"""

import json
import os
import socket
import sys


TOOL_MAP_PATH = os.environ.get("DORA_TOOL_MAP", "nav_tool_map.json")
BRIDGE_SOCK = "/tmp/octos_dora_bridge.sock"


def load_tool_map(path):
    """Load tool definitions from nav_tool_map.json."""
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("tools", [])
    except FileNotFoundError:
        return []


def tools_to_mcp_schema(tools):
    """Convert tool map entries to MCP tool list response."""
    mcp_tools = []
    for t in tools:
        schema = {
            "type": "object",
            "properties": {},
        }
        # navigate_to has a waypoint parameter
        if t["name"] == "navigate_to":
            schema["properties"]["waypoint"] = {
                "type": "string",
                "description": "Station name: A, B, or home",
            }
            schema["required"] = ["waypoint"]

        mcp_tools.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "inputSchema": schema,
        })
    return mcp_tools


def call_dora_tool(tool_name, args):
    """Forward a tool call to the dora dataflow via Unix socket.

    If the socket is not available (dora not running), return a simulated
    response so octos can still test the flow.
    """
    request = {"tool": tool_name, "args": args}

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(30.0)
        sock.connect(BRIDGE_SOCK)
        sock.sendall(json.dumps(request).encode() + b"\n")
        response = sock.recv(65536)
        sock.close()
        return json.loads(response.decode())
    except (socket.error, FileNotFoundError):
        # Dora not running — return simulated response
        return [
            f"[simulated] {tool_name}({args}) — dora dataflow not running. "
            f"Start the dataflow first: dora start dataflow_nav_sim_py.yaml",
            {"simulated": True, "tool": tool_name},
        ]


def handle_request(req):
    """Handle a single MCP JSON-RPC request."""
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "octos-dora-bridge",
                    "version": "0.1.0",
                },
            },
        }

    elif method == "notifications/initialized":
        return None  # No response for notifications

    elif method == "tools/list":
        tools = load_tool_map(TOOL_MAP_PATH)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools_to_mcp_schema(tools)},
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        result = call_dora_tool(tool_name, tool_args)

        content = result[0] if isinstance(result, list) else str(result)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": content}],
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main():
    """MCP stdio server main loop."""
    sys.stderr.write("[mcp-dora-bridge] Started\n")
    sys.stderr.write(f"[mcp-dora-bridge] Tool map: {TOOL_MAP_PATH}\n")
    sys.stderr.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_request(req)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
