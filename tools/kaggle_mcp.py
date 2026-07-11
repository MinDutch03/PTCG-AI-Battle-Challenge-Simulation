"""Minimal client for the Kaggle MCP server (https://www.kaggle.com/mcp).

Speaks JSON-RPC over streamable HTTP directly — no MCP client runtime needed.
Reads KAGGLE_API_TOKEN from the environment or a .env file in the repo root.

Usage:
    python tools/kaggle_mcp.py <tool_name> '<json_args>'
    python tools/kaggle_mcp.py list_submission_episodes '{"competition":"pokemon-tcg-ai-battle"}'
"""

import json
import os
import subprocess
import sys

URL = "https://www.kaggle.com/mcp"


def _token() -> str:
    tok = os.environ.get("KAGGLE_API_TOKEN")
    if not tok:
        env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env):
            for line in open(env):
                if line.startswith("KAGGLE_API_TOKEN"):
                    tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        sys.exit("KAGGLE_API_TOKEN not set (env or .env)")
    return tok


def call(tool: str, arguments: dict, timeout: int = 120):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    # system curl (macOS python.org builds often lack root certs for urllib)
    body = subprocess.run(
        ["curl", "-sS", "--max-time", str(timeout), "-X", "POST", URL,
         "-H", f"Authorization: Bearer {_token()}",
         "-H", "Content-Type: application/json",
         "-H", "Accept: application/json, text/event-stream",
         "-d", json.dumps(payload)],
        capture_output=True, text=True, check=True,
    ).stdout
    # streamable-HTTP: one or more "data: {...}" SSE lines, or bare JSON
    result = None
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            line = line[6:]
        if line.startswith("{"):
            d = json.loads(line)
            if "result" in d or "error" in d:
                result = d
    if result is None:
        raise RuntimeError(f"no JSON-RPC result in response: {body[:400]}")
    if "error" in result:
        raise RuntimeError(json.dumps(result["error"])[:800])
    r = result["result"]
    # tool results carry content blocks; unwrap single text blocks to JSON
    content = r.get("content")
    if isinstance(content, list) and content and content[0].get("type") == "text":
        text = content[0]["text"]
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text
    return r


if __name__ == "__main__":
    tool = sys.argv[1]
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    out = call(tool, args)
    print(json.dumps(out, indent=2) if not isinstance(out, str) else out)
