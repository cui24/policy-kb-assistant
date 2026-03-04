# MCP

This project includes a stdio MCP server that exposes ticket tools without bypassing the existing backend safety model.

## Scope

The current implementation is intentionally narrow:
- transport: `stdio`
- auth model: fixed single actor from environment variables
- tools:
  - `lookup_ticket`
  - `add_ticket_comment`
  - `escalate_ticket`
  - `request_cancel_ticket`
  - `confirm_cancel_ticket`

Not in scope yet:
- multi-user OAuth or bearer-token mapping
- Streamable HTTP deployment
- automatic packaging for Claude Desktop extensions

## Design Principle

MCP is only a transport and tool-exposure layer here.

The server does not get its own privileged execution path. Tool calls still go through:
- the shared skills contract
- existing workflow validation
- confirmation-state checks
- audit logging

That keeps the Web path and the MCP path aligned.

## Demo Security Model

Current MCP runs in demo mode:
- actor identity is fixed by `MCP_ACTOR_USER_ID`
- tool inputs do not accept an external `actor` field
- high-risk cancellation still requires two calls

This protects against the simplest class of host-side impersonation in stdio mode.

It is not a production identity model.

## Start the Server

Install dependencies first:

```bash
python -m pip install -r requirements.txt
```

Then set environment variables:

```bash
export MCP_ACTOR_USER_ID=alice
export MCP_DEPARTMENT=IT
export PYTHONPATH=$(pwd)
```

Start the server:

```bash
python -m src.mcp_stdio_server
```

Important:
- Do not print logs to `stdout` in stdio mode
- Use `stderr` or logging instead

## Smoke Test Without a Host

You can validate the local MCP behavior without Claude or Cursor:

```bash
PYTHONPATH=$(pwd) python scripts/mcp_smoke.py --actor alice
```

This runs an in-memory session and verifies:
- tool discovery
- lookup
- append-only comment
- escalation
- two-step cancellation

## Generic Host Configuration

Use the same pattern in any MCP host that supports stdio:

- command: your Python interpreter
- args: `["-m", "src.mcp_stdio_server"]`
- env:
  - `PYTHONPATH`: absolute path to the repo root
  - `MCP_ACTOR_USER_ID`: fixed demo actor
  - `MCP_DEPARTMENT`: optional department label

For Cursor, the repository includes a template:
- `.cursor/mcp.json.example`

Replace the placeholder paths with your local paths before use.

## Why Cancellation Is Split Into Two Tools

The MCP UX is clearer if cancellation is explicit:
- `request_cancel_ticket` returns a confirmation token
- `confirm_cancel_ticket` consumes that token and performs the cancellation

This keeps human confirmation visible in host tool logs and mirrors the backend’s pending-action model.

## Roadmap

Natural next steps:
- expose `kb_answer` as a read-only MCP tool
- add Streamable HTTP transport
- move from fixed actor to token-to-actor mapping
- promote audit source from JSON payload to a first-class column
