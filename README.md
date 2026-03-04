# Policy KB Assistant

[English](README.md) | [简体中文](README.zh-CN.md)

Enterprise KB + ITSM Agent. This project combines policy Q&A, ticket workflows, and MCP-exposed tools behind one governed execution layer.

The core design is intentional:
- The model proposes a route or tool plan.
- The backend validates schema, auth, and confirmation state.
- Only validated actions execute, and every step is auditable.

Two practical differentiators:
- The same skills registry drives both `/agent` and MCP tools.
- High-risk actions use two-step confirmation instead of direct execution.

## Prerequisites

- Python 3.10+
- Conda or Miniconda is recommended for a clean local environment
- `make` is recommended because the repository already ships a `Makefile`
- Docker + Docker Compose are only required for the full demo path

## What You Can Demo

- Policy Q&A with citations (the web UI routes natural-language requests through `POST /agent`; `POST /ask` remains available as a direct API endpoint)
- One-shot agent routing via `POST /agent`
- Ticket creation, lookup, comments, escalation, and cancellation
- Streamlit web UI that calls the real HTTP API
- MCP stdio server exposing governed ticket tools
- Replay via `kb_queries` and `audit_logs`

## Quickstart (Minimal Local Run)

Use this path if you want the fastest local setup for tickets, the web UI, and MCP tools. It does not require Docker, Postgres, or Qdrant.

1. Install dependencies.

```bash
conda create -n policy-kb python=3.10 -y
conda activate policy-kb
python -m pip install -r requirements.txt
cp .env.example .env
```

2. In `.env`, override these values for a SQLite-based local run.

```dotenv
DATABASE_URL=sqlite:///./policy_kb_l2.db
POLICY_API_KEY=local-dev-key
AUTO_MIGRATE_ON_STARTUP=true
DEV_DB_FALLBACK_CREATE_ALL=false
```

3. Start the API.

```bash
make api
```

4. Start the web UI in another terminal.

```bash
make ui
```

5. Open the app.

- Web UI: `http://localhost:8501`
- API health: `http://localhost:8080/health`

Use these UI values:
- `API Base URL`: `http://localhost:8080`
- `API Key`: `local-dev-key`
- `User`: `alice`
- `Department`: `IT`

What works in this mode:
- Manual ticket creation
- Ticket lookup and management
- Trace replay for created tickets
- MCP ticket tools

What is intentionally limited in this mode:
- Knowledge-backed Q&A (`POST /ask` or `/agent` routed to `ASK`) needs a valid `OPENAI_API_KEY`
- Retrieval-backed Q&A also needs Qdrant and ingested documents

## Full Demo (Postgres + Qdrant + Document Ingestion)

Use this path if you want the complete Q&A + ticket + audit demo.

1. Start infrastructure.

```bash
make l2-up
```

This brings up:
- Qdrant
- Postgres
- Redis

2. Keep the default Postgres-style `DATABASE_URL` from `.env.example`, and set:

```dotenv
POLICY_API_KEY=local-dev-key
OPENAI_API_KEY=YOUR_REAL_OPENAI_COMPATIBLE_KEY
```

3. The repository ships one redistributable demo document:

- `data/demo/ACME_IT_Admin_Handbook_v1.0_Demo.pdf`

Optional extra source documents are not redistributed. If you want them, download them into `data/raw/`:

```bash
./scripts/download_pdfs.sh
```

`data/raw/` is git-ignored by default, so those extra documents stay local.

4. Ingest the bundled demo document plus any local `data/raw/` PDFs into Qdrant.

```bash
make ingest
```

5. Start the API and UI.

```bash
make api
make ui
```

In this mode, you can demo:
- KB-assisted `/agent` from the web UI
- `POST /ask` if you want to call the API directly
- Draft continuation
- Existing-ticket tool actions
- Web UI + MCP side by side

For document provenance and optional source URLs, see [docs/demo_data.md](docs/demo_data.md).

## MCP

Run the stdio server in demo mode:

```bash
export MCP_ACTOR_USER_ID=alice
export MCP_DEPARTMENT=IT
PYTHONPATH=$(pwd) python -m src.mcp_stdio_server
```

You can also validate the MCP tool chain without any external host:

```bash
PYTHONPATH=$(pwd) python scripts/mcp_smoke.py --actor alice
```

## Security Model

This repository implements a demo-grade safety model, not a production multi-tenant auth system.

- `/agent` and write APIs use a shared `X-API-Key`
- MCP stdio runs in fixed-actor mode via `MCP_ACTOR_USER_ID`
- High-risk cancellation is two-step: request confirmation, then confirm
- Audit entries record source information in payloads so Web and MCP calls are distinguishable

Current non-goals:
- OAuth
- Per-user bearer-token auth
- Remote HTTP MCP with multi-user identity mapping

## Testing

Run the local regression suite:

```bash
PYTHONPATH=$(pwd) pytest -q tests
```

The repository includes:
- service-layer tests
- API smoke tests
- MCP in-memory tool tests
- Streamlit UI smoke tests

GitHub Actions runs the core regression subset on pushes and pull requests.

## Troubleshooting

- `401 Unauthorized`: set `POLICY_API_KEY` in `.env`, then use the same value in the UI sidebar.
- `Qdrant connection refused`: the full demo path needs `make l2-up` before `make ingest` or KB-backed `/agent`.
- No citations or empty KB answers: you likely skipped `make ingest`, or no valid `OPENAI_API_KEY` is configured.
- MCP host cannot connect: in stdio mode, do not print to `stdout`; use `stderr` for logs.
- FastAPI `on_event` deprecation warnings: known warning, not a runtime failure; migration to `lifespan` is a follow-up cleanup.

## Repository Layout

- `src/api/`: FastAPI app, services, skills registry, storage logic
- `src/ui/`: Streamlit app and HTTP client
- `src/kb/`: retrieval and answer generation
- `src/agent/`: ticket extraction logic
- `src/mcp_stdio_server.py`: MCP stdio entrypoint
- `tests/`: regression coverage
- `scripts/`: smoke and release helpers
- `docs/`: public documentation

## Documentation

- [Architecture](docs/architecture.md)
- [Demo Data](docs/demo_data.md)
- [MCP](docs/mcp.md)
