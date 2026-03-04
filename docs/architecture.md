# Architecture

This document explains the current public architecture: what happens from a user action to a returned result, and why the system is structured around governed tool execution instead of direct model control.

## System Shape

There are two entrypoints:
- Web UI: Streamlit in `src/ui/app.py`
- MCP: stdio server in `src/mcp_stdio_server.py`

Both converge on the same backend concepts:
- shared skills registry
- validator-style checks
- workflow execution
- audit logging

That is the key design choice in this repository: entrypoints differ, safety boundaries do not.

## Main Layers

1. UI or MCP host
2. Thin transport layer
3. FastAPI or MCP wrapper
4. Service orchestration in `src/api/services.py` and `src/api/services_mcp.py`
5. CRUD + ORM persistence in `src/api/crud.py` and `src/api/models.py`
6. Supporting engines:
   - retrieval and answer generation
   - ticket extraction
   - planner
   - skills registry

## Core Runtime Contract

The runtime contract is:

1. Interpret the request.
2. Produce a route or tool plan.
3. Validate the planned action.
4. Execute only the validated action.
5. Persist business state and audit state.
6. Return a structured response.

In practice this means:
- the model can suggest
- the backend decides what is allowed

## Web Request Flow

The default web flow is:

1. The user clicks a button in Streamlit.
2. `src/ui/api_client.py` turns that action into an HTTP request.
3. FastAPI receives the request and creates a DB session.
4. `src/api/services.py` routes the request.
5. The selected workflow writes business rows and audit rows.
6. JSON is returned to the UI.
7. The UI stores the result in `st.session_state` and renders it.

The web UI is intentionally thin. It should not contain business rules that can bypass the backend.

## `/agent` Flow

`POST /agent` is the main orchestration entrypoint.

It can end up in one of four broad outcomes:
- knowledge-base answer
- create ticket
- continue ticket draft
- act on an existing ticket

The service layer first normalizes request context, then uses either rules or the planner path, depending on configuration. Regardless of planner mode, execution still lands in explicit workflows.

## Skills Registry and Tool Orchestration

The skills registry in `src/api/skills.py` is the unifying contract for tool-like actions.

It defines, per skill:
- name and description
- input and output schema
- risk level
- auth rule
- handler semantics
- callable handler, when executable

That registry is used in two places:
- `/agent`, where the system plans and dispatches governed actions
- MCP, where the server exposes thin wrappers over the same governed actions

This is why the project is better described as a tool-orchestrated agent than a simple RAG demo.

## Validation and Confirmation

Before execution, the backend enforces:
- schema validation
- ticket existence checks
- ownership or admin checks where relevant
- confirmation requirements for high-risk actions

Cancellation is intentionally two-step:
- first request creates a `pending_actions` record
- second request confirms using a token

This prevents “single model response = destructive side effect”.

## Persistence Model

The main persistent objects are:
- `tickets`
- `ticket_comments`
- `ticket_drafts`
- `pending_actions`
- `kb_queries`
- `audit_logs`
- `agent_conversation_memory`
- `user_memory`

Each exists for a specific runtime reason:
- comments are append-only to avoid concurrent overwrite
- drafts support incomplete ticket collection
- pending actions support confirmation
- memories support follow-up references and default values

## Why It Is Structured This Way

The project is optimized for controllability and explainability:
- model output is not trusted as final authority
- write actions are explicit workflows
- risky actions are confirmable
- actions are replayable through audit data
- the same business contract can be reused across Web and MCP

That is the main engineering story of the repository.

## Related Docs

- [MCP](mcp.md)
