"""Operational metrics routes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from src.api.deps import get_db
from src.api.ops_metrics import get_ops_metrics


router = APIRouter(tags=["ops"])


@router.get("/ops/metrics", response_model=None)
def read_ops_metrics(
    hours: int = Query(default=24, ge=1, le=24 * 7),
    pretty: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> Any:
    """Return a lightweight observability summary for ASK and ticket traffic."""
    metrics = get_ops_metrics(db, hours=hours)
    if not pretty:
        return metrics
    return Response(
        content=json.dumps(metrics, ensure_ascii=False, indent=2),
        media_type="application/json",
    )
