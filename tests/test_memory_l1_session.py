from __future__ import annotations

from datetime import timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import crud, models, services
from src.api.db import Base


def _build_test_session() -> Session:
    engine = create_engine(
        "sqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    local_session = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    return local_session()


def test_l1_session_memory_records_draft_pending_task() -> None:
    db = _build_test_session()
    try:
        services._update_short_term_memory_from_response(
            db,
            actor_user_id="alice",
            text="我宿舍断网了，帮我报修工单。",
            response={
                "route": "NEED_MORE_INFO",
                "missing_fields": ["location", "contact"],
                "draft": {
                    "draft_id": "DRF-2026-ABC123",
                    "status": "open",
                    "missing_fields": ["location", "contact"],
                },
            },
        )

        record = crud.get_agent_conversation_memory(db, "alice")
        assert record is not None
        assert record.last_draft_id == "DRF-2026-ABC123"
        assert record.current_goal == "补全信息后创建工单"
        assert record.pending_task_json == {
            "type": "ticket_draft",
            "draft_id": "DRF-2026-ABC123",
            "missing_fields": ["location", "contact"],
            "updated_from_route": "NEED_MORE_INFO",
        }
        assert record.expires_at is not None
        assert len(record.recent_turns_json) == 1
        assert record.recent_turns_json[0]["draft_id"] == "DRF-2026-ABC123"
    finally:
        db.close()


def test_l1_session_memory_clears_pending_task_after_ticket_created() -> None:
    db = _build_test_session()
    try:
        services._update_short_term_memory_from_response(
            db,
            actor_user_id="alice",
            text="我宿舍断网了，帮我报修工单。",
            response={
                "route": "NEED_MORE_INFO",
                "missing_fields": ["location"],
                "draft": {"draft_id": "DRF-2026-ABC123", "status": "open"},
            },
        )
        services._update_short_term_memory_from_response(
            db,
            actor_user_id="alice",
            text="地点在金明校区 3 号楼，电话 13812345678。",
            response={
                "route": "CREATE_TICKET",
                "ticket": {"ticket_id": "TCK-2026-XYZ789"},
                "draft": {"draft_id": "DRF-2026-ABC123", "status": "consumed"},
            },
        )

        record = crud.get_agent_conversation_memory(db, "alice")
        assert record is not None
        assert record.last_ticket_id == "TCK-2026-XYZ789"
        assert record.last_draft_id is None
        assert record.current_goal == "工单已创建"
        assert record.pending_task_json is None
        assert len(record.recent_turns_json) == 2
        assert record.recent_turns_json[-1]["ticket_id"] == "TCK-2026-XYZ789"
    finally:
        db.close()


def test_l1_session_memory_expired_snapshot_is_not_loaded() -> None:
    db = _build_test_session()
    try:
        record = models.AgentConversationMemory(
            user_id="alice",
            last_ticket_id="TCK-2026-OLD001",
            last_tool="LOOKUP_TICKET",
            expires_at=services._utc_now() - timedelta(minutes=1),
        )
        db.add(record)
        db.commit()

        snapshot = services._load_short_term_memory(db, "alice")
        assert snapshot == {}
        assert services._infer_ticket_id_from_memory("帮我查一下上一单", snapshot) is None
    finally:
        db.close()


def test_l1_session_memory_recent_turns_are_capped() -> None:
    db = _build_test_session()
    try:
        for index in range(7):
            services._update_short_term_memory_from_response(
                db,
                actor_user_id="alice",
                text=f"第 {index} 轮问题",
                response={"route": "ASK", "kb": {"request_id": f"req_{index}"}},
            )

        record = crud.get_agent_conversation_memory(db, "alice")
        assert record is not None
        assert len(record.recent_turns_json) == 5
        assert record.recent_turns_json[0]["summary"] == "第 2 轮问题"
        assert record.recent_turns_json[-1]["summary"] == "第 6 轮问题"
    finally:
        db.close()


def test_l1_session_memory_stores_only_confirmation_token_prefix() -> None:
    db = _build_test_session()
    try:
        services._update_short_term_memory_from_response(
            db,
            actor_user_id="alice",
            text="取消上一单",
            response={
                "route": "NEED_CONFIRMATION",
                "confirm_token": "12345678-SECRET-FULL-TOKEN",
            },
        )

        record = crud.get_agent_conversation_memory(db, "alice")
        assert record is not None
        assert record.pending_task_json == {
            "type": "pending_confirmation",
            "confirm_token_prefix": "12345678",
            "updated_from_route": "NEED_CONFIRMATION",
        }
        assert "SECRET" not in str(record.pending_task_json)
    finally:
        db.close()
