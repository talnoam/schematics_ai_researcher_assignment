"""Thread-safe in-memory session store for questionnaire state."""

from __future__ import annotations

from threading import Lock
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, ConfigDict

from backend.core_logic.deterministic_rules import PartialUserProfile
from backend.data_generation.enums import TargetField


class SessionState(BaseModel):
    """Represent per-session state required by adaptive question routing."""

    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: UUID
    profile: PartialUserProfile
    marginal_probabilities: dict[TargetField, dict[str, float]]


class InMemorySessionStore:
    """Manage questionnaire sessions with lock-protected in-memory storage."""

    def __init__(self) -> None:
        """Initialize empty session storage and synchronization lock."""
        self._sessions: dict[str, SessionState] = {}
        self._lock: Lock = Lock()

    def create_session(self, session_state: SessionState) -> SessionState:
        """Create a new session entry and return stored state."""
        session_key: str = str(session_state.session_id)
        with self._lock:
            self._sessions[session_key] = session_state
        logger.info("Created session", session_id=session_key)
        return session_state

    def get_session(self, session_id: UUID) -> SessionState | None:
        """Retrieve session state by identifier when present."""
        session_key: str = str(session_id)
        with self._lock:
            session_state: SessionState | None = self._sessions.get(session_key)
        return session_state

    def update_session(self, session_state: SessionState) -> SessionState:
        """Overwrite session state and return updated payload."""
        session_key: str = str(session_state.session_id)
        with self._lock:
            self._sessions[session_key] = session_state
        logger.info("Updated session", session_id=session_key)
        return session_state
