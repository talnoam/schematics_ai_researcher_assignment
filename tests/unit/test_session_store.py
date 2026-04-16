"""Unit tests for in-memory API session storage."""

from uuid import uuid4

from backend.api.session_store import InMemorySessionStore, SessionState
from backend.core_logic.deterministic_rules import PartialUserProfile
from backend.data_generation.enums import PropertyUse, TargetField


def _build_marginals() -> dict[TargetField, dict[str, float]]:
    """Build minimal valid marginals for session state tests."""
    return {
        TargetField.PROPERTY_USE: {"primary_residence": 0.7, "investment": 0.3},
    }


def test_create_and_get_session_roundtrip() -> None:
    """Verify created session can be fetched by id."""
    store = InMemorySessionStore()
    session_id = uuid4()
    session_state = SessionState(
        session_id=session_id,
        profile=PartialUserProfile(),
        marginal_probabilities=_build_marginals(),
    )

    store.create_session(session_state)
    fetched_state = store.get_session(session_id)

    assert fetched_state is not None
    assert fetched_state.session_id == session_id


def test_update_session_overwrites_profile() -> None:
    """Verify update operation replaces previously stored profile values."""
    store = InMemorySessionStore()
    session_id = uuid4()
    initial_state = SessionState(
        session_id=session_id,
        profile=PartialUserProfile(),
        marginal_probabilities=_build_marginals(),
    )
    store.create_session(initial_state)

    updated_state = SessionState(
        session_id=session_id,
        profile=PartialUserProfile(property_use=PropertyUse.INVESTMENT),
        marginal_probabilities=_build_marginals(),
    )
    store.update_session(updated_state)
    fetched_state = store.get_session(session_id)

    assert fetched_state is not None
    assert fetched_state.profile.property_use == PropertyUse.INVESTMENT
