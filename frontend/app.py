"""Interactive Streamlit frontend for adaptive questionnaire sessions."""

from __future__ import annotations

from uuid import UUID

from loguru import logger
import streamlit as st

from backend.api.schemas import QuestionnaireResponse
from backend.core_logic.field_mappings import (
    BOOLEAN_TARGET_FIELDS,
    FIELD_NAME_BY_TARGET,
    TARGET_VALUE_ENUM_BY_FIELD,
)
from backend.data_generation.config import DEFAULT_COHORT_SAMPLING_WEIGHTS
from backend.data_generation.enums import TargetField
from backend.data_generation.schemas import ProfileFieldValue
from config import settings
from frontend.api_client import FrontendApiClientError, QuestionnaireApiClient

SESSION_RESPONSE_KEY: str = "questionnaire_response"
SESSION_EXPLICIT_FIELDS_KEY: str = "explicit_answered_fields"
SESSION_CLIENT_KEY: str = "api_client"
SESSION_CHAT_HISTORY_KEY: str = "chat_history"
SESSION_COHORT_KEY: str = "selected_cohort_name"
CHAT_ROLE_ASSISTANT: str = "assistant"
CHAT_ROLE_USER: str = "user"


def get_explicit_options(target_field: TargetField) -> list[str]:
    """Return explicit-answer UI options for the selected target field."""
    if target_field in BOOLEAN_TARGET_FIELDS:
        return ["true", "false"]
    enum_type = TARGET_VALUE_ENUM_BY_FIELD[target_field]
    return [enum_member.value for enum_member in enum_type]


def coerce_explicit_option(target_field: TargetField, selected_option: str) -> ProfileFieldValue:
    """Coerce selected UI option value into the expected field type."""
    if target_field in BOOLEAN_TARGET_FIELDS:
        return selected_option == "true"
    enum_type = TARGET_VALUE_ENUM_BY_FIELD[target_field]
    return enum_type(selected_option)


def split_profile_fields(
    response: QuestionnaireResponse,
    explicit_fields: set[TargetField],
) -> tuple[dict[str, object], dict[str, object]]:
    """Split populated profile values into explicit and inferred groups."""
    profile_payload: dict[str, object] = response.current_profile.model_dump(mode="json")
    explicit_values: dict[str, object] = {}
    inferred_values: dict[str, object] = {}
    for target_field, field_name in FIELD_NAME_BY_TARGET.items():
        value: object = profile_payload[field_name]
        if value is None:
            continue
        if target_field in explicit_fields:
            explicit_values[field_name] = value
        else:
            inferred_values[field_name] = value
    return explicit_values, inferred_values


def render_app() -> None:
    """Render the full adaptive questionnaire user interface."""
    st.set_page_config(page_title=settings.app_name, page_icon=":robot_face:", layout="wide")
    _initialize_state()
    api_client = _get_client()

    st.title("Adaptive Questionnaire Demo")
    st.caption("The agent can skip questions when confidence is already sufficient.")

    with st.sidebar:
        _render_debug_sidebar()

    response = st.session_state[SESSION_RESPONSE_KEY]
    if response is None:
        _render_start_screen(api_client)
        return

    _render_chat_history()

    if response.is_complete:
        st.success("Questionnaire complete. The adaptive agent decided further questions are unnecessary.")
        st.subheader("Final Profile")
        st.json(response.current_profile.model_dump(mode="json"))
        return

    next_question = response.next_question
    if next_question is None:
        st.warning("No follow-up question is available. The session may already be complete.")
        return

    st.markdown("#### Explicit Fallback Answer")
    explicit_options: list[str] = get_explicit_options(next_question.target_field)
    selected_option: str = st.selectbox(
        "Choose an explicit answer",
        explicit_options,
        key=f"explicit_option_{next_question.target_field.value}",
    )
    if st.button("Submit Explicit Answer", use_container_width=True):
        _submit_explicit_answer(
            api_client=api_client,
            response=response,
            target_field=next_question.target_field,
            selected_option=selected_option,
        )

    user_message: str | None = st.chat_input("Type your answer here...")
    if user_message is None:
        return
    if not user_message.strip():
        st.warning("Type an answer before submitting.")
        return

    _append_chat_message(CHAT_ROLE_USER, user_message.strip())
    with st.spinner("Agent is thinking..."):
        _submit_text_answer(api_client, response.session_id, user_message.strip())


def _initialize_state() -> None:
    """Initialize Streamlit session state containers for frontend flow."""
    if SESSION_CLIENT_KEY not in st.session_state:
        st.session_state[SESSION_CLIENT_KEY] = QuestionnaireApiClient()
    if SESSION_RESPONSE_KEY not in st.session_state:
        st.session_state[SESSION_RESPONSE_KEY] = None
    if SESSION_EXPLICIT_FIELDS_KEY not in st.session_state:
        st.session_state[SESSION_EXPLICIT_FIELDS_KEY] = set()
    if SESSION_CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[SESSION_CHAT_HISTORY_KEY] = []
    if SESSION_COHORT_KEY not in st.session_state:
        st.session_state[SESSION_COHORT_KEY] = "Automatic (weighted)"


def _get_client() -> QuestionnaireApiClient:
    """Return API client instance from session state."""
    return st.session_state[SESSION_CLIENT_KEY]


def _render_start_screen(api_client: QuestionnaireApiClient) -> None:
    """Render controls for creating a new questionnaire session."""
    st.subheader("Start Questionnaire")
    cohort_options: list[str] = ["Automatic (weighted)"] + sorted(DEFAULT_COHORT_SAMPLING_WEIGHTS)
    selected_option: str = st.selectbox("Optional cohort", cohort_options, index=0)
    if st.button("Start Questionnaire", type="primary"):
        selected_cohort: str | None = None
        if selected_option != "Automatic (weighted)":
            selected_cohort = selected_option
        try:
            response: QuestionnaireResponse = api_client.start_session(cohort_name=selected_cohort)
        except FrontendApiClientError as error:
            st.error(str(error))
            return
        st.session_state[SESSION_COHORT_KEY] = selected_option
        st.session_state[SESSION_RESPONSE_KEY] = response
        st.session_state[SESSION_EXPLICIT_FIELDS_KEY] = set()
        st.session_state[SESSION_CHAT_HISTORY_KEY] = []
        if response.next_question is not None:
            _append_chat_message(CHAT_ROLE_ASSISTANT, response.next_question.question_text)
        elif response.is_complete:
            _append_chat_message(
                CHAT_ROLE_ASSISTANT,
                "Thanks, we already have enough information to complete your profile.",
            )
        logger.info("Started frontend questionnaire session", session_id=str(response.session_id))
        st.rerun()


def _render_debug_sidebar() -> None:
    """Render agent debug panel with profile and adaptive progress insights."""
    st.markdown("### Agent's Brain - Debug View")
    response = st.session_state[SESSION_RESPONSE_KEY]
    if response is None:
        st.info("No active session.")
        return

    selected_cohort: str = st.session_state[SESSION_COHORT_KEY]
    st.markdown("#### Cohort Context")
    st.code(selected_cohort)

    explicit_fields: set[TargetField] = st.session_state[SESSION_EXPLICIT_FIELDS_KEY]
    explicit_values, inferred_values = split_profile_fields(response, explicit_fields)
    st.metric("Completion", "Complete" if response.is_complete else "In Progress")
    st.metric("Explicitly answered fields", len(explicit_values))
    st.metric("Inferred or extracted fields", len(inferred_values))
    st.markdown("#### Current Partial Profile")
    st.json(response.current_profile.model_dump(mode="json"))
    st.markdown("#### Explicit Answers")
    st.json(explicit_values)
    st.markdown("#### Inferred / LLM-Extracted")
    st.json(inferred_values)
    st.markdown("#### Inference Trace")
    st.json([inferred_field.model_dump(mode="json") for inferred_field in response.inferred_fields])


def _submit_text_answer(
    api_client: QuestionnaireApiClient,
    session_id: UUID,
    user_text: str,
) -> None:
    """Submit free-text answer and update session state safely."""
    try:
        response: QuestionnaireResponse = api_client.answer_question_via_text(session_id, user_text)
    except FrontendApiClientError as error:
        st.error(str(error))
        return
    st.session_state[SESSION_RESPONSE_KEY] = response
    if response.is_complete:
        _append_chat_message(
            CHAT_ROLE_ASSISTANT,
            "Great, I have everything I need and can finalize your profile now.",
        )
    elif response.next_question is not None:
        _append_chat_message(CHAT_ROLE_ASSISTANT, response.next_question.question_text)
    st.rerun()


def _submit_explicit_answer(
    api_client: QuestionnaireApiClient,
    response: QuestionnaireResponse,
    target_field: TargetField,
    selected_option: str,
) -> None:
    """Submit explicit answer and track field as directly answered."""
    answer_value: ProfileFieldValue = coerce_explicit_option(target_field, selected_option)
    _append_chat_message(CHAT_ROLE_USER, f"(Explicit) {selected_option}")
    try:
        updated_response: QuestionnaireResponse = api_client.answer_question_explicitly(
            session_id=response.session_id,
            target_field=target_field,
            answer_value=answer_value,
        )
    except FrontendApiClientError as error:
        st.error(str(error))
        return
    explicit_fields: set[TargetField] = st.session_state[SESSION_EXPLICIT_FIELDS_KEY]
    explicit_fields.add(target_field)
    st.session_state[SESSION_EXPLICIT_FIELDS_KEY] = explicit_fields
    st.session_state[SESSION_RESPONSE_KEY] = updated_response
    if updated_response.is_complete:
        _append_chat_message(
            CHAT_ROLE_ASSISTANT,
            "Great, I have everything I need and can finalize your profile now.",
        )
    elif updated_response.next_question is not None:
        _append_chat_message(CHAT_ROLE_ASSISTANT, updated_response.next_question.question_text)
    st.rerun()


def _render_chat_history() -> None:
    """Render all chat turns collected in session state."""
    chat_history: list[dict[str, str]] = st.session_state[SESSION_CHAT_HISTORY_KEY]
    for message in chat_history:
        role: str = message.get("role", CHAT_ROLE_ASSISTANT)
        content: str = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


def _append_chat_message(role: str, content: str) -> None:
    """Append one message into chat history."""
    chat_history: list[dict[str, str]] = st.session_state[SESSION_CHAT_HISTORY_KEY]
    chat_history.append({"role": role, "content": content})
    st.session_state[SESSION_CHAT_HISTORY_KEY] = chat_history


def main() -> None:
    """Run the frontend application."""
    logger.info("Frontend app started", app_name=settings.app_name)
    render_app()


if __name__ == "__main__":
    main()
