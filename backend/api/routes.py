"""FastAPI routes exposing session-based adaptive questionnaire actions."""

from __future__ import annotations

import random
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.api.config import COHORT_ZIPCODE_PRIORS, SESSIONS_PREFIX
from backend.api.schemas import (
    AnswerQuestionRequest,
    AnswerTextRequest,
    InferredFieldResponse,
    QuestionnaireResponse,
    StartSessionRequest,
)
from backend.api.session_store import InMemorySessionStore, SessionState
from backend.core_logic.agent import (
    AdaptiveQuestionnaireAgent,
    NextActionDecision,
)
from backend.core_logic.deterministic_rules import PartialUserProfile
from backend.core_logic.field_mappings import FIELD_NAME_BY_TARGET, coerce_target_field_value
from backend.core_logic.question_bank import QUESTION_BANK, QuestionMetadata
from backend.data_generation.cohort_loader import CohortLoader
from backend.data_generation.config import DEFAULT_COHORT_SAMPLING_WEIGHTS
from backend.data_generation.enums import TargetField
from backend.data_generation.schemas import CohortBaseProbabilities, CohortDefinition, ProfileFieldValue
from backend.llm.extractor import extract_fields_from_text
from backend.llm.generator import generate_conversational_question

router = APIRouter(prefix=SESSIONS_PREFIX, tags=["sessions"])
_agent = AdaptiveQuestionnaireAgent()
_cohort_loader = CohortLoader()
_session_store = InMemorySessionStore()
_cohort_randomizer = random.Random()


@router.post("/start", response_model=QuestionnaireResponse)
async def start_session(request: StartSessionRequest) -> QuestionnaireResponse:
    """Create a new questionnaire session and return the first action."""
    cohorts = _cohort_loader.load_definitions()
    selected_cohort = _resolve_cohort_definition(
        available_cohorts=cohorts,
        requested_cohort_name=request.cohort_name,
    )
    marginal_probabilities = _to_marginal_probabilities(
        selected_cohort.base_probabilities,
        selected_cohort.cohort_name,
    )
    session_id: UUID = uuid4()

    decision: NextActionDecision = _agent.get_next_action(PartialUserProfile(), marginal_probabilities)

    _session_store.create_session(
        SessionState(
            session_id=session_id,
            profile=decision.updated_profile,
            marginal_probabilities=marginal_probabilities,
        )
    )

    logger.info(
        "Started questionnaire session",
        session_id=str(session_id),
        cohort_name=selected_cohort.cohort_name,
        action_type=decision.action_type,
    )
    return await _build_questionnaire_response(
        session_id=session_id,
        profile=decision.updated_profile,
        decision=decision,
    )


@router.post("/{session_id}/answer", response_model=QuestionnaireResponse)
async def answer_question(
    session_id: UUID,
    request: AnswerQuestionRequest,
) -> QuestionnaireResponse:
    """Store one answer for a session and return the next decision."""
    session_state = _session_store.get_session(session_id)
    if session_state is None:
        msg = f"Session not found: {session_id}"
        raise HTTPException(status_code=404, detail=msg)

    field_name: str = FIELD_NAME_BY_TARGET[request.target_field]
    coerced_value = _coerce_answer_value(request.target_field, request.answer_value)
    updated_profile = session_state.profile.model_copy(update={field_name: coerced_value})

    decision: NextActionDecision = _agent.get_next_action(
        partial_profile=updated_profile,
        marginal_probabilities=session_state.marginal_probabilities,
    )

    _session_store.update_session(
        SessionState(
            session_id=session_state.session_id,
            profile=decision.updated_profile,
            marginal_probabilities=session_state.marginal_probabilities,
        )
    )
    logger.info(
        "Processed answer for session",
        session_id=str(session_id),
        target_field=request.target_field.value,
        action_type=decision.action_type,
    )
    return await _build_questionnaire_response(
        session_id=session_state.session_id,
        profile=decision.updated_profile,
        decision=decision,
    )


@router.post("/{session_id}/answer_text", response_model=QuestionnaireResponse)
async def answer_text(
    session_id: UUID,
    request: AnswerTextRequest,
) -> QuestionnaireResponse:
    """Extract structured answers from text and return next questionnaire action."""
    session_state = _session_store.get_session(session_id)
    if session_state is None:
        msg = f"Session not found: {session_id}"
        raise HTTPException(status_code=404, detail=msg)

    working_profile: PartialUserProfile = session_state.profile.model_copy(deep=True)
    missing_fields: list[TargetField] = _get_missing_fields(working_profile)
    extracted_values: dict[TargetField, ProfileFieldValue] = await extract_fields_from_text(
        user_text=request.user_text,
        target_fields=missing_fields,
    )

    for target_field, extracted_value in extracted_values.items():
        field_name: str = FIELD_NAME_BY_TARGET[target_field]
        working_profile = working_profile.model_copy(update={field_name: extracted_value})

    decision: NextActionDecision = _agent.get_next_action(
        partial_profile=working_profile,
        marginal_probabilities=session_state.marginal_probabilities,
    )
    _session_store.update_session(
        SessionState(
            session_id=session_state.session_id,
            profile=decision.updated_profile,
            marginal_probabilities=session_state.marginal_probabilities,
        )
    )

    logger.info(
        "Processed text answer for session",
        session_id=str(session_id),
        extracted_count=len(extracted_values),
        action_type=decision.action_type,
    )
    return await _build_questionnaire_response(
        session_id=session_state.session_id,
        profile=decision.updated_profile,
        decision=decision,
    )


async def _build_questionnaire_response(
    session_id: UUID,
    profile: PartialUserProfile,
    decision: NextActionDecision,
) -> QuestionnaireResponse:
    """Build a consistent API response from a decision and profile state."""
    next_question: QuestionMetadata | None = None
    if decision.action_type == "ask_question" and decision.selected_field is not None:
        static_question: QuestionMetadata = QUESTION_BANK[decision.selected_field]
        conversational_text: str = await generate_conversational_question(
            target_field=decision.selected_field,
            profile=profile,
        )
        next_question = QuestionMetadata(
            target_field=static_question.target_field,
            question_text=conversational_text,
            friction_cost=static_question.friction_cost,
        )
    inferred_fields = [
        InferredFieldResponse(
            field_name=inferred_field.field_name,
            inferred_value=inferred_field.inferred_value,
            confidence=inferred_field.confidence,
            inference_reason=inferred_field.inference_reason,
        )
        for inferred_field in decision.inferred_fields
    ]
    return QuestionnaireResponse(
        session_id=session_id,
        is_complete=decision.action_type == "stop_and_infer",
        next_question=next_question,
        current_profile=profile,
        inferred_fields=inferred_fields,
    )


def _resolve_cohort_definition(
    available_cohorts: list[CohortDefinition],
    requested_cohort_name: str | None,
) -> CohortDefinition:
    """Resolve cohort definition from request input or weighted random selection."""
    cohort_by_name = {cohort_definition.cohort_name: cohort_definition for cohort_definition in available_cohorts}
    if requested_cohort_name is not None:
        selected = cohort_by_name.get(requested_cohort_name)
        if selected is None:
            msg = f"Unknown cohort_name: {requested_cohort_name}"
            raise HTTPException(status_code=404, detail=msg)
        return selected

    cohort_names: list[str] = list(cohort_by_name)
    weights: list[float] = [DEFAULT_COHORT_SAMPLING_WEIGHTS.get(name, 0.0) for name in cohort_names]
    selected_name: str = _cohort_randomizer.choices(cohort_names, weights=weights, k=1)[0]
    return cohort_by_name[selected_name]


def _to_marginal_probabilities(
    base_probabilities: CohortBaseProbabilities,
    cohort_name: str,
) -> dict[TargetField, dict[str, float]]:
    """Convert cohort base probabilities to target-field keyed marginals."""
    serialized_probabilities: dict[str, dict[str, float]] = base_probabilities.model_dump(mode="json")
    zipcode_priors: dict[str, float] = COHORT_ZIPCODE_PRIORS.get(cohort_name, {})
    return {
        TargetField.CREDIT_SCORE_RATE: serialized_probabilities["credit_score_rate"],
        TargetField.LOAN_PRIMARY_PURPOSE: serialized_probabilities["loan_primary_purpose"],
        TargetField.PROPERTY_TYPE: serialized_probabilities["property_type"],
        TargetField.PROPERTY_USE: serialized_probabilities["property_use"],
        TargetField.ANNUAL_INCOME_BAND: serialized_probabilities["annual_income_band"],
        TargetField.PROPERTY_VALUE_BAND: serialized_probabilities["property_value_band"],
        TargetField.CREDIT_LINE_BAND: serialized_probabilities["credit_line_band"],
        TargetField.AGE_BAND: serialized_probabilities["age_band"],
        TargetField.CURRENTLY_HAVE_MORTGAGE: serialized_probabilities["currently_have_mortgage"],
        TargetField.MILITARY_VETERAN: serialized_probabilities["military_veteran"],
        TargetField.ZIPCODE: zipcode_priors,
    }


def _coerce_answer_value(target_field: TargetField, raw_value: str | bool) -> ProfileFieldValue:
    """Coerce answer payload value to the expected typed field value."""
    try:
        return coerce_target_field_value(target_field, raw_value)
    except ValueError as error:
        msg = str(error)
        raise HTTPException(status_code=422, detail=msg) from error


def _get_missing_fields(profile: PartialUserProfile) -> list[TargetField]:
    """Return missing fields for extraction based on current profile state."""
    return [
        target_field
        for target_field, field_name in FIELD_NAME_BY_TARGET.items()
        if getattr(profile, field_name) is None
    ]
