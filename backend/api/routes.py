"""FastAPI routes exposing session-based adaptive questionnaire actions."""

from __future__ import annotations

import random
from enum import StrEnum
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.api.config import SESSIONS_PREFIX
from backend.api.schemas import AnswerQuestionRequest, QuestionnaireResponse, StartSessionRequest
from backend.api.session_store import InMemorySessionStore, SessionState
from backend.core_logic.agent import (
    FIELD_NAME_BY_TARGET,
    AdaptiveQuestionnaireAgent,
    NextActionDecision,
)
from backend.core_logic.deterministic_rules import DeterministicRulesEngine, PartialUserProfile
from backend.core_logic.question_bank import QUESTION_BANK, QuestionMetadata
from backend.data_generation.cohort_loader import CohortLoader
from backend.data_generation.config import DEFAULT_COHORT_SAMPLING_WEIGHTS
from backend.data_generation.enums import (
    AgeBand,
    AnnualIncomeBand,
    CreditLineBand,
    CreditScoreRate,
    LoanPrimaryPurpose,
    PropertyType,
    PropertyUse,
    PropertyValueBand,
    TargetField,
)
from backend.data_generation.schemas import CohortBaseProbabilities
from backend.data_generation.schemas import CohortDefinition
from backend.data_generation.schemas import ProfileFieldValue

TARGET_VALUE_ENUM_BY_FIELD: dict[TargetField, type[StrEnum]] = {
    TargetField.CREDIT_SCORE_RATE: CreditScoreRate,
    TargetField.LOAN_PRIMARY_PURPOSE: LoanPrimaryPurpose,
    TargetField.PROPERTY_TYPE: PropertyType,
    TargetField.PROPERTY_USE: PropertyUse,
    TargetField.ANNUAL_INCOME_BAND: AnnualIncomeBand,
    TargetField.PROPERTY_VALUE_BAND: PropertyValueBand,
    TargetField.CREDIT_LINE_BAND: CreditLineBand,
    TargetField.AGE_BAND: AgeBand,
}
BOOLEAN_TARGET_FIELDS: set[TargetField] = {
    TargetField.CURRENTLY_HAVE_MORTGAGE,
    TargetField.MILITARY_VETERAN,
}

router = APIRouter(prefix=SESSIONS_PREFIX, tags=["sessions"])
_agent = AdaptiveQuestionnaireAgent()
_deterministic_rules_engine = DeterministicRulesEngine()
_cohort_loader = CohortLoader()
_session_store = InMemorySessionStore()
_cohort_randomizer = random.Random()


@router.post("/start", response_model=QuestionnaireResponse)
def start_session(request: StartSessionRequest) -> QuestionnaireResponse:
    """Create a new questionnaire session and return the first action."""
    cohorts = _cohort_loader.load_definitions()
    selected_cohort = _resolve_cohort_definition(
        available_cohorts=cohorts,
        requested_cohort_name=request.cohort_name,
    )
    marginal_probabilities = _to_marginal_probabilities(selected_cohort.base_probabilities)
    session_id: UUID = uuid4()

    profile = PartialUserProfile()
    deterministic_result = _deterministic_rules_engine.apply_rules(profile)
    resolved_profile = deterministic_result.updated_profile
    decision: NextActionDecision = _agent.get_next_action(resolved_profile, marginal_probabilities)

    _session_store.create_session(
        SessionState(
            session_id=session_id,
            profile=resolved_profile,
            marginal_probabilities=marginal_probabilities,
        )
    )

    logger.info(
        "Started questionnaire session",
        session_id=str(session_id),
        cohort_name=selected_cohort.cohort_name,
        action_type=decision.action_type,
    )
    return _build_questionnaire_response(
        session_id=session_id,
        profile=resolved_profile,
        decision=decision,
    )


@router.post("/{session_id}/answer", response_model=QuestionnaireResponse)
def answer_question(
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

    deterministic_result = _deterministic_rules_engine.apply_rules(updated_profile)
    resolved_profile = deterministic_result.updated_profile
    decision: NextActionDecision = _agent.get_next_action(
        partial_profile=resolved_profile,
        marginal_probabilities=session_state.marginal_probabilities,
    )

    _session_store.update_session(
        SessionState(
            session_id=session_state.session_id,
            profile=resolved_profile,
            marginal_probabilities=session_state.marginal_probabilities,
        )
    )
    logger.info(
        "Processed answer for session",
        session_id=str(session_id),
        target_field=request.target_field.value,
        action_type=decision.action_type,
    )
    return _build_questionnaire_response(
        session_id=session_state.session_id,
        profile=resolved_profile,
        decision=decision,
    )


def _build_questionnaire_response(
    session_id: UUID,
    profile: PartialUserProfile,
    decision: NextActionDecision,
) -> QuestionnaireResponse:
    """Build a consistent API response from a decision and profile state."""
    next_question: QuestionMetadata | None = None
    if decision.action_type == "ask_question" and decision.selected_field is not None:
        next_question = QUESTION_BANK[decision.selected_field]
    return QuestionnaireResponse(
        session_id=session_id,
        is_complete=decision.action_type == "stop_and_infer",
        next_question=next_question,
        current_profile=profile,
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
) -> dict[TargetField, dict[str, float]]:
    """Convert cohort base probabilities to target-field keyed marginals."""
    serialized_probabilities: dict[str, dict[str, float]] = base_probabilities.model_dump(mode="json")
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
    }


def _coerce_answer_value(target_field: TargetField, raw_value: str | bool) -> ProfileFieldValue:
    """Coerce answer payload value to the expected typed field value."""
    if target_field in BOOLEAN_TARGET_FIELDS:
        if not isinstance(raw_value, bool):
            msg = f"Answer for '{target_field.value}' must be boolean."
            raise HTTPException(status_code=422, detail=msg)
        return raw_value

    expected_type: type[StrEnum] = TARGET_VALUE_ENUM_BY_FIELD[target_field]
    if not isinstance(raw_value, str):
        msg = f"Answer for '{target_field.value}' must be a string enum value."
        raise HTTPException(status_code=422, detail=msg)
    try:
        return expected_type(raw_value)
    except ValueError as error:
        msg = f"Invalid value '{raw_value}' for field '{target_field.value}'."
        raise HTTPException(status_code=422, detail=msg) from error
