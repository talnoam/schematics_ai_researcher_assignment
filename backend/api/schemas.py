"""API request and response schemas for session-based questionnaire routes."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.core_logic.deterministic_rules import PartialUserProfile
from backend.core_logic.question_bank import QuestionMetadata
from backend.data_generation.enums import TargetField


class StartSessionRequest(BaseModel):
    """Represent payload for creating a new questionnaire session."""

    model_config = ConfigDict(extra="forbid")

    cohort_name: str | None = None


class AnswerQuestionRequest(BaseModel):
    """Represent payload for submitting one field answer."""

    model_config = ConfigDict(extra="forbid")

    target_field: TargetField
    answer_value: str | bool


class AnswerTextRequest(BaseModel):
    """Represent payload for submitting free-text questionnaire content."""

    model_config = ConfigDict(extra="forbid", strict=True)

    user_text: str = Field(min_length=1)


class QuestionnaireResponse(BaseModel):
    """Represent API state returned after start or answer operations."""

    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: UUID
    is_complete: bool
    next_question: QuestionMetadata | None
    current_profile: PartialUserProfile
