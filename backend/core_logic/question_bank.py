"""Question metadata repository for adaptive questionnaire scoring."""

from pydantic import BaseModel, ConfigDict, Field

from backend.data_generation.enums import TargetField


class QuestionMetadata(BaseModel):
    """Represent metadata for one adaptive-question candidate."""

    model_config = ConfigDict(extra="forbid", strict=True)

    target_field: TargetField
    question_text: str = Field(min_length=1)
    friction_cost: float = Field(ge=0.0, le=1.0)


QUESTION_BANK: dict[TargetField, QuestionMetadata] = {
    TargetField.CREDIT_SCORE_RATE: QuestionMetadata(
        target_field=TargetField.CREDIT_SCORE_RATE,
        question_text="What best describes your current credit score range?",
        friction_cost=0.88,
    ),
    TargetField.LOAN_PRIMARY_PURPOSE: QuestionMetadata(
        target_field=TargetField.LOAN_PRIMARY_PURPOSE,
        question_text="What is your primary loan purpose?",
        friction_cost=0.24,
    ),
    TargetField.PROPERTY_TYPE: QuestionMetadata(
        target_field=TargetField.PROPERTY_TYPE,
        question_text="What type of property are you financing?",
        friction_cost=0.18,
    ),
    TargetField.PROPERTY_USE: QuestionMetadata(
        target_field=TargetField.PROPERTY_USE,
        question_text="How will the property be used?",
        friction_cost=0.16,
    ),
    TargetField.ANNUAL_INCOME_BAND: QuestionMetadata(
        target_field=TargetField.ANNUAL_INCOME_BAND,
        question_text="Which annual income band best matches your household?",
        friction_cost=0.92,
    ),
    TargetField.PROPERTY_VALUE_BAND: QuestionMetadata(
        target_field=TargetField.PROPERTY_VALUE_BAND,
        question_text="Which property value band best matches the home?",
        friction_cost=0.46,
    ),
    TargetField.CREDIT_LINE_BAND: QuestionMetadata(
        target_field=TargetField.CREDIT_LINE_BAND,
        question_text="What is your available credit line range?",
        friction_cost=0.73,
    ),
    TargetField.AGE_BAND: QuestionMetadata(
        target_field=TargetField.AGE_BAND,
        question_text="Which age band do you fall into?",
        friction_cost=0.28,
    ),
    TargetField.CURRENTLY_HAVE_MORTGAGE: QuestionMetadata(
        target_field=TargetField.CURRENTLY_HAVE_MORTGAGE,
        question_text="Do you currently have a mortgage?",
        friction_cost=0.31,
    ),
    TargetField.MILITARY_VETERAN: QuestionMetadata(
        target_field=TargetField.MILITARY_VETERAN,
        question_text="Have you served in the military as a veteran?",
        friction_cost=0.41,
    ),
}
