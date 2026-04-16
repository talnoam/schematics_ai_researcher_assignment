"""Generate and persist synthetic users for local experimentation."""

from __future__ import annotations

from time import perf_counter

from loguru import logger

from backend.data_generation.config import DEFAULT_MOCK_USER_COUNT, DEFAULT_MOCK_USERS_OUTPUT_PATH
from backend.data_generation.generator import MockDataGenerator


def main() -> None:
    """Generate synthetic users and save them as a CSV file."""
    started_at: float = perf_counter()
    generator: MockDataGenerator = MockDataGenerator()
    generated_dataframe = generator.generate_dataframe(user_count=DEFAULT_MOCK_USER_COUNT)

    output_path = DEFAULT_MOCK_USERS_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_dataframe.to_csv(output_path, index=False)

    elapsed_seconds: float = perf_counter() - started_at
    logger.info(
        "Mock data generation completed",
        output_path=str(output_path),
        records_generated=len(generated_dataframe),
        elapsed_seconds=round(elapsed_seconds, 3),
    )


if __name__ == "__main__":
    main()
