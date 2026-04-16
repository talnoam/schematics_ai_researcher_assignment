"""Frontend Streamlit bootstrap entrypoint."""

from loguru import logger
import streamlit as st

from config import settings


def render_app() -> None:
    """Render the initial Streamlit placeholder page."""
    st.set_page_config(page_title=settings.app_name, page_icon=":robot_face:", layout="wide")
    st.title("Adaptive Questionnaire")
    st.caption("Frontend shell is ready. Core questionnaire UI will be added in Phase 4.")
    st.success("Backend URL configuration is loaded. You can now run the container stack.")


def main() -> None:
    """Run the frontend application."""
    logger.info("Frontend app started", app_name=settings.app_name)
    render_app()


if __name__ == "__main__":
    main()
