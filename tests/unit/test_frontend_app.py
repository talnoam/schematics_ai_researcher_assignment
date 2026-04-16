"""Unit tests for frontend bootstrap page."""

from unittest.mock import MagicMock

from _pytest.monkeypatch import MonkeyPatch

from frontend.app import main, render_app


def test_render_app_invokes_streamlit_widgets(monkeypatch: MonkeyPatch) -> None:
    """Verify render function uses expected Streamlit widgets."""
    from frontend import app as frontend_app

    set_page_config_mock = MagicMock()
    title_mock = MagicMock()
    caption_mock = MagicMock()
    success_mock = MagicMock()

    monkeypatch.setattr(frontend_app.st, "set_page_config", set_page_config_mock)
    monkeypatch.setattr(frontend_app.st, "title", title_mock)
    monkeypatch.setattr(frontend_app.st, "caption", caption_mock)
    monkeypatch.setattr(frontend_app.st, "success", success_mock)

    render_app()

    set_page_config_mock.assert_called_once()
    title_mock.assert_called_once()
    caption_mock.assert_called_once()
    success_mock.assert_called_once()


def test_main_calls_render_app(monkeypatch: MonkeyPatch) -> None:
    """Verify main delegates page rendering."""
    from frontend import app as frontend_app

    render_app_mock = MagicMock()
    monkeypatch.setattr(frontend_app, "render_app", render_app_mock)

    main()

    render_app_mock.assert_called_once()
