import os
import sys

import pytest

pytest.importorskip("customtkinter")  # the UI toolkit isn't installed in the light CI environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chat_panel import tidy_markdown  # noqa: E402


def test_tidy_markdown_bullets_and_headings():
    text = "### Decisions\n*   Launch stays\n  - nested item\n+ plus item\nNot * a bullet"
    assert tidy_markdown(text).splitlines() == [
        "**Decisions**", "•  Launch stays", "  •  nested item", "•  plus item", "Not * a bullet"]


def test_tidy_markdown_keeps_bold_for_the_renderer():
    assert tidy_markdown("*   **Owner:** Marcus") == "•  **Owner:** Marcus"
