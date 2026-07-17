"""Consistency checks for the GitHub Pages site in docs/ — it must stay
self-contained and not drift from what the app/README actually say."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS = (ROOT / "docs" / "index.html").read_text()
README = (ROOT / "README.md").read_text()


def test_docs_page_is_self_contained():
    # GitHub Pages serves docs/ alone — no reaching into /static or external JS.
    assert "/static/" not in DOCS
    assert "<script" not in DOCS


def test_docs_mentions_all_three_languages():
    for lang in ("German", "English", "Spanish"):
        assert lang in DOCS, f"landing page no longer mentions {lang}"


def test_docs_quickstart_matches_readme():
    # If the README's install steps change, the site must change with them.
    for cmd in ("brew install uv ollama", "ollama pull gemma3:12b",
                "uv sync", "uv run uvicorn server:app"):
        assert cmd in README and cmd in DOCS, f"quick-start drift: {cmd}"
    assert "8710" in DOCS  # the port users will open


def test_docs_links_point_at_this_repo():
    repos = set(re.findall(r"github\.com/([\w-]+/[\w-]+)", DOCS))
    assert "ismayc/AllKlaro" in repos
    # No placeholder like <your-username> may leak onto the public site.
    assert "<your-username>" not in DOCS
