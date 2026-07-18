"""Consistency checks for the GitHub Pages site in docs/ — it must stay
self-contained and not drift from what the app/README actually say."""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS = (ROOT / "docs" / "index.html").read_text()
README = (ROOT / "README.md").read_text()


def test_docs_page_is_self_contained():
    # GitHub Pages serves docs/ alone — no reaching into /static, and any
    # script must be inline (no external sources).
    assert "/static/" not in DOCS
    assert not re.search(r"<script[^>]*\bsrc\s*=", DOCS)


def test_language_toggle_covers_de_and_es():
    # Toggle buttons + a translation entry for every data-i18n key.
    for lang in ("en", "de", "es"):
        assert f'data-lang="{lang}"' in DOCS
    keys = set(re.findall(r'data-i18n="([^"]+)"', DOCS))
    assert len(keys) > 30, "translatable regions vanished?"
    for lang in ("de", "es"):
        block = re.search(lang + r":\s*\{(.*?)\n    \},", DOCS, re.S).group(1)
        missing = {k for k in keys if not re.search(rf"\b{k}:", block)}
        assert not missing, f"{lang} translation missing keys: {missing}"
    # Deep-linkable for sharing (?lang=de) and persisted across visits.
    assert "URLSearchParams" in DOCS and '.get("lang")' in DOCS
    assert "localStorage" in DOCS
    assert "documentElement.lang" in DOCS  # <html lang> follows the toggle
    # Spot-check the translations are actually there.
    assert "Schnellstart" in DOCS and "Inicio rápido" in DOCS
    # Browser translate prompts are suppressed — the page translates itself.
    assert 'name="google" content="notranslate"' in DOCS


def test_features_are_grouped_into_sections():
    # The tile wall outgrew one screenful; group headings keep it scannable.
    groups = re.findall(r'class="group-h" data-i18n="(g_\w+)"', DOCS)
    assert len(groups) >= 4 and len(set(groups)) == len(groups)
    # Each group heading introduces its own tile grid ("Under the hood"
    # keeps a grid of its own, hence the +1).
    assert DOCS.count('<div class="grid">') == len(groups) + 1
    # The newest user-facing features made it onto the site.
    for key in ("f_copy_h", "f_big_h", "f_lookup_h"):
        assert f'data-i18n="{key}"' in DOCS


def test_docs_mentions_all_three_languages():
    for lang in ("German", "English", "Spanish"):
        assert lang in DOCS, f"landing page no longer mentions {lang}"


def test_docs_quickstart_matches_readme():
    # If the README's install steps change, the site must change with them.
    for cmd in ("brew install uv ollama", "ollama pull gemma3:12b",
                "uv sync", "uv run uvicorn server:app"):
        assert cmd in README and cmd in DOCS, f"quick-start drift: {cmd}"
    assert "8710" in DOCS  # the port users will open


def _png_size(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    return (int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"))


def test_docs_link_preview_card():
    # iMessage/Slack previews come from Open Graph tags + a hosted image.
    m = re.search(r'property="og:image" content="([^"]+)"', DOCS)
    assert m, "og:image tag missing"
    name = m.group(1).rsplit("/", 1)[-1]
    img = ROOT / "docs" / name
    assert img.exists(), f"og:image points at missing file {name}"
    assert _png_size(img) == (1200, 630)
    for tag in ('property="og:title"', 'property="og:description"',
                'name="twitter:card" content="summary_large_image"'):
        assert tag in DOCS, f"missing {tag}"
    assert _png_size(ROOT / "docs" / "apple-touch-icon.png") == (180, 180)


def test_docs_links_point_at_this_repo():
    repos = set(re.findall(r"github\.com/([\w-]+/[\w-]+)", DOCS))
    assert "ismayc/AllKlaro" in repos
    # No placeholder like <your-username> may leak onto the public site.
    assert "<your-username>" not in DOCS
