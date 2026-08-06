"""Which Ollama models the UI picks, exercised as real JavaScript.

`resolvePair` is pure precisely so this can run it — the rest of app.js needs
a browser, but the pairing decision is the one piece where being wrong is
expensive and silent: an inverted pair still translates, just four times
slower, so nothing fails and nobody notices.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = (Path(__file__).parent.parent / "static" / "app.js").read_text()

# What's actually installed on the development machine, in bytes.
SIZES = {
    "gemma3:12b": 8.1e9,
    "translategemma:12b": 8.1e9,
    "translategemma:4b": 3.3e9,
    "qwen2.5:32b-instruct": 19e9,
    "qwen2.5:14b-instruct": 9.0e9,
    "qwen2.5:7b-instruct": 4.7e9,
    "llama3.2:3b": 2.0e9,
}
MODELS = list(SIZES)
DEFAULT = "gemma3:12b"


def extract(name: str) -> str:
    """Pull one top-level function out of app.js by name, closing brace and
    all. Relies on app.js keeping top-level functions unindented."""
    m = re.search(rf"^function {name}\(.*?^}}$", APP_JS, re.S | re.M)
    assert m, f"{name}() not found in app.js — was it renamed or indented?"
    return m.group(0)


def resolve_pair(saved, models=None, sizes=None, default=DEFAULT):
    """Run the real app.js resolvePair() under node."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    consts = re.search(r"^const MIN_DRAFT_BYTES = .*?;$", APP_JS, re.M)
    assert consts, "MIN_DRAFT_BYTES not found in app.js"
    script = f"""
{consts.group(0)}
{extract("resolvePair")}
const i = JSON.parse(require("fs").readFileSync(0, "utf8"));
console.log(JSON.stringify(resolvePair(i.saved, i.models, i.sizes, i.def)));
"""
    payload = json.dumps({"saved": saved,
                          "models": MODELS if models is None else models,
                          "sizes": SIZES if sizes is None else sizes,
                          "def": default})
    out = subprocess.run([node, "-e", script], input=payload, text=True,
                         capture_output=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_inverted_saved_pair_is_discarded():
    """The real setting found on the development machine: the 19 GB model as
    the *draft* for a 4.7 GB main. Measured over a 54-minute conversation,
    that is a first-word lag p50 of 57 s against 15 s for the right way round,
    so a saved pairing like this must not survive a reload."""
    pair = resolve_pair({"model": "qwen2.5:7b-instruct",
                         "draft": "qwen2.5:32b-instruct"})
    assert pair == {"model": "gemma3:12b", "draft": "qwen2.5:7b-instruct"}


def test_equal_sized_draft_is_also_inverted():
    """A draft the same size as the main model buys nothing and costs a second
    model resident in memory."""
    pair = resolve_pair({"model": "gemma3:12b",
                         "draft": "translategemma:12b"})
    assert pair["draft"] != "translategemma:12b"
    assert SIZES[pair["draft"]] < SIZES[pair["model"]]


def test_a_sane_saved_pair_is_left_alone():
    """Control: the repair must not touch a pairing that is already right."""
    saved = {"model": "gemma3:12b", "draft": "qwen2.5:7b-instruct"}
    assert resolve_pair(saved) == saved


def test_smaller_hand_picked_draft_survives():
    """Any smaller draft is the user's call, even one below MIN_DRAFT_BYTES
    that the automatic default would never have chosen."""
    saved = {"model": "gemma3:12b", "draft": "llama3.2:3b"}
    assert resolve_pair(saved) == saved


def test_draft_turned_off_stays_off():
    """"" is a real choice — single-pass translation — not a missing setting."""
    assert resolve_pair({"model": "gemma3:12b", "draft": ""})["draft"] == ""


def test_no_saved_settings_uses_the_server_default_and_a_fast_draft():
    pair = resolve_pair({})
    assert pair["model"] == DEFAULT
    assert pair["draft"] == "qwen2.5:7b-instruct"   # smallest above 4 GB


def test_uninstalled_saved_model_falls_back_to_the_default():
    pair = resolve_pair({"model": "mistral:7b", "draft": "gone:1b"})
    assert pair["model"] == DEFAULT
    assert pair["draft"] == "qwen2.5:7b-instruct"


def test_default_not_installed_falls_back_to_a_listed_model():
    models = ["llama3.2:3b", "qwen2.5:7b-instruct"]
    pair = resolve_pair({}, models=models)
    assert pair["model"] in models


def test_only_tiny_models_still_gets_a_draft():
    """With nothing above MIN_DRAFT_BYTES the largest small model is the
    least-bad draft — better than silently dropping the draft pass."""
    models = ["llama3.2:3b", "translategemma:4b", "qwen2.5:7b-instruct"]
    pair = resolve_pair({}, models=models, default="qwen2.5:7b-instruct")
    assert pair == {"model": "qwen2.5:7b-instruct", "draft": "translategemma:4b"}


def test_single_model_gets_no_draft():
    pair = resolve_pair({}, models=["gemma3:12b"])
    assert pair == {"model": "gemma3:12b", "draft": ""}
