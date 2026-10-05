"""The feed follows the newest text only while the reader is already there.

Every token used to force the feed to the bottom, so scrolling up to reread
a card was undone a moment later. This runs the page's own follow logic
against a stand-in feed.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "static"

HARNESS = """
const listeners = {};
const feed = { scrollHeight: 1000, scrollTop: 600, clientHeight: 400,
               addEventListener: (name, fn) => { listeners[name] = fn; } };
const hidden = new Set(["hidden"]);
const latestBtn = { classList: { add: (c) => hidden.add(c),
                                 remove: (c) => hidden.delete(c) } };
%s
const out = [];
const state = () => ({ top: feed.scrollTop, button: !hidden.has("hidden") });
feed.scrollHeight = 1200; followFeed(); out.push(state());      // at the end
feed.scrollTop = 100; listeners.scroll();                       // reader scrolls up
feed.scrollHeight = 1500; followFeed(); out.push(state());      // new text
latestBtn.onclick(); out.push(state());                         // back to latest
feed.scrollHeight = 1700; followFeed(); out.push(state());
feed.scrollTop = 1700 - 400 - 50; listeners.scroll();           // within the slack
feed.scrollHeight = 1800; followFeed(); out.push(state());
console.log(JSON.stringify(out));
"""


def follow_logic() -> str:
    js = (STATIC / "app.js").read_text()
    start = js.index("const FOLLOW_SLACK_PX")
    end = js.index("};", js.index("latestBtn.onclick", start)) + 2
    return js[start:end]


@pytest.fixture
def run():
    if not shutil.which("node"):
        pytest.skip("node is not installed")
    out = subprocess.run(["node", "-e", HARNESS % follow_logic()],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def test_it_follows_new_text_while_the_reader_is_at_the_end(run):
    assert run[0] == {"top": 1200, "button": False}


def test_a_reader_who_scrolled_up_stays_put_and_is_offered_the_way_back(run):
    assert run[1] == {"top": 100, "button": True}


def test_the_button_returns_to_the_newest_text_and_following_resumes(run):
    assert run[2] == {"top": 1500, "button": False}
    assert run[3] == {"top": 1700, "button": False}


def test_a_small_nudge_near_the_end_does_not_stop_following(run):
    assert run[4] == {"top": 1800, "button": False}


def test_live_text_no_longer_forces_the_feed_down():
    """Only the two user-initiated jumps (Center latest, Summarize) and the
    follow logic itself may set the scroll position directly."""
    js = (STATIC / "app.js").read_text()
    assert js.count("feed.scrollTop = feed.scrollHeight") == 4
    assert js.count("followFeed();") >= 5
    assert 'id="latestBtn"' in (STATIC / "index.html").read_text()
