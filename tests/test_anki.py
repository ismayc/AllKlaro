"""The Anki vocab export (/api/anki) and the word-selection behind it.

Turns a conversation into a flashcard deck: each heard German word reduced to
its dictionary headword, the trivial high-frequency ones and the listener's
known words filtered out, one card per lemma.
"""
import json

import build_wiktionary_lexicon as bwl
import pytest

import server as srv

# Zipf frequencies (checked against wordfreq's German list): Tisch 4.83 and
# Gartenhaus 3.17 survive the 5.0 gate; Haus 5.41 and der/und/im do not.
TISCH = {
    "word": "Tisch", "lang_code": "de", "pos": "noun",
    "sounds": [{"ipa": "/tɪʃ/"}],
    "forms": [{"form": "Tische", "tags": ["plural"]}],
    "senses": [{"glosses": ["dining table"], "tags": ["masculine"]},
               {"glosses": ["desk"], "tags": ["masculine"]}],
}
GARTENHAUS = {
    "word": "Gartenhaus", "lang_code": "de", "pos": "noun",
    "senses": [{"glosses": ["garden house"], "tags": ["neuter"]}],
}
HAUS = {
    "word": "Haus", "lang_code": "de", "pos": "noun",
    "senses": [{"glosses": ["house"], "tags": ["neuter"]}],
}
GING = {
    "word": "ging", "lang_code": "de", "pos": "verb",
    "senses": [{"glosses": ["preterite of gehen"], "tags": ["form-of"],
                "form_of": [{"word": "gehen"}]}],
}
GEHEN = {
    "word": "gehen", "lang_code": "de", "pos": "verb",
    "senses": [{"glosses": ["to go, to walk"]}],
}
ENTRIES = (TISCH, GARTENHAUS, HAUS, GING, GEHEN)


@pytest.fixture
def wikt_de(tmp_path):
    path = tmp_path / "wikt_de.sqlite"
    bwl.compile_lexicon(iter(json.dumps(e) for e in ENTRIES), "de", path)
    srv.WIKTIONARY_PATHS["de"] = path
    srv._wikt_conns.clear()
    return srv.wiktionary_conn("de")


@pytest.fixture
def known(tmp_path, monkeypatch):
    def use(words):
        p = tmp_path / "known.txt"
        p.write_text("# mine\n" + "\n".join(words) + "\n")
        monkeypatch.setattr(srv, "KNOWN_WORDS_PATH", p)
        monkeypatch.setattr(srv, "_known_cache",
                            {"mtime": None, "words": frozenset()})
    return use


def de(*texts):
    return [{"source": "de", "text": t} for t in texts]


# ----------------------------------------------------------- lemma resolution

def test_lemma_from_the_noun_paradigm_table(wikt_de, noun_forms):
    noun_forms([("tische", "Tisch", "m", "np,ap")])
    assert srv._vocab_lemma("Tische", wikt_de, srv.load_noun_forms()) == "Tisch"


def test_lemma_from_an_inflected_wiktionary_entry(wikt_de):
    assert srv._vocab_lemma("ging", wikt_de, {}) == "gehen"


def test_lemma_falls_back_to_the_word_itself(wikt_de):
    # A base word links to no other lemma, and an unknown word has no entry.
    assert srv._vocab_lemma("Tisch", wikt_de, {}) == "Tisch"
    assert srv._vocab_lemma("Quztl", wikt_de, {}) == "Quztl"


# ------------------------------------------------------------- card faces

def test_a_noun_gets_its_article_and_ipa(wikt_de):
    front, back = srv._card_faces(srv._vocab_entry(wikt_de, "Tisch"), "de")
    assert front == "der Tisch"
    assert back == "dining table; desk  /tɪʃ/"


def test_a_verb_has_no_article_and_may_lack_ipa(wikt_de):
    front, back = srv._card_faces(srv._vocab_entry(wikt_de, "gehen"), "de")
    assert front == "gehen" and back == "to go, to walk"


def test_a_word_with_no_glossed_entry_is_none(wikt_de):
    assert srv._vocab_entry(wikt_de, "Quztl") is None


# ---------------------------------------------------------- word selection

def test_collect_keeps_rare_content_words_in_spoken_order(wikt_de):
    cards = srv.collect_vocab(de("Das Gartenhaus.", "Der Tisch."), "de")
    assert [c["front"] for c in cards] == ["das Gartenhaus", "der Tisch"]
    assert cards[0]["example"] == "Das Gartenhaus."


def test_collect_drops_common_words_and_words_without_an_entry(wikt_de):
    # Haus (Zipf 5.41) is over the gate; "Quztl" has no entry; "im"/"der" are
    # both common and short. Nothing review-worthy is left.
    assert srv.collect_vocab(de("Der im Haus. Quztl."), "de") == []


def test_collect_collapses_inflected_forms_onto_one_card(wikt_de, noun_forms):
    noun_forms([("tische", "Tisch", "m", "np,ap")])
    cards = srv.collect_vocab(de("Ein Tisch, zwei Tische."), "de")
    assert [c["front"] for c in cards] == ["der Tisch"]


def test_collect_skips_the_known_list(wikt_de, known):
    known(["gartenhaus"])           # matched on the lemma
    assert srv.collect_vocab(de("Das Gartenhaus."), "de") == []


def test_collect_skips_a_known_surface_form(wikt_de, known):
    known(["tische"])               # matched on the heard token, not the lemma
    assert srv.collect_vocab(de("Zwei Tische."), "de") == []


def test_collect_ignores_non_german_and_malformed_items(wikt_de):
    items = ["not a dict",
             {"source": "en", "text": "The Tisch."},   # wrong language
             {"source": "de", "text": None},            # no text
             {"source": "de", "text": "a Tisch"}]       # "a" is too short
    cards = srv.collect_vocab(items, "de")
    assert [c["front"] for c in cards] == ["der Tisch"]


def test_collect_without_a_dictionary_is_none():
    # The autouse fixture points the paths at files that do not exist.
    assert srv.collect_vocab(de("Der Tisch."), "de") is None


# ------------------------------------------------------------- the endpoint

def test_export_returns_a_tab_separated_deck(client, wikt_de):
    body = client.post("/api/anki",
                       json={"items": de("Der Tisch im Gartenhaus.")}).json()
    assert body["count"] == 2
    lines = body["deck"].splitlines()
    assert lines[:2] == ["#separator:tab", "#html:true"]
    assert lines[2] == ("der Tisch\tdining table; desk  /tɪʃ/"
                        "<br>Der Tisch im Gartenhaus.")
    assert lines[3].startswith("das Gartenhaus\tgarden house<br>")


def test_export_caps_the_deck_size(client, wikt_de, monkeypatch):
    monkeypatch.setattr(srv, "ANKI_MAX_CARDS", 1)
    body = client.post("/api/anki",
                       json={"items": de("Der Tisch im Gartenhaus.")}).json()
    assert body["count"] == 1


def test_export_rejects_an_unknown_language(client, wikt_de):
    assert "error" in client.post("/api/anki", json={"lang": "fr"}).json()


def test_export_without_a_dictionary_says_how_to_build_it(client):
    body = client.post("/api/anki", json={"items": de("Der Tisch.")}).json()
    assert "build_wiktionary_lexicon.py de" in body["error"]


def test_export_with_nothing_worth_keeping_says_so(client, wikt_de):
    body = client.post("/api/anki", json={"items": de("Der im Haus.")}).json()
    assert body["error"] == "No new vocabulary to export."


def test_export_tolerates_a_non_list_items_field(client, wikt_de):
    body = client.post("/api/anki", json={"items": "oops"}).json()
    assert body["error"] == "No new vocabulary to export."


def test_anki_field_neutralises_tabs_and_newlines():
    assert srv._anki_field("a\tb\r\nc\nd") == "a b<br>c<br>d"
