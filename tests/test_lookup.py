"""The Wiktionary lexicon builder and the /api/lookup endpoint behind the
long-press word popup."""
import json

import build_wiktionary_lexicon as bwl
import server

TISCH = {
    "word": "Tisch", "lang_code": "de", "pos": "noun",
    "sounds": [{"rhymes": "-ɪʃ"}, {"ipa": "/tɪʃ/"}],
    "forms": [
        {"form": "Tisches", "tags": ["genitive"]},
        {"form": "Tische", "tags": ["plural"]},
        {"form": "Tischlein", "tags": ["diminutive", "neuter"]},
        {"form": "Tischen", "tags": ["dative", "plural"]},
        {"form": "de-ndecl", "tags": ["inflection-template"]},
    ],
    "senses": [
        {"glosses": ["table (a piece of furniture); specific uses include:",
                     "dining table"], "tags": ["masculine", "strong"]},
        {"glosses": ["table (a piece of furniture); specific uses include:",
                     "desk"], "tags": ["masculine", "strong"]},
    ],
}
GING = {
    "word": "ging", "lang_code": "de", "pos": "verb",
    "sounds": [{"ipa": "/ɡɪŋ/"}],
    "senses": [{"glosses": ["first/third-person singular preterite of gehen"],
                "tags": ["form-of", "preterite"],
                "form_of": [{"word": "gehen"}]}],
}
GEHEN = {
    "word": "gehen", "lang_code": "de", "pos": "verb",
    "senses": [{"glosses": ["to go, to walk"]}],
}


def test_extract_noun_gets_gender_ipa_and_citation_plural():
    row = bwl.extract_entry(TISCH, "de")
    word, word_lc, pos, gender, ipa, plural, senses, lemma = row
    assert (word, word_lc, pos, gender, ipa) == \
        ("Tisch", "tisch", "noun", "m", "/tɪʃ/")
    # The most specific gloss is kept; the shared parent gloss is not.
    assert json.loads(senses) == ["dining table", "desk"]
    # Genitive, dative plural, diminutive, and template rows are not plurals.
    assert plural == "Tische"
    assert lemma == ""


def test_extract_inflected_form_links_to_its_lemma():
    row = bwl.extract_entry(GING, "de")
    assert row[0] == "ging" and row[-1] == "gehen"


def test_extract_rejects_other_languages_and_glossless_entries():
    assert bwl.extract_entry({**TISCH, "lang_code": "en"}, "de") is None
    assert bwl.extract_entry({**TISCH, "senses": []}, "de") is None


def test_extract_reports_all_genders_of_a_wobbly_noun():
    joghurt = {"word": "Joghurt", "lang_code": "de", "pos": "noun",
               "senses": [{"glosses": ["yogurt"],
                           "tags": ["masculine", "neuter"]}]}
    assert bwl.extract_entry(joghurt, "de")[3] == "m/n"


def test_extract_ignores_plural_forms_on_verbs():
    verb = {"word": "gehen", "lang_code": "de", "pos": "verb",
            "senses": [{"glosses": ["to go"]}],
            "forms": [{"form": "gehen", "tags": ["plural", "present"]}]}
    assert bwl.extract_entry(verb, "de")[5] == ""


def build_db(tmp_path, entries=(TISCH, GING, GEHEN)):
    path = tmp_path / "wikt_de.sqlite"
    lines = [json.dumps(e) for e in entries] + ["not json"]
    kept = bwl.compile_lexicon(iter(lines), "de", path)
    assert kept == len(entries)  # the junk line is skipped, not fatal
    return path


def test_lookup_returns_entry_case_insensitively(client, tmp_path):
    server.WIKTIONARY_PATHS["de"] = build_db(tmp_path)
    data = client.get("/api/lookup", params={"word": "tisch", "lang": "de"}).json()
    entry = data["entries"][0]
    assert entry["word"] == "Tisch" and entry["gender"] == "m"
    assert entry["plural"] == "Tische" and "dining table" in entry["senses"]


def test_lookup_chains_an_inflected_form_to_its_lemma(client, tmp_path):
    server.WIKTIONARY_PATHS["de"] = build_db(tmp_path)
    data = client.get("/api/lookup", params={"word": "ging", "lang": "de"}).json()
    words = [e["word"] for e in data["entries"]]
    assert words == ["ging", "gehen"]  # the base word rides along
    assert "to go, to walk" in data["entries"][1]["senses"]


def test_lookup_without_a_built_lexicon_says_how_to_build_it(client):
    data = client.get("/api/lookup", params={"word": "Tisch", "lang": "de"}).json()
    assert "build_wiktionary_lexicon.py de" in data["error"]


def test_lookup_rejects_bad_input(client):
    assert "error" in client.get(
        "/api/lookup", params={"word": "Tisch", "lang": "fr"}).json()
    assert "error" in client.get(
        "/api/lookup", params={"word": " ", "lang": "de"}).json()
    assert "error" in client.get(
        "/api/lookup", params={"word": "x" * 65, "lang": "de"}).json()


def test_rebuild_replaces_the_database_atomically(tmp_path):
    path = build_db(tmp_path)
    build_db(tmp_path)  # second build must overwrite, not append
    conn = __import__("sqlite3").connect(path)
    assert conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 3
    conn.close()
