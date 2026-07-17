"""Gender lexicons: dictionary compilation rules, lookup, and prompt injection."""
import io

import server
from build_gender_lexicon import (build_lexicons, loose_match, parse_dictcc,
                                  parse_tei)


def dictcc(en, de):
    return f"{en}\t{de}\tnoun\t[gastr.]\n"


def tei(body):
    xml = ("<TEI xmlns='http://www.tei-c.org/ns/1.0'><text><body>"
           f"{body}</body></text></TEI>")
    return io.BytesIO(xml.encode())


def build(*pair_iters):
    return build_lexicons(p for it in pair_iters for p in it)


# ------------------------------------------------------------------ matching


def test_loose_match_is_conservative():
    assert loose_match("caipirinha", "Caipirinha")   # case
    assert loose_match("email", "E-Mail")            # hyphens
    assert loose_match("problem", "problema")        # trailing romance vowel
    assert loose_match("map", "mapa")
    assert not loose_match("gift", "Geschenk")
    assert not loose_match("embarrassed", "embarazada")
    assert not loose_match("map", "mapea")           # only ONE trailing vowel


# ------------------------------------------------------------------- dict.cc


def test_dictcc_same_spelling_loanwords_kept():
    lex = build(parse_dictcc([dictcc("caipirinha [cocktail]", "Caipirinha {m} [Cocktail]"),
                              dictcc("email", "E-Mail {f}")]))
    assert lex["de"]["caipirinha"] == ("Caipirinha", "m")
    assert lex["de"]["email"] == ("E-Mail", "f")


def test_dictcc_translations_and_false_friends_excluded():
    lex = build(parse_dictcc([dictcc("gift", "Geschenk {n}"),
                              dictcc("poison", "Gift {n}")]))
    assert lex["de"] == {}   # "gift -> das Gift" must never be teachable


def test_multi_gender_entries_are_unusable():
    # dict.cc "Margarita {m} {f}" means both genders are correct.
    lex = build(parse_dictcc([dictcc("margarita", "Margarita {m} {f} [Getränk]")]))
    assert lex["de"] == {}


def test_conflicting_gender_across_lines_dropped():
    lex = build(parse_dictcc([dictcc("joghurt", "Joghurt {m}"),
                              dictcc("joghurt", "Joghurt {n}")]))
    assert lex["de"] == {}


# ------------------------------------------------------------------ TEI forms


def test_tei_headword_gender_deu_eng_style():
    pairs = parse_tei(tei("""
      <entry><form><orth>Problem</orth></form>
        <gramGrp><gen>neut</gen><pos>n</pos></gramGrp>
        <sense><cit type="trans"><quote>problem</quote></cit></sense>
      </entry>"""), "de", "en")
    assert build(pairs)["de"]["problem"] == ("Problem", "n")


def test_tei_per_translation_gender_eng_deu_style():
    pairs = parse_tei(tei("""
      <entry><form><orth>problem</orth><gramGrp><pos>n</pos></gramGrp></form>
        <sense>
          <cit type="trans"><quote>Fragestellung</quote>
            <gramGrp><gen>fem</gen></gramGrp></cit>
          <cit type="trans"><quote>Problem</quote>
            <gramGrp><gen>neut</gen></gramGrp></cit>
        </sense>
      </entry>"""), "en", "de")
    lex = build(pairs)
    assert lex["de"]["problem"] == ("Problem", "n")   # same-spelling only
    assert "fragestellung" not in lex["de"]


def test_tei_sense_level_gender_feeds_both_targets():
    # spa-deu style: Spanish head gender + German translation gender.
    pairs = list(parse_tei(tei("""
      <entry><form><orth>problema</orth>
          <gramGrp><pos>n</pos><gen>m</gen></gramGrp></form>
        <sense><cit type="trans"><quote>Problem</quote></cit>
          <gramGrp><pos>n</pos><gen>n</gen></gramGrp></sense>
      </entry>"""), "es", "de"))
    lex = build(pairs)
    assert lex["de"]["problema"] == ("Problem", "n")  # for ES -> DE speech
    assert lex["es"]["problem"] == ("problema", "m")  # for DE -> ES speech


def test_gender_pool_join_for_genderless_dictionaries():
    # eng-spa has no genders; spa-deu's observation of "mapa" supplies one.
    eng_spa = parse_tei(tei("""
      <entry><form><orth>map</orth></form><gramGrp><pos>noun</pos></gramGrp>
        <sense><cit type="trans"><quote>mapa</quote></cit></sense>
      </entry>"""), "en", "es")
    spa_deu = parse_tei(tei("""
      <entry><form><orth>mapa</orth>
          <gramGrp><pos>n</pos><gen>m</gen></gramGrp></form>
        <sense><cit type="trans"><quote>Landkarte</quote></cit>
          <gramGrp><gen>f</gen></gramGrp></sense>
      </entry>"""), "es", "de")
    assert build(eng_spa, spa_deu)["es"]["map"] == ("mapa", "m")


def test_pool_disagreement_blocks_the_join():
    # la radio (broadcast) vs el radio (radius): nothing safe to teach.
    eng_spa = parse_tei(tei("""
      <entry><form><orth>radio</orth></form><gramGrp><pos>noun</pos></gramGrp>
        <sense><cit type="trans"><quote>radio</quote></cit></sense>
      </entry>"""), "en", "es")
    spa_deu = parse_tei(tei("""
      <entry><form><orth>radio</orth>
          <gramGrp><pos>n</pos><gen>f</gen></gramGrp></form>
        <sense><cit type="trans"><quote>Rundfunk</quote></cit></sense>
      </entry>
      <entry><form><orth>radio</orth>
          <gramGrp><pos>n</pos><gen>m</gen></gramGrp></form>
        <sense><cit type="trans"><quote>Radius</quote></cit></sense>
      </entry>"""), "es", "de")
    assert "radio" not in build(eng_spa, spa_deu)["es"]


def test_multiword_and_non_noun_entries_skipped():
    pairs = parse_tei(tei("""
      <entry><form><orth>problem child</orth></form>
        <gramGrp><pos>n</pos></gramGrp>
        <sense><cit type="trans"><quote>Problemkind</quote>
          <gramGrp><gen>neut</gen></gramGrp></cit></sense>
      </entry>
      <entry><form><orth>cool</orth></form><gramGrp><pos>adj</pos></gramGrp>
        <sense><cit type="trans"><quote>cool</quote></cit></sense>
      </entry>"""), "en", "de")
    assert build(pairs)["de"] == {}


# ------------------------------------------------------- lookup & injection


def test_notes_for_german_and_spanish_targets(gender_lexicon):
    gender_lexicon([("caipirinha", "Caipirinha", "m")], target="de")
    gender_lexicon([("problem", "problema", "m")], target="es")
    assert "caipirinha → der Caipirinha" in server.gender_notes(
        "A caipirinha, please.", "de")
    assert "problem → el problema" in server.gender_notes(
        "This problem is hard.", "es")
    assert server.gender_notes("A caipirinha, please.", "en") is None
    assert server.gender_notes("This problem is hard.", "de") is None


def test_no_matches_or_no_lexicon_mean_no_note(gender_lexicon):
    gender_lexicon([("caipirinha", "Caipirinha", "m")])
    assert server.gender_notes("Nice weather today.", "de") is None
    assert server.gender_notes("A caipirinha, please.", "es") is None


def test_notes_deduplicate_and_cap(gender_lexicon):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliett", "kilo", "lima"]
    gender_lexicon([(w, w.capitalize(), "n") for w in words])
    note = server.gender_notes("alpha alpha " + " ".join(words), "de")
    assert note.count("→") == server.GENDER_NOTE_LIMIT
    assert note.count("alpha") == 1


def test_prompt_gets_note_with_precedence_over_grammar_rule(gender_lexicon):
    gender_lexicon([("caipirinha", "Caipirinha", "m")])
    msgs = server.translation_messages("I'd like a caipirinha.", "en", "de")
    system = msgs[0]["content"]
    assert "caipirinha → der Caipirinha" in system
    assert "overriding any general rule" in system
    # Static parts (grammar rule, base prompt) still precede the dynamic note.
    assert system.index("drinks and cocktails") < system.index("der Caipirinha")


def test_prompt_unchanged_when_nothing_matches(gender_lexicon):
    gender_lexicon([("caipirinha", "Caipirinha", "m")])
    msgs = server.translation_messages("Schönes Wetter.", "de", "en")
    assert "Dictionary genders" not in msgs[0]["content"]
