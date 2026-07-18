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


# ------------------------------------------------------- output maps & guard


def test_output_maps_cover_both_sides_and_drop_ambiguity():
    pairs = list(parse_dictcc([dictcc("margarita", "Margarita {m} {f}"),
                               dictcc("appointment", "Termin {m}")]))
    from build_gender_lexicon import build_output_maps
    maps = build_output_maps(pairs)
    # Termin never same-spells "appointment" (no source-keyed entry), but
    # the output map still knows its gender; ambiguous Margarita is dropped.
    assert maps["de"]["termin"] == ("Termin", "m")
    assert "margarita" not in maps["de"]


def test_agreement_flags_impossible_german_combinations(output_gender_map):
    output_gender_map([("Margarita", "f"), ("Termin", "m"), ("Meeting", "n")])
    # einen + feminine, das + feminine, eine + masculine, das + masculine:
    assert server.agreement_issues("Ich hätte gerne einen Margarita.", "de")
    assert server.agreement_issues("Das Margarita ist gut.", "de")
    assert server.agreement_issues("Ich habe eine Termin.", "de")
    assert server.agreement_issues("Das Termin ist morgen.", "de")
    assert server.agreement_issues("Mit einem Margarita feiern wir.", "de")
    # The message carries the corrective fact for the retry prompt.
    (issue,) = server.agreement_issues("Das Termin ist morgen.", "de")
    assert "masculine" in issue and "der Termin" in issue


def test_agreement_accepts_all_valid_german_cases(output_gender_map):
    output_gender_map([("Margarita", "f"), ("Termin", "m"), ("Meeting", "n"),
                       ("Lehrer", "m"), ("Fenster", "n")])
    for ok in (
        "Die Margarita schmeckt gut.",           # nominative feminine
        "Mit der Margarita stoßen wir an.",      # dative feminine ("der"!)
        "Wegen einer Margarita bleiben wir.",    # genitive feminine
        "Ich hätte gerne eine Margarita.",
        "Der Termin ist morgen.",
        "Ich habe einen Termin.",                # accusative masculine
        "Das Meeting beginnt gleich.",
        "Die Lehrer sind da.",                   # plural via "die" (unchecked)
        "Der Fenster wegen bleiben wir.",        # genitive plural via "der"
        "Ein bisschen Ruhe täte gut.",           # measure construction
        "Wir treffen uns um ein Uhr.",           # time idiom exception
    ):
        assert server.agreement_issues(ok, "de") == [], ok


def test_agreement_checks_adjective_endings(output_gender_map):
    output_gender_map([("Tag", "m"), ("Haus", "n")])
    assert server.agreement_issues("Das war ein schöne Tag.", "de")
    assert server.agreement_issues("Das ist das schönes Haus.", "de")
    for ok in ("Das war ein schöner Tag.",
               "Das ist das schöne Haus.",
               "In einem schönen Haus wohnen wir.",
               "Ein rosa Haus steht da.",):      # undeclinable color adj
        assert server.agreement_issues(ok, "de") == [], ok


def test_agreement_spanish_articles(output_gender_map):
    output_gender_map([("problema", "m"), ("casa", "f"), ("agua", "f")],
                      target="es")
    assert server.agreement_issues("La problema es difícil.", "es")
    assert server.agreement_issues("Es un casa bonita.", "es")
    for ok in ("El problema es difícil.",
               "La casa es bonita.",
               "El agua está fría.",):           # stressed a- feminine
        assert server.agreement_issues(ok, "es") == [], ok


def test_agreement_silent_without_output_map():
    assert server.agreement_issues("Das Margarita ist gut.", "de") == []


async def test_enforce_agreement_retries_and_verifies(output_gender_map,
                                                      fake_ollama):
    output_gender_map([("Margarita", "f")])
    # Violating candidate -> corrective call; fake returns a clean text.
    final, changed = await server.enforce_agreement(
        "I'd like a margarita.", "en", "de", "gemma3:12b", [],
        "Ich hätte gerne einen Margarita.")
    assert changed and final == "Refined translation."
    revise_turn = fake_ollama["chat"]["messages"][-1]
    assert "feminine" in revise_turn["content"]  # facts reached the model


async def test_enforce_agreement_no_issues_no_call(output_gender_map,
                                                   fake_ollama):
    output_gender_map([("Margarita", "f")])
    final, changed = await server.enforce_agreement(
        "x", "en", "de", "gemma3:12b", [], "Ich hätte gerne eine Margarita.")
    assert not changed and "chat" not in fake_ollama  # zero extra latency


# ------------------------------------------------------- prepositional case


def test_preposition_determines_case(output_gender_map):
    output_gender_map([("Frau", "f"), ("Termin", "m"), ("Handy", "n")])
    # Wrong case after a case-governing preposition:
    for bad in ("Ich tanze mit die Frau.",        # mit -> dative: der Frau
                "Ich tanze mit dem Frau.",
                "Das Geschenk ist für der Frau.",  # für -> acc: die Frau
                "Wir kommen ohne dem Termin aus.",
                "Ich rufe von das Handy an.",):
        assert server.agreement_issues(bad, "de"), bad
    # The message names the preposition's case and the required article.
    (issue,) = server.agreement_issues("Ich tanze mit die Frau.", "de")
    assert "dat" in issue and '"der" Frau' in issue


def test_preposition_correct_cases_accepted(output_gender_map):
    output_gender_map([("Frau", "f"), ("Termin", "m"), ("Handy", "n"),
                       ("Uhr", "f"), ("Deutsch", "n")])
    for ok in ("Ich tanze mit der Frau.",
                "Das Geschenk ist für die Frau.",
                "Wir reden nach dem Termin.",
                "Ich komme ohne das Handy.",
                "Wegen des Termins bleiben wir hier.",
                "Wegen dem Termin bleiben wir hier.",   # colloquial dative
                "Wir treffen uns um ein Uhr.",          # time idiom
                "Ich möchte mit ihr Deutsch üben.",     # ihr = pronoun here
                ):
        assert server.agreement_issues(ok, "de") == [], ok


def test_preposition_checks_possessive_determiners(output_gender_map):
    output_gender_map([("Handy", "n"), ("Frau", "f")])
    assert server.agreement_issues("Ich rufe mit mein Handy an.", "de")
    assert server.agreement_issues("Er kommt ohne seinem Handy.", "de")
    for ok in ("Ich rufe mit meinem Handy an.",
               "Er kommt ohne sein Handy.",
               "Sie tanzt mit ihrer Frau."):
        assert server.agreement_issues(ok, "de") == [], ok


def test_contractions_encode_gender(output_gender_map):
    output_gender_map([("Frau", "f"), ("Termin", "m"), ("Kino", "n")])
    assert server.agreement_issues("Wir gehen zur Termin.", "de")
    assert server.agreement_issues("Ich bin beim Frau.", "de")
    for ok in ("Wir gehen ins Kino.", "Ich bin beim Termin.",
               "Sie geht zur Frau.", "Wir sind im Kino."):
        assert server.agreement_issues(ok, "de") == [], ok


def test_prep_and_impossible_passes_do_not_double_flag(output_gender_map):
    output_gender_map([("Frau", "f")])
    # "mit das Frau" violates both passes; one clear message is enough.
    assert len(server.agreement_issues("Ich rede mit das Frau.", "de")) == 1
