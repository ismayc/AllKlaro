"""Gender lexicon: dict.cc compilation rules, lookup, and prompt injection."""
import server
from build_gender_lexicon import compile_lexicon


# ------------------------------------------------------------- compilation


def line(en, de, pos="noun"):
    return f"{en}\t{de}\t{pos}\t[gastr.]\n"


def test_same_spelling_loanwords_kept():
    lex = compile_lexicon([line("caipirinha [cocktail]", "Caipirinha {m} [Cocktail]"),
                           line("email", "E-Mail {f}")])
    assert lex["caipirinha"] == ("Caipirinha", "m")
    assert lex["email"] == ("E-Mail", "f")     # hyphen-insensitive match


def test_false_friends_and_translations_excluded():
    lex = compile_lexicon([line("gift", "Geschenk {n}"),      # different word
                           line("poison", "Gift {n}")])
    assert lex == {}   # "gift -> das Gift" must never be teachable


def test_multi_gender_entries_are_unusable():
    # dict.cc "Margarita {m} {f}" means both genders are correct.
    lex = compile_lexicon([line("margarita", "Margarita {m} {f} [Getränk]"),
                           line("cola", "Cola {f} {n}")])
    assert lex == {}


def test_conflicting_entries_across_lines_dropped():
    lex = compile_lexicon([line("joghurt", "Joghurt {m}"),
                           line("joghurt", "Joghurt {n}")])
    assert lex == {}


def test_non_nouns_plurals_and_comments_skipped():
    lex = compile_lexicon(["# comment line\n",
                           line("cool", "cool", pos="adj"),
                           line("chips", "Chips {pl}")])
    assert lex == {}


# ------------------------------------------------------- lookup & injection


def test_notes_only_when_translating_into_german(gender_lexicon):
    gender_lexicon([("caipirinha", "Caipirinha", "m")])
    note = server.gender_notes("A caipirinha, please.", "de")
    assert "caipirinha → der Caipirinha" in note
    assert server.gender_notes("A caipirinha, please.", "en") is None
    assert server.gender_notes("A caipirinha, please.", "es") is None


def test_no_matches_or_no_lexicon_mean_no_note(gender_lexicon):
    gender_lexicon([("caipirinha", "Caipirinha", "m")])
    assert server.gender_notes("Nice weather today.", "de") is None


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
