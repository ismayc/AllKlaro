"""Dialect detection and translation hints (German and Spanish regions)."""
import server


def test_dialects_file_parses_with_ambiguity_flags():
    lex = server.load_dialects()
    assert lex["de"]["ick"] == ("ich", False, frozenset({"berlin"}))
    assert lex["de"]["nochemol"][1] is False
    # Untagged is still a meaningful state: it means "every dialect", which is
    # correct for a contraction that is colloquial rather than regional.
    assert lex["de"]["haste"] == ("hast du", False, None)
    assert lex["de"]["nett"][1] is True          # real standard word too
    assert lex["de"]["des"][1] is True
    # The [es] section holds the Spanish regional entries.
    assert lex["es"]["chido"][1] is False
    assert lex["es"]["vale"][1] is True          # everyday standard word
    assert "chido" not in lex["de"]              # sections don't bleed


def test_ambiguous_entries_name_the_dialect_they_belong_to():
    """Which dialect an ambiguous word belongs to only matters once a dialect
    is selected — and then it matters a lot: "mehr" is Rhine-Hessian "mer"
    (wir), and offering that reading to someone listening to a Berliner would
    be a new error rather than a fix."""
    lex = server.load_dialects()
    assert lex["de"]["nett"][2] == frozenset({"hessian", "worms"})
    assert lex["de"]["mehr"][2] == frozenset({"hessian", "worms"})
    assert "berlin" not in lex["de"]["des"][2]
    # Unambiguous markers name their dialect too, which is what makes the
    # style selector change what gets coloured in the heard text.
    assert lex["de"]["ick"][2] == frozenset({"berlin"})
    assert lex["de"]["isch"][2] == frozenset({"hessian", "worms"})


def test_markers_trigger_note_with_mishearing_hints():
    note = server.dialect_notes(
        "Ich hab des nett verstarne, kannste des nochemol saache?", "de")
    assert '"nochemol" = noch einmal' in note
    assert '"nett" = net (nicht)' in note        # hinted thanks to markers
    assert '"des" = das' in note


def test_ambiguous_words_alone_prove_nothing():
    # "nett", "des", "mehr" are everyday standard German — no note without
    # an unambiguous dialect marker in the same sentence.
    assert server.dialect_notes("Das ist nett von dir.", "de") is None
    assert server.dialect_notes("Wegen des Termins brauche ich mehr Zeit.",
                                "de") is None


def test_note_only_for_german_sources():
    assert server.dialect_notes("Ick sage wat.", "en") is None
    assert server.dialect_notes("Ick sage wat.", "es") is None
    assert server.dialect_notes("Ick sage wat.", "de") is not None


def test_note_reaches_the_translation_prompt():
    msgs = server.translation_messages("Dit is ooch jut, wa?", "de", "en")
    system = msgs[0]["content"]
    assert "regional dialect" in system
    assert '"dit" = das' in system
    plain = server.translation_messages("Das ist auch gut.", "de", "en")
    assert "regional dialect" not in plain[0]["content"]


def test_hints_are_deduplicated_and_capped():
    text = "Ick ick " + " ".join(["dit", "wat", "ooch", "keene", "nüscht",
                                  "kiek", "jut", "janz", "jenau", "morjen",
                                  "uff", "een"])
    note = server.dialect_notes(text, "de")
    assert note.count('"ick"') == 1
    assert note.count("=") <= 10


def test_wormser_platt_markers_are_recognized():
    # Rheinhessisch/Wormser forms from WhatsApp messages get the same
    # intended-meaning hints as spoken dialect.
    note = server.dialect_notes(
        "Alla, hoscht du die Grumbeere un de Woi geholt?", "de")
    assert '"alla"' in note
    assert '"hoscht" = hast' in note
    assert '"grumbeere" = Kartoffeln (potatoes)' in note
    assert '"woi" = Wein (wine)' in note
    # "nää" — the signature Wormser word per the Stadt Worms dialect page.
    assert '"nää" = nein' in server.dialect_notes(
        "Nää, dabber gehe mer heim.", "de")


def test_spanish_regional_markers_are_recognized():
    note = server.dialect_notes(
        "Órale güey, ¿me ayudas con la chamba ahorita?", "es")
    assert '"chamba" = trabajo (job)' in note
    assert '"ahorita"' in note
    assert "Mexican" in note
    barna = server.dialect_notes("Flipo tío, vale, plego ya.", "es")
    assert '"plego"' in barna and '"tío"' in barna


def test_spanish_ambiguous_words_alone_prove_nothing():
    # "vale", "tío", "padre" are everyday standard Spanish — no note
    # without an unambiguous regional marker in the same sentence.
    assert server.dialect_notes("Vale, nos vemos mañana.", "es") is None
    assert server.dialect_notes("Mi padre y mi tío llegan hoy.", "es") is None


# ------------------------------- dialect the user selected, not one we guess


def test_selected_dialect_reaches_the_source_side():
    """`dialect_notes` infers dialect from spelling, which speech never
    supplies: Whisper normalises dialect to standard orthography. Measured
    over 514 word tokens of the real Berlin recording, not one unambiguous
    marker appeared, so the inferred hint could never fire. The flavor the
    user picked is the missing signal — an assertion, not a guess."""
    plain = server.translation_messages("Ich habe das nett verstanden.",
                                        "de", "en")[0]["content"]
    assert "Berlinerisch" not in plain

    heard = server.translation_messages("Ich habe das nett verstanden.",
                                        "de", "en",
                                        heard_flavor="berlin")[0]["content"]
    assert "Berlinerisch" in heard
    assert "transcribed speech" in heard


def test_source_side_hint_is_keyed_on_the_source_language():
    """Writing replies in Berlinerisch says nothing about the Spanish being
    spoken to you — the note must follow the source, not the target."""
    msgs = server.translation_messages("¿Vale, quedamos ahorita?", "es", "en",
                                       heard_flavor="berlin")[0]["content"]
    assert "Berlinerisch" not in msgs
    mex = server.translation_messages("¿Vale, quedamos ahorita?", "es", "en",
                                      heard_flavor="mexico")[0]["content"]
    assert "Mexican Spanish" in mex


def test_source_side_hint_warns_against_reading_dialect_into_plain_text():
    """The risk of an always-on hint is the opposite failure: inventing
    dialect where the speaker used none."""
    heard = server.translation_messages("Guten Tag.", "de", "en",
                                        heard_flavor="hessian")[0]["content"]
    assert "exactly as written" in heard


def test_source_side_hint_stays_in_the_cacheable_prefix():
    """It is static per connection, so it must sit above the per-sentence
    additions or every sentence invalidates Ollama's prefix cache."""
    system = server.translation_messages("Ick sage wat.", "de", "en",
                                         heard_flavor="berlin")[0]["content"]
    assert "transcribed speech" in system
    # The inferred, per-sentence note still fires for typed dialect...
    assert "regional dialect" in system
    # ...and comes after the static one.
    assert system.index("transcribed speech") < system.index("regional dialect")


def test_unknown_or_absent_flavor_adds_nothing():
    for bogus in ("", None, "swabian"):
        system = server.translation_messages("Guten Tag.", "de", "en",
                                             heard_flavor=bogus)[0]["content"]
        assert "transcribed speech" not in system


def test_asserted_dialect_activates_only_its_own_ambiguous_words():
    """The whole point of the tags. A Berlin speaker saying "mehr" means
    more; a Wormser saying it may mean "mer" (wir)."""
    text = "Ich habe mehr Zeit und des ist gut."
    assert server.dialect_notes(text, "de") is None            # inferring: silent
    assert server.dialect_notes(text, "de", asserted="berlin") is None
    worms = server.dialect_notes(text, "de", asserted="worms")
    assert worms and '"mehr" = mer (wir)' in worms


def test_asserted_hint_is_hedged_not_a_command():
    """"nett" really can just mean nice; an unhedged gloss would invert
    "das war nett von dir" instead of fixing anything."""
    note = server.dialect_notes("Das war nett.", "de", asserted="hessian")
    assert note and "also ordinary German" in note
    assert "whichever fits" in note


def test_marker_path_still_glosses_ambiguous_words_of_any_dialect():
    """With an unambiguous marker present the sentence vouches for itself,
    so the tags must not start filtering the typed-input path."""
    note = server.dialect_notes("Nochemol, des war nett.", "de")
    assert note and '"des" = das' in note and '"nett" = net (nicht)' in note


def test_the_lexicon_keys_on_what_whisper_actually_writes():
    """The first entries that can fire on the audio path.

    Every other unambiguous entry keys on dialect spelling — `kiek`, `ick`,
    `wat` — which Whisper never produces, so they were dormant on speech by
    construction. These key on the mis-hearing instead. Evidence from the real
    Berlin recording: two independent decoders both wrote "gekickt" and
    "kicke", and the next sentence gives the meaning away by using the
    standard word ("Meistens *guckt* der Arvid nach").
    """
    lex = server.load_dialects()["de"]
    for heard in ("kicke", "kickt", "gekickt"):
        gloss, ambiguous, flavors = lex[heard]
        assert ambiguous, f"{heard} is ordinary German too — must stay hedged"
        assert flavors == frozenset({"berlin"})
        assert "kiek" in gloss or "jekiek" in gloss


def test_a_mis_hearing_is_only_offered_to_a_berlin_listener():
    """kicken is an ordinary German verb. Offering the Berlin reading
    unprompted would break football; offering it to someone who told us they
    are listening to a Berliner is the whole point."""
    heard = "Ich kicke ja immer nicht."
    assert server.dialect_notes(heard, "de") is None
    assert server.dialect_notes(heard, "de", asserted="hessian") is None
    note = server.dialect_notes(heard, "de", asserted="berlin")
    assert note and "kieke" in note
    # ...and hedged, never asserted: it really might be football.
    assert "whichever fits" in note


def test_the_mis_hearings_are_never_coloured():
    """They are ordinary German on the page, so marking them red would put a
    dialect claim on a sentence about football."""
    assert server.dialect_markers("Ich kicke ja immer nicht", "de",
                                  "berlin") == []
