"""Dialect detection and translation hints (German and Spanish regions)."""
import server


def test_dialects_file_parses_with_ambiguity_flags():
    lex = server.load_dialects()
    assert lex["de"]["ick"] == ("ich", False)    # unambiguous marker
    assert lex["de"]["nochemol"][1] is False
    assert lex["de"]["nett"][1] is True          # real standard word too
    assert lex["de"]["des"][1] is True
    # The [es] section holds the Spanish regional entries.
    assert lex["es"]["chido"][1] is False
    assert lex["es"]["vale"][1] is True          # everyday standard word
    assert "chido" not in lex["de"]              # sections don't bleed


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
