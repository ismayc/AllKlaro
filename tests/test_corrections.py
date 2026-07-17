"""User-edited translations: storage, retrieval, and prompt injection."""
import json

import server


def entry(text, corrected, source="de", target="en", **extra):
    return {"source": source, "target": target, "text": text,
            "corrected": corrected, **extra}


# ------------------------------------------------------------- storage & cache


def test_save_and_load_roundtrip(corrections_file):
    server.save_correction(entry("Guten Morgen.", "Good morning!"))
    items = server.load_corrections()
    assert len(items) == 1
    assert items[0]["corrected"] == "Good morning!"


def test_missing_file_is_empty(corrections_file):
    assert server.load_corrections() == []


def test_reedit_supersedes_earlier_correction(corrections_file):
    server.save_correction(entry("Guten Morgen.", "Good morning."))
    server.save_correction(entry("Guten Morgen.", "Morning!"))
    items = server.load_corrections()
    assert len(items) == 1                    # same utterance edited twice
    assert items[0]["corrected"] == "Morning!"


def test_malformed_and_incomplete_lines_are_skipped(corrections_file):
    corrections_file.write_text(
        "not json at all\n"
        + json.dumps({"source": "de", "target": "en", "text": "no correction"})
        + "\n" + json.dumps(entry("Gut.", "Good.")) + "\n")
    items = server.load_corrections()
    assert [i["corrected"] for i in items] == ["Good."]


# ----------------------------------------------------------------- retrieval


def test_retrieval_requires_word_overlap_and_same_direction(corrections_file):
    server.save_correction(entry("Der Termin ist am Montag.", "The slot is on Monday."))
    server.save_correction(entry("Ganz etwas anderes hier.", "Something else entirely."))
    server.save_correction(entry("El plazo es el lunes.", "The deadline is Monday.",
                                 source="es"))
    found = server.relevant_corrections("Wann ist der Termin?", "de", "en")
    assert [c["corrected"] for c in found] == ["The slot is on Monday."]
    assert server.relevant_corrections("Wann ist der Termin?", "es", "en") == []


def test_retrieval_keeps_best_k_by_overlap(corrections_file):
    server.save_correction(entry("Termin morgen.", "Appointment tomorrow."))
    server.save_correction(entry("Der Termin für das Projekt ist morgen.",
                                 "The project appointment is tomorrow."))
    for i in range(4):
        server.save_correction(entry(f"Anderes Thema {i} hier.", f"Other {i}."))
    found = server.relevant_corrections(
        "Der Termin für das Projekt", "de", "en", k=1)
    assert [c["corrected"] for c in found] == ["The project appointment is tomorrow."]


# ------------------------------------------------------------ prompt injection


def test_corrections_become_few_shot_pairs_before_history(corrections_file):
    server.save_correction(entry("Das Meeting ist verschoben.",
                                 "The stand-up is postponed."))
    history = [{"source": "de", "target": "en", "text": "Hallo zusammen.",
                "translation": "Hi everyone."}]
    msgs = server.translation_messages(
        "Wann ist das Meeting?", "de", "en", history)
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    # Example pair first (imitation target), then history, then the sentence.
    assert msgs[1]["content"] == "Das Meeting ist verschoben."
    assert msgs[2]["content"] == "The stand-up is postponed."
    assert msgs[3]["content"] == "Hallo zusammen."
    assert msgs[-1] == {"role": "user", "content": "Wann ist das Meeting?"}


def test_unrelated_sentence_gets_no_examples(corrections_file):
    server.save_correction(entry("Das Meeting ist verschoben.",
                                 "The stand-up is postponed."))
    msgs = server.translation_messages("Schönes Wetter heute.", "de", "en")
    assert len(msgs) == 2                     # system + the sentence only
