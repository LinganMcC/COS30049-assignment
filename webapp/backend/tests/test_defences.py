"""The two nominated 2.7 defences (denial of service, misconfiguration) and the 2.8 evasion defence."""

import random
import re
import threading
import time


from app.ml.text_pipeline import (
    HTML_TAG_RE,
    HTML_TAG_RE_ASSIGNMENT2,
    clean_text,
    normalise_for_evasion,
)

from .conftest import AI_TEXT


# --- Defence A: denial of service -------------------------------------------
def test_regex_fix_is_equivalent_to_assignment2():
    random.seed(7)
    alphabet = ["<", ">", "/", " ", "\n", "\x0c", "p", "b", "r", "a", "x", "=", "'", "div", "BR", "h1", "h7", "TD"]
    for _ in range(20_000):
        s = "".join(random.choice(alphabet) for _ in range(random.randint(1, 14)))
        assert HTML_TAG_RE.sub("", s) == HTML_TAG_RE_ASSIGNMENT2.sub("", s), repr(s)


def test_regex_fix_is_fast_on_attack_input():
    attack = "<p" + "\x0c" * 11_000 + "x"
    start = time.perf_counter()
    clean_text(attack)
    assert time.perf_counter() - start < 0.1


def test_busy_server_returns_503(client):
    guard = client.app.state.analysis_guard
    taken = [guard.try_acquire() for _ in range(guard.capacity)]
    assert all(taken)
    try:
        r = client.post("/api/detect", json={"text": AI_TEXT})
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "busy"
        assert r.headers["retry-after"] == "2"
        # Cheap endpoints still answer while analyses are saturated.
        assert client.get("/api/health").status_code == 200
    finally:
        for _ in taken:
            guard.release()
    assert client.post("/api/detect", json={"text": AI_TEXT}).status_code == 200


def test_attack_text_is_handled_quickly_end_to_end(client):
    attack = "Ordinary words here to pass the word count check. " * 3 + "<p" + "\x0c" * 9_000 + "x"
    start = time.perf_counter()
    r = client.post("/api/detect", json={"text": attack})
    assert r.status_code == 200
    assert time.perf_counter() - start < 2


# --- 2.8 detection evasion -------------------------------------------------
def _zero_width(text: str) -> str:
    return re.sub(r"([A-Za-z]{4})(?=[A-Za-z]{3})", "\\1\u200b", text)


def _homoglyphs(text: str) -> str:
    return text.translate(str.maketrans({"a": "а", "e": "е", "o": "о", "c": "с", "p": "р"}))


def test_normalisation_reverses_attacks():
    for attack in (_zero_width, _homoglyphs):
        restored, report = normalise_for_evasion(attack(AI_TEXT))
        assert restored == AI_TEXT
        assert report.changed


def test_normalisation_leaves_plain_text_alone():
    restored, report = normalise_for_evasion(AI_TEXT)
    assert restored == AI_TEXT and not report.changed


def test_evasion_does_not_change_score(client):
    clean = client.post("/api/detect", json={"text": AI_TEXT}).json()
    for attack in (_zero_width, _homoglyphs):
        attacked = client.post("/api/detect", json={"text": attack(AI_TEXT)}).json()
        assert attacked["score"] == clean["score"]
        assert attacked["normalisation"]["invisible_removed"] + attacked["normalisation"]["homoglyphs_replaced"] > 0


def test_concurrent_requests_are_safe(client):
    """Several threads at once give the same answer (spaCy/model are shared)."""
    expected = client.post("/api/detect", json={"text": AI_TEXT}).json()["score"]
    results = []

    def worker():
        from fastapi.testclient import TestClient

        r = TestClient(client.app).post("/api/detect", json={"text": AI_TEXT})
        if r.status_code == 200:
            results.append(r.json()["score"])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results and all(s == expected for s in results)


def test_genuine_accents_are_kept_and_not_flagged(client):
    text = AI_TEXT.replace("Remote work", "Remote work at the café") + " Việt and Zoë wrote their résumé."
    body = client.post("/api/detect", json={"text": text}).json()
    shown = "".join(seg["text"] for seg in body["passage"])
    assert "café" in shown and "Việt" in shown and "résumé" in shown
    n = body["normalisation"]
    assert n["invisible_removed"] == n["homoglyphs_replaced"] == n["marks_removed"] == 0


def test_stray_marks_are_removed_and_flagged():
    restored, report = normalise_for_evasion("q\u0301uick e\u0301\u0301\u0301xample")
    assert restored == "quick example"
    assert report.marks_removed == 4


def test_accented_text_matches_training_features(client):
    """Genuine accents reach the model unchanged, exactly as in Assignment 2
    (whose word rule [A-Za-z']+ splits "résumé" into "r" and "sum")."""
    detector = client.app.state.detector
    x = detector.featurise(["A résumé provides a useful summary.", "My name is Việt."])
    assert list(x[:, 0]) == [7.0, 5.0]  # n_words under the Assignment 2 rule
