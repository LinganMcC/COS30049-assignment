"""API behaviour: results, validation, the shared error format and CORS."""

from .conftest import AI_TEXT, HUMAN_TEXT


def assert_error_shape(response, status, code=None):
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "fields"}
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]
    if code:
        assert body["error"]["code"] == code
    text = response.text.lower()
    assert "traceback" not in text and 'file "' not in text
    return body["error"]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_detect_ai_text(client):
    r = client.post("/api/detect", json={"text": AI_TEXT})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "ai"
    assert body["range"]["low"] <= body["score"] <= body["range"]["high"]
    assert body["sentence_count"] == len(body["sentences"]) == 3
    # The passage segments rebuild the analysed text and point at every sentence once.
    indexes = [s["sentence_index"] for s in body["passage"] if s["sentence_index"] is not None]
    assert indexes == [0, 1, 2]
    for s in body["sentences"]:
        assert 0 <= s["score"] <= 1
        assert all(f["contribution"] > 0 for f in s["toward_ai"])
        assert all(f["contribution"] < 0 for f in s["toward_human"])


def test_detect_human_text(client):
    r = client.post("/api/detect", json={"text": HUMAN_TEXT})
    assert r.status_code == 200
    assert r.json()["label"] == "human"


def test_result_is_deterministic(client):
    a = client.post("/api/detect", json={"text": AI_TEXT}).json()
    b = client.post("/api/detect", json={"text": AI_TEXT}).json()
    assert a == b


def test_too_few_words(client):
    err = assert_error_shape(client.post("/api/detect", json={"text": "Too short to judge."}), 422, "validation_error")
    assert "at least 20" in err["message"]
    assert err["fields"][0]["field"] == "text"


def test_too_many_words(client):
    err = assert_error_shape(client.post("/api/detect", json={"text": "word " * 1001}), 422, "validation_error")
    assert "limit is 1000" in err["message"]


def test_accented_words_count_once(client):
    ok = client.post("/api/detect", json={"text": "Le résumé était très détaillé. " * 200})  # 1,000 words
    assert ok.status_code == 200
    err = assert_error_shape(client.post("/api/detect", json={"text": "résumé " * 1001}), 422)
    assert "1001 words" in err["message"]


def test_too_large_body_still_reaches_the_browser(client):
    """CORS is the outermost middleware, so the browser can read the 413."""
    r = client.post(
        "/api/detect",
        content=b"x" * 140_000,
        headers={"Content-Type": "application/json", "Origin": "http://localhost:5173"},
    )
    assert r.status_code == 413
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_too_many_characters(client):
    assert_error_shape(client.post("/api/detect", json={"text": "word " * 999 + "a" * 15_100}), 422, "validation_error")


def test_whitespace_only(client):
    assert_error_shape(client.post("/api/detect", json={"text": "     \n\n   "}), 422)


def test_no_letters(client):
    text = " ".join(["2024", "12.5", "%", "100"] * 10)
    assert_error_shape(client.post("/api/detect", json={"text": text}), 422)


def test_missing_and_extra_fields(client):
    assert_error_shape(client.post("/api/detect", json={}), 422)
    assert_error_shape(client.post("/api/detect", json={"text": AI_TEXT, "admin": True}), 422)


def test_wrong_type(client):
    assert_error_shape(client.post("/api/detect", json={"text": 12345}), 422)


def test_invalid_json(client):
    r = client.post("/api/detect", content=b"{not json", headers={"Content-Type": "application/json"})
    assert_error_shape(r, 400, "invalid_json")


def test_body_too_large(client):
    r = client.post("/api/detect", content=b"x" * 140_000, headers={"Content-Type": "application/json"})
    assert_error_shape(r, 413, "payload_too_large")


def test_validation_error_does_not_echo_input(client):
    secret = "my-private-sentence-" * 3
    r = client.post("/api/detect", json={"text": secret})
    assert secret not in r.text


def test_unknown_route_and_method(client):
    assert_error_shape(client.get("/api/nope"), 404, "not_found")
    assert_error_shape(client.get("/api/detect"), 405, "method_not_allowed")


def test_unexpected_error_is_generic(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("internal detail /secret/path")

    monkeypatch.setattr(client.app.state.detector, "analyse", boom)
    from fastapi.testclient import TestClient

    # No "with": reuse the already-started app instead of running lifespan again.
    r = TestClient(client.app, raise_server_exceptions=False).post("/api/detect", json={"text": AI_TEXT})
    assert_error_shape(r, 500, "internal_error")
    assert "secret" not in r.text


def test_model_info(client):
    r = client.get("/api/model-info")
    assert r.status_code == 200
    body = r.json()
    assert len(body["features"]) == 15
    assert body["threshold"] == 0.5
    assert "accuracy" in body["evaluation"]["drcat_test"]
    assert body["limits"]["max_words"] == 1000


def test_cors_allows_only_our_front_end(client):
    ok = client.options(
        "/api/detect",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"

    evil = client.options(
        "/api/detect",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in evil.headers

    simple = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in simple.headers


def test_security_headers(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"
    assert "default-src 'none'" in r.headers["content-security-policy"]
