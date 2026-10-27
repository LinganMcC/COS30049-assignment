"""API behaviour: status codes, the shared error format, and input limits.

Run from backend/:  python -m pytest
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

AI_LIKE = ('Renewable energy has become a central topic in discussions about the future of '
           'our planet. Solar and wind power offer clean alternatives to fossil fuels.')


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:   # `with` runs the lifespan, so the model is loaded once
        yield c


def assert_error_shape(response, status, code):
    assert response.status_code == status
    body = response.json()
    assert set(body) == {'error'}
    assert body['error']['code'] == code
    assert isinstance(body['error']['message'], str)
    assert 'Traceback' not in response.text


def test_analyse_returns_scores(client):
    r = client.post('/api/analyse', json={'text': AI_LIKE + ' Ok.'})
    assert r.status_code == 200
    body = r.json()
    doc = body['document']
    assert 0 <= doc['low'] <= doc['score'] <= doc['high'] <= 1
    assert doc['scored_sentences'] == 2
    assert body['excluded_sentences'] == 1
    assert [s['scored'] for s in body['sentences']] == [True, True, False]


def test_same_input_same_scores(client):   # Req 10
    first = client.post('/api/analyse', json={'text': AI_LIKE}).json()
    second = client.post('/api/analyse', json={'text': AI_LIKE}).json()
    assert first == second


@pytest.mark.parametrize('payload, code', [
    ({'text': ''}, 'invalid_input'),
    ({'text': '    '}, 'invalid_input'),
    ({'text': 5}, 'invalid_input'),
    ({}, 'invalid_input'),
    ({'text': 'hello', 'extra': 1}, 'invalid_input'),
    ({'text': 'word ' * 1001}, 'invalid_input'),
    ({'text': 'x' * 8001}, 'invalid_input'),
    ({'text': 'Too short.'}, 'no_scorable_sentences'),
])
def test_invalid_input_is_422(client, payload, code):
    assert_error_shape(client.post('/api/analyse', json=payload), 422, code)


def test_malformed_json_is_422(client):
    r = client.post('/api/analyse', content=b'{"text":', headers={'Content-Type': 'application/json'})
    assert_error_shape(r, 422, 'invalid_input')


def test_oversized_body_is_413(client):
    r = client.post('/api/analyse', content=b'x' * 40_000, headers={'Content-Type': 'application/json'})
    assert_error_shape(r, 413, 'payload_too_large')


def test_unknown_route_is_404(client):
    assert_error_shape(client.get('/api/nope'), 404, 'not_found')


def test_wrong_method_is_405(client):
    assert_error_shape(client.get('/api/analyse'), 405, 'method_not_allowed')


def test_unexpected_error_hides_details(client, monkeypatch):
    def explode(_text):
        raise RuntimeError('secret internal detail')
    with TestClient(app, raise_server_exceptions=False) as c:
        monkeypatch.setattr(app.state.detector, 'analyse', explode)
        r = c.post('/api/analyse', json={'text': AI_LIKE})
    assert_error_shape(r, 500, 'internal_error')
    assert 'secret' not in r.text


def test_cors_only_allows_frontend(client):
    preflight = {'Access-Control-Request-Method': 'POST'}
    ok = client.options('/api/analyse', headers={'Origin': 'http://localhost:5173', **preflight})
    bad = client.options('/api/analyse', headers={'Origin': 'http://evil.example', **preflight})
    assert ok.headers.get('access-control-allow-origin') == 'http://localhost:5173'
    assert 'access-control-allow-origin' not in bad.headers


def test_security_headers_present(client):
    r = client.get('/api/health')
    assert r.headers['x-content-type-options'] == 'nosniff'
    assert r.headers['content-security-policy'].startswith("default-src 'none'")
