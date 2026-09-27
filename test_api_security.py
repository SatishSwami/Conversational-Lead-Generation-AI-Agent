import hashlib
import hmac

import re

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import webhook_server


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------


def test_api_key_accepts_valid_key(monkeypatch):
    monkeypatch.setattr(
        webhook_server,
        "API_KEY",
        "test-secret",
    )

    webhook_server._verify_api_key(
        x_api_key="test-secret",
    )


def test_api_key_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(
        webhook_server,
        "API_KEY",
        "test-secret",
    )

    with pytest.raises(HTTPException) as exc:
        webhook_server._verify_api_key(
            x_api_key=None,
        )

    assert exc.value.status_code == 401


def test_api_key_rejects_invalid_key(monkeypatch):
    monkeypatch.setattr(
        webhook_server,
        "API_KEY",
        "test-secret",
    )

    with pytest.raises(HTTPException) as exc:
        webhook_server._verify_api_key(
            x_api_key="wrong-secret",
        )

    assert exc.value.status_code == 401


def test_production_requires_api_key_configuration(monkeypatch):
    monkeypatch.setattr(
        webhook_server,
        "API_KEY",
        None,
    )

    monkeypatch.setattr(
        webhook_server,
        "ENVIRONMENT",
        "production",
    )

    with pytest.raises(HTTPException) as exc:
        webhook_server._verify_api_key(
            x_api_key=None,
        )

    assert exc.value.status_code == 503


def test_development_can_run_without_api_key(monkeypatch):
    monkeypatch.setattr(
        webhook_server,
        "API_KEY",
        None,
    )

    monkeypatch.setattr(
        webhook_server,
        "ENVIRONMENT",
        "development",
    )

    webhook_server._verify_api_key(
        x_api_key=None,
    )


# ---------------------------------------------------------------------------
# Request Validation
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(webhook_server.app)


def test_chat_rejects_invalid_session_id(client):
    response = client.post(
        "/chat",
        json={
            "session_id": "invalid session id!",
            "message": "Hello",
        },
    )

    assert response.status_code == 422


def test_chat_rejects_empty_message(client):
    response = client.post(
        "/chat",
        json={
            "session_id": "test-session",
            "message": "",
        },
    )

    assert response.status_code == 422


def test_chat_rejects_oversized_message(client):
    response = client.post(
        "/chat",
        json={
            "session_id": "test-session",
            "message": "x" * 4001,
        },
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# WhatsApp Signature Verification
# ---------------------------------------------------------------------------


def test_whatsapp_signature_accepts_valid_signature(monkeypatch):
    secret = "test-whatsapp-secret"
    payload = b'{"test":"payload"}'

    monkeypatch.setattr(
        webhook_server,
        "WA_APP_SECRET",
        secret,
    )

    expected_signature = (
        "sha256="
        + hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()
    )

    assert webhook_server._verify_whatsapp_signature(
        payload,
        expected_signature,
    )


def test_whatsapp_signature_rejects_invalid_signature(monkeypatch):
    monkeypatch.setattr(
        webhook_server,
        "WA_APP_SECRET",
        "test-whatsapp-secret",
    )

    assert not webhook_server._verify_whatsapp_signature(
        b'{"test":"payload"}',
        "sha256=invalid",
    )


def test_production_rejects_missing_whatsapp_secret(monkeypatch):
    monkeypatch.setattr(
        webhook_server,
        "WA_APP_SECRET",
        "",
    )

    monkeypatch.setattr(
        webhook_server,
        "ENVIRONMENT",
        "production",
    )

    assert not webhook_server._verify_whatsapp_signature(
        b'{"test":"payload"}',
        "",
    )


def test_development_allows_missing_whatsapp_secret(monkeypatch):
    monkeypatch.setattr(
        webhook_server,
        "WA_APP_SECRET",
        "",
    )

    monkeypatch.setattr(
        webhook_server,
        "ENVIRONMENT",
        "development",
    )

    assert webhook_server._verify_whatsapp_signature(
        b'{"test":"payload"}',
        "",
    )

# ---------------------------------------------------------------------------
# Request Observability
# ---------------------------------------------------------------------------


def test_request_id_is_preserved(client):
    request_id = "test-request-123"

    response = client.post(
        "/chat",
        headers={
            "X-Request-ID": request_id,
        },
        json={
            "session_id": "test-session",
            "message": "",
        },
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == request_id


def test_request_id_is_generated_when_missing(client):
    response = client.post(
        "/chat",
        json={
            "session_id": "test-session",
            "message": "",
        },
    )

    assert response.status_code == 422

    request_id = response.headers.get("X-Request-ID")

    assert request_id is not None
    assert re.fullmatch(
        r"[0-9a-f]{32}",
        request_id,
    )


def test_invalid_request_id_is_replaced(client):
    response = client.post(
        "/chat",
        headers={
            "X-Request-ID": "invalid request id!",
        },
        json={
            "session_id": "test-session",
            "message": "",
        },
    )

    assert response.status_code == 422

    request_id = response.headers.get("X-Request-ID")

    assert request_id is not None
    assert request_id != "invalid request id!"
    assert re.fullmatch(
        r"[0-9a-f]{32}",
        request_id,
    )


def test_request_id_is_returned_for_health_endpoint(client):
    request_id = "health-check-001"

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": request_id,
        },
    )

    assert response.headers["X-Request-ID"] == request_id
