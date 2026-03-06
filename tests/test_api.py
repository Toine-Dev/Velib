"""
Tests unitaires — api/main.py
Couvre : /health, /auth/token, /auth/me, /client/predict,
         /admin/users (POST/DELETE), /admin/pipeline/trigger
Toutes les dépendances DB et auth sont mockées.
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Bloquer la connexion DB qui s'exécute au niveau module à l'import
# ---------------------------------------------------------------------------
_mock_engine = MagicMock()
_mock_conn = MagicMock()
_mock_conn.__enter__ = lambda s: _mock_conn
_mock_conn.__exit__ = MagicMock(return_value=False)
_mock_conn.execute.return_value.scalar.return_value = 1
_mock_engine.begin.return_value = _mock_conn

with patch("sqlalchemy.create_engine", return_value=_mock_engine):
    from api.main import app, create_access_token, verify_password

from fastapi.testclient import TestClient
client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper : fabricate a valid JWT for test users
# ---------------------------------------------------------------------------
def make_token(username="testuser", role="client") -> str:
    return create_access_token(sub=username, role=role)

def make_admin_token() -> str:
    return create_access_token(sub="admin", role="admin")

def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ===========================================================================
# 1. /health
# ===========================================================================
class TestHealth:
    def test_status_ok(self):
        resp = client.get("/system/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_no_auth_required(self):
        resp = client.get("/health")
        assert resp.status_code != 401


# ===========================================================================
# 2. /auth/token
# ===========================================================================
class TestAuthToken:
    def test_wrong_password_returns_401(self):
        with patch("api.main.get_user") as mock_get_user:
            mock_get_user.return_value = {
                "username": "alice",
                "password_hash": "$2b$12$FAKEHASH",
                "role": "client",
            }
            with patch("api.main.verify_password", return_value=False):
                resp = client.post(
                    "/auth/token",
                    data={"username": "alice", "password": "wrong"},
                )
        assert resp.status_code == 401

    def test_unknown_user_returns_401(self):
        with patch("api.main.get_user", return_value=None):
            resp = client.post(
                "/auth/token",
                data={"username": "nobody", "password": "x"},
            )
        assert resp.status_code == 401

    def test_valid_credentials_return_token(self):
        hashed = "$2b$12$placeholder"  # won't be verified; we mock verify_password
        with patch("api.main.get_user") as mock_get_user, \
             patch("api.main.verify_password", return_value=True):
            mock_get_user.return_value = {
                "username": "alice",
                "password_hash": hashed,
                "role": "client",
            }
            resp = client.post(
                "/auth/token",
                data={"username": "alice", "password": "correct"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"] == "bearer"

    def test_token_is_string(self):
        with patch("api.main.get_user") as mock_get_user, \
             patch("api.main.verify_password", return_value=True):
            mock_get_user.return_value = {
                "username": "bob",
                "password_hash": "h",
                "role": "admin",
            }
            resp = client.post(
                "/auth/token",
                data={"username": "bob", "password": "p"},
            )
        assert isinstance(resp.json()["access_token"], str)


# ===========================================================================
# 3. /auth/me
# ===========================================================================
class TestAuthMe:
    def test_me_without_token_returns_401(self):
        resp = client.get("/auth/me")
        assert resp.status_code in (401, 403)

    def test_me_with_valid_token_returns_username(self):
        token = make_token("alice", "client")
        resp = client.get("/auth/me", headers=auth_header(token))
        assert resp.status_code == 200
        assert resp.json()["username"] == "alice"
        assert resp.json()["role"] == "client"

    def test_me_with_invalid_token_returns_401(self):
        resp = client.get("/auth/me", headers={"Authorization": "Bearer INVALID"})
        assert resp.status_code == 401






# ===========================================================================
# 4. /client/predict
# ===========================================================================
class TestClientPredict:
    def test_missing_auth_returns_401(self):
        resp = client.post("/client/predict", json={"datetime": "2025-03-01 08:00:00"})
        assert resp.status_code in (401, 403)

    def test_invalid_datetime_returns_400(self):
        token = make_token()
        resp = client.post(
            "/client/predict",
            json={"datetime": "not-a-date"},
            headers=auth_header(token),
        )
        assert resp.status_code == 400

    def test_no_forecast_returns_404(self):
        token = make_token()
        empty_df = pd.DataFrame(columns=[
            "identifiant_du_site_de_comptage", "nom_du_site_de_comptage",
            "latitude", "longitude", "comptage_horaire"
        ])
        with patch("api.main.pd.read_sql", return_value=empty_df):
            resp = client.post(
                "/client/predict",
                json={"datetime": "2025-03-01 08:00:00"},
                headers=auth_header(token),
            )
        assert resp.status_code == 404

    def test_valid_forecast_returns_list(self):
        token = make_token()
        mock_df = pd.DataFrame({
            "identifiant_du_site_de_comptage": [100056789],
            "nom_du_site_de_comptage": ["Site A"],
            "latitude": [48.85],
            "longitude": [2.35],
            "comptage_horaire": [42.0],
        })
        with patch("api.main.pd.read_sql", return_value=mock_df):
            resp = client.post(
                "/client/predict",
                json={"datetime": "2025-03-01 08:00:00"},
                headers=auth_header(token),
            )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert body[0]["identifiant_du_site_de_comptage"] == 100056789
        assert body[0]["comptage_horaire"] == 42.0


# ===========================================================================
# 5. /admin/users  (POST + DELETE)
# ===========================================================================
class TestAdminUsers:
    def test_create_user_requires_admin(self):
        token = make_token("alice", "client")  # client, NOT admin
        resp = client.post(
            "/admin/users",
            json={"username": "newuser", "password": "pw"},
            headers=auth_header(token),
        )
        assert resp.status_code == 403

    def test_create_user_conflict_if_exists(self):
        token = make_admin_token()
        existing_user = {"username": "alice", "password_hash": "h", "role": "client"}
        with patch("api.main.get_user", return_value=existing_user):
            resp = client.post(
                "/admin/users",
                json={"username": "alice", "password": "pw"},
                headers=auth_header(token),
            )
        assert resp.status_code == 409

    def test_create_user_success(self):
        token = make_admin_token()
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute = MagicMock()

        with patch("api.main.get_user", return_value=None), \
            patch("api.main.engine.begin", return_value=mock_conn), \
            patch("api.main.pwd_context.hash", return_value="hashed_pw"):
            resp = client.post(
                "/admin/users",
                json={"username": "newuser", "password": "pw", "role": "client"},
                headers=auth_header(token),
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "created"

    def test_delete_user_requires_admin(self):
        token = make_token("alice", "client")
        resp = client.delete("/admin/users/alice", headers=auth_header(token))
        assert resp.status_code == 403

    def test_delete_nonexistent_user_returns_404(self):
        token = make_admin_token()
        mock_result = MagicMock()
        mock_result.rowcount = 0

        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute = MagicMock(return_value=mock_result)

        with patch("api.main.engine.begin", return_value=mock_conn):
            resp = client.delete("/admin/users/ghost", headers=auth_header(token))
        assert resp.status_code == 404


# ===========================================================================
# 6. /admin/pipeline/trigger
# ===========================================================================
class TestAdminPipelineTrigger:
    def test_trigger_requires_admin(self):
        token = make_token("alice", "client")
        resp = client.post(
            "/admin/pipeline/trigger",
            json={"job_type": "data"},
            headers=auth_header(token),
        )
        assert resp.status_code == 403

    def test_invalid_job_type_returns_422(self):
        token = make_admin_token()
        resp = client.post(
            "/admin/pipeline/trigger",
            json={"job_type": "unknown_job"},
            headers=auth_header(token),
        )
        assert resp.status_code == 422

    @pytest.mark.parametrize("job_type", ["data", "model", "forecast"])
    def test_valid_job_types_queued(self, job_type):
        token = make_admin_token()
        mock_conn = MagicMock()
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute = MagicMock(return_value=MagicMock(scalar=lambda: 1))

        with patch("api.main.engine.begin", return_value=mock_conn):
            resp = client.post(
                "/admin/pipeline/trigger",
                json={"job_type": job_type},
                headers=auth_header(token),
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"
        assert resp.json()["job_type"] == job_type


# ===========================================================================
# 7. Auth helpers (unit level)
# ===========================================================================
class TestAuthHelpers:
    def test_create_access_token_returns_string(self):
        token = create_access_token("user1", "client")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_password_correct(self):
        with patch("api.main.pwd_context.verify", return_value=True):
            assert verify_password("abc", "any_hash") is True

    def test_verify_password_incorrect(self):
        with patch("api.main.pwd_context.verify", return_value=False):
            assert verify_password("wrong", "any_hash") is False