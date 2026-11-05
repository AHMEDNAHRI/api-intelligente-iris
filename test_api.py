from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

USER = {"X-API-Key": "user-key-123"}
ADMIN = {"X-API-Key": "admin-key-456"}
SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5,
          "petal_length": 1.4, "petal_width": 0.2}
VIRGINICA = {"sepal_length": 6.7, "sepal_width": 3.0,
             "petal_length": 5.2, "petal_width": 2.3}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_prediction_ok():
    r = client.post("/predict", json=SETOSA, headers=USER)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "setosa"
    assert "request_id" in body and "timestamp" in body


def test_sans_cle_401():
    r = client.post("/predict", json=SETOSA)
    assert r.status_code == 401


def test_mauvaise_cle_401():
    r = client.post("/predict", json=SETOSA, headers={"X-API-Key": "faux"})
    assert r.status_code == 401


def test_entree_invalide_422():
    bad = dict(SETOSA, sepal_length=-5)
    r = client.post("/predict", json=bad, headers=USER)
    assert r.status_code == 422


def test_batch_probabilites():
    r = client.post("/predict/batch", json={"flowers": [SETOSA, VIRGINICA]},
                    headers=USER)
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 2
    assert results[0]["prediction"] == "setosa"
    assert set(results[0]["probabilities"].keys()) == {"setosa", "versicolor", "virginica"}


def test_user_interdit_admin_403():
    r = client.get("/admin/stats", headers=USER)
    assert r.status_code == 403


def test_admin_autorise_200():
    r = client.get("/admin/stats", headers=ADMIN)
    assert r.status_code == 200
    assert "total_predictions" in r.json()


def test_limite_debit_429():
    codes = [client.get("/limited").status_code for _ in range(6)]
    assert codes[:5] == [200] * 5
    assert codes[5] == 429
