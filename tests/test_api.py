import io, numpy as np, cv2
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def jpeg_bytes(w=300, h=300):
    ok, buf = cv2.imencode(".jpg", np.random.randint(0, 255, (h, w, 3), np.uint8))
    return buf.tobytes()

def test_health():
    assert client.get("/health").status_code == 200

def test_valid_image():
    r = client.post("/predict", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in {"normal", "defective"}
    assert 0.0 <= body["confidence"] <= 1.0

def test_wrong_type():
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415

def test_corrupt_image():
    r = client.post("/predict", files={"file": ("a.jpg", b"notanimage", "image/jpeg")})
    assert r.status_code == 400