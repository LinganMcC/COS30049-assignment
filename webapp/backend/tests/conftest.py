import os
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
MODEL = BACKEND / "model" / "logistic_regression.joblib"

# Tests set their own configuration so they do not depend on a local .env.
os.environ["FRONTEND_ORIGIN"] = "http://localhost:5173"
os.environ["MAX_CONCURRENT_ANALYSES"] = "2"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

AI_TEXT = (
    "Remote work has fundamentally transformed the modern workplace, offering both significant "
    "opportunities and notable challenges. Furthermore, employees now enjoy greater flexibility, "
    "which can lead to improved work-life balance and increased productivity. However, the absence "
    "of in-person interaction may weaken team cohesion and hinder spontaneous collaboration."
)
HUMAN_TEXT = (
    "I am typing about why drivers should not use phones while driving. This is my opinion and I "
    "will be backing it up. It is also a law all over the world. Not texting or being on your phone "
    "can save lives if you are driving."
)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

