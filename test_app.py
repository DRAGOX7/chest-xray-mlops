import pytest
from fastapi.testclient import TestClient
from app import app  # Imports your FastAPI app

client = TestClient(app)


def test_read_root():
    """Test if the API home page is reachable"""
    response = client.get("/")
    assert response.status_code == 200


def test_model_info():
    """Test if the model metadata is loaded correctly"""
    # Adjust this endpoint name to match whatever your info route is called
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
