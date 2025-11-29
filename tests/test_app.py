import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app

@pytest.fixture
def client():
    with flask_app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "healthy"}

def test_ask_llm_success(client, mocker):
    
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Esta é uma resposta simulada."
    mock_model.generate_content.return_value = mock_response

    mocker.patch("app.model", mock_model)

    response = client.post("/ask", json={"question": "Qual a cor do céu?"})
    
    assert response.status_code == 200
    assert response.json == {"answer": "Esta é uma resposta simulada."}

def test_ask_llm_bad_request(client, mocker):
    
    mocker.patch("app.model", MagicMock())
    
    response = client.post("/ask", json={"prompt": "Pergunta errada"})
    
    assert response.status_code == 400
    assert "error" in response.json
    assert response.json["error"] == "Requisição inválida. 'question' não encontrada no JSON."

def test_ask_llm_api_error(client, mocker):
    
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("Falha na API")
    
    mocker.patch("app.model", mock_model)
    
    response = client.post("/ask", json={"question": "Isso vai falhar"})
    
    assert response.status_code == 503
    assert "error" in response.json
    assert "Falha na API" in response.json["error"]

def test_ask_llm_no_model(client, mocker):
    
    mocker.patch("app.model", None)
    
    response = client.post("/ask", json={"question": "O modelo existe?"})
    
    assert response.status_code == 500
    assert "error" in response.json
    assert "Modelo LLM não foi inicializado" in response.json["error"]