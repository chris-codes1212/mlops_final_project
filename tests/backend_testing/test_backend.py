from fastapi.testclient import TestClient
from unittest.mock import patch

# Import the FastAPI app
# import back_end.main as main_module
from back_end import main as main_module
# from back_end import utils


client = TestClient(main_module.app)


# Test /health endpoint
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# Test /predict when model is None
@patch("back_end.main.model", None)
def test_predict_model_not_loaded():
    payload = {"comment": "This is a test comment"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded. Cannot make predictions"


# Test /predict with mocked model and utils
@patch("back_end.main.write_logs.write_log")
@patch("back_end.main.utils.preprocess_user_input")
def test_predict_success(mock_preprocess, mock_write_log):
    # Mock the model object
    class DummyModel:
        def predict(self, X):
            # Return a 2D array like Keras predict()
            return [[0.6, 0.4, 0.7]]

    # Patch model, tokenizer, maxlen, and labels
    main_module.model = DummyModel()
    main_module.tokenizer = "dummy_tokenizer"
    main_module.maxlen = 100
    main_module.labels = ["toxic", "threat", "insult"]

    # Mock preprocessing to return any value (we don't use it)
    mock_preprocess.return_value = "processed_input"

    payload = {"comment": "This is a test comment"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    # Labels with prob > 0.5 are "toxic" and "insult"
    assert response.json() == {"labels": ["toxic", "insult"]}

    # Ensure write_log was called
    assert mock_write_log.called
