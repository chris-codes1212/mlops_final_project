from unittest.mock import patch, MagicMock

# PATCH WANDB BEFORE IMPORTING BACK_END

# Fake WandB artifact object
fake_artifact = MagicMock()
fake_artifact.download.return_value = None

# Fake WandB API object
fake_api = MagicMock()
fake_api.artifact.return_value = fake_artifact

# Mock wandb.login() to do nothing
patch("back_end.utils.wandb.login", return_value=None).start()

# Mock wandb.Api() to return our fake API object
patch("back_end.utils.wandb.Api", return_value=fake_api).start()

# Mock environment variable
patch.dict("os.environ", {"WANDB_API_KEY": "dummy"}, clear=False).start()

# ONLY NOW import FastAPI app

from fastapi.testclient import TestClient
import back_end.main as main_module

client = TestClient(main_module.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("back_end.main.model", None)
def test_predict_model_not_loaded():
    payload = {"comment": "hello"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded. Cannot make predictions"


@patch("back_end.main.write_logs.write_log")
@patch("back_end.main.utils.preprocess_user_input")
def test_predict_success(mock_preprocess, mock_write_log):
    # Dummy model
    class DummyModel:
        def predict(self, X):
            return [[0.6, 0.4, 0.9]]

    main_module.model = DummyModel()
    main_module.tokenizer = "dummy"
    main_module.maxlen = 100
    main_module.labels = ["toxic", "threat", "insult"]

    mock_preprocess.return_value = "processed"

    payload = {"comment": "This is a test"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"labels": ["toxic", "insult"]}

    assert mock_write_log.called
