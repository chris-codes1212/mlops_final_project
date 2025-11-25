from unittest.mock import patch, MagicMock
import importlib

@patch("front_end.app.st")
@patch("front_end.app.requests.post")
@patch("front_end.app.requests.get")
def test_run_app_smoke(mock_get, mock_post, mock_st):
    # Mock health check to always succeed
    mock_health = MagicMock()
    mock_health.raise_for_status.return_value = None
    mock_get.return_value = mock_health

    # Mock Streamlit inputs
    mock_st.button.return_value = True      # Simulate clicking "Submit"
    mock_st.text_input.return_value = "test comment"

    # Mock POST response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"labels": []}
    mock_post.return_value = mock_response

    # Reload app module so patched objects are used
    import front_end.app as app_module
    importlib.reload(app_module)

    # Run the app
    app_module.run_app()