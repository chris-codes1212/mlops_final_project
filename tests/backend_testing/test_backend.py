from unittest.mock import patch, MagicMock

# The module to test is 'front_end.app'
import front_end.app as app_module


@patch("front_end.app.st")
@patch("front_end.app.requests.post")
@patch("front_end.app.requests.get")
def test_backend_health_ready(mock_get, mock_post, mock_st):
    # Mock /health GET request
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    # The health check loop runs at import, so we just assert GET was called
    app_module.BACKEND_URL = "http://fake-backend"
    import importlib
    importlib.reload(app_module)  # reload to trigger health check

    mock_get.assert_called()
    # Print outputs are captured by default in pytest, we could assert print call if needed


@patch("front_end.app.st")
@patch("front_end.app.requests.post")
def test_submit_comment_non_toxic(mock_post, mock_st):
    # Mock Streamlit button press
    mock_st.button.return_value = True
    mock_st.text_input.return_value = "This is a comment"

    # Mock /predict POST response with empty labels
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"labels": []}
    mock_post.return_value = mock_response

    # Run the part of the script that handles submission
    import importlib
    import front_end.app as app_module
    importlib.reload(app_module)

    # Check that st.subheader was called with non-toxic message
    mock_st.subheader.assert_any_call("This comment is :green[non-toxic]")


@patch("front_end.app.st")
@patch("front_end.app.requests.post")
def test_submit_comment_toxic(mock_post, mock_st):
    # Mock Streamlit button press
    mock_st.button.return_value = True
    mock_st.text_input.return_value = "This is a toxic comment"

    # Mock /predict POST response with some labels
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"labels": ["toxic", "threat"]}
    mock_post.return_value = mock_response

    # Reload app to trigger code execution
    import importlib
    import front_end.app as app_module
    importlib.reload(app_module)

    # Check that st.subheader was called with the correct classification
    mock_st.subheader.assert_any_call(
        "This comment is classified as :red[Toxic, Threat]"
    )


@patch("front_end.app.st")
@patch("front_end.app.requests.post")
def test_submit_comment_backend_error(mock_post, mock_st):
    # Mock Streamlit button press
    mock_st.button.return_value = True
    mock_st.text_input.return_value = "Error comment"

    # Mock POST request raising RequestException
    from requests.exceptions import RequestException
    mock_post.side_effect = RequestException("Connection error")

    import importlib
    import front_end.app as app_module
    importlib.reload(app_module)

    # Check that st.error was called
    mock_st.error.assert_any_call("Error connecting to backend: Connection error")
