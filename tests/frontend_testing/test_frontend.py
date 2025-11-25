from unittest.mock import patch, MagicMock
import importlib

@patch("front_end.app.requests.get")
@patch("front_end.app.requests.post")
@patch("front_end.app.st")
def test_submit_comment_non_toxic(mock_st, mock_post, mock_get):
    # Mock health check to succeed
    mock_health = MagicMock()
    mock_health.raise_for_status.return_value = None
    mock_get.return_value = mock_health

    # Mock Streamlit button press
    mock_st.button.return_value = True
    mock_st.text_input.return_value = "This is a comment"

    # Mock /predict POST response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"labels": []}
    mock_post.return_value = mock_response

    # Reload module so patched objects are used
    import front_end.app as app_module
    importlib.reload(app_module)

    mock_st.subheader.assert_any_call("This comment is :green[non-toxic]")
