from unittest.mock import patch, MagicMock
import importlib

@patch("front_end.app.st")
@patch("front_end.app.requests.post")
@patch("front_end.app.requests.get")
def test_submit_comment_non_toxic(mock_get, mock_post, mock_st):
    mock_health = MagicMock()
    mock_health.raise_for_status.return_value = None
    mock_get.return_value = mock_health

    mock_st.button.return_value = True
    mock_st.text_input.return_value = "This is a comment"

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"labels": []}
    mock_post.return_value = mock_response

    import front_end.app as app_module
    importlib.reload(app_module)

    app_module.run_app()

    mock_st.subheader.assert_any_call("This comment is :green[non-toxic]")

