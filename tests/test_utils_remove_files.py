import pandas as pd
from unittest.mock import MagicMock, patch
from monitoring import utils


# test remove_files function
def test_remove_files(tmp_path):
    # Create a temp file
    f = tmp_path / "temp.csv"
    f.write_text("hello")

    # Ensure it exists
    assert f.exists()

    # Remove file
    utils.remove_files(str(f))

    # File should be gone
    assert not f.exists()


# patch api call and pandas read_csv
@patch("monitoring.utils.wandb.Api")
@patch("monitoring.utils.pd.read_csv")
def test_laod_train_data_and_labels(mock_read_csv, mock_wandb_api, tmp_path):
    # create fake dataframe
    fake_df = pd.DataFrame(
        {
            "comment": ["this is a test comment"],
            "toxic": [1],
            "obscene": [0],
            "severe_toxic": [1],
            "threat": [0],
        }
    )

    # we replaced pd.read_csv with mock_read_csv
    mock_read_csv.return_value = fake_df

    # we replaced wandb.Api with mock_wandb_api
    mock_api_instance = mock_wandb_api.return_value

    # create a MagicMock object to mock the behavior of the artifact object
    mock_artifact = MagicMock()
    mock_artifact.metadata = {"labels": ["toxic", "threat", "insult"]}

    # replace artifact returned by wandb.Api
    mock_api_instance.artifact.return_value = mock_artifact

    # create a temporary train.csv file and fake text entry
    fake_csv_path = tmp_path / "train.csv"
    fake_csv_path.write_text("this is a comment,0,0,1,0,1,0")

    # create a mock object to replace get_entry return value and get_entry.download
    mock_entry = MagicMock()
    mock_entry.download.return_value = str(fake_csv_path)
    mock_artifact.get_entry.return_value = mock_entry

    # call function as test
    df, labels, file_path = utils.load_train_data_and_labels(
        entity="my_entity", project="my_project", data_set_name="toxic-data"
    )

    # test assertions
    assert labels == ["toxic", "threat", "insult"]
    assert isinstance(df, pd.DataFrame)
    assert "comment" in df.columns
    assert file_path == str(fake_csv_path)

    mock_wandb_api.assert_called_once()
    mock_api_instance.artifact.assert_called_once_with(
        "my_entity/my_project/toxic-data:latest", type="dataset"
    )
    mock_artifact.get_entry.assert_called_once_with("train.csv")
    mock_entry.download.assert_called_once()
    mock_read_csv.assert_called_once_with(str(fake_csv_path))
