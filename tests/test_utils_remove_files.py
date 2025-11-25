import os
import builtins
import pandas as pd
import pytest
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
@patch('monitoring.utils.wandb.Api')
@patch('monitoring.utils.pd.read_csv')
def test_laod_train_data_and_labels(mock_read_csv, mock_wandb_api, tmp_path):
    # create fake dataframe
    fake_df = pd.DataFrame({
            'comment':['this is a test comment'],
            'toxic':[1],
            'obscene':[0],
            'severe_toxic':[1],
            'threat':[0]})
    
    # we replaced pd.read_csv with mock_read_csv, now we make the return value of this mock equal to a fake dataframe
    mock_read_csv.return_value = fake_df

    # we replaced wandb.Api with mock_wandb_api, now we create a variable equal to the return value of this mock function
    mock_api_instance = mock_wandb_api.return_value

    # create a MagicMock object to mock the behavior of the artifact object
    mock_artifact = MagicMock()
    mock_artifact.metadata = {'labels':['toxic','threat','insult']}

    # here we replace the artifact returned by the wandb.Api equal to the mock artifact we created earlier
    mock_api_instance.artifact.return_value = mock_artifact

    # create a temporary train.csv file and fake text entry
    fake_csv_path = tmp_path / 'train.csv'
    fake_csv_path.write_text('this is a comment,0,0,1,0,1,0')

    # create a mock object to replace the get_entry return value and the get_entry.download return value
    mock_entry = MagicMock()
    mock_entry.download.return_value = str(fake_csv_path)
    mock_artifact.get_entry.return_value = mock_entry

    # call function as test
    df, labels, file_path = utils.load_train_data_and_labels(
        entity="my_entity",
        project="my_project",
        data_set_name="toxic-data"
    )

    # test assertions
    assert labels == ['toxic','threat','insult']
    assert isinstance(df, pd.DataFrame)
    assert "comment" in df.columns
    assert file_path == str(fake_csv_path)

    mock_wandb_api.assert_called_once()
    mock_api_instance.artifact.assert_called_once_with(
        "my_entity/my_project/toxic-data:latest",
        type="dataset"
    )
    mock_artifact.get_entry.assert_called_once_with("train.csv")
    mock_entry.download.assert_called_once()
    mock_read_csv.assert_called_once_with(str(fake_csv_path))