# tests/test_utils.py
import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from monitoring import utils



# Test remove_files
def test_remove_existing_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    utils.remove_files(str(test_file))
    assert not test_file.exists()


def test_remove_nonexistent_file(tmp_path):
    test_file = tmp_path / "nonexistent.txt"
    # Should not raise an error
    utils.remove_files(str(test_file))
    assert not test_file.exists()


# Test load_train_data_and_labels
@patch("utils.wandb.Api")
def test_load_train_data_and_labels(mock_wandb):
    # Mock artifact object
    mock_artifact = MagicMock()
    mock_artifact.metadata.get.return_value = ["label1", "label2"]
    mock_artifact.get_entry.return_value.download.return_value = "train.csv"

    mock_api_instance = MagicMock()
    mock_api_instance.artifact.return_value = mock_artifact
    mock_wandb.return_value = mock_api_instance

    # Patch pd.read_csv to return a simple DataFrame
    with patch("pandas.read_csv", return_value=pd.DataFrame({"a": [1, 2]})) as mock_read_csv:
        df, labels, path = utils.load_train_data_and_labels("entity", "project")
        assert isinstance(df, pd.DataFrame)
        assert labels == ["label1", "label2"]
        assert path == "train.csv"



# Test dynamodb_to_dataframe
@patch("utils.boto3.client")
def test_dynamodb_to_dataframe(mock_boto):
    # Mock DynamoDB response
    mock_table = MagicMock()
    mock_table.scan.side_effect = [
        {
            "Items": [
                {
                    "timestamp": {"S": "2025-11-24T23:42:49.010154+00:00"},
                    "latency_seconds": {"N": "0.123"},
                    "comment": {"S": "Test comment"},
                    "prediction_labels": {"L": [{"S": "label1"}]}
                }
            ]
        }
    ]
    mock_boto.return_value = mock_table

    df = utils.dynamodb_to_dataframe("table_name", ["label1", "label2"])
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "latency_seconds" in df.columns
    assert "label1" in df.columns
    assert "label2" in df.columns
    assert df.iloc[0]["latency_seconds"] == 0.123
    assert df.iloc[0]["label1"] == 1
    assert df.iloc[0]["label2"] == 0
