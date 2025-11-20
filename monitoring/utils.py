import streamlit as st
import boto3
import pandas as pd
import os
import wandb
from boto3.dynamodb.conditions import Key, Attr


def remove_files(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} deleted successfully")
    else:
        print(f"{file_path} does not exist")

def load_train_data_and_labels(entity, project, data_set_name="toxic-data"):
    api = wandb.Api()

    # Fetch the latest dataset artifact
    artifact = api.artifact(
        f"{entity}/{project}/{data_set_name}:latest",
        type="dataset"
    )
    labels = artifact.metadata.get("labels", None)

    # Download the artifact to a temporary directory
    train_csv_path = artifact.get_entry("train.csv").download(".")

    # Load into pandas
    df = pd.read_csv(train_csv_path)

    return df, labels, train_csv_path

# load new data from DynamoDB
def dynamodb_to_dataframe(table_name, labels, region="us-east-1"):
    dynamodb = boto3.client("dynamodb", region_name=region)

    # Scan entire table (DynamoDB automatically paginates)
    items = []
    response = dynamodb.scan(TableName=table_name)
    items.extend(response["Items"])

    while "LastEvaluatedKey" in response:
        response = dynamodb.scan(
            TableName=table_name,
            ExclusiveStartKey=response["LastEvaluatedKey"]
        )
        items.extend(response["Items"])

    # Convert DynamoDB response → Python dict → DataFrame
    data = []
    for item in items:
        row = {}
        row["timestamp"] = item["timestamp"]["S"]
        row["comment"] = item["comment"]["S"]

        # Convert list of {"S": label} into a Python list of strings
        raw_labels = [entry["S"] for entry in item["prediction_labels"]["L"]]

        # Create dummy variables for each label
        for label in labels:
            row[label] = 1 if label in raw_labels else 0

        data.append(row)

    df = pd.DataFrame(data)
    return df
