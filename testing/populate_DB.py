import json
import requests
import pandas as pd
from sklearn.metrics import accuracy_score
import boto3
import os

import re

def load_data(): 
    # Initialize S3 client (EC2 role credentials will be used automatically)
    s3 = boto3.client('s3')

    # S3 path
    bucket_name = "toxic-comment-classifier-training-data"
    key = "test.csv"

    # Local filename to save as (current directory)
    data_file_path = os.path.join(os.getcwd(), "test.csv")

    # Download
    s3.download_file(bucket_name, key, data_file_path)

    return data_file_path

data_file_path = load_data()

df = pd.read_csv(data_file_path)

df = df.sample(n=1000, random_state=40) 

df.head()

import re

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # lower-case
    text = text.lower()

    # remove wiki headings like "== something =="
    text = re.sub(r"==+[^=]+==+", " ", text)

    # remove bullet markers like "*" at beginning of line
    text = re.sub(r"^\s*\*\s*", " ", text)

    # remove weird triple quotes and repeated quotes
    text = text.replace('"""', ' ').replace("''", " ")

    # remove stray slashes used by wiki formatting
    text = re.sub(r"\s*/\s*", " ", text)

    # normalize punctuation spacing
    text = re.sub(r"([.,!?;:]){2,}", r" \1 ", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df['comment_text']= df['comment_text'].apply(clean_text)

df.head()

def remove_files(data_file_path):
    if os.path.exists(data_file_path):
        os.remove(data_file_path)
        print("Data file deleted successfully")
    else:
        print("Data file does not exist")

payload_list = [{"comment": text} for text in df["comment_text"]]

url = "http://localhost:8000/predict" 

print("Adding records to DynamoDB...")
for payload in payload_list:
    try:
        # make call to /predict endpoint
        response = requests.post(url, json=payload)
    # print error message on exception
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

print("Finished adding records")

remove_files(data_file_path)
print(f"Removed {data_file_path} from disk")