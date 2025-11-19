import matplotlib.pyplot as plt
import seaborn as sns

import datetime

import pandas as pd
import tensorflow as tf

import wandb
import pickle
import re
import os

def load_production_model_and_tokenizer(entity, project, model_name="toxic-comment-multilabel"):
    
    # Login to W&B (will use WANDB_API_KEY env variable)
    wandb.login(key=os.environ["WANDB_API_KEY"])
    
    api = wandb.Api()

    # Fetch the production artifact
    artifact = api.artifact(
        f"{entity}/{project}/{model_name}:production",
        type="model"
    )

    # Download the artifact locally
    artifact_path = artifact.download()

    # Load Keras model
    model_file = f"{artifact_path}/best_model.keras"
    model = tf.keras.models.load_model(model_file)

    # Load tokenizer
    tokenizer_file = f"{artifact_path}/tokenizer.pkl"
    with open(tokenizer_file, "rb") as f:
        tokenizer = pickle.load(f)
    
    # Get MAX_LEN parameter for tokenizing user data later, set to 300 by default
    maxlen = artifact.metadata.get("MAX_LEN", 300)

    return model, tokenizer, maxlen

def load_labels_from_dataset(entity, project, data_set_name = "toxic-data"):
    api = wandb.Api()

    # Fetch latest dataset artifact
    artifact = api.artifact(f"{entity}/{project}/{data_set_name}:latest", type="dataset")

    # Load labels from metadata
    labels = artifact.metadata.get("labels", None)

    if labels is None:
        raise ValueError("Dataset artifact does not contain label metadata")

    return labels

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

def preprocess_user_input(user_input, tokenizer, maxlen):
    user_input_cleaned = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([user_input_cleaned])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen)
    return padded_seq



