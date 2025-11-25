import os
import re
import pickle
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import wandb


def load_production_model_and_tokenizer(
    entity, project, model_name="toxic-comment-multilabel"
):
    """
    Load production Keras model and tokenizer from W&B artifact.
    Returns: model, tokenizer, maxlen
    """

    # Login to W&B (uses WANDB_API_KEY env variable)
    wandb.login(key=os.environ["WANDB_API_KEY"])

    api = wandb.Api()

    # Fetch the production artifact
    artifact = api.artifact(
        f"{entity}/{project}/{model_name}:production", type="model"
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

    # Get MAX_LEN parameter for tokenizing user data later (default 300)
    maxlen = artifact.metadata.get("MAX_LEN", 300)

    return model, tokenizer, maxlen


def load_labels_from_dataset(entity, project, data_set_name="toxic-data"):
    """
    Load labels metadata from latest W&B dataset artifact.
    """

    api = wandb.Api()

    # Fetch latest dataset artifact
    artifact = api.artifact(
        f"{entity}/{project}/{data_set_name}:latest", type="dataset"
    )

    # Load labels from metadata
    labels = artifact.metadata.get("labels", None)

    if labels is None:
        raise ValueError("Dataset artifact does not contain label metadata")

    return labels


def clean_text(text):
    """
    Clean text for model input:
    - lowercase
    - remove wiki headings and bullets
    - remove repeated quotes and stray slashes
    - normalize punctuation and whitespace
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()
    # Remove wiki headings like "== something =="
    text = re.sub(r"==+[^=]+==+", " ", text)
    # Remove bullet markers like "*" at beginning of line
    text = re.sub(r"^\s*\*\s*", " ", text)
    # Remove triple quotes and repeated quotes
    text = text.replace('"""', " ").replace("''", " ")
    # Remove stray slashes
    text = re.sub(r"\s*/\s*", " ", text)
    # Normalize repeated punctuation
    text = re.sub(r"([.,!?;:]){2,}", r" \1 ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_user_input(user_input, tokenizer, maxlen):
    """
    Clean and tokenize user input, returning padded sequences.
    """

    user_input_cleaned = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([user_input_cleaned])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen)
    return padded_seq
