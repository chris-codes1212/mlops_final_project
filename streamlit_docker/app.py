import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import tensorflow as tf

import wandb
import pickle
import re

def load_production_model_and_tokenizer(entity, project, model_name="toxic-comment-multilabel"):
    
    # Login to W&B (will use WANDB_API_KEY env variable)
    wandb.login()
    
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

    return model, tokenizer


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.write("Loading production model...")

ENTITY = 'chris-r-thompson1212-university-of-denver'
PROJECT = "toxic-comment-multilabel"
model, tokenizer = load_production_model(ENTITY, PROJECT)

st.success("Model Loaded!")

user_input = st.text_input("insert comment")

if st.button("predict"):
    st.text("making prediction...")
    user_input_cleaned = clean(tes)
    model.predict(tokenized_user_input)
# from sklearn.metrics import accuracy_score, precision_score

st.title('Toxicity Analysis App')

# model = load_model('../model.keras')
