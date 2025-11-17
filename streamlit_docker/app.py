import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import tensorflow as tf

import wandb

def load_production_model(entity, project, model_name="toxic-comment-multilabel"):
    # Authenticate using environment variable WANDB_API_KEY
    wandb.login()

    # Using the public API
    api = wandb.Api()

    # Get the production artifact
    artifact = api.artifact(
        f"{entity}/{project}/{model_name}:production",
        type="model"
    )

    # Download to a local directory
    model_path = artifact.download()

    # Artifact contains best_model.keras
    model_file = f"{model_path}/best_model.keras"

    # Load Keras model
    model = tf.keras.models.load_model(model_file)

    return model


st.write("Loading production model...")

ENTITY = 'chris-r-thompson1212-university-of-denver'
PROJECT = "toxic-comment-multilabel"
model = load_production_model(ENTITY, PROJECT)

st.success("Model Loaded!")

user_input = st.text_input("insert comment")

if st.button("predict"):
    st.text("making prediction...")
# from sklearn.metrics import accuracy_score, precision_score

st.title('Toxicity Analysis App')

# model = load_model('../model.keras')
