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

st.title('Toxicity Analysis App')

st.write("Loading production model...")

ENTITY = 'chris-r-thompson1212-university-of-denver'
PROJECT = "toxic-comment-multilabel"
model, tokenizer = load_production_model_and_tokenizer(ENTITY, PROJECT)

st.success("Model Loaded!")

user_input = st.text_input("insert comment")

if st.button("predict"):

    user_input_cleaned = clean_text(user_input)
    seq = tokenizer.text_to_sequences([user_input_cleaned])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=300)

    prediction = model.predict(padded_seq)
    st.write("Predicted Probabilities:", prediction)


# if comment:
#     # Preprocess comment (clean + tokenize + pad)
#     def clean_text(text):
#         import re
#         text = text.lower()
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text

#     MAX_LEN = 300  # same as training
#     cleaned_comment = clean_text(comment)
#     seq = tokenizer.texts_to_sequences([cleaned_comment])
#     padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)

#     # Predict
#     prediction = model.predict(padded_seq)
#     st.write("Predicted probabilities:", prediction)
# # from sklearn.metrics import accuracy_score, precision_score



# model = load_model('../model.keras')
