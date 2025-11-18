import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import requests
import json
import wandb
import pickle
import re

back_end_url = "http://44.202.157.103:8000/predict"

user_input = st.text_input("Insert comment")

if st.button("Predict"):
    payload = {"comment": user_input}
    try:
        response = requests.post(back_end_url, json=payload, timeout=5)
        response.raise_for_status()  # raise exception for HTTP errors
        json_response = response.json()
        st.write(json_response)
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")



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
