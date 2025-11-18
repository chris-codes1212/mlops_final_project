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

# create title and subtitle
st.title("Comment Toxicity Classifier")
st.subheader("A site to grade the toxicity of online comments")
st.text("Enter a comment in the text box below. Press 'Submit' after typing a comment. You will see what classes of toxicity your comment falls into")

# set url for the backend api (FastAPI) to make requests to
back_end_url = "http://44.202.157.103:8000/predict"

# get example comment from the user
user_input = st.text_input("Insert comment")

# if the user presses the 'Submit' button, we will give the user predictions for the toxicity labels of their comment
if st.button("Predict"):
    # create payload which needs to be in json format
    payload = {"comment": user_input}
    # try to call the /predict Fastapi endpoint with the new comment
    try:
        response = requests.post(back_end_url, json=payload, timeout=5)
        response.raise_for_status()  # raise exception for HTTP errors
        json_response = response.json()
        st.write(json_response['labels'])
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
