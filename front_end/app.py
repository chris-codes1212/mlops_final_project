import streamlit as st
import requests
import os
import time

BACKEND_URL = os.getenv(
    "BACKEND_URL", "http://3.215.45.153:8000"
)

for i in range(10):
    try:
        r = requests.get(f"{BACKEND_URL}/health")
        r.raise_for_status()
        print("Backend ready!")
        break
    except requests.exceptions.RequestException:
        print("Backend not ready, retrying...")
        time.sleep(2)


# Create title and subtitle
st.title("Comment Toxicity Classifier")
st.subheader("An app to grade the toxicity of online comments")
st.text(
    "Enter a comment in the text box below. Press 'Submit' after typing "
    "a comment. You will see what classes of toxicity your comment "
    "falls into"
)

# Get example comment from the user
user_input = st.text_input("Insert comment")

# If the user presses the 'Submit' button, we will give the user predictions
# for the toxicity labels of their comment
if st.button("Submit"):
    # Create payload which needs to be in JSON format
    payload = {"comment": user_input}
    # Try to call the /predict FastAPI endpoint with the new comment
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict", json=payload, timeout=10
        )
        response.raise_for_status()  # raise exception for HTTP errors
        json_response = response.json()
        labels_list = json_response["labels"]
        if len(labels_list) == 0:
            st.subheader("This comment is :green[non-toxic]")
        else:
            labels_str = ", ".join(label.capitalize() for label in labels_list)
            st.subheader(f"This comment is classified as :red[{labels_str}]")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
