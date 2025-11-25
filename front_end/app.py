import streamlit as st
import requests
import os
import time


BACKEND_URL = os.getenv(
    "BACKEND_URL", "http://3.215.45.153:8000"
)

# Health check on backend loop
def wait_for_backend():
    for _ in range(10):
        try:
            r = requests.get(f"{BACKEND_URL}/health")
            r.raise_for_status()
            print("Backend ready!")
            return True
        except requests.exceptions.RequestException:
            print("Backend not ready, retrying...")
            time.sleep(2)
    return False

# Wrap all Streamlit UI logic inside a testable function.
def run_app():
    """"""
    
    # Title and instructions
    st.title("Comment Toxicity Classifier")
    st.subheader("An app to grade the toxicity of online comments")
    st.text(
        "Enter a comment in the text box below. Press 'Submit' after typing "
        "a comment. You will see what classes of toxicity your comment "
        "falls into"
    )

    user_input = st.text_input("Insert comment")

    if st.button("Submit"):
        payload = {"comment": user_input}

        try:
            response = requests.post(
                f"{BACKEND_URL}/predict", json=payload, timeout=10
            )
            response.raise_for_status()

            labels = response.json().get("labels", [])

            if len(labels) == 0:
                st.subheader("This comment is :green[non-toxic]")
            else:
                labels_str = ", ".join(label.capitalize() for label in labels)
                st.subheader(f"This comment is classified as :red[{labels_str}]")

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {e}")


# Run the health check and UI ONLY when launched normally
if __name__ == "__main__":
    wait_for_backend()
    run_app()
