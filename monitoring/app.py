import streamlit as st
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import requests

st.title("Model Monitoring")

BACKEND_URL = "http://52.90.174.40:8000"

# try and load model data labels
# load latest training data from weights and biases
try:
    ENTITY = 'chris-r-thompson1212-university-of-denver'
    PROJECT = "toxic-comment-multilabel"
    train_df, labels, train_csv_path = utils.load_train_data_and_labels(ENTITY, PROJECT)
    print("Train Data Loaded Successfully")
    utils.remove_files(train_csv_path)
except FileNotFoundError:
    print("Error: unable to load training data or training labels")
    train_df= None
    labels = None
    artifact_dir = None

try:
    TABLE_NAME = 'toxicity_app'
    REGION = 'us-east-1'
    new_df = utils.dynamodb_to_dataframe(TABLE_NAME, labels, REGION)
    print("New Data Loaded Successfully")
except:
    print("Error: unable to load training data or training labels")
    new_df= None

# Count how many times each label appears
label_counts_new = new_df[labels].sum().reset_index()
label_counts_new.columns = ["label", "count"]

label_counts_train = train_df[labels].sum().reset_index()
label_counts_train.columns = ["label", "count"]

# create plots
fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.barplot(data=label_counts_new, x="label", y="count", ax=axes[0])
sns.barplot(data=label_counts_train, x="label", y="count", ax=axes[1], color = 'orange')

axes[0].set_title('Predicted Labels')
axes[1].set_title('Train Data Labels')

axes[0].set_xlabel("Labels")
axes[1].set_xlabel("Labels")

st.pyplot(fig)

st.subheader("An app to grade the toxicity of online comments")
st.text("Enter a comment in the text box below. Press 'Submit' after typing a comment. You will see what classes of toxicity your comment falls into")

# get example comment from the user
user_input = st.text_input("Insert comment")

# if the user presses the 'Submit' button, we will give the user predictions for the toxicity labels of their comment
if st.button("Submit"):
    # create payload which needs to be in json format
    payload = {"comment": user_input}
    # try to call the /predict Fastapi endpoint with the new comment
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
        # response = requests.post(BACKEND_URL, json=payload, timeout=5)
        response.raise_for_status()  # raise exception for HTTP errors
        json_response = response.json()
        labels_list = json_response['labels']
        if len(labels_list) == 0:
            st.subheader("This comment is :green[non-toxic]")
        else:
            labels_str = ", ".join(label.capitalize() for label in labels_list)
            st.subheader(f"This comment is classified as :red[{labels_str}]")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
    
