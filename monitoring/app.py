import streamlit as st
import utils
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Model Monitoring")

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

# Select the 6 dummy label columns
label_cols = ["toxic", "obscene", "insult", "threat", "identity_hate", "severe_toxic"]

# Count how many times each label appears
label_counts = new_df[label_cols].sum().reset_index()
label_counts.columns = ["label", "count"]

# Plot
fig, ax = plt.figure(figsize=(10, 6))
sns.barplot(data=label_counts, x="label", y="count")
ax.title("Frequency of Toxicity Labels")
plt.ylabel("Count")
plt.xlabel("Label")
plt.xticks(rotation=45)
plt.show()

st.pyplot(fig)