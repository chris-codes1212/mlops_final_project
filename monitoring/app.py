import streamlit as st
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

st.set_page_config(layout="wide")

st.title("Model Monitoring")

BACKEND_URL = "http://3.215.45.153:8000"

# Try and load model data labels
try:
    ENTITY = 'chris-r-thompson1212-university-of-denver'
    PROJECT = "toxic-comment-multilabel"
    train_df, labels, train_csv_path = utils.load_train_data_and_labels(
        ENTITY, PROJECT
    )
    utils.remove_files(train_csv_path)
except FileNotFoundError:
    train_df = None
    labels = None

try:
    TABLE_NAME = 'toxicity_app'
    REGION = 'us-east-1'
    new_df = utils.dynamodb_to_dataframe(TABLE_NAME, labels, REGION)
except Exception:
    new_df = None

if new_df is not None and "latency_seconds" in new_df.columns:

    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])

    # Remove _ from labels
    labels_clean = [label.replace("_", " ") for label in labels]

    # Count how many times each label appears
    label_counts_new = new_df[labels].sum().reset_index()
    label_counts_new.columns = ["label", "count"]
    label_counts_new['percent'] = label_counts_new['count'] / len(new_df)

    label_counts_train = train_df[labels].sum().reset_index()
    label_counts_train.columns = ["label", "count"]
    label_counts_train['percent'] = label_counts_train['count'] / len(train_df)

    st.subheader("Distrubution of Classes")
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=label_counts_new, x=labels_clean, y="count", ax=axes[0])
    sns.barplot(
        data=label_counts_train, x=labels_clean, y="count", ax=axes[1], color='orange'
    )

    axes[0].set_title('Predicted Labels')
    axes[1].set_title('Train Data Labels')
    axes[0].tick_params(axis='x', labelrotation=60)
    axes[1].tick_params(axis='x', labelrotation=60)

    st.pyplot(fig)

    fig_lat, ax_lat = plt.subplots(figsize=(12, 4))

    sns.lineplot(
        data=new_df.sort_values("timestamp"),
        x="timestamp",
        y="latency_seconds",
        ax=ax_lat,
    )

    ax_lat.set_title("Model Inference Latency Over Time")
    ax_lat.set_xlabel("Timestamp")
    ax_lat.set_ylabel("Latency (seconds)")
    ax_lat.tick_params(axis='x', labelrotation=45)

    ax_lat.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax_lat.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))

    st.pyplot(fig_lat)

else:
    st.warning("Latency data not available yet.")