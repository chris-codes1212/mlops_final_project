import streamlit as st
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import requests

st.set_page_config(layout="wide")

st.title("Model Monitoring")

# custom css for 3-D buttons
st.markdown("""
<style>
.rating-container {
    text-align: center;
    margin-top: 25px;
}

.rating-question {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 15px;
}

.rate-btn {
    display: inline-block;
    padding: 14px 30px;
    margin: 10px;
    font-size: 22px;
    font-weight: 600;
    color: white;
    border-radius: 14px;
    cursor: pointer;
    border: 3px solid transparent;
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    box-shadow: 0px 6px 14px rgba(0,0,0,0.4), inset 0px 2px 5px rgba(255,255,255,0.25);
    transition: all 0.25s ease-in-out;
    user-select: none;
}

.rate-btn:hover {
    transform: translateY(-5px);
    box-shadow: 0px 12px 22px rgba(0,0,0,0.35), inset 0px 2px 8px rgba(255,255,255,0.25);
}

.rate-btn:active {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #667eea, #764ba2);
}

# Thumbs up special gradient
.up-btn {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    border: 3px solid rgba(0, 255, 150, 0.7);
}

.down-btn {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    border: 3px solid rgba(255, 100, 120, 0.7);
}

</style>
""", unsafe_allow_html=True)
# ----------------------------------------------------------


BACKEND_URL = "http://52.90.174.40:8000"

# try and load model data labels
try:
    ENTITY = 'chris-r-thompson1212-university-of-denver'
    PROJECT = "toxic-comment-multilabel"
    train_df, labels, train_csv_path = utils.load_train_data_and_labels(ENTITY, PROJECT)
    utils.remove_files(train_csv_path)
except FileNotFoundError:
    train_df= None
    labels = None

try:
    TABLE_NAME = 'toxicity_app'
    REGION = 'us-east-1'
    new_df = utils.dynamodb_to_dataframe(TABLE_NAME, labels, REGION)
except:
    new_df= None

# remove _ from labels
labels_clean = [label.replace("_", " ") for label in labels]

# Count how many times each label appears
label_counts_new = new_df[labels].sum().reset_index()
label_counts_new.columns = ["label", "count"]
label_counts_new['percent'] = label_counts_new['count']/len(new_df)


label_counts_train = train_df[labels].sum().reset_index()
label_counts_train.columns = ["label", "count"]
label_counts_train['percent'] = label_counts_train['count']/len(train_df)

# create plots
fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.barplot(data=label_counts_new, x=labels_clean, y="count", ax=axes[0])
sns.barplot(data=label_counts_train, x=labels_clean, y="count", ax=axes[1], color='orange')

axes[0].set_title('Predicted Labels')
axes[1].set_title('Train Data Labels')
axes[0].tick_params(axis='x', labelrotation=60)
axes[1].tick_params(axis='x', labelrotation=60)

st.pyplot(fig)

st.subheader("An app to grade the toxicity of online comments")
st.text("Enter a comment in the text box below. Press 'Submit' after typing a comment.")

user_input = st.text_input("Insert comment")

# --- STORE RATING STATE ---
if "user_rating" not in st.session_state:
    st.session_state.user_rating = None


# submit button
if st.button("Submit"):

    payload = {"comment": user_input}

    try:
        response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        json_response = response.json()
        labels_list = json_response['labels']

        if len(labels_list) == 0:
            st.subheader("This comment is :green[non-toxic]")
        else:
            labels_str = ", ".join(label.capitalize() for label in labels_list)
            st.subheader(f"This comment is classified as :red[{labels_str}]")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")

    # rating buttons
    st.markdown("<div class='rating-container'>", unsafe_allow_html=True)

    st.markdown("<div class='rating-question'>Do you agree with the category?</div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("👍 Thumbs Up", key="thumb_up"):
            ###############################################################
            #                       CALLBACK FOR 👍 BUTTON               #
            #               (Place your code here later)                 #
            ###############################################################
            st.session_state.user_rating = "up"

    with col2:
        if st.button("👎 Thumbs Down", key="thumb_down"):
            ###############################################################
            #                       CALLBACK FOR 👎 BUTTON               #
            #               (Place your code here later)                 #
            ###############################################################
            st.session_state.user_rating = "down"

    st.markdown("</div>", unsafe_allow_html=True)
    # ---------------------------------------------------------------