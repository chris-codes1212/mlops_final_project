import matplotlib.pyplot as plt
import seaborn as sns

import datetime
from pydantic import BaseModel

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
    
    # Get MAX_LEN parameter for tokenizing user data later, set to 300 by default
    maxlen = artifact.metadata.get("MAX_LEN", 300)

    return model, tokenizer, maxlen

def load_labels_from_dataset(entity, project, data_set_name = "toxic-data"):
    api = wandb.Api()

    # Fetch latest dataset artifact
    artifact = api.artifact(f"{entity}/{project}/{data_set_name}:latest", type="dataset")

    # Load labels from metadata
    labels = artifact.metadata.get("labels", None)

    if labels is None:
        raise ValueError("Dataset artifact does not contain label metadata")

    return labels

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_user_input(user_input, tokenizer, maxlen):
    user_input_cleaned = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([user_input_cleaned])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=maxlen)
    return padded_seq


class log(BaseModel):
    timestamp: datetime
    request_text: str
    predicted_label: str
    true_label: str
    
# # create a funciton to handle writing json logs
def write_logs(input_data, prediction):

    # create new log object
    new_log = log( 
        timestamp = datetime.now().astimezone().isoformat(),
        request_text = input_data.text,
        predicted_label = prediction[0],
        true_label = input_data.true_label
        )

    # format log object as a json object
    new_log_json = new_log.model_dump_json()
    
#     # set log file path -- Using .jsonl b/c will be cleaner than using a list format
#     file_path = "../logs/prediction_logs.ndjson"

#     # check if logs directory exists, if not create it and logs file and write new log to file
#     if not os.path.isdir("../logs"):
#         os.makedirs("../logs", exist_ok=True)
#         with open(file_path, "w") as log_file:
#             log_file.write(new_log_json)
#             return 

#     # check if log file exists, if not, create it and write new log to file then return
#     # also checks if file is empty, if so, write new log then return
#     if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
#         with open(file_path, "w") as log_file:
#             log_file.write(new_log_json)
#             return 
        
#     # otherwise, write new log with new line
#     else:
#         existing_logs_list = []
#         with open(file_path, "a") as log_file:
#             log_file.write(f'\n{new_log_json}')


# user_input = st.text_input("insert comment")

# if st.button("predict"):

#     user_input_cleaned = clean_text(user_input)
#     seq = tokenizer.texts_to_sequences([user_input_cleaned])
#     padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=300)

#     prediction = model.predict(padded_seq)
#     st.write("Predicted Probabilities:", prediction)


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
