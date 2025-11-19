import re
import boto3
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
import pickle
import os
import wandb
from wandb.integration.keras import WandbCallback
from wandb.integration.keras import WandbMetricsLogger

np.random.seed(42) # NEVER change this line

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# create function to clearn the input text
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # lower-case
    text = text.lower()

    # remove wiki headings like "== something =="
    text = re.sub(r"==+[^=]+==+", " ", text)

    # remove bullet markers like "*" at beginning of line
    text = re.sub(r"^\s*\*\s*", " ", text)

    # remove weird triple quotes and repeated quotes
    text = text.replace('"""', ' ').replace("''", " ")

    # remove stray slashes used by wiki formatting
    text = re.sub(r"\s*/\s*", " ", text)

    # normalize punctuation spacing
    text = re.sub(r"([.,!?;:]){2,}", r" \1 ", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def load_data():

    # Initialize S3 client (EC2 role credentials will be used automatically)
    s3 = boto3.client('s3')

    # S3 path
    bucket_name = "toxic-comment-classifier-training-data"
    key = "train.csv"

    # Local filename to save as (current directory)
    data_file_path = os.path.join(os.getcwd(), "train.csv")

    # Download
    s3.download_file(bucket_name, key, data_file_path)

    return data_file_path

# load data and create tensorflow dataset objects for train, val, and test data sets
def prepare_data(file_path, MAX_WORDS, MAX_LEN):

    # import data

    # Make sure s3fs is installed: pip install s3fs
    train_df = pd.read_csv(file_path, engine='python')

    # get labels
    labels = train_df.drop(columns=['id','comment_text']).columns.to_list()

    # create dataset artifact
    data_artifact = wandb.Artifact("toxic-data", type="dataset")
    data_artifact.add_file(file_path)
    # add labels as dataset artifact metadata
    data_artifact.metadata = {"labels": labels}
    
    # log the dataset artifact
    logged_data_artifact = wandb.log_artifact(data_artifact)
    logged_data_artifact.wait()

    # create Keras tokenizer and set num_words parameter
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    
    cleaned_comments = train_df['comment_text'].apply(clean_text)

    tokenizer.fit_on_texts(cleaned_comments)

        # --- Save tokenizer for inference ---
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    X_data = pad_sequences(
        tokenizer.texts_to_sequences(cleaned_comments),
        maxlen=MAX_LEN
        )
    
    # select the label columns for our y_train ds and convert to numpy matrix where each row corresponds to a single comments labels
    # make data type float for gradient calculation and creating sample weights
    y_data = train_df[labels].values.astype('float32')

    # using iterative stratification, which balances the label combinations across splits.

    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X_data, 
        y_data, 
        test_size=0.2
    )

    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_train, 
        y_train, 
        test_size=0.25
    )

    class_totals = np.sum(y_train, axis=0)

    total_samples = y_train.shape[0]

    class_weights = {i: total_samples / (len(labels) * class_totals[i]) for i in range(len(labels))}

    sample_weights = np.ones_like(y_train, dtype='float32')
    for i in range(len(labels)):
        sample_weights[:, i] = (
            y_train[:, i] * class_weights[i] + 
            (1 - y_train[:, i]) * 1.0  
            )

    sample_weights_flat = np.mean(sample_weights, axis=1).astype('float32')

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights_flat)) \
        .shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
        .batch(128).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
        .batch(128).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, labels, logged_data_artifact

def build_model(MAX_WORDS, MAX_LEN, labels):
    model = Sequential([
        Embedding(MAX_WORDS + 1, 200, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True, dtype='float32')),  # force FP32
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(labels), activation='sigmoid', dtype='float32')
    ])


    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True)]
    )
    return model



def build_callbacks():
    callbacks = [
        ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, save_weights_only=False),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        WandbMetricsLogger()
        # WandbCallback(log_weights=True, log_graph=False)
    ]

    return callbacks

def fit_model(model, train_ds, val_ds, epochs, callbacks):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    return model, history

def evaluate_model(model, test_ds):
    loss, auc = model.evaluate(test_ds)

    # Log to wandb
    log_data = {"test_loss": loss}
    log_data["test_auc"] = auc
    wandb.log(log_data)

    return log_data

def promote_best_model(test_results, logged_dataset_artifact, logged_model_artifact, model_name="toxic-comment-multilabel"):
    
    # get current test auc and the dataset used in this experiment
    current_auc = test_results.get("test_auc", 0)
    # dataset_used = test_results.get("dataset_artifact", None)

    ENTITY = wandb.run.entity
    PROJECT = wandb.run.project
    api = wandb.Api()

    # Load current production model and its test_auc score
    try:
        prod_artifact = api.artifact(f"{ENTITY}/{PROJECT}/{model_name}:production")
        prod_auc = prod_artifact.metadata.get("test_auc", 0)
    except wandb.CommError:
        prod_auc = 0  # If no production model exists yet

    # Compare metrics
    if current_auc > prod_auc:
        print(f"Promoting model! AUC {current_auc:.4f} > {prod_auc:.4f}")

        # Tag the model "production"
        logged_model_artifact.aliases.append("production")
        logged_model_artifact.save()

        print("Model promoted to :production")

        # promote current data set to production if model was promoted to production
        if logged_dataset_artifact is not None:
            try:
                # This will fetch the artifact from the same project as the run
    
                logged_dataset_artifact.aliases.append("production")
               
                print("Dataset also tagged as production")

            except wandb.CommError:
                print(f"Warning: Could not load dataset artifact")

    else:
        print(f"Model not better than current production (AUC {prod_auc:.4f}). No promotion.")


def remove_files(model_path, tokenizer_path, data_file_path):
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Model file deleted successfully")
    else:
        print("Model file does not exist")

    if os.path.exists(tokenizer_path):
        os.remove(tokenizer_path)
        print("Tokenizer file deleted successfully")
    else:
        print("Tokenizer file does not exist")

    if os.path.exists(data_file_path):
        os.remove(data_file_path)
        print("Data file deleted successfully")
    else:
        print("Data file does not exist")   


def main():

    # --- W&B Init ---
    wandb.init(
        project="toxic-comment-multilabel",
        config={
            "MAX_WORDS": 20000,
            "MAX_LEN": 300,
            "embedding_dim": 200,
            "lstm_units": 64,
            "dropout": 0.3,
            "batch_size": 128,
            "epochs": 3,
            "optimizer": "adam"
        }
    )
    config = wandb.config

    # load data from s3
    data_file_path = load_data()
    
    # Prepare data for model training
    train_ds, val_ds, test_ds, labels, logged_dataset_artifact = prepare_data(
        data_file_path, config.MAX_WORDS, config.MAX_LEN
    )

    # Build model using hyperparams from wandb.config
    model = build_model(config.MAX_WORDS, config.MAX_LEN, labels)

    # Train model
    callbacks = build_callbacks()
    model, history = fit_model(model, train_ds, val_ds, config.epochs, callbacks)

    # Evaluate and log to wandb
    test_results = evaluate_model(model, test_ds)
    print(test_results)
    
    # Create a model artifact
    model_artifact = wandb.Artifact(
        name="toxic-comment-multilabel",
        type="model",
        metadata=test_results
    )

    model_artifact.add_file("best_model.keras")  # already saved by ModelCheckpoint
    model_artifact.add_file("tokenizer.pkl") # now adding tokenizer pipeline
    logged_model_artifact = wandb.log_artifact(model_artifact)
    logged_model_artifact.wait()

    # Promote best model and its associated dataset to production
    promote_best_model(test_results, logged_dataset_artifact, logged_model_artifact, wandb.run.project)

    wandb.finish()

    # remove model and tokenizer files from disk
    remove_files("best_model.keras", "tokenizer.pkl", data_file_path)


if __name__ == "__main__":
    main()


