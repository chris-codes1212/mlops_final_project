import re

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D

from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split

import numpy as np

import pickle

import wandb
from wandb.integration.keras import WandbCallback
from wandb.integration.keras import WandbMetricsLogger

np.random.seed(42) # NEVER change this line

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# create function to clearn the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# load data and create tensorflow dataset objects for train, val, and test data sets
def load_and_prepare_data(file_path, MAX_WORDS, MAX_LEN):

    # import data
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

    # new
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

# def promote_best_model(test_results, model_name="toxic-comment-multilabel"):
#     current_auc = test_results.get("test_auc", 0)

#     # Get the current production artifact
#     ENTITY = wandb.run.entity
#     PROJECT = wandb.run.project
#     api = wandb.Api()
    
#     try:
#         prod_artifact = api.artifact(f"{ENTITY}/{PROJECT}/{model_name}:production")
#         prod_auc = prod_artifact.metadata.get("test_auc", 0)
#     except wandb.CommError:
#         prod_auc = 0  # No production model exists yet

#     if current_auc > prod_auc:
#         print(f"Promoting model! AUC {current_auc:.4f} > {prod_auc:.4f}")

#         # Create new artifact for this run
#         model_artifact = wandb.Artifact(
#             name=model_name,
#             type="model",
#             metadata=test_results
#         )
#         model_artifact.add_file("best_model.keras")

#         # Log the artifact
#         wandb.log_artifact(model_artifact)

#         # Wait until artifact is fully logged
#         model_artifact.wait()

#         # Add the "production" alias
#         model_artifact.aliases.append("production")
#         model_artifact.save()
#     else:
#         print(f"Model not better than current production (AUC {prod_auc:.4f}). No promotion.")

def promote_best_model(test_results, model_name="toxic-comment-multilabel"):
    current_auc = test_results.get("test_auc", 0)
    dataset_used = test_results.get("dataset_artifact", None)

    ENTITY = wandb.run.entity
    PROJECT = wandb.run.project
    api = wandb.Api()

    # Load current production model
    try:
        prod_artifact = api.artifact(f"{ENTITY}/{PROJECT}/{model_name}:production")
        prod_auc = prod_artifact.metadata.get("test_auc", 0)
    except wandb.CommError:
        prod_auc = 0  # If no production model exists yet

    # Compare metrics
    if current_auc > prod_auc:
        print(f"Promoting model! AUC {current_auc:.4f} > {prod_auc:.4f}")

        # --- Promote MODEL ---
        model_artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata=test_results
        )
        model_artifact.add_file("best_model.keras")

        logged_artifact = wandb.log_artifact(model_artifact)
        logged_artifact.wait()

        # Tag the model "production"
        logged_artifact.aliases.append("production")
        logged_artifact.save()

        print("Model promoted to :production")

        if dataset_used is not None:
            try:
                # This will fetch the artifact from the same project as the run
                dataset_artifact = api.artifact(dataset_used)
                dataset_artifact.aliases.append("production")
                dataset_artifact.save()
                print(f"Dataset {dataset_used} also tagged as :production")

            except wandb.CommError:
                print(f"Warning: Could not load dataset artifact {dataset_used}")

    else:
        print(f"Model not better than current production (AUC {prod_auc:.4f}). No promotion.")


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

    # Load data
    train_ds, val_ds, test_ds, labels, logged_data_artifact = load_and_prepare_data(
        "train.csv", config.MAX_WORDS, config.MAX_LEN
    )

    # Build model using hyperparams from wandb.config
    model = build_model(config.MAX_WORDS, config.MAX_LEN, labels)

    # Train
    callbacks = build_callbacks()
    model, history = fit_model(model, train_ds, val_ds, config.epochs, callbacks)

    # Evaluate and log to wandb
    test_results = evaluate_model(model, test_ds)
    print(test_results)

    # add current dataset_artifact key value pair to test_results for promote_best_model function
    test_results["dataset_artifact"] = f"{logged_data_artifact.name}:latest"
    
    # Create a model artifact
    model_artifact = wandb.Artifact(
        name="toxic-comment-multilabel",
        type="model",
        metadata=test_results
    )

    model_artifact.add_file("best_model.keras")  # already saved by ModelCheckpoint
    model_artifact.add_file("tokenizer.pkl") # now adding tokenizer pipeline
    wandb.log_artifact(model_artifact)

    # Promote best model and its associated dataset to production
    promote_best_model(test_results, wandb.run.project)

    wandb.finish()


if __name__ == "__main__":
    main()


