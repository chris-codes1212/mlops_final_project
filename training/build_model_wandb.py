import re
import io
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from skmultilearn.model_selection import iterative_train_test_split
import wandb
from wandb.integration.keras import WandbMetricsLogger

# ---------------------------------------
# SETTINGS
# ---------------------------------------
np.random.seed(42)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ---------------------------------------
# TEXT CLEANING
# ---------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------
# DATA LOADING
# ---------------------------------------
def load_and_prepare_data(file_path, MAX_WORDS, MAX_LEN):
    train_df = pd.read_csv(file_path, engine='python')
    labels = train_df.drop(columns=['id','comment_text']).columns.to_list()

    # Log dataset to W&B
    data_artifact = wandb.Artifact("toxic-data", type="dataset")
    data_artifact.add_file(file_path)
    data_artifact.metadata = {"labels": labels}
    logged_data_artifact = wandb.log_artifact(data_artifact)
    logged_data_artifact.wait()

    # Tokenizer
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    cleaned_comments = train_df['comment_text'].apply(clean_text)
    tokenizer.fit_on_texts(cleaned_comments)

    X_data = pad_sequences(tokenizer.texts_to_sequences(cleaned_comments), maxlen=MAX_LEN)
    y_data = train_df[labels].values.astype('float32')

    # Iterative stratification
    X_train, y_train, X_test, y_test = iterative_train_test_split(X_data, y_data, test_size=0.2)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_train, y_train, test_size=0.25)

    # Class/sample weights
    class_totals = np.sum(y_train, axis=0)
    total_samples = y_train.shape[0]
    class_weights = {i: total_samples / (len(labels) * class_totals[i]) for i in range(len(labels))}
    sample_weights = np.ones_like(y_train, dtype='float32')
    for i in range(len(labels)):
        sample_weights[:, i] = y_train[:, i] * class_weights[i] + (1 - y_train[:, i])
    sample_weights_flat = np.mean(sample_weights, axis=1).astype('float32')

    # TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights_flat)) \
        .shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(128).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, labels, tokenizer, logged_data_artifact

# ---------------------------------------
# MODEL
# ---------------------------------------
def build_model(MAX_WORDS, MAX_LEN, labels):
    model = Sequential([
        Embedding(MAX_WORDS + 1, 200, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True, dtype='float32')),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(labels), activation='sigmoid', dtype='float32')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    return model

# ---------------------------------------
# CALLBACKS (no ModelCheckpoint)
# ---------------------------------------
def build_callbacks():
    return [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        WandbMetricsLogger()
    ]

# ---------------------------------------
# TRAINING
# ---------------------------------------
def fit_model(model, train_ds, val_ds, epochs, callbacks):
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model, history

# ---------------------------------------
# EVALUATION
# ---------------------------------------
def evaluate_model(model, test_ds):
    loss, auc = model.evaluate(test_ds)
    wandb.log({"test_loss": loss, "test_auc": auc})
    return {"test_loss": loss, "test_auc": auc}

# ---------------------------------------
# PROMOTION
# ---------------------------------------
def promote_best_model(test_results, logged_dataset_artifact, logged_model_artifact, model_name="toxic-comment-multilabel"):
    current_auc = test_results.get("test_auc", 0)
    ENTITY = wandb.run.entity
    PROJECT = wandb.run.project
    api = wandb.Api()

    try:
        prod_artifact = api.artifact(f"{ENTITY}/{PROJECT}/{model_name}:production")
        prod_auc = prod_artifact.metadata.get("test_auc", 0)
    except wandb.CommError:
        prod_auc = 0

    if current_auc > prod_auc:
        print(f"Promoting model! AUC {current_auc:.4f} > {prod_auc:.4f}")
        logged_model_artifact.aliases.append("production")
        logged_model_artifact.save()
        print("Model promoted to :production")

        if logged_dataset_artifact:
            logged_dataset_artifact.aliases.append("production")
            print("Dataset also tagged as production")
    else:
        print(f"Model not better than current production (AUC {prod_auc:.4f}). No promotion.")

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    wandb.init(
        project="toxic-comment-multilabel",
        config={
            "MAX_WORDS": 20000,
            "MAX_LEN": 300,
            "embedding_dim": 200,
            "lstm_units": 64,
            "dropout": 0.3,
            "batch_size": 128,
            "epochs": 1,
            "optimizer": "adam"
        }
    )
    config = wandb.config

    # Load data
    train_ds, val_ds, test_ds, labels, tokenizer, logged_dataset_artifact = load_and_prepare_data(
        "train.csv", config.MAX_WORDS, config.MAX_LEN
    )

    # Build and train model
    model = build_model(config.MAX_WORDS, config.MAX_LEN, labels)
    callbacks = build_callbacks()
    model, _ = fit_model(model, train_ds, val_ds, config.epochs, callbacks)

    # Evaluate
    test_results = evaluate_model(model, test_ds)

    # --- Log model & tokenizer to W&B without writing to disk ---
    model_artifact = wandb.Artifact(
        name="toxic-comment-multilabel",
        type="model",
        metadata=test_results
    )

    # Model
    model_buffer = io.BytesIO()
    tf.keras.models.save_model(model, model_buffer, save_format="keras")
    model_buffer.seek(0)
    with open("best_model.keras", "wb") as f:
        f.write(model_buffer.read())
    model_artifact.add_file("best_model.keras")

    # Tokenizer
    tokenizer_buffer = io.BytesIO()
    pickle.dump(tokenizer, tokenizer_buffer)
    tokenizer_buffer.seek(0)
    with open("tokenizer.pkl", "wb") as f:
        f.write(tokenizer_buffer.read())
    model_artifact.add_file("tokenizer.pkl")

    logged_model_artifact = wandb.log_artifact(model_artifact)
    logged_model_artifact.wait()

    # Promote if better
    promote_best_model(test_results, logged_dataset_artifact, logged_model_artifact, wandb.run.project)

    wandb.finish()

if __name__ == "__main__":
    main()
