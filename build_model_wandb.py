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

    # create Keras tokenizer and set num_words parameter
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    
    cleaned_comments = train_df['comment_text'].apply(clean_text)

    tokenizer.fit_on_texts(cleaned_comments)

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
        
        #sample_weights[:, i] = y_train[:, i] * class_weights[i]





    sample_weights_flat = np.mean(sample_weights, axis=1).astype('float32')

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights_flat)) \
        .shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
        .batch(128).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
        .batch(128).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, labels


# create funciton to build our deep learning model
# def build_model(MAX_WORDS, MAX_LEN, labels):
#     model = Sequential([
#         Embedding(input_dim=MAX_WORDS + 1, output_dim=128, input_length=MAX_LEN),
#         LSTM(128, recurrent_activation='sigmoid', use_bias=True),
#         Dropout(0.2),
#         Dense(len(labels), activation='sigmoid', dtype='float32')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy']  # list or dict matching output names
#     )
    
#     return model



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
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(curve="ROC", multi_label=True)]
    )
    return model


def build_callbacks():
    callbacks = [
        ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, save_weights_only=False),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    return callbacks

def fit_model(model, train_ds, val_ds, callbacks):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks
    )

    return model, history

def evaluate_model(model, test_ds):
    test_accuracy, test_loss = model.evaluate(test_ds)

def main():

    # Get the 20000 most common words to tokenize
    MAX_WORDS = 20000
    # For each comment, we want the length to be 200. If it is more, it will be cut short, if it is less, it will be padded
    MAX_LEN = 300

    # get training, validation, and test data (tensorflow dataset objects)
    train_ds, val_ds, test_ds, labels = load_and_prepare_data('train.csv', MAX_WORDS, MAX_LEN)

    model = build_model(MAX_WORDS, MAX_LEN, labels)

    callbacks = build_callbacks()

    model, history = fit_model(model, train_ds, val_ds, callbacks)

    # imported_model = keras.models.load_model("model.keras")

    evaluate_model(model, test_ds)

if __name__ == "__main__":
    main()


