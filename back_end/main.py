from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import json
import os
import sklearn

import utils
import write_logs

# create FastAPI app
app = FastAPI(
    title="Toxic Comment Moderation"
)

# try and load model and tokenizer pipeline, if not successful, print error message
try:
    ENTITY = 'chris-r-thompson1212-university-of-denver'
    PROJECT = "toxic-comment-multilabel"
    model, tokenizer, maxlen = utils.load_production_model_and_tokenizer(ENTITY, PROJECT)
    print("Model Loaded Successfully")
except FileNotFoundError:
    print("Error: unable to load model or tokenizer pipeline")
    model = None
    tokenizer = None
    maxlen = None

# try and load model data labels
try:
    ENTITY = 'chris-r-thompson1212-university-of-denver'
    PROJECT = "toxic-comment-multilabel"
    labels = utils.load_labels_from_dataset(ENTITY, PROJECT)
    print("Data Labels Loaded Successfully")
except FileNotFoundError:
    print("Error: could not load data labels.")
    labels = None

# create a class for the /predict endpoint
class predict_input(BaseModel):
    comment: str
    # true_label: str

# create a class for the logs we will create
# class log(BaseModel):
#     timestamp: datetime
#     request_text: str
#     # predicted_label: str
#     # true_label: str



# create startup event to print if model is not loaded
@app.on_event("startup")
def startup_event():
    if model is None:
        print("WARNING: Model is not loaded. Prediction endpoints will not work properly")

# create health get endpoint to show 'status: ok' to confirm API is working
@app.get('/health')
async def root():
    return{'status': 'ok'}

# create predict endpoint to make predictions using loaded model
@app.post("/predict")
async def make_prediction(input_data: predict_input):

    # if model did not load properly, give 503 error
    if model is None:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail = 'Model is not loaded. Cannot make predictions'
        )
    
    # # turn input text into an array of words for correct model input data type
    # text_array = [input_data.comment]

    # make prediction with the model
    # processed_input = utils.preprocess_user_input(input_data.comment, tokenizer)
    prediction = model.predict(utils.preprocess_user_input(input_data.comment, tokenizer, maxlen))
    
    # create list of prediction probabilities
    prediction_list = prediction.tolist()

    # create dictionary of prediction probabilities
    pred_proba_dict = {}
    for i in labels:
        pred_proba_dict[i] = prediction_list[0][i]

    # get predicted lables (threshold >0.5)
    pred_labels = []
    for label in pred_proba_dict:
        if label.value > 0.5:
            pred_labels.append(label)

    # write log to DynamoDB
    write_logs.write_log(input_data, pred_labels, pred_proba_dict, labels)

    # return the prediction from the model
    return {labels: pred_labels}