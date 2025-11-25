from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import time

import utils
import write_logs

# Create FastAPI app
app = FastAPI(title="Toxic Comment Moderation")

# Try and load model and tokenizer pipeline
try:
    ENTITY = 'chris-r-thompson1212-university-of-denver'
    PROJECT = "toxic-comment-multilabel"
    model, tokenizer, maxlen = utils.load_production_model_and_tokenizer(
        ENTITY, PROJECT
    )
    print("Model Loaded Successfully")
except FileNotFoundError:
    print("Error: unable to load model or tokenizer pipeline")
    model = None
    tokenizer = None
    maxlen = None

# Try and load model data labels
try:
    ENTITY = 'chris-r-thompson1212-university-of-denver'
    PROJECT = "toxic-comment-multilabel"
    labels = utils.load_labels_from_dataset(ENTITY, PROJECT)
    print("Data Labels Loaded Successfully")
except FileNotFoundError:
    print("Error: could not load data labels.")
    labels = None

# Create a class for the /predict endpoint
class PredictInput(BaseModel):
    comment: str

# Startup event to print if model is not loaded
@app.on_event("startup")
def startup_event():
    if model is None:
        print("WARNING: Model is not loaded. Prediction endpoints will not work properly")


# Health get endpoint
@app.get("/health")
async def root():
    return {"status": "ok"}


# Predict endpoint to make predictions using loaded model
@app.post("/predict")
async def make_prediction(input_data: PredictInput):
    # If model did not load properly, give 503 error
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Cannot make predictions",
        )

    # Make prediction with the model
    start = time.time()
    prediction = model.predict(
        utils.preprocess_user_input(input_data.comment, tokenizer, maxlen)
    )
    latency = time.time() - start

    # Create list of prediction probabilities
    prediction_list = prediction.tolist()

    # Create dictionary of prediction probabilities
    pred_proba_dict = {label: prediction_list[0][idx] for idx, label in enumerate(labels)}

    # Get predicted labels (threshold > 0.5)
    pred_labels = [label for label, prob in pred_proba_dict.items() if prob > 0.5]

    # Write log to DynamoDB
    write_logs.write_log(input_data, pred_labels, pred_proba_dict, latency, labels)

    # Return the prediction from the model
    return {"labels": pred_labels}
