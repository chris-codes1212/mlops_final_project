# MLOps Final Project: Toxic Comment Classification

This repository contains a full-stack MLOps project for multi-label toxicity classification of online comments using a deep learning LSTM model. The project includes **training**, a **FastAPI backend**, a **Streamlit frontend**, a **monitoring dashboard**, and infrastructure automation with **CloudFormation**, **Ansible**, and **CI/CD via GitHub Actions**.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Training the Model](#training-the-model)  
- [FastAPI Backend](#fastapi-backend)  
- [Streamlit Frontend](#streamlit-frontend)  
- [Monitor Dashboard](#monitor-dashboard)  
- [Docker and Deployment](#docker-and-deployment)  
- [CI/CD Workflow](#cicd-workflow)  
- [Infrastructure](#infrastructure)  
- [Testing](#testing)  
- [Populate DynamoDB](#populate-dynamodb)  
- [Access](#access)  

---

## Project Structure

```
/
├── training/             # Model training scripts
│   └── build_model.py
├── back_end/             # FastAPI backend
│   ├── main.py
│   ├── utils.py
│   └── write_logs.py
├── front_end/            # Streamlit frontend
│   └── app.py
├── monitor/              # Streamlit monitoring dashboard
│   └── app.py
├── infra/                # Infrastructure automation
│   ├── fastapi_backend_ec2/
│   ├── streamlit_frontend_ec2/
│   └── model_monitor_ec2/
├── tests/                # Unit tests
│   ├── backend_testing/
│   ├── frontend_testing/
│   └── monitor_testing/
└── populate_DB/          # Script to populate DynamoDB with test data
    └── populate_DB.py
```

---

## Training the Model

- `build_model.py` downloads the training data from **S3** (`train.csv`) as a pandas DataFrame.  
- Data is cleaned of unnecessary symbols and tokenized.  
- The dataset is split into training, validation, and test sets.  
- Class weights are computed to handle imbalanced classes.  
- An LSTM model is trained with callbacks for validation loss.  
- The model is evaluated on the test set.  
- If the model outperforms the previous "production" model (based on AUC), it is **tagged as production** in **Weights & Biases**.  
- The `tokenizer.pkl` is saved with the model artifact.

---

## FastAPI Backend

- `/health` endpoint for health checks.  
- `/predict` endpoint takes a text comment, preprocesses it, tokenizes it, and uses the best LSTM model from **Weights & Biases** to make predictions.  
- Predicted labels are returned as a list.  
- Each prediction is logged to **DynamoDB** (`toxic_app`) via `write_logs.py` with:  
  - Timestamp  
  - Comment text  
  - Predicted labels  
  - Probabilities for each label  
  - Latency of prediction  

---

## Streamlit Frontend

- `front_end/app.py` allows users to enter a comment and get predicted toxicity labels.  
- Calls the FastAPI `/predict` endpoint.  
- Displays predicted labels in a user-friendly interface.

---

## Monitor Dashboard

- `monitor/app.py` downloads logged data from DynamoDB and training data from **Weights & Biases**.  
- Features include:  
  - Frequency histogram of predicted toxicity labels vs. training labels  
  - Prediction latency over time  
- Provides visual insights into model performance and distribution shifts.

---

## Docker and Deployment

- Each component (`backend`, `frontend`, `monitor`) has its **own Dockerfile**.  
- Docker images are built and pushed to **AWS ECR**.  
- Containers are run on separate **EC2 instances** provisioned with Ansible playbooks.  

---

## CI/CD Workflow

The project now includes a **GitHub Actions workflow** that automates testing, building, and deploying:

1. **Trigger:** Workflow runs on **pull requests or merges to the `main` branch**.
2. **Test:** Linting (`flake8`) and unit tests (`pytest`) are executed. Failure blocks deployment.
3. **Docker Build & Push:** Images for the FastAPI backend, Streamlit frontend, and monitoring dashboard are built and pushed to **AWS ECR**.
4. **Deployment:** EC2 instances pull the new images and run the containers.

### Simple CI/CD Flow

```
GitHub PR / Merge to main
          │
          ▼
   GitHub Actions Workflow
          │
   ┌──────┴──────┐
   │   Tests     │
   │ (pytest)    │
   └──────┬──────┘
          ▼
   Docker Build & Push
     (Backend/Frontend/Monitor)
          ▼
  EC2 Instances pull & deploy
```

**How to trigger a deployment:**  
- Simply create a **pull request to `main`**.  
- Once tests pass, the workflow automatically builds new images and deploys them.

---

## Infrastructure

- `infra/` contains CloudFormation templates and Ansible playbooks for EC2 provisioning:  
  - `fastapi_backend_ec2/`  
  - `streamlit_frontend_ec2/`  
  - `model_monitor_ec2/`  
- Playbooks install Docker, pull the relevant images, and run containers on the EC2 instances.

---

## Testing

- Unit tests in `tests/` use **pytest**.  
- GitHub Actions workflow runs on pull requests:  
  - Linting with **flake8** (warnings only)  
  - Unit tests with **pytest** (failures block merge)

---

## Populate DynamoDB

- `populate_DB/populate_DB.py` uses additional CSV-formatted comment data.  
- Calls the backend `/predict` endpoint to generate predictions and populate DynamoDB.  
- Used for testing and generating meaningful monitor dashboard metrics.

---

## Access

- **Frontend:** [http://50.17.169.167:8501](http://50.17.169.167:8501)  
- **Monitor Dashboard:** [http://3.229.230.170:8501](http://3.229.230.170:8501)  
- **FastAPI Backend:** [http://3.215.45.153:8000](http://3.215.45.153:8000) (can test in Postman)

---

## Notes

- Make sure your **AWS credentials / IAM roles** have access to ECR, S3, and DynamoDB.  
- Backend relies on the production-tagged model in **Weights & Biases** for predictions.  
- Creating a pull request to `main` will automatically trigger the **CI/CD workflow** to build, push, and deploy new containers.

