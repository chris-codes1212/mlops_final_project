# MLOps Final Project

This repository implements an end‑to‑end MLOps pipeline featuring model
training, deployment, monitoring, and CI/CD automation.

## 🚀 Project Structure

    /training
        build_model.py        # Builds and trains LSTM toxicity classifier
        data/                 # Raw and processed training data
        models/               # Saved model artifacts

    /backend
        app.py                # FastAPI inference service
        utils/                # Preprocessing + model loading helpers

    /frontend
        streamlit_app.py      # Monitoring dashboard UI
        utils/                # Shared functions (fetch model stats, plots)

    /.github/workflows
        ci.yml                # CI/CD automation (tests + deployment)

------------------------------------------------------------------------

## 🧠 Model Training

-   The project trains an **LSTM-based multi-label toxicity
    classifier**.
-   Training uses **Weights & Biases** for experiment tracking.
-   Training is designed to run on **GPU-backed infrastructure**\
    (Recommended: AWS `g4dn.xlarge` or any NVIDIA GPU environment).

Run training:

``` bash
cd training
python build_model.py
```

Artifacts are stored locally and can be synced to W&B.

------------------------------------------------------------------------

## ⚙️ Backend (FastAPI Inference Service)

The backend exposes a `/predict` endpoint:

``` bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Loads the trained LSTM model and performs real‑time inference.

------------------------------------------------------------------------

## 📊 Monitoring Dashboard (Streamlit)

The `/frontend` directory contains a monitoring dashboard with:

-   Real-time model latency
-   Recent predictions
-   Error monitoring
-   Model confidence distribution

Run locally:

``` bash
streamlit run streamlit_app.py
```

------------------------------------------------------------------------

## 🐳 Dockerization

Both backend and frontend services are fully containerized.

Build images:

``` bash
docker build -t backend ./backend
docker build -t frontend ./frontend
```

------------------------------------------------------------------------

## 🔄 CI/CD Pipeline (GitHub Actions → EC2)

The workflow includes:

### ✔️ Steps

1.  Run unit tests
2.  Build & push Docker images to ECR
3.  SSH into EC2
4.  Pull & restart backend container
5.  Use GitHub secrets:
    -   `EC2_SSH_KEY`
    -   `BACK_END_EC2`
    -   `BACK_END_ECR`
    -   `WANDB_API_KEY`

Example deployment step:

``` yaml
- name: SSH and deploy backend service
  uses: appleboy/ssh-action@v0.1.7
  with:
    host: ${{ secrets.BACK_END_EC2 }}
    username: ubuntu
    key: ${{ secrets.EC2_SSH_KEY }}
    script: |
      docker pull $REPO_URI:$IMAGE_TAG
      docker stop backend || true
      docker rm backend || true
      docker run -d --name backend -p 8000:8000 --restart unless-stopped         -e WANDB_API_KEY=${{ secrets.WANDB_API_KEY }}         $REPO_URI:$IMAGE_TAG
  env:
    IMAGE_TAG: latest
    REPO_URI: ${{ secrets.BACK_END_ECR }}
```

------------------------------------------------------------------------

## 📁 Requirements

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🧪 Tests

Unit tests are automatically run through GitHub Actions.

Run locally:

``` bash
pytest
```

------------------------------------------------------------------------

## 🤝 Contributing

Pull requests are welcome!\
Please follow formatting + linting rules before opening a PR.

------------------------------------------------------------------------

## 📄 License

MIT License.
