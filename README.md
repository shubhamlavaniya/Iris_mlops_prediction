# Team - Group 64
SHALINI DIXIT - 2023ac05474@wilp.bits-pilani.ac.in – Contribution (100%)

SHUBHAM LAVANIYA  - 2023ac05899@wilp.bits-pilani.ac.in – Contribution (100%)

RAYAPUREDDYAVVMANOJKUMAR - 2023ad05128@wilp.bits-pilani.ac.in  – Contribution (100%)

RAVISH KUMAR J N - 2023ac05178@wilp.bits-pilani.ac.in - Contribution (100%)

# End-to-End MLOps Pipeline – Iris Classifier Project

This project demonstrates a complete **End-to-End MLOps workflow** using the Iris dataset. It includes data versioning with DVC, model training, hyperparameter tuning, experiment tracking, serving via FastAPI, containerization with Docker, CI/CD with GitHub Actions, and deployment to AWS EC2 with logging and monitoring.

---

## Objective

Automate and scale the lifecycle of an ML model from experimentation to production deployment, using real-world MLOps tools and cloud infrastructure.

---

## Tech Stack

- **Data & Modeling**: scikit-learn, pandas, Optuna
- **Data Version Control**: DVC
- **Experiment Tracking**: MLflow
- **Model Serving**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Deployment**: AWS EC2 + ECR
- **Logging**: SQLite + Grafana Dashboard
- **Database Viewer**: SQLite Web UI
- **Monitoring**: Prometheus & Grafana (local setup)

---

## Folder Structure

```
.
├── .dvc/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── app/
│     └── app.py
│     └── predict.py
│   ├── train_baseline.py
│   ├── train_model.py
│   ├── data_loader.py
│   ├── app_logging.py
│   ├── model_builder.py
│   ├── preprocess.py
│   ├── utils.py
│   └── config/
│       └── model_config.yaml
├── prometheus/
│   ├── prometheus.yml
├── logs/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
└── README.md
```

## Features

- **Version Data and Models** with DVC for full reproducibility.
- Train ML models and track experiments with MLflow.
- Optimize hyperparameters with Optuna.
- Deploy API using FastAPI.
- Track error and prediction logs in SQLite.
- Visualize metrics and logs in Grafana.
- Access SQLite DB via SQLite Web.
- Prometheus scrapes metrics exposed by FastAPI.
- End-to-end pipeline automated using GitHub Actions.

---

## Setup Instructions

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## DVC

# 1. Initialize DVC in the project root
dvc init

# 2. Configure a remote storage (e.g., a local directory)
# This directory should be outside your project folder.
mkdir ~/dvc_remote
dvc remote add -d <remote_name> <remote_path>

# 3. Run the DVC pipeline to generate and version data
dvc repro

# 4. Push the data to the remote storage
dvc push

# 5. Commit the DVC files to Git
git add .
git commit -m "Initialize DVC and run pipeline"

## Model Training (with MLflow + Optuna)

```bash
python src/train_model.py
```

MLflow logs are saved under `mlruns/`. The best model is saved to `models/` or registered in MLflow.

---

## Run Locally with FastAPI

```bash
uvicorn app.main:app --reload
```

- Endpoint: `http://localhost:8000/predict`
- Input:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
- Output:
```json
{
  "prediction": "class label"
}
```

---

## Docker Usage

```bash
# Build Docker image
docker build -t iris-fastapi-app:latest .

# Run container
docker run -p 8000:8000 iris-fastapi-app

# docker directory access
docker exec -it ef5e2dc2fd3d /bin/bash

# Run with Docker Compose (Locally)
```bash
docker-compose up --build
```

- Access FastAPI: [http://localhost:8000](http://localhost:8000)
- Access Grafana: [http://localhost:3000](http://localhost:3000) (Login: `admin` / `admin`)
- Access Prometheus: [http://localhost:9090](http://localhost:9090)
- Access SQLite Web: [http://localhost:8080](http://localhost:8080)

```

---

```

## CI/CD & EC2 Deployment(Optional)


- GitHub Actions workflow builds the image and pushes to AWS ECR
- SSHs into EC2 instance
- Pulls the latest image and runs the container
- Secrets are managed using GitHub Repository Settings
Secrets to add in GitHub:
- `DOCKER_USERNAME`, `DOCKER_PASSWORD`
- `EC2_HOST`, `EC2_USER`, `EC2_KEY` # for AWS cloud set up

---


## Logging & Monitoring

- SQLite + Grafana Dashboard
- Prometheus + Grafana are set up locally (to be ported to EC2)

---

## ➕ Extending to a New Dataset

To use a new dataset:
1. Replace preprocessing logic in your notebook or `preprocess.py`
2. Update `data_loader.py` to match new schema
3. Adjust `model_config.yaml` with new features and target


---
