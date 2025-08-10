# 🤖 End-to-End MLOps Pipeline – Iris Classifier Project - Group 64

This project demonstrates a complete **End-to-End MLOps workflow** using the Iris dataset. It includes model training, hyperparameter tuning, experiment tracking, serving via FastAPI, containerization with Docker, CI/CD with GitHub Actions, and deployment to AWS EC2 with logging and monitoring.

---

## 📌 Objective 

Automate and scale the lifecycle of an ML model from experimentation to production deployment, using real-world MLOps tools and cloud infrastructure.

---

## 🧰 Tech Stack

- **Data & Modeling**: scikit-learn, pandas, Optuna
- **Experiment Tracking**: MLflow
- **Model Serving**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Deployment**: AWS EC2 + ECR
- **Logging**: Amazon CloudWatch
- **Monitoring**: Prometheus & Grafana (local setup)
- **A/B Testing**: Supported locally

---

## 🗂️ Folder Structure

```
.
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── baseline.py
│   ├── train_model.py
│   ├── data_loader.py
│   ├── model_builder.py
│   ├── utils.py
│   └── config/
│       └── model_config.yaml
├── app/
│   ├── main.py
│   └── predict.py
├── Dockerfile
├── .github/workflows/ci-cd.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Model Training (with MLflow + Optuna)

```bash
python src/train_model.py
```

MLflow logs are saved under `mlruns/`. The best model is saved to `models/` or registered in MLflow.

---

## 🌐 Run Locally with FastAPI

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
  "prediction": "setosa"
}
```

---

## 🐳 Docker Usage

```bash
# Build Docker image
docker build -t iris-classifier-app .

# Run container
docker run -d -p 8000:8000 iris-classifier-app
```

---

## 🚀 CI/CD & EC2 Deployment

- GitHub Actions workflow builds the image and pushes to AWS ECR
- SSHs into EC2 instance
- Pulls the latest image and runs the container
- Secrets are managed using GitHub Repository Settings

---

## 📊 Logging & Monitoring

- ✅ Logs are streamed to **Amazon CloudWatch**
- ⚙️ Prometheus + Grafana are set up locally (to be ported to EC2)

---

## 🔁 A/B Testing

- `baseline.py` runs a simpler baseline model
- Run both models and expose endpoints `/predict-a` and `/predict-b`

---

## 🧱 Architecture Diagram

```
User ─▶ FastAPI ─▶ Docker ─▶ Model ─▶ MLflow
           │           │         │
           ▼           ▼         ▼
     GitHub Actions → AWS EC2 → CloudWatch
```

---

## ➕ Extending to a New Dataset

To use a new dataset:
1. Replace preprocessing logic in your notebook or `preprocess.py`
2. Update `data_loader.py` to match new schema
3. Adjust `model_config.yaml` with new features and target

Optional: Parameterize dataset name for multi-dataset support.

---

## ✅ Completion Checklist

| Task                           | Status     |
|--------------------------------|------------|
| Data Preprocessing             | ✅ Completed |
| Model Training                 | ✅ Completed |
| MLflow Integration             | ✅ Completed |
| Hyperparameter Tuning (Optuna)| ✅ Completed |
| FastAPI API                    | ✅ Completed |
| Dockerization                  | ✅ Completed |
| CI/CD with GitHub Actions      | ✅ Completed |
| EC2 Deployment                 | ✅ Completed |
| CloudWatch Logging             | ✅ Completed |
| Monitoring (Grafana/Prometheus)| 🔄 Local Testing |
| A/B Testing                    | 🔄 Local Setup |
| Final README                   | ✅ You're here! |

---

## 🧠 Author

**Shubh** – Building hands-on expertise in full MLOps workflows for production-ready machine learning systems.
