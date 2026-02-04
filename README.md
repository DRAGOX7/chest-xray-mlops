# 🩻 Chest X-Ray Abnormality Detection (MLOps Pipeline)

[![Build and Deploy to Azure](https://github.com/DRAGOX7/chest-xray-mlops/actions/workflows/main.yml/badge.svg)](https://github.com/DRAGOX7/chest-xray-mlops/actions/workflows/main.yml)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Azure](https://img.shields.io/badge/Azure-Cloud-0078D4)

A full-stack Deep Learning application that detects abnormalities in chest X-rays using a fine-tuned **DenseNet121** model. The project demonstrates a complete **MLOps pipeline**, including model training, API development, containerization, and automated cloud deployment.

👉 **[Live Demo (API Docs)](https://aljaf-xray-final.azurewebsites.net/docs)**

---

## 🏗️ Architecture
The project follows a modern cloud-native architecture:
1.  **Model:** PyTorch DenseNet121 (Transfer Learning from ImageNet).
2.  **API:** FastAPI (Python) for serving real-time predictions.
3.  **Container:** Docker for consistent runtime environment.
4.  **Registry:** Azure Container Registry (ACR) for storing images.
5.  **Deployment:** Azure Web App for Containers (Serverless PaaS).
6.  **CI/CD:** GitHub Actions for automated build and deploy on every push.

---

## 🚀 How to Run Locally

### Prerequisites
* Docker Desktop installed
* Git installed

### Steps
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/DRAGOX7/chest-xray-mlops.git](https://github.com/DRAGOX7/chest-xray-mlops.git)
    cd chest-xray-mlops
    ```

2.  **Build the Docker image**
    ```bash
    docker build -t xray-app .
    ```

3.  **Run the container**
    ```bash
    docker run -p 8000:8000 xray-app
    ```

4.  **Test the API**
    Open your browser to `http://localhost:8000/docs` and upload a chest X-ray image.

---

## 🤖 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check (Returns "Hello World"). |
| `POST` | `/predict` | Upload an image file to get the probability of abnormality. |

**Example Response:**
```json
{
  "filename": "patient_xray.jpg",
  "abnormality_probability": "0.9821",
  "diagnosis": "Abnormal"
}
