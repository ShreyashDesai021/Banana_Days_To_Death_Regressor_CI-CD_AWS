# 🍌 Banana Days-to-Death Predictor

An end-to-end Machine Learning and Computer Vision pipeline designed to predict the remaining edible shelf-life of bananas from images. 

This project fuses **YOLOv8** Instance Segmentation with **EfficientNet-B0** Regression within a production-grade, reproducible MLOps architecture.

## 🌟 Key Features
- **Two-Stage Architecture:** 
  - **Stage A:** **YOLOv8** isolates and crops individual bananas out of bunches.
  - **Stage B:** **EfficientNet-B0** predicts the continuous "days remaining" for each crop.
- **Custom Robust Loss Function:** Combines Huber Loss (noise robust) and an Ordinal Penalty (penalizing safety-critical over-prediction errors).
- **MLOps & Reproducibility:** 5-stage data processing pipeline governed by **DVC**, with **MLflow** integrated for hyperparameter and metric tracking.
- **REST API & Web UI:** **Flask-based** application that returns structured JSON payloads and base64-encoded annotated image rendering.
- **Containerized & CI/CD Ready:** Packaged in **Docker** with **GitHub Actions** workflows for automated testing and continuous integration.

## 📊 Ripeness Categories
The model's continuous regression output maps to four discrete produce maturity stages:
* 🟢 **Green:** > 6 days left (Not ripe)
* 🟡 **Yellow:** 4–6 days left (Ready to eat)
* 🟠 **Spotted:** 1–3 days left (Eat soon)
* 🟤 **Overripe:** ≤ 0 days left (Inedible)

## 🚀 Setup & Installation

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/ShreyashDesai021/Banana_Days_To_Death_Regressor_CI-CD_AWS.git
cd Banana_Days_To_Death_Regressor_CI-CD_AWS
pip install -r requirements.txt
pip install -e .  # Install local package
```

**2. Configure the Segmentation Model:**
Download your trained YOLOv8 model (`best.pt`) from Roboflow and place it here:
```bash
model/segmentation_model/weights/best.pt
```

## 🧠 Running the MLOps Pipeline
This project uses **DVC** to track the extensive ML pipeline. To execute the entire 5-stage pipeline (Data Ingestion $\rightarrow$ Segmentation Validation $\rightarrow$ BaseModel Prep $\rightarrow$ Training $\rightarrow$ Evaluation):

```bash
dvc repro
```
*(Alternatively, you can run `python main.py`)*

To view experiment training metrics and logged hyperparameters:
```bash
mlflow ui --port 5000
```

## 🌐 Running the Inference API (Web App)
To launch the Flask REST API and access the drag-and-drop Web UI:
```bash
python app.py
```
Access the dashboard at `http://localhost:8080`.

**API Usage Example (cURL):**
```bash
curl -X POST -F "image=@your_banana_photo.jpg" http://localhost:8080/predict
```

## 🐳 Docker Deployment
Easily build and run the containerized application across environments:
```bash
docker build -t banana-predictor:latest .
docker run -p 8080:8080 banana-predictor:latest
```

## 🔧 Hyperparameter Configuration
All model hyperparameters (Epochs, Learning Rates, Augmentations, Architecture configurations) are abstracted out into `params.yaml` for extreme ease-of-use. Modifying variables in this file will automatically invalidate only the necessary DVC pipeline stages on your next `dvc repro` run.
