# Waste Classification AI ♻️

A machine learning–powered web application that classifies images of waste into:

- **Organic (O)**
- **Recyclable (R)**

The app is built with **Streamlit** on the frontend and uses a **TensorFlow / Keras** image classification model.  
A lightweight **Flask backend** is used for model inference when deployed.


## 🚀 Live Demo

Frontend (Streamlit Cloud):

- Production: https://ml-01-classify-waste-appuctsgit-r89m87u8kffmauhgtvpakb.streamlit.app/
- Testing: https://ml-01-classify-waste-appuctsgit-nzm8crshdcp9bw2upagazz.streamlit.app/

Backend (Flask API):

- Deployed on **Render.com**


## 🧠 What the App Does

1. User uploads an image of waste (or selects an example image)
2. The image is sent to the backend API
3. A trained CNN model predicts whether the waste is:
   - **Organic**
   - **Recyclable**
4. The result is displayed in the Streamlit UI with confidence scores


## 📁 Project Structure

├── app/
│ └── streamlit_app.py # Streamlit frontend
│
├── server/
│ └── server.py # Flask backend (model inference API)
│
├── model/
│ └── waste_classifier.keras # Trained Keras model
│
├── notebooks/
│ └── training.ipynb # Model training & experimentation
│
├── data/
│ └── examples/ # Example images for demo/testing
│
├── requirements.txt # pip dependencies (deployment)
├── base_environment.yml # Base conda environment
├── waste_env.yml # Development conda environment
├── .env # Environment variables (not committed)
└── README.md



## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Flask (REST API)
- **ML Framework:** TensorFlow / Keras
- **Model Type:** CNN image classifier
- **Deployment:**
  - Streamlit Cloud (frontend)
  - Render.com (backend)
- **Environments:** Conda (dev), venv (deployment)


## ⚙️ Running Locally (Development)

### 1. Create the Conda environment
```
conda env create -f waste_env.yml
conda activate waste-classifier
```
### 2. Set environment variables
Create a .env file:
```
BACKEND_URL=http://localhost:5000
```
(Used by the Streamlit app to call the Flask API)
### 3. Run the backend (Flask)
```python server/server.py```
Backend will be available at:
```http://localhost:5000```
### 4. Run the frontend (Streamlit)
```streamlit run app/streamlit_app.py```

## 📦 Deployment Setup
**Frontend – Streamlit Cloud**
- Uses ```requirements.txt```
- Entry point:
```app/streamlit_app.py```
- Backend URL is configured via Streamlit secrets or environment variables
**Backend – Render.com**
- Flask app deployed separately
- Environment variables configured in Render dashboard
- Uses the same model file under model/

## 🧪 Environment Strategy (Important)
Context	Environment
Development	Conda (waste_env.yml)
Local testing	venv (ship-venv)
Frontend prod	Streamlit Cloud
Backend prod	Render.com

Conda is used for development and experimentation.
A lightweight ```venv``` + ```requirements.txt``` is used to mirror production.

## 🔒 Environment Variables
Variables are never committed to GitHub.
Used environments:
- ```.env``` locally
- GitHub Codespaces secrets
- Streamlit Cloud secrets
- Render environment variables

## 🧠 Model Training
- Model was trained in a Jupyter notebook under notebooks/
- Dataset preprocessing, augmentation, and evaluation are documented in the notebook
- Final model exported as .h5 and stored under model/

## 📌 Future Improvements
- Multi-class waste classification
- Model confidence explanations
- Batch image upload
- Dataset expansion
- Model versioning

## 👤 Author
Built by **Lampros Velentzas**
Machine Learning / Data Science Project
