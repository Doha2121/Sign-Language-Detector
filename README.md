# Sign Language Detector

**Real-time Arabic Sign Language detection using MediaPipe hand landmarks and a custom SVM model.**

This project allows detection and classification of Arabic hand signs from a webcam or video input. It uses MediaPipe to extract hand landmarks and a trained Support Vector Machine (SVM) for classification. A live Streamlit web demo enables real-time recognition.

---

## 🚀 Key Features

- Real-time hand sign detection via webcam.
- Feature extraction using **MediaPipe Hands** landmarks (42 features per hand).
- Classification using **Support Vector Machine (SVM)**.
- Web deployment using **Streamlit + streamlit-webrtc**.
- Handles multiple image formats (JPG, PNG).
- Clean, organized dataset preparation for training and testing.

---

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **NumPy** – Feature arrays
- **OpenCV** – Image/video processing
- **MediaPipe** – Hand landmark detection
- **scikit-learn** – SVM training and scaling
- **Streamlit + streamlit-webrtc** – Real-time web demo
- **Pickle** – Model & feature storage

## 🎯 Usage Instructions
 I used this dataset Arabic Sign Language ArSL dataset
from kaggle Arabic Sign Language ArSL dataset
https://www.kaggle.com/datasets/sabribelmadoui/arabic-sign-language-unaugmented-dataset
### 1️⃣ Dataset Preparation
- Organize your YOLO dataset using `1_organize_dataset.py`.
- This will create `data_classification/` with one folder per sign.

### 2️⃣ Feature Extraction
```bash
python src/2_extract_features.py
3️⃣ Model Training
python src/3_train_model.py


Trains SVM on features and saves the trained model & scaler as model.p.

Prints classification accuracy on a test split.

4️⃣ Real-Time Detection (Local)
python src/4_real_time_detect.py


Opens webcam and predicts hand signs in real-time.

Press q to quit.

5️⃣ Streamlit Web Demo
streamlit run web_demo/app.py


Accesses live webcam in browser.

Displays real-time predictions with confidence and FPS.

Works on Hugging Face Spaces or local deployment.

🎨 Demo

Live Demo (Hugging Face Spaces):
https://huggingface.co/spaces/Doha000/arabic-sign-language-detector



