import streamlit as st
import pickle
import numpy as np
import cv2
import mediapipe as mp
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- 1. Load Model and Setup ---
@st.cache_resource
def load_resources():
    if not os.path.exists('./model.p'):
        st.error("❌ model.p file not found in current directory.")
        st.stop()

    model_data = pickle.load(open('./model.p', 'rb'))
    model = model_data.get('model', None)
    scaler = model_data.get('scaler', None)

    # Load labels
    if os.path.exists('./data.pickle'):
        data_dict = pickle.load(open('./data.pickle', 'rb'))
        unique_labels = sorted(list(set(data_dict['labels'])))
        label_map = {i: label for i, label in enumerate(unique_labels)}
    else:
        label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    # Mediapipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    return model, scaler, label_map, hands, mp_hands


model, scaler, label_map, hands, mp_hands = load_resources()
MAX_FEATURES = 42


# --- 2. Video Processing ---
class SignDetector(VideoTransformerBase):
    def __init__(self, model, scaler, label_map, hands, mp_hands):
        self.model = model
        self.scaler = scaler
        self.label_map = label_map
        self.hands = hands
        self.mp_hands = mp_hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.predicted_sign = "WAITING..."

    def transform(self, frame):
        frame_bgr = frame.to_ndarray(format="bgr24")
        H, W, _ = frame_bgr.shape
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        data_aux, x_coords, y_coords = [], [], []
        self.predicted_sign = "NO HAND DETECTED"

        try:
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                for landmark in hand_landmarks.landmark:
                    data_aux.extend([landmark.x, landmark.y])
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)

                self.mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                )

                if len(data_aux) == MAX_FEATURES:
                    input_data = np.asarray(data_aux).reshape(1, -1)
                    try:
                        scaled_input = self.scaler.transform(input_data) if self.scaler else input_data
                    except Exception:
                        scaled_input = input_data

                    prediction = self.model.predict(scaled_input)[0]
                    if isinstance(prediction, (int, np.integer)):
                        self.predicted_sign = self.label_map.get(prediction, f"CLASS_{prediction}")
                    else:
                        self.predicted_sign = str(prediction)

                    # Draw prediction overlay
                    x1 = max(int(min(x_coords) * W) - 20, 0)
                    y1 = max(int(min(y_coords) * H) - 20, 0)
                    x2 = min(int(max(x_coords) * W) + 20, W)
                    y2 = min(int(max(y_coords) * H) + 20, H)

                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(frame_bgr, (x1, y1 - 40), (x2, y1), (30, 30, 30), -1)
                    cv2.putText(
                        frame_bgr,
                        self.predicted_sign,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
        except Exception as e:
            print("Runtime error:", e)

        return frame_bgr


# --- 3. UI ---
st.set_page_config(page_title="🤟 Sign Language Detector", page_icon="🖐️", layout="centered")

# Elegant gradient + smaller camera view
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #8EC5FC, #E0C3FC);
    }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    h1, h2, h3, p {color: #222; text-align: center;}
    video {
        width: 60% !important;
        height: auto !important;
        border-radius: 20px;
        box-shadow: 0 0 25px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤟 Real-Time Sign Language Detection")
st.markdown("### Powered by MediaPipe + Your Trained Model")

webrtc_streamer(
    key="sign-detector",
    video_transformer_factory=lambda: SignDetector(model, scaler, label_map, hands, mp_hands),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
