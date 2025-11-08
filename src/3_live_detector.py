import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- Load Model and Scaler ---
try:
    model_data = pickle.load(open('./model.p', 'rb'))
    model = model_data['model']
    scaler = model_data['scaler']
except Exception as e:
    print(f"Error loading model: {e}")
    os._exit(1)

# --- Load Labels ---
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    unique_labels = sorted(list(set(data_dict['labels'])))
except Exception as e:
    print(f"Error loading labels: {e}")
    os._exit(1)

max_data_len = 42

# --- Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    os._exit(1)

print("✅ Arabic Sign Language Detector started. Press 'q' to quit.")

prev_time = 0  # for FPS calculation

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    current_time = time.time()

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Extract 42 features (x, y)
        data_aux, x_coords, y_coords = [], [], []
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x)
            data_aux.append(landmark.y)
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)

        if len(data_aux) == max_data_len:
            try:
                # Prepare input and scale
                input_data = np.asarray(data_aux).reshape(1, -1)
                scaled_input = scaler.transform(input_data)

                # Predict label and probability (SVM doesn't return probs unless probability=True)
                prediction = model.predict(scaled_input)[0]
                predicted_sign = str(prediction)

                # Optional: approximate confidence using decision function
                if hasattr(model, "decision_function"):
                    scores = model.decision_function(scaled_input)
                    confidence = np.max(scores)
                    # Normalize roughly to 0–1 range
                    confidence = (confidence - np.min(scores)) / (np.ptp(scores) + 1e-6)
                else:
                    confidence = 1.0  # fallback

                # Bounding box (keep inside frame)
                x1 = max(int(min(x_coords) * W) - 20, 0)
                y1 = max(int(min(y_coords) * H) - 20, 0)
                x2 = min(int(max(x_coords) * W) + 20, W)
                y2 = min(int(max(y_coords) * H) + 20, H)

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

                # Draw black background for label
                cv2.rectangle(frame, (x1, y1 - 50), (x2, y1), (0, 0, 0), -1)

                # Label text
                cv2.putText(frame, f"{predicted_sign}", (x1 + 10, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

                # Confidence bar (under label)
                bar_x1, bar_y1 = x1, y1 - 55
                bar_x2 = int(bar_x1 + (x2 - x1) * confidence)
                cv2.rectangle(frame, (bar_x1, bar_y1 - 10), (bar_x2, bar_y1), (0, 255, 0), -1)
                cv2.putText(frame, f"{int(confidence * 100)}%", (x2 - 70, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"[Warning] Frame skipped: {e}")

    # --- FPS Calculation ---
    fps = 1 / (current_time - prev_time + 1e-6)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Display
    cv2.imshow('Arabic Sign Language Detector', frame)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
