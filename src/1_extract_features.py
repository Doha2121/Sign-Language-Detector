import os
import cv2
import mediapipe as mp
import numpy as np
import pickle

# --- Configuration ---
# IMPORTANT: This must match your final organized data path.
DATA_DIR = r'D:\Sign Langauage Detector\Datasets\data_classification' 

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

data = []
labels = []
print("Starting feature extraction from:", DATA_DIR)

# Iterate through each sign directory (e.g., 'Alef', 'Baa')
for sign_dir in os.listdir(DATA_DIR):
    sign_path = os.path.join(DATA_DIR, sign_dir)
    if not os.path.isdir(sign_path):
        continue
        
    class_label = sign_dir # The folder name is the label (e.g., 'Alef')
    print(f"Processing class: {class_label}")

    # Iterate through each image in the sign directory
    for img_file in os.listdir(sign_path):
        data_aux = []
        img_path = os.path.join(sign_path, img_file)
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # We only process the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract the X and Y coordinates for all 21 landmarks
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)
                
                # Check if the feature vector length is consistent (should be 42)
                if len(data_aux) == 42: 
                    data.append(data_aux)
                    labels.append(class_label)
        
        except Exception as e:
            # You can log specific errors if needed
            pass 

# Save the extracted features to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ Extraction complete. Saved {len(data)} valid samples to data.pickle.")