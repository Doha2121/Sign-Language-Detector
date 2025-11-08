import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC              # New Model Import
from sklearn.preprocessing import StandardScaler # New Preprocessing Import
from sklearn.metrics import accuracy_score

# Load the data
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except FileNotFoundError:
    print("Error: 'data.pickle' not found. Please run 1_extract_features.py first.")
    exit()

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
print(f"Total samples loaded: {len(data)}")

# --- 1. Scale the Data (Crucial for SVM) ---
print("Scaling features...")
scaler = StandardScaler()
# Note: Scaling is done on the full dataset before splitting
data_scaled = scaler.fit_transform(data) 

# Split data (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    data_scaled, labels,  # Use the scaled data
    test_size=0.2, 
    shuffle=True, 
    stratify=labels,
    random_state=42 # Keep fixed for consistent testing
)

# --- 2. Initialize and Train the Support Vector Classifier ---
# kernel='rbf' is powerful for non-linear classification
# C and gamma are hyperparameters, starting with common values
model = SVC(kernel='rbf', C=10, gamma='auto', random_state=42) 
print("Training Support Vector Classifier...")

# Training the model
model.fit(x_train, y_train)
print("Training complete.")

# Test the model and print the accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"\n--- SVM Model Performance ---")
print(f"Accuracy: {score * 100:.2f}%")
print(f"-----------------------------")

# Save the trained model
with open('model.p', 'wb') as f:
    # Save the model
    pickle.dump({'model': model, 'scaler': scaler}, f) # Save the scaler too!

print("✅ Model saved as 'model.p'")