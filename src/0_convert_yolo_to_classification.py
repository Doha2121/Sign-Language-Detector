import os
import shutil

# --- Configuration ---
# Set the base directory where your 'train', 'test', 'valid' folders are located
BASE_DATA_DIR = r'D:\Sign Langauage Detector\archive\unaugmented\416' 

# The YAML file contains the sign names (classes)
YAML_PATH = os.path.join(BASE_DATA_DIR, 'data.yaml')

# The new directory to hold the organized images for feature extraction
OUTPUT_DIR = os.path.join(os.getcwd(), 'data_classification')

# --- Load Class Names from data.yaml ---
class_names = []
try:
    with open(YAML_PATH, 'r') as f:
        content = f.read()
    
    # Simple parsing to find the 'names:' list
    if 'names:' in content:
        names_start = content.find('names:') + len('names:')
        names_end = content.find('\n', names_start)
        # Extract the list string (e.g., ['alef', 'baa', ...])
        names_list_str = content[names_start:names_end].strip().replace('[', '').replace(']', '').replace("'", "").split(', ')
        class_names = [name.strip() for name in names_list_str if name.strip()]
        
    if not class_names:
        raise ValueError("Could not parse class names from data.yaml. Check its format.")
        
except FileNotFoundError:
    print(f"Error: data.yaml not found at {YAML_PATH}")
    exit()
except Exception as e:
    print(f"Error parsing data.yaml: {e}")
    exit()

print(f"Found {len(class_names)} classes: {class_names}")

# --- Conversion Logic ---
if os.path.exists(OUTPUT_DIR):
    print(f"Warning: Deleting existing output directory: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)
    
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Created output directory: {OUTPUT_DIR}")

# Create sub-folders for each sign
for name in class_names:
    os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)

# Process train, test, and valid splits
for split in ['train', 'test', 'valid']:
    split_path = os.path.join(BASE_DATA_DIR, split)
    images_path = os.path.join(split_path, 'images')
    labels_path = os.path.join(split_path, 'labels')
    
    if not os.path.exists(images_path):
        print(f"Skipping {split} split. Images folder not found.")
        continue
        
    print(f"\nProcessing {split} split...")
    
    for label_file in os.listdir(labels_path):
        if label_file.endswith('.txt'):
            label_filepath = os.path.join(labels_path, label_file)
            image_filename = label_file.replace('.txt', '.jpg') # Assuming images are JPG
            image_filepath = os.path.join(images_path, image_filename)
            
            if not os.path.exists(image_filepath):
                image_filename = label_file.replace('.txt', '.png') # Try PNG as backup
                image_filepath = os.path.join(images_path, image_filename)
                if not os.path.exists(image_filepath):
                     # If neither JPG nor PNG is found, skip this label
                    continue

            # Read the YOLO label file
            # YOLO format: <class_index> <x_center> <y_center> <width> <height>
            with open(label_filepath, 'r') as f:
                yolo_data = f.readline().strip().split()
                if not yolo_data:
                    continue
                
                class_index = int(yolo_data[0])
                
                # Get the actual sign name
                if 0 <= class_index < len(class_names):
                    sign_name = class_names[class_index]
                    target_dir = os.path.join(OUTPUT_DIR, sign_name)
                    
                    # Copy the image to the correct classification folder
                    shutil.copy(image_filepath, os.path.join(target_dir, image_filename))
                    
print("\nConversion complete! Images are organized in the 'data_classification' folder.")
print("You can now proceed to feature extraction.")