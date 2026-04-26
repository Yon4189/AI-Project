import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam 

# 1. Pipeline Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'dataset', 'CropImages')
model_dir = os.path.join(BASE_DIR, 'ml_model')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(dataset_path):
    print(f" Error: Dataset folder missing! Please place your images in: {dataset_path}")
    print("Structure should be:")
    print("  dataset/")
    print("    CropImages/")
    print("      Class_Name_1/ (e.g., Tomato_Early_blight/)")
    print("        img1.jpg")
    print("        img2.jpg")
    print("      Class_Name_2/")
    exit(1)

# 2. Hyperparameters
img_height, img_width = 128, 128
batch_size = 32
epochs = 15

print(f" Loading image data from: {dataset_path}")

# 3. Advanced Data Augmentation (This makes the AI practice on harder, rotated, and flipped images!)
datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# --- NEW LOGIC: Save the exact class name order to avoid label swiping bugs ---
class_indices = train_data.class_indices
# Invert dictionary to get index: class_name
idx_to_class = {v: k for k, v in class_indices.items()}
# Convert to a list sorted by index
sorted_class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

json_path = os.path.join(model_dir, 'class_names.json')
with open(json_path, 'w') as f:
    json.dump(sorted_class_names, f)
print(f" Auto-saved class map to {json_path}")
# -----------------------------------------------------------------------------

print(f"Found {len(train_data.class_indices)} distinct plant disease classes.")

# 4. Build Custom Deep CNN Architecture
print("Building Deep Convolutional Neural Network architecture...")
model = Sequential([
#  filter layer identifies simple lines.  
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2,2)),
#  filter layer identifies textures and small shapes.      
    Conv2D(64, (3,3), activation='relu'),
#   making the data smaller and easier to process.          
    MaxPooling2D(pool_size=(2,2)),
#   filter layer identifies textures and small shapes.          
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
#  filter layer identifies complex structures like the specific pattern of a fungus or mold.          
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
# turn the 2D image into a 1D list of numbers.    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])
# The Learning Process (Training)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train the Model
print(f"Training model for {epochs} iterations (Epochs)...")
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# 6. Save raw Keras Model (.h5)
h5_path = os.path.join(model_dir, 'plant_disease_model.h5')
model.save(h5_path)
print(f"Standard Model saved to {h5_path}")

# 7. Convert and Compress to TFLite (.tflite) for Web Inference
print("Converting to compressed TFLite format for rapid web Server processing...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = os.path.join(model_dir, 'plant_disease_model.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
    
print(f"TFLite Model saved to {tflite_path}")
print("Setup Pipeline 100% Complete! You can now launch web app with: python server.py")
