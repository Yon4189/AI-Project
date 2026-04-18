import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

interpreter = None
input_details = None
output_details = None

# Class names for the plant disease
class_names = [
    'Pepper__bell___Bacterial_spot', 
    'Pepper__bell___healthy', 
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Potato___healthy', 
    'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 
    'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

def load_tflite_model():
    global interpreter, input_details, output_details
    # Load model from the exact same directory (once you run train.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'ml_model', 'plant_disease_model.tflite')
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def predict_disease(image_bytes):
    global interpreter, input_details, output_details
    
    if interpreter is None:
        load_tflite_model()

    # Open image directly from memory (bytes)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize and preprocess just like the exact logic before
    image = pil_image.resize((128, 128))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction)) * 100

    return class_names[predicted_class], confidence
