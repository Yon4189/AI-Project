import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import json
import os

interpreter = None
input_details = None
output_details = None
class_names = []

def load_tflite_model():
    global interpreter, input_details, output_details, class_names
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'ml_model', 'plant_disease_model.tflite')
    json_path = os.path.join(base_dir, 'ml_model', 'class_names.json')
    try:
        with open(json_path, 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
<<<<<<< HEAD
        print("class_names.json missing! Models might not have been fully retrained yet.")
=======
        print(" class_names.json missing! Models might not have been fully retrained yet.")
>>>>>>> 126696c12a11e82aabe6a29ff991d05e72358a94
        class_names = []

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def predict_disease(image_bytes):
    global interpreter, input_details, output_details
    
    if interpreter is None:
        load_tflite_model()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
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

    if not class_names:
        raise Exception("Your AI Labels are missing! Please run 'python train.py' until it completely finishes to generate your mapping file.")
        
    return class_names[predicted_class], confidence
