

## 1. Project Overview
The **Smart Crop** project is an end-to-end AI application designed to identify plant diseases from images. It uses a custom-trained Deep Learning model to analyze leaf patterns and provide a diagnosis with a confidence score.

---

## 2. How the AI Model Works

### Type of Machine Learning
This project uses **Supervised Learning**. 
*   **Why?** Because the model is trained on a labeled dataset (the "PlantVillage" dataset). Each image in the dataset is already tagged with the correct disease name (e.g., "Tomato Early Blight", "Potato Healthy"). The model learns by comparing its guesses against these correct labels.

### Model Architecture: CNN
The model uses a **Convolutional Neural Network (CNN)**. CNNs are the gold standard for computer vision because they:
1.  **Extract Features**: Use "filters" to detect edges, shapes, and textures (spots, wilting patterns).
2.  **Downsample**: Use "Pooling" layers to simplify the data while keeping important features.
3.  **Classify**: Use "Dense" layers at the end to decide which disease category the image belongs to.

---

## 3. File-by-File Breakdown

### 📂 `train.py` (The Brain Creator)
This is where the AI is "born" and trained.
*   **Loading Data**: It looks into the `dataset/CropImages` folder.
*   **Data Augmentation**: It "stretches" the training data by rotating, flipping, and zooming images so the AI learns to recognize diseases even if the photo is blurry or tilted.
*   **Building the CNN**: It defines the layers of the neural network (Conv2D, MaxPooling, Flatten, Dense).
*   **Saving the Model**: 
    *   Saves a standard `.h5` model.
    *   Converts it to a **.tflite** (TensorFlow Lite) version, which is smaller and faster for the web server to use.
    *   Saves `class_names.json` so the server knows which ID corresponds to which disease name.

### 📂 `predict.py` (The Inference Engine)
This file handles the "thinking" when a new image arrives.
*   **Loading the Model**: It loads the `.tflite` model and the class names.
*   **Preprocessing**: It resizes any uploaded image to **128x128 pixels** and scales the colors (normalization) to match what the AI saw during training.
*   **Prediction**: It runs the image through the model and returns the **Disease Name** and the **Confidence Percentage** (how sure the AI is).

### 📂 `server.py` (The Postman & Librarian)
This is a pure Python web server that connects everything.
*   **Hosting**: It serves the `index.html` frontend to your browser.
*   **Handling Uploads**: When you click "Upload" on the website, this file receives the image data.
*   **Database (MongoDB)**: After a prediction is made, it saves the result (Filename, Disease, Confidence, Date) into a MongoDB database called `smart_crop_db`.
*   **Communication**: It sends the final result back to the website to show the user.

### 📂 `index.html` (The User Interface)
The "Face" of the project. It provides:
*   A clean, modern dashboard.
*   An image upload area.
*   Real-time display of the AI's diagnosis.

---

## 4. How to Run the Project
1.  **Step 1: Train the Model**
    ```bash
    python train.py
    ```
    *(Wait for it to finish and save the `.tflite` file)*

2.  **Step 2: Start the Server**
    ```bash
    python server.py
    ```

3.  **Step 3: Open the App**
    Go to `http://localhost:8080` in your web browser.

---

## 5. Technical Stack
*   **Language**: Python 3
*   **AI Framework**: TensorFlow / Keras
*   **Image Processing**: Pillow (PIL)
*   **Database**: MongoDB
*   **Frontend**: HTML5 / CSS3 / JavaScript
*   **Server**: Python `http.server` (No external frameworks like Flask/Django used for the core server)
