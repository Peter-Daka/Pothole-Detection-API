from flask import Flask, render_template, request, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load your trained CNN model
MODEL_PATH = './modelcnn.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define the classes
class_labels = ['No Pothole', 'Pothole']

# Create a directory for uploaded images if it doesn't exist
UPLOAD_FOLDER = './static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(img_path):
    """Preprocess the image to the required format, maintaining aspect ratio."""
    img = Image.open(img_path)
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array



def predict_image(img_path):
    """Predict the class of the image using the loaded model."""
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]
    return class_labels[predicted_class_index], confidence

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        imagefile = request.files['imagefile']
        image_path = os.path.join(UPLOAD_FOLDER, imagefile.filename)
        imagefile.save(image_path)

        # Perform prediction
        predicted_class, confidence = predict_image(image_path)
        confidence_percentage = confidence * 100

        return render_template('index.html', 
                               prediction_text=f"The image is classified as: {predicted_class} with a confidence of {confidence_percentage:.2f}%",
                               image_url=url_for('static', filename='uploads/' + imagefile.filename))
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(port=5000, debug=True)
