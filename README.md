# Pothole Detection API using Flask and CNN

This project is a Flask-based web application that detects whether an image contains a pothole using a pre-trained Convolutional Neural Network (CNN) model. The app allows users to upload an image via a web form and returns a prediction with confidence.



## Overview

This Flask app allows users to upload an image and uses a pre-trained CNN model to classify the image as either containing a pothole or not. It returns the prediction with a confidence score.

## Technologies

- Python 3.8+
- Flask 2.x
- TensorFlow/Keras (for the CNN model)
- HTML/CSS (Frontend for the web form)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/pothole-detection-api.git
    cd pothole-detection-api
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Add your CNN model**:
    - Place your pre-trained model in the root directory and name it `modelcnn.h5`.

5. **Run the Flask app**:
    ```bash
    python app.py
    ```

6. **Access the app**:
    - Open your browser and navigate to `http://127.0.0.1:5000/`.

## Usage

1. Navigate to the homepage (`/`), and you'll see an upload form.
2. Upload an image containing a road.
3. The model will classify the image as either containing a pothole or not, and the result will be displayed with the confidence level.

## Project Structure

