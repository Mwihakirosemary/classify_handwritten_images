import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
model = load_model('Models/mnist_cnn_model.h5')
# model = load_model('Models/mnist_mlp_model.joblib')
# model = load_model('Models/mnist_logistic_model.joblib')
# model = load_model('Models/mnist_knn_model.joblib')
# model = load_model('Models/mnist_svm_model.joblib')

# Streamlit UI
st.title("MNIST Handwritten Digit Classifier")
st.write("Upload a 28x28 pixel grayscale image of a digit, and the model will classify it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

# Prediction function
def predict_digit(image):
    # Preprocess the image to match model input
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0  # Normalize to 0-1
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model input
    
    # Get prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100  # Confidence in percentage
    return predicted_class, confidence

# Show prediction if an image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Make prediction
    predicted_class, confidence = predict_digit(image)
    st.write(f"Predicted Digit: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
