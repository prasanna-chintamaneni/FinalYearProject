import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model('trained_plant_disease_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128)) 
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict_disease(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    result_index = np.argmax(predictions)
    return result_index

# Function to predict fertilizer based on disease
def predict_fertilizer(disease):
    # Define a mapping between disease and fertilizer
    fertilizer_mapping = {
    'Apple___Apple_scab': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Apple___Black_rot': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Apple___Cedar_apple_rust': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Apple___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Blueberry___healthy': 'Acidic Fertilizer for Acid-Loving Plants',
    'Cherry_(including_sour)___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Cherry_(including_sour)___Powdery_mildew': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Nitrogen, Phosphorus, Potassium (NPK) Fertilizer',
    'Corn_(maize)___Common_rust_': 'Urea Fertilizer',
    'Corn_(maize)___Early_blight': 'Ammonium Nitrate Fertilizer',
    'Corn_(maize)___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Corn_(maize)___Northern_Leaf_Blight': 'Potassium Sulfate Fertilizer',
    'Grape___Black_rot': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Grape___Esca_(Black_Measles)': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Grape___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Orange___Haunglongbing_(Citrus_greening)': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Peach___Bacterial_spot': 'Sulphate of Potash Fertilizer',
    'Peach___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Pepper,_bell___Bacterial_spot': 'Potassium Nitrate Fertilizer',
    'Pepper,_bell___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Potato___Early_blight': 'Ammonium Phosphate Fertilizer',
    'Potato___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Potato___Late_blight': 'Potassium Chloride Fertilizer',
    'Potato___Leaf_mold': 'Urea Fertilizer',
    'Potato___Septoria_leaf_spot': 'Calcium Ammonium Nitrate Fertilizer',
    'Potato___Spider_mites': 'Monoammonium Phosphate Fertilizer',
    'Potato___Target_spot': 'Ammonium Sulfate Fertilizer',
    'Potato___Yellow_leaf_curl_virus': 'Potassium Sulfate Fertilizer',
    'Raspberry___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Soybean___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Squash___Powdery_mildew': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Strawberry___healthy': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Strawberry___Leaf_scorch': 'Balanced N-P-K Fertilizer with Micronutrients',
    'Tomato___Bacterial_spot': 'Ammonium Nitrate Fertilizer',
    'Tomato___Early_blight': 'Triple Superphosphate Fertilizer',
    'Tomato___Late_blight': 'Calcium Nitrate Fertilizer',
    'Tomato___Leaf_mold': 'Potassium Sulfate Fertilizer',
    'Tomato___Septoria_leaf_spot': 'Potassium Nitrate Fertilizer',
    'Tomato___Spider_mites': 'Urea Fertilizer',
    'Tomato___Target_spot': 'Diammonium Phosphate Fertilizer',
    'Tomato___Yellow_leaf_curl_virus': 'Magnesium Sulfate Fertilizer',
}

    # Lookup the fertilizer based on the predicted disease
    return fertilizer_mapping.get(disease, 'Unknown fertilizer')

# Streamlit App
def main():
    st.title("Plant Disease Detection App")

    # Option to upload image
    uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"])

    # Load the validation set to access class names
    validation_set = tf.keras.utils.image_dataset_from_directory(
        'data',
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
    )
    class_names = validation_set.class_names

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Center the image using columns
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction and display result
        if st.button("Classify"):
            st.write("Classified Disease")
            prediction = predict_disease(image)
            class_name = class_names[prediction]
            st.success(f"Predicted Disease: {class_name}")

            # Predict fertilizer based on the disease
            fertilizer = predict_fertilizer(class_name)
            st.success(f"Suggested Fertilizer: {fertilizer}")

if __name__ == '__main__':
    main()
