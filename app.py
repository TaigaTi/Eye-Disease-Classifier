import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="auto"
)

# Placeholder for initial loading message
loading_message_placeholder = st.empty()
loading_message_placeholder.info("🚀 Setting up the classifier... Please wait.")


# Load model
@st.cache_resource # Cache the model to avoid reloading on each interaction
def load_model():
    model = tf.keras.models.load_model("models/eye_classifier_model.keras")
    return model

model = load_model()

# Once the model is loaded, clear the loading message
loading_message_placeholder.empty()

# Your class names (ensure this matches your model!)
class_names = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Normal"  # Example: Update with your actual disease classes
]

IMG_SIZE = 256

# --- Header Section ---
st.title("👁️ Eye Disease Classifier")
st.markdown("""
    Welcome to the **Eye Disease Classifier**! This tool helps to identify potential eye conditions
    from retinal images using a powerful AI model.
    Simply upload an image, and our system will provide a prediction.
""")
st.info("💡 **Disclaimer**: This tool is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns.")

st.write("---")

# --- Image Upload Section ---
st.subheader("Upload Retinal Image")
uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of the retina for the best prediction."
)

col1, col2 = st.columns(2) # Create two columns for layout

if uploaded_file is not None:
    # Display the uploaded image
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    with col2:
        st.subheader("Analysis Results")
        with st.spinner("Analyzing image..."):
            img = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Preprocessing should match your model's training
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

            prediction = model.predict(img_array)
            pred_class_index = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100 # Get confidence score

            st.markdown(f"**Predicted Condition:** <span style='font-size: 24px; color: #4CAF50;'>**{class_names[pred_class_index]}**</span>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {confidence:.2f}%")

            st.markdown("---")
            st.subheader("Understanding the Prediction")
            st.write(f"""
                Based on the uploaded image, the model predicts the condition to be **{class_names[pred_class_index]}** with a confidence level of **{confidence:.2f}%**.
                For accurate diagnosis, please consult with a medical professional.
            """)

            # Display all class probabilities
            st.markdown("---")
            st.subheader("All Class Probabilities")
            for i, class_name in enumerate(class_names):
                st.write(f"- **{class_name}**: {prediction[0][i]*100:.2f}%")

else:
    st.info("👆 Please upload an image to get a prediction.")

st.write("---")

# --- Footer ---
st.markdown("""
    <div style="text-align: center; color: gray;">
        Developed with ❤️ using Streamlit and TensorFlow.
    </div>
""", unsafe_allow_html=True)