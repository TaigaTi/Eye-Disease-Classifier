import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Custom CSS for button and image styling ---
st.markdown("""
<style>
div.stButton > button {
    background-color: var(--primary-color);
    color: white; 
    border-radius: 0.5rem;
    font-weight: bold;
    width: 100%;
    transition: background-color 0.2s, color 0.2s, border-color 0.2s;
}
div.stButton > button:hover {
    background-color: orange;
    color: black; 
    border: 1px solid orange;
}
div.stImage > img {
    object-fit: cover;
    width: 100%;
    height: auto;
    border-radius: 0.5rem;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="auto"
)

if 'selected_sample_image_path' not in st.session_state:
    st.session_state.selected_sample_image_path = None

loading_message_placeholder = st.empty()
loading_message_placeholder.info("🚀 Setting up the classifier... Please wait.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/eye_classifier_model.keras")

model = load_model()
loading_message_placeholder.empty()

class_names = [
    "Cataract",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Normal",
]
IMG_SIZE = 256

st.title("👁️ Eye Disease Classifier")
st.markdown("""
    Welcome to the **Eye Disease Classifier**! This tool helps to identify potential eye conditions
    from retinal images using a powerful AI model.
    Simply upload an image, or select a sample, and our system will provide a prediction.
""")
st.info("💡 **Disclaimer**: This tool is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns.")
st.write("---")

def process_and_predict_image(image_to_analyze):
    st.subheader("Analysis Results")
    with st.spinner("Analyzing image..."):
        img = image_to_analyze.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        prediction = model.predict(img_array)
        pred_class_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        st.markdown(f"**Predicted Condition:** <span style='font-size: 24px; color: #4CAF50;'>**{class_names[pred_class_index]}**</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.markdown("---")
        st.subheader("Understanding the Prediction")
        st.write(f"""
            Based on the provided image, the model predicts the condition to be **{class_names[pred_class_index]}** with a confidence level of **{confidence:.2f}%**.
            For accurate diagnosis, please consult with a medical professional.
        """)
        st.markdown("---")
        st.subheader("All Class Probabilities")
        for i, class_name in enumerate(class_names):
            st.write(f"- **{class_name}**: {prediction[0][i]*100:.2f}%")

st.subheader("Upload Retinal Image")
uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of the retina for the best prediction."
)
st.write("---")

st.subheader("Or Try with Sample Images")
sample_image_dir = "samples"

if not os.path.exists(sample_image_dir):
    os.makedirs(sample_image_dir, exist_ok=True)
    st.warning(f"The '{sample_image_dir}' folder was just created. Please add some sample images (JPG, JPEG, PNG) into it to use this feature.")

all_sample_image_files = [f for f in os.listdir(sample_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
display_sample_image_files = sorted(all_sample_image_files)[:6]

if not display_sample_image_files:
    st.info("No sample images found in the 'samples' folder. Please add some to test this feature.")
else:
    cols = st.columns(len(display_sample_image_files))
    for i, sample_file in enumerate(display_sample_image_files):
        sample_image_path = os.path.join(sample_image_dir, sample_file)
        with cols[i]:
            try:
                sample_image = Image.open(sample_image_path).convert("RGB")
                st.image(sample_image, use_container_width=True)
                if st.button(f"Sample {i+1}", key=f"sample_btn_{sample_file}", use_container_width=True):
                    st.session_state.selected_sample_image_path = sample_image_path
                    st.session_state.uploaded_file = None
                    st.rerun()
            except FileNotFoundError:
                st.error(f"Sample image '{sample_file}' not found at '{sample_image_path}'.")
            except Exception as e:
                st.error(f"Error loading sample image '{sample_file}': {e}")

col1, col2 = st.columns(2)
image_to_display_and_process = None
image_caption = ""

if uploaded_file is not None:
    image_to_display_and_process = Image.open(uploaded_file).convert("RGB")
    image_caption = "Uploaded Image"
    st.session_state.selected_sample_image_path = None
elif st.session_state.selected_sample_image_path:
    if os.path.exists(st.session_state.selected_sample_image_path):
        try:
            image_to_display_and_process = Image.open(st.session_state.selected_sample_image_path).convert("RGB")
            image_caption = f"Sample Image: {os.path.basename(st.session_state.selected_sample_image_path)}"
        except Exception as e:
            st.error(f"Error loading previously selected sample image: {e}")
            st.session_state.selected_sample_image_path = None
    else:
        st.warning(f"Previously selected sample image not found at '{st.session_state.selected_sample_image_path}'. It might have been moved or deleted.")
        st.session_state.selected_sample_image_path = None

if image_to_display_and_process is not None:
    with col1:
        st.image(image_to_display_and_process, caption=image_caption, use_container_width=True)
    with col2:
        process_and_predict_image(image_to_display_and_process)
else:
    st.info("👆 Please upload an image or select a sample image to get a prediction.")

st.write("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
        Developed with ❤️ using Streamlit and TensorFlow.
    </div>
""", unsafe_allow_html=True)