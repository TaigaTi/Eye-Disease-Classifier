import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Custom CSS for button styling ---
# We'll use Streamlit's built-in CSS variables for better theme compatibility.
# --primary-color: usually an accent color (e.g., red-orange)
# --primary-background: the main app background color (e.g., light gray, dark blue)
# --secondary-background: often used for sidebar or secondary elements (slightly different shade than primary)
# --text-color: main text color
# --font-family: main font

custom_button_css = """
<style>
/* Target all Streamlit buttons */
div.stButton > button {
    /* Set default background to Streamlit's primary accent color */
    background-color: var(--primary-color);
    /* Set text color to white for clear visibility against the primary color */
    color: white; 
    border-radius: 0.5rem; /* Standard Streamlit border radius */
    font-weight: bold; /* Make text bolder */
    /* Ensure the button fills the column width */
    width: 100%;
    /* Smooth transition for visual feedback */
    transition: background-color 0.2s, color 0.2s, border-color 0.2s;
}

/* Style on hover */
div.stButton > button:hover {
    /* On hover, background becomes orange as requested */
    background-color: orange;
    /* Text color changes to black on hover for strong contrast */
    color: black; 
    border: 1px solid orange; /* Match border to new background */
}

/* Adjust image size in columns to ensure they look good with buttons */
div.stImage > img {
    object-fit: cover; /* or 'contain' depending on desired cropping */
    width: 100%;
    height: auto; /* Maintain aspect ratio */
    border-radius: 0.5rem; /* Optional: match button corners */
    margin-bottom: 5px; /* Add a small margin between image and button */
}
</style>
"""
st.markdown(custom_button_css, unsafe_allow_html=True)


# --- Page Configuration ---
st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize session state for selected sample image.
if 'selected_sample_image_path' not in st.session_state:
    st.session_state.selected_sample_image_path = None

# Placeholder for initial loading message
loading_message_placeholder = st.empty()
loading_message_placeholder.info("🚀 Setting up the classifier... Please wait.")


# Load model
@st.cache_resource # Cache the model to avoid reloading on each interaction
def load_model():
    # Make sure the path to your model is correct relative to where you run the Streamlit app
    model = tf.keras.models.load_model("models/eye_classifier_model.keras")
    return model

model = load_model()

# Once the model is loaded, clear the loading message
loading_message_placeholder.empty()

# Your class names
class_names = [
    "Cataract",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Normal",
]

IMG_SIZE = 256

# --- Header Section ---
st.title("👁️ Eye Disease Classifier")
st.markdown("""
    Welcome to the **Eye Disease Classifier**! This tool helps to identify potential eye conditions
    from retinal images using a powerful AI model.
    Simply upload an image, or select a sample, and our system will provide a prediction.
""")
st.info("💡 **Disclaimer**: This tool is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult with a qualified healthcare provider for any health concerns.")

st.write("---")

# --- Refactor Prediction Logic into a Function ---
def process_and_predict_image(image_to_analyze):
    """
    Preprocesses an image and runs a prediction using the loaded model,
    then displays the results.
    """
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

# --- Image Upload Section ---
st.subheader("Upload Retinal Image")
uploaded_file = st.file_uploader(
    "Choose an image file (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of the retina for the best prediction."
)

st.write("---")

# --- Sample Images Section ---
st.subheader("Or Try with Sample Images")

sample_image_dir = "samples"

# Create the 'samples' directory if it doesn't exist and provide guidance
if not os.path.exists(sample_image_dir):
    os.makedirs(sample_image_dir, exist_ok=True)
    st.warning(f"The '{sample_image_dir}' folder was just created. Please add some sample images (JPG, JPEG, PNG) into it to use this feature.")

# Get list of sample images. Filter out non-image files.
all_sample_image_files = [f for f in os.listdir(sample_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# Limit to a maximum of 6 images for display in a row
display_sample_image_files = sorted(all_sample_image_files)[:6]

if not display_sample_image_files:
    st.info("No sample images found in the 'samples' folder. Please add some to test this feature.")
else:
    # Create columns for sample images. The number of columns will match the number of images to display.
    cols = st.columns(len(display_sample_image_files))

    for i, sample_file in enumerate(display_sample_image_files):
        sample_image_path = os.path.join(sample_image_dir, sample_file)
        with cols[i]: # Place content in the current column
            try:
                sample_image = Image.open(sample_image_path).convert("RGB")
                # Display a smaller preview, allowing the button to take more space
                st.image(sample_image, use_container_width=True)

                # Create a button for each image. Unique key is essential for Streamlit.
                # Button label changed to "Sample X"
                # use_container_width=True makes the button fill the column
                if st.button(f"Sample {i+1}", key=f"sample_btn_{sample_file}", use_container_width=True):
                    # Set the session state variable and force a rerun
                    st.session_state.selected_sample_image_path = sample_image_path
                    st.session_state.uploaded_file = None # Ensure uploaded file is cleared
                    st.rerun() # Rerun the app to process the newly selected sample image
            except FileNotFoundError:
                st.error(f"Sample image '{sample_file}' not found at '{sample_image_path}'.")
            except Exception as e:
                st.error(f"Error loading sample image '{sample_file}': {e}")


# Create two columns for the main image display and results
col1, col2 = st.columns(2)

image_to_display_and_process = None
image_caption = ""

# --- Determine which image to process ---
# Priority: 1. Newly uploaded file, 2. Previously selected sample image
if uploaded_file is not None:
    image_to_display_and_process = Image.open(uploaded_file).convert("RGB")
    image_caption = "Uploaded Image"
    # If a new file is uploaded, clear any previously selected sample
    st.session_state.selected_sample_image_path = None
elif st.session_state.selected_sample_image_path:
    # If a sample image was selected via button click, load it
    if os.path.exists(st.session_state.selected_sample_image_path):
        try:
            image_to_display_and_process = Image.open(st.session_state.selected_sample_image_path).convert("RGB")
            image_caption = f"Sample Image: {os.path.basename(st.session_state.selected_sample_image_path)}"
        except Exception as e:
            st.error(f"Error loading previously selected sample image: {e}")
            st.session_state.selected_sample_image_path = None # Clear state on error
    else:
        st.warning(f"Previously selected sample image not found at '{st.session_state.selected_sample_image_path}'. It might have been moved or deleted.")
        st.session_state.selected_sample_image_path = None # Clear state if file is gone

# --- Display and Process the selected/uploaded image ---
if image_to_display_and_process is not None:
    with col1:
        st.image(image_to_display_and_process, caption=image_caption, use_container_width=True)
    with col2:
        # Call the refactored function to process and predict
        process_and_predict_image(image_to_display_and_process)
else:
    st.info("👆 Please upload an image or select a sample image to get a prediction.")

st.write("---")

# --- Footer ---
st.markdown("""
    <div style="text-align: center; color: gray;">
        Developed with ❤️ using Streamlit and TensorFlow.
    </div>
""", unsafe_allow_html=True)