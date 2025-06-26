import streamlit as st
import tensorflow as tf
import openpyxl
import numpy as np
from PIL import Image
import os
import io

st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Fixed-position scroll prompt with transition and mobile-friendly padding ---
if st.session_state.get("show_scroll_prompt", False):
    st.markdown("""
        <style>
        #scroll-prompt-banner {
            display: none;
        }
        #scroll-prompt-banner.hide {
            opacity: 0;
            pointer-events: none;
        }
        @media (max-width: 600px) {
            #scroll-prompt-banner {
                display: block;
                padding-top: 2.5em;
                position: fixed;
                top: 0; left: 0; right: 0; z-index: 1000;
                background: #002244;
                color: #fff;
                font-size: 1.3rem;
                text-align: center;
                padding: 2.5em 0 0.7em 0;
                box-shadow: 0 2px 6px rgba(0,0,0,0.12);
                opacity: 1;
                transition: opacity 1s ease;
            }
        }
        </style>
        <div id="scroll-prompt-banner">
            ⬇️ Scroll down to see the analysis results!
        </div>
        <div style="height: 3.1em;"></div>
        <script>
        setTimeout(function() {
            var banner = window.parent.document.getElementById("scroll-prompt-banner");
            if(banner){ banner.classList.add("hide"); }
        }, 3000);
        </script>
        """, unsafe_allow_html=True)

# --- Session state initialization ---
if "selected_image_bytes" not in st.session_state:
    st.session_state.selected_image_bytes = None
if "selected_image_caption" not in st.session_state:
    st.session_state.selected_image_caption = None
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0
if "scroll_to_results" not in st.session_state:
    st.session_state.scroll_to_results = False
if "show_scroll_prompt" not in st.session_state:
    st.session_state.show_scroll_prompt = False

# --- Model loading ---
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
    st.subheader("Analysis Results", anchor="results")
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
    help="Upload a clear image of the retina for the best prediction.",
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)
st.write("---")

if uploaded_file is not None:
    st.session_state.selected_image_bytes = uploaded_file.read()
    st.session_state.selected_image_caption = "Uploaded Image"
    st.session_state.scroll_to_results = True
    st.session_state.show_scroll_prompt = True

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
                    with open(sample_image_path, "rb") as f:
                        st.session_state.selected_image_bytes = f.read()
                    st.session_state.selected_image_caption = f"Sample Image: {os.path.basename(sample_image_path)}"
                    st.session_state.file_uploader_key += 1 
                    st.session_state.scroll_to_results = True
                    st.session_state.show_scroll_prompt = True
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading sample image '{sample_file}': {e}")

col1, col2 = st.columns(2)
image_to_display_and_process = None
image_caption = st.session_state.selected_image_caption

if st.session_state.selected_image_bytes is not None:
    try:
        image_to_display_and_process = Image.open(
            io.BytesIO(st.session_state.selected_image_bytes)
        ).convert("RGB")
    except Exception as e:
        st.error(f"Error loading selected image: {e}")
        st.session_state.selected_image_bytes = None
        st.session_state.selected_image_caption = None

if image_to_display_and_process is not None:
    if st.session_state.scroll_to_results:
        st.session_state.scroll_to_results = False
        st.markdown("""
            <script>
            window.location.hash = "results";
            </script>
            """, unsafe_allow_html=True)
    with col1:
        st.image(image_to_display_and_process, caption=image_caption, use_container_width=True)
    with col2:
        process_and_predict_image(image_to_display_and_process)
    st.session_state.show_scroll_prompt = False
else:
    st.info("👆 Please upload an image or select a sample image to get a prediction.")

st.write("---")
st.subheader("Model Performance Summary")

results_file = "eye_classification_results.xlsx"

def try_format(val, stat_name=None):
    if stat_name and "epoch" in stat_name.lower():
        try:
            return str(int(round(float(val))))
        except Exception:
            return str(val)
    try:
        f = float(val)
        return f"{f:.3f}"
    except Exception:
        return str(val)

if os.path.exists(results_file):
    wb = openpyxl.load_workbook(results_file)
    ws = wb.active
    stats = []
    for row in ws.iter_rows(min_row=2, max_col=3, values_only=True):
        if not any(row):
            break
        stat_name, _, best_val = row
        if stat_name is not None and best_val is not None:
            stats.append((stat_name, try_format(best_val, stat_name)))
    confusion_matrix = stats.pop()[1] if stats else None
    if stats:
        for stat_name, best_val in stats:
            st.markdown(f"- **{stat_name}**: `{best_val}`")
    else:
        st.info("No summary statistics found in results file.")
    if confusion_matrix:
        cm_path = confusion_matrix
        if not os.path.isabs(cm_path) and not cm_path.startswith("confusion_matrices/"):
            cm_path = os.path.join("confusion_matrices", cm_path)
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix (Best Model)", use_container_width=True)
else:
    st.info("No model results file (`eye_classification_results.xlsx`) found. Please train and evaluate your model first.")

st.write("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
        Developed with ❤️ using Streamlit and TensorFlow.
    </div>
""", unsafe_allow_html=True)