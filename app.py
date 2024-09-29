import streamlit as st
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os
import torch

# Cache the model and tokenizer to optimize resource usage
@st.cache_resource
def load_model():
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        
        # Load the model on CPU
        model = AutoModel.from_pretrained(
            'ucaslcl/GOT-OCR2_0',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='cpu',  # Force CPU usage
            use_safetensors=True,
            pad_token_id=tokenizer.eos_token_id
        )
        model = model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model and tokenizer
tokenizer, model = load_model()

# Define the OCR function
def perform_ocr(image):
    try:
        # Convert PIL image to RGB format if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save the image to a temporary path
        image_file_path = 'temp_image.jpg'
        image.save(image_file_path)

        # Perform OCR using the model
        res = model.chat(tokenizer, image_file_path, ocr_type='ocr')  # Ensure this method exists and works as expected

        # Remove the temporary image file after processing
        if os.path.exists(image_file_path):
            os.remove(image_file_path)

        return res

    except Exception as e:
        st.error(f"Error during OCR processing: {e}")
        return None

# Streamlit UI
st.title("OCR and Document Search Web Application")
st.write("Upload an image to extract text using the GOT-OCR2_0 model.")

# File uploader for image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform OCR and display results
    if st.button("Extract Text"):
        with st.spinner("Processing..."):
            extracted_text = perform_ocr(image)
            if extracted_text:
                st.success("Text extracted successfully!")
                st.text_area("Extracted Text", extracted_text, height=300)
            else:
                st.error("Failed to extract text from the image.")
