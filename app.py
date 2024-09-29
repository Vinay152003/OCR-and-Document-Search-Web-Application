import torch
import streamlit as st
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os

# Check if a GPU is available and load the model accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'ucaslcl/GOT-OCR2_0', 
    trust_remote_code=True, 
    low_cpu_mem_usage=True, 
    device_map='auto', 
    use_safetensors=True, 
    pad_token_id=tokenizer.eos_token_id
)
model = model.eval().to(device)

# Define the OCR function
def perform_ocr(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save the image to a temporary path
    image_file_path = 'temp_image.jpg'
    image.save(image_file_path)

    # Perform OCR using the model
    res = model.chat(tokenizer, image_file_path, ocr_type='ocr')

    # Remove the temporary image file after processing
    os.remove(image_file_path)

    return res

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
            st.success("Text extracted successfully!")
            st.text_area("Extracted Text", extracted_text, height=300)
