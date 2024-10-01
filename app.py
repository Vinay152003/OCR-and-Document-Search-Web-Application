import gradio as gr
from transformers import AutoModel, AutoTokenizer
from PIL import Image

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

# Define the OCR function
def perform_ocr(image):
    # Convert PIL image to RGB format (if necessary)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save the image to a temporary path
    image_file_path = 'temp_image.jpg'
    image.save(image_file_path)

    # Perform OCR using the model
    res = model.chat(tokenizer, image_file_path, ocr_type='ocr')

    return res

# Define the search function
def search_keyword(extracted_text, keyword):
    # Check if keyword is provided
    if not keyword.strip():
        return "Please enter a keyword."

    # Search for the keyword in the extracted text
    if keyword.lower() in extracted_text.lower():
        return f"Keyword '{keyword}' found in the extracted text!"
    else:
        return f"Keyword '{keyword}' not found in the extracted text."

# Define the interface with both OCR and keyword search functionality
def ocr_and_search(image, keyword):
    # Perform OCR to extract text from the image
    extracted_text = perform_ocr(image)

    # Perform keyword search within the extracted text
    search_result = search_keyword(extracted_text, keyword)

    # Return both the extracted text and the search result
    return extracted_text, search_result

# Define the Gradio interface
interface = gr.Interface(
    fn=ocr_and_search,
    inputs=[gr.Image(type="pil", label="Upload Image"), gr.Textbox(label="Enter Keyword to Search")],
    outputs=[gr.Textbox(label="Extracted Text"), gr.Textbox(label="Search Result")],
    title="OCR and Document Search Web Application",
    description="Upload an image to extract text using the GOT-OCR2_0 model and search for a keyword within the extracted text."
)

# Launch the Gradio app
interface.launch()
