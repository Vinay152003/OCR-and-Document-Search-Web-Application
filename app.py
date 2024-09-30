import gradio as gr
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torch  # Importing torch to check CUDA availability

# Check CUDA availability
def check_cuda():
    if torch.cuda.is_available():
        device_info = f"CUDA is available. GPU device: {torch.cuda.get_device_name(0)}"
    else:
        device_info = "CUDA is not available. Running on CPU."
    return device_info

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, device_map="auto", use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval()  # No need for .cuda() with device_map="auto"

# Define the OCR function
def perform_ocr(image):
    # Check for CUDA availability and print the result
    cuda_info = check_cuda()
    print(cuda_info)  # This will be logged in the output

    # Convert PIL image to RGB format (if necessary)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save the image to a temporary path
    image_file_path = 'temp_image.jpg'
    image.save(image_file_path)

    # Perform OCR using the model
    res = model.chat(tokenizer, image_file_path, ocr_type='ocr')

    return res

# Define the Gradio interface
interface = gr.Interface(
    fn=perform_ocr,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Extracted Text"),
    title="OCR and Document Search Web Application",
    description="Upload an image to extract text using the GOT-OCR2_0 model."
)

# Launch the Gradio app
interface.launch()
