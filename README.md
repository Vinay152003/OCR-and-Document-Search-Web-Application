# OCR Web Application Prototype

This repository contains the code for a web-based prototype that performs Optical Character Recognition (OCR) on uploaded images containing text in both Hindi and English. The application also implements a basic keyword search functionality based on the extracted text.

## Table of Contents
- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [OCR Model Integration](#ocr-model-integration)
- [Web Application Development](#web-application-development)
- [Deployment](#deployment)
- [Usage](#usage)
- [License](#license)

## Introduction

The goal of this project is to create a simple web application that allows users to upload a single image, process the image to extract text using OCR, and provide a basic search feature for the extracted text.

## Environment Setup

### Requirements

- Python >= 3.8
- Necessary Libraries:
  - `torch`
  - `transformers`
  - `gradio`

### Installation Steps

1. **Clone this repository**:
   ```bash
   git clone https://github.com/Vinay152003/OCR-and-Document-Search-Web-Application.git
   cd OCR-and-Document-Search-Web-Application
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required libraries**:
   ```bash
   pip install torch torchvision transformers gradio
   ```

## OCR Model Integration

In this project, we have implemented the OCR model using **General OCR Theory (GOT)**. The selected model is integrated to extract text from the uploaded images.

### Model Loading Example
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("your-model-name")
model = AutoModelForTokenClassification.from_pretrained("your-model-name")
```

## Web Application Development

The web application is developed using [Gradio](https://gradio.app/).

### Application Features
- Users can upload an image file for OCR processing.
- The application displays the extracted text from the image.
- Users can enter keywords to search within the extracted text, with results displayed on the same page.

### Code Example
```python
import gradio as gr

def extract_text(image):
    # Your OCR code here
    return extracted_text

interface = gr.Interface(fn=extract_text, inputs="image", outputs="text")
interface.launch()
```

## Deployment

The web application is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/Vinay15/OCR).

### Live URL
You can access the live web application at: [https://huggingface.co/spaces/Vinay15/OCR](https://huggingface.co/spaces/Vinay15/OCR)

## Usage

1. Open the web application in your browser.
2. Upload an image file (JPEG, PNG, etc.) containing text in Hindi and English.
3. The extracted text will be displayed on the page.
4. Enter a keyword to search within the extracted text and view the results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
