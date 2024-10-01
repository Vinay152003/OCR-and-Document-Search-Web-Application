<h3><a href="">OCR and Document Search Web Application Using General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model</a></h3>
<a href="https://huggingface.co/spaces/Vinay15/OCR"><img src="https://img.shields.io/badge/Huggingface-yellow"></a>

# Live URL
You can access the Live Web Application at My Hugging Face Space: [https://huggingface.co/spaces/Vinay15/OCR](https://huggingface.co/spaces/Vinay15/OCR)
[Note: The Hugging Face space may be in a sleeping state due to inactivity. Please click the link to restart the space]

This repository contains the code for a web-based application that performs Optical Character Recognition (OCR) on uploaded images containing text in both Hindi and English. The application also implements a basic keyword search functionality based on the extracted text.

- English Image Output Text 1
![OCR1](https://github.com/user-attachments/assets/4ac16c5c-e2bb-43b6-b5e2-26060f49eeb7)
- Chinese and English Image Output Text 2
![Screenshot 2024-10-01 202502](https://github.com/user-attachments/assets/8f83a2f9-503b-4510-b2c6-4febeb3b33fe)
- Chinese Image Output Text 3
![Screenshot 2024-10-01 202609](https://github.com/user-attachments/assets/a6c9e892-1131-4314-acbb-c36ca90e2023)

## Table of Contents
- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [Install](#Install)
- [OCR Model Integration](#ocr-model-integration)
- [Web Application Development](#web-application-development)
- [Deployment](#deployment)
- [Live URL](#Live-URL)
- [Usage](#usage)
- [License](#license)

## Introduction

The goal of this project is to create a simple web application that allows users to upload a single image, process the image to extract text using GOT-OCR2.0.

## Environment Setup

### Requirements

- Python >= 3.8
- Necessary Libraries:
  - `torch`
  - `torchvision`
  - `transformers`
  - `tiktoken`
  - `verovio`
  - `accelerate`
  - `gradio`

## Install
0. Our environment is cuda11.8+torch2.0.1
1. Clone this repository and navigate to the GOT folder
```bash
git clone https://github.com/Ucas-HaoranWei/GOT-OCR2.0.git
cd 'the GOT folder'
```
2. Install Package
```Shell
conda create -n got python=3.10 -y
conda activate got
pip install -e .
```

3. Install Flash-Attention
```
pip install ninja
pip install flash-attn --no-build-isolation
```
## GOT Weights
- [Huggingface](https://huggingface.co/ucaslcl/GOT-OCR2_0)
- [Google Drive](https://drive.google.com/drive/folders/1OdDtsJ8bFJYlNUzCQG4hRkUL6V-qBQaN?usp=sharing)
- [BaiduYun](https://pan.baidu.com/s/1G4aArpCOt6I_trHv_1SE2g) code: OCR2

## OCR Model Integration

In this project, we have implemented the OCR model using **General OCR Theory (GOT)**. The selected model is integrated to extract text from the uploaded images.

### Model Loading
```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()


# input your test image
image_file = 'xxx.jpg'

# plain texts OCR
res = model.chat(tokenizer, image_file, ocr_type='ocr')

# format texts OCR:
# res = model.chat(tokenizer, image_file, ocr_type='format')

# fine-grained OCR:
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

# multi-crop OCR:
# res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
# res = model.chat_crop(tokenizer, image_file, ocr_type='format')

# render the formatted OCR results:
# res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

print(res)

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
2. Upload an image file (JPEG, PNG, etc.) containing text in chinese and English.
3. The extracted text will be displayed on the page.
4. Enter a keyword to search within the extracted text and view the results.


## License

This project is licensed under the MIT License. See the following for details:

```
MIT License

Copyright (c) 2024 Vinay15

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```"

