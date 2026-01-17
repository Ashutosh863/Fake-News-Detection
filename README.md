Multimodal Fake News Detection System (From Scratch):

This project is a from-scratch multimodal fake news detection system that uses:
a custom-built Transformer encoder for text
a custom-built CNN for image processing
a fusion network to combine both modalities

The project demonstrates end-to-end ML system design, including model architecture, inference pipeline, debugging, and API deployment.

Project Overview

Fake news often spreads using both misleading text and images.
This project tackles the problem by combining textual and visual information to make predictions.

The model predicts whether a news article is:

FAKE
REAL

Architecture

Text and image inputs are processed separately and then fused:

Text → Transformer Encoder
Image → CNN Encoder

Both embeddings are concatenated and passed through a classifier.

Project Structure
Fake News detection/
│
├── app.py                  # FastAPI inference server
├── test_model.py           # Local model testing (no API)
├── requirements.txt
│
├── model/
│   ├── __init__.py
│   ├── transformer.py      # Transformer built from scratch
│   ├── cnn.py              # CNN built from scratch
│   └── fusion.py           # Multimodal fusion model
│
├── utils/
│   ├── __init__.py
│   └── image_utils.py      # Image preprocessing
│
└── test.jpg                # Sample image for testing

Installation
Create virtual environment
python -m venv .venv

Activate virtual environment (Windows)
.venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Testing the Model (Without API)

This project supports direct model testing in VS Code / terminal.

Step 1: Add an image
Place any image in the project root and name it:
test.jpg

Step 2: Run test script
python test_model.py

Sample Output
LOGITS: tensor([[ 0.02, -0.01]])
PROBABILITIES: tensor([[0.51, 0.49]])
PREDICTION: FAKE


Note:
The model is not trained, so predictions are random.
This project focuses on architecture and system design, not accuracy.

Running FastAPI Server
Start the server
python -m uvicorn app:app --reload

Open API documentation
http://127.0.0.1:8000/docs


Use the /predict endpoint to send:
news text
image file
Model Training
Training is not included in the API by design.

Training should be done separately using:
train.py

Training requires:
labeled dataset
loss function
optimizer
backpropagation
saving trained weights
The API is used only for inference.

Technologies Used:
Python
PyTorch
FastAPI
TorchVision
NumPy
PIL

Learning Outcomes:
Implemented Transformer from scratch
Implemented CNN from scratch
Built multimodal fusion architecture
Understood training vs inference separation
Debugged real-world Python import issues

Built production-style ML project structure

Deployed inference using FastAPI
