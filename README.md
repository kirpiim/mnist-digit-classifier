# MNIST Handwritten Digit Recognizer (CNN)

A Convolutional Neural Network trained on the MNIST dataset (handwritten digits 0–9).  
Achieved ~99% test accuracy.

## Overview
- **Dataset:** MNIST (60,000 train / 10,000 test images)
- **Framework:** TensorFlow / Keras
- **Model:** Convolutional Neural Network (Conv2D → ReLU → MaxPool → Dense)
- **Outputs:**
  - `mnist_cnn.h5` → trained model (ignored in repo, must be trained locally)
  - `docs/training_curves.png` → accuracy & loss visualization

## Training Curves
Below are the training and validation accuracy & loss curves produced during training:

![Training Curves](docs/training_curves.png)

## Setup & Usage

### 1. Clone the repository
```bash
  git clone https://github.com/kirpiim/mnist-cnn-pytorch.git
  cd mnist-cnn-pytorch
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```


### 3. Train the model
```bash
python -m src.train
```
### 4. Evaluate the model
```bash
python mnist_test.py
```
### 5. Predict a single image
```bash
python predict.py path/to/image.png
```
### 6. Export the model

# Export to TensorFlow Lite
```bash
python export_tflite.py
```
# Export to ONNX
```bash
python export_onnx.py
```

## Results
- Final test accuracy: ~99%  
- Training/validation curves saved in `docs/training_curves.png`

**Example prediction:**

python predict.py samples/7.png
# Predicted: 7

## Project Structure

```text
.
├── src/
│   ├── models/
│   │   └── cnn.py          # CNN model definition
│   ├── train.py            # Training script
│   ├── predict.py          # CLI predictor
│   ├── export_tflite.py    # Export model to TFLite
│   ├── export_onnx.py      # Export model to ONNX
├── mnist_test.py           # Evaluation script
├── requirements.txt
├── docs/
│   └── training_curves.png # Training visualization
└── README.md
```

## Extra Features
CLI Predictor: Classify digits from PNG/JPG
Deployment Ready: Export to ONNX / TensorFlow Lite for mobile & edge devices
