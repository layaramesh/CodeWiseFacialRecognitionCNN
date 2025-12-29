# Facial Expression Recognition Using Convolutional Neural Networks

## Overview
This project implements a Convolutional Neural Network (CNN) for classifying human facial expressions.  
It was developed as part of an independent research study exploring the intersection of affective computing, accessibility, and real‑world educational applications.

The model is trained on a curated dataset of labeled facial images and optimized for accuracy, generalization, and deployment feasibility.

## Motivation
Many emotion‑recognition systems fail on real‑world data or underrepresented groups.  
This project investigates:
- How CNN architectures perform across diverse facial expressions  
- What preprocessing steps improve robustness  
- How lightweight models can be deployed in educational tools  

This work supports a research paper currently under revision for submission. And will be used in real-world to test feasibility.

## Features
- Custom CNN architecture implemented in Python  
- Data preprocessing pipeline (normalization, augmentation, resizing)  
- Training, validation, and testing scripts  
- Accuracy, loss, and confusion matrix visualizations  
- Modular code structure for experimentation  

## Dataset
- Source: Publicly available facial expression datasets  
- Classes: e.g., Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust  
- Preprocessing:  
  - Grayscale conversion  
  - Histogram equalization  
  - Augmentation (flip, rotation, shift)

## Results
- Best model accuracy (emotion-ferplus-8): 34.8%
- emotion model accuracy: 0%
- emotion_cnn model accuracy: 0%
- Evaluation dataset size: 23 classroom-like images

## Immediate Future Work
- Improve dataset diversity  
- Group emotions into distinct groups: positive (happy, neutral, surprise) and negative (sad, fear, anger, disgust) and check if improves accuracy
- Add Webcam module to get real world signal
- Deploy in a small pilot in real classroom to see it working

## How to Run
```powershell
# From repo root, build the project
cd <project root>
dotnet restore
dotnet build -c Release

# Run (provide an ONNX model and an input image). Optional: provide Haarcascade XML for face detection
dotnet run --project . --model .\Models\model.onnx --image .\Data\happy.jpg --cascade 
.\haarcascade_frontalface_default.xml
```