# Facial Expression Recognition System

A complete pipeline for facial expression recognition using ONNX models, from image collection to model evaluation.

## Research paper and Motivation

https://drive.google.com/drive/folders/1dIsPLGT7txuxNzNkvnmNJDLxU-91quDM

This research explores whether CNN-based FER can be applied meaningfully within CodeWise’s online
learning environment. Many CodeWise instructors are high school students with limited teaching
experience, and younger learners oŌen hesitate to ask for help. A system capable of detecting confusion
or disengagement in real-time could support instructors in adjusting their teaching and providing timely
intervention.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       DATA COLLECTION PHASE                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
         ┌──────────▼──────────┐       ┌───────────▼──────────┐
         │  Pexels API         │       │  Bing Images API     │
         │  (Free)             │       │  (Alternative)       │
         └──────────┬──────────┘       └───────────┬──────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │ download_free_images.py│
                         │ - Fetch by emotion     │
                         │ - Organize in folders  │
                         └──────────┬─────────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │   cleanup_images.py    │
                         │ - Quality control      │
                         │ - Remove low quality   │
                         └──────────┬─────────────┘
                                    │
                                    ▼
                         [ Images/ Folder ]
                         (7 emotion folders)
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                       INFERENCE PHASE                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
         ┌──────────▼──────────┐       ┌───────────▼──────────┐
         │  Python Batch       │       │  C# Single Image     │
         │  (batch_predict.py) │       │  (Program.cs)        │
         └──────────┬──────────┘       └──────────────────────┘
                    │
                    │  ┌──────────────────────────┐
                    ├─▶│ emotion_cnn.onnx        │
                    │  │ (17.53% accuracy)       │
                    │  └──────────────────────────┘
                    │
                    │  ┌──────────────────────────┐
                    ├─▶│ emotion-ferplus-8.onnx  │
                    │  │ (37.64% accuracy)       │
                    │  └──────────────────────────┘
                    │
                    │  ┌──────────────────────────┐
                    └─▶│ emotion.onnx            │
                       │ (13.79% accuracy)       │
                       └──────────────────────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │     output.csv         │
                         │ - Raw predictions      │
                         │ - All 3 models         │
                         └──────────┬─────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION PHASE                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │ add_classification.py  │
                         │ - Apply Hybrid Logic   │
                         │ - Add labels           │
                         └──────────┬─────────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │ HYBRID MODEL LOGIC     │
                         │                        │
                         │ Needs Help emotions:   │
                         │ Use emotion_cnn        │
                         │ (Fear, Anger, Surprise,│
                         │  Disgust, Sadness)     │
                         │                        │
                         │ Understands Material:  │
                         │ Use emotion_ferplus    │
                         │ (Happiness, Neutral)   │
                         └──────────┬─────────────┘
                                    │
                  ┌─────────────────▼─────────────────┐
                  │ output_with_classification.csv    │
                  │ - Binary labels                   │
                  │ - "Needs Help" vs "Understands"   │
                  └─────────────────┬─────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                     EVALUATION PHASE                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │confusion_matrix_per_   │
                         │model.py                │
                         │ - 7x7 matrices         │
                         │ - 2x2 custom labels    │
                         │ - All metrics          │
                         └──────────┬─────────────┘
                                    │
                         ┌──────────▼─────────────┐
                         │ confusion_matrix.csv   │
                         │                        │
                         │ TP, TN, FP, FN         │
                         │ Precision, Recall, F1  │
                         │ Per-emotion metrics    │
                         │                        │
                         │ RESULT: 92.24% accuracy│
                         └────────────────────────┘
```

### Pipeline Summary

1. **Data Collection**: Download emotion-labeled images from free APIs
2. **Quality Control**: Filter and validate images
3. **Inference**: Run 3 ONNX models on all images
4. **Hybrid Classification**: Combine models for educational needs assessment
5. **Evaluation**: Generate comprehensive confusion matrices and metrics

### Key Innovation: Hybrid Model Strategy

The hybrid model achieves **92.24% accuracy** by intelligently selecting which base model to use:
- **emotion_cnn.onnx**: Better at detecting "Needs Help" emotions
- **emotion-ferplus-8.onnx**: Better at detecting "Understanding Material" emotions
- **Note**: The 92.24% accuracy refers to the custom label classification task (“Needs Help” vs “Understands Material”), not 7‑class emotion recognition.

## Components

### C# ONNX Inference
- Uses `Microsoft.ML.OnnxRuntime` for ONNX inference
- Uses `OpenCvSharp4` for image I/O and face detection
- Supports 7 emotion labels: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### Python Analysis Pipeline
See [READMe_Metrics_Calculations.md](READMe_Metrics_Calculations.md) for the complete data collection and analysis workflow.

## Quick Start - C# Inference

```powershell
# Build the project
dotnet restore
dotnet build -c Release

# Run inference on a single image
dotnet run --project . --model .\Models\emotion.onnx --image .\Images\angry\angry_001.jpg
```

## Model Requirements

- Input: 48x48 grayscale image in NCHW format `[1,1,48,48]`
- Output: 7 emotion probabilities
- Format: ONNX

To convert Keras/TensorFlow models to ONNX, use `keras2onnx` or `tf2onnx`.

## Model Performance

Performance comparison on 348 test images across 7 emotion categories.

### Hybrid Model Logic

- **Needs Help emotions** (Fear, Anger, Surprise, Disgust, Sadness): Uses emotion_cnn predictions
- **Understands Material emotions** (Happiness, Neutral): Uses emotion_ferplus predictions

### Emotion-Level Accuracy (7 Classes)

| Model | Overall Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|------------------|-----------------|--------------|----------|
| emotion-ferplus-8.onnx | 37.64% | 59.87% | 37.49% | 34.50% |
| emotion_cnn.onnx | 17.53% | 20.03% | 17.43% | 10.19% |
| emotion.onnx | 13.79% | 9.03% | 13.72% | 5.41% |
| Hybrid Model | **92.24%** | **97.02%** | **91.94%** | **94.41%** |

Note: Running the model on a different device results in:
Accuracy- 0.9598
Precision (Needs Help)- 0.9835
Recall (Needs Help)- 0.9597
F1 Score- 0.9714


**Key Findings**:
- Hybrid model achieves **92.24% accuracy** for educational needs assessment (identifying students who need help)
- Data is still low to latch on to this 92.24% accuracy rate, more studies needed
- Best individual model: emotion-ferplus-8 (37.64% emotion-level accuracy)

**Metrics post manual cleanup of images**:
A visual inspection of all 350 images was done, and images of the wrong classification were re-classified. Most of the images in "Disgust" were not disgust. I attempted to download 200 more disgust images to see if I can get better disgust image count, but there weren't great results. Since disgust is not an emotion that generally appears in a classroom setting anyways, the disgust count can be 4 images.

| Model | Overall Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|------------------|-----------------|--------------|----------|
| emotion-ferplus-8.onnx | 37.32% | 51.91% | 29.96% | 26.76% |
| emotion_cnn.onnx | 18.48% | 15.81% | 15.03% | 06.69% |
| emotion.onnx | 15.58% | 08.49% | 13.26% | 05.35% |
| Hybrid Model | **96.38** | **96.99%** | **96.99** | **96.99%** |

** Hybrid model accuracy and other metrics are still giving good results, indicating that this approach is working across different dataset.**

## Troubleshooting

**Native runtime errors (OpenCvSharp)**

If you see "The type initializer for 'OpenCvSharp.Internal.NativeMethods' threw an exception":

1. Install [Microsoft Visual C++ Redistributable for Visual Studio 2015-2022 (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

2. Build with Windows runtime identifier:
   ```powershell
   dotnet build -c Release
   dotnet run --project . -c Release -r win-x64 --model .\Models\emotion.onnx --image .\Images\angry\angry_001.jpg
   ```

3. Ensure process architecture (x64) matches native binaries

## API Keys Setup

This project uses the Pexels API for image download in the Python pipeline.

1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Get a free API key from [Pexels API](https://www.pexels.com/api/)

3. Edit `.env` and add your API key

4. The `.env` file is in `.gitignore` and will not be committed

**Note**: Never commit `.env` to GitHub. Only commit `.env.example` with placeholder values.

## Project Structure

```
├── Program.cs                  # C# ONNX inference code
├── scripts/                    # Python analysis pipeline
│   ├── download_free_images.py
│   ├── cleanup_images.py
│   ├── batch_predict.py
│   ├── add_classification.py
│   ├── confusion_matrix_per_model.py
│   └── emotion_utils.py
├── Models/                     # ONNX models
├── Images/                     # Training/test images
├── Data/                       # Haarcascade files
├── output.csv                  # Raw model predictions
├── output_with_classification.csv  # Enriched predictions
└── confusion_matrix.csv        # Performance metrics
```

## Documentation

- [READMe_Metrics_Calculations.md](READMe_Metrics_Calculations.md) - Complete Python analysis pipeline
- [.env.example](.env.example) - API key configuration template

