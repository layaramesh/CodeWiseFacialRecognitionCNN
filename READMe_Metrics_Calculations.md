# CSV Generation Scripts

This folder contains the complete workflow from downloading images to generating classification results.

## Required Files

### Scripts Folder (`scripts/`)
All Python scripts are organized in the `scripts/` folder:
1. **download_free_images.py** - Downloads facial expression images
2. **cleanup_images.py** - Cleans up downloaded images (optional quality control)
3. **batch_predict.py** - Runs all 3 models on images to generate predictions
4. **add_classification.py** - Adds classification columns to predictions
5. **emotion_utils.py** - Shared utility functions

### Input
- Images downloaded to `Images/` folder (organized by emotion)

### Output
- **output.csv** - Raw predictions from 3 emotion recognition models
- **output_with_classification.csv** - Enriched with classification columns:
  - Classification: Ground truth emotion from filename
  - Label: "Needs Help" or "Understanding Material"
  - Predicted Classification: Hybrid model prediction
  - Predicted Label: Hybrid model's label prediction

## Complete Workflow

```bash
# Navigate to scripts folder
cd scripts

# Step 1: Download images
python download_free_images.py

# Step 2 (Optional): Clean up images
python cleanup_images.py

# Step 3: 
# 1. Install [Microsoft Visual C++ Redistributable for Visual Studio 2015-2022 (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

# 2. Build with Windows runtime identifier:
#    ```powershell
#    dotnet build ..\ -c Release
#    dotnet run --project ..\ -c Release --model ..\Models\emotion.onnx --image ..\Images\angry\angry_001.jpg
#    ```

# Step 4: Run models on all images to generate predictions
python batch_predict.py

# Step 5: Add classification columns
python add_classification.py

# Step 6: Generate confusion matrices and metrics
python confusion_matrix_per_model.py
```

All scripts are in the `scripts/` folder. 
- Data files (CSV outputs) are saved to the root folder
- `confusion_matrix.csv` contains all per-model and custom label analysis

This will generate both `output.csv` and `output_with_classification.csv`.

## Hybrid Model Logic

- **Needs Help emotions** (Fear, Anger, Surprise, Disgust, Sadness): Uses emotion_cnn predictions
- **Understands Material emotions** (Happiness, Neutral): Uses emotion_ferplus predictions

## Archive Folder

The `archive/` folder contains analysis scripts and output files that were used during development but are not needed for generating the two main CSV files.
