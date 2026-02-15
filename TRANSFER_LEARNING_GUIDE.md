# Transfer Learning Implementation Guide

## Overview

This implementation uses **ResNet50** (pretrained on ImageNet) for binary emotion classification: **Confused vs Neutral**.

### Two Training Approaches

#### **Approach A: Frozen Weights**
- **Concept**: Freeze all pretrained layers, only train the final classification layer
- **Use case**: When you have limited data and want fast training
- **Trainable parameters**: ~2,050 (only the final FC layer)
- **Training time**: Faster (fewer parameters to update)
- **Memory usage**: Lower

```
ResNet50 backbone (frozen) → [Output Layer] (trainable)
                                     ↓
                            2 classes (confused/neutral)
```

#### **Approach B: Fine-tuning**
- **Concept**: Unfreeze all layers, but use a much lower learning rate
- **Use case**: When you want better performance and have sufficient data
- **Trainable parameters**: ~25.5 million (entire model)
- **Training time**: Slower (more parameters to update)
- **Memory usage**: Higher
- **Learning rate**: 10x lower than Approach A to avoid catastrophic forgetting

```
ResNet50 backbone (trainable) → [Output Layer] (trainable)
    ↓
  (learns features specific to emotions)
```

## Setup

### 1. Install Dependencies
```powershell
cd c:\Users\laya_\OneDrive\Documents\GitHub\CodeWiseFacialRecognitionCNN
pip install -r scripts/transfer_learning_requirements.txt
```

### 2. Ensure Data Structure
Your `Images/` folder should have:
```
Images/
├── confused/       (downloaded from Pexels)
│   ├── confused_001.jpg
│   ├── confused_002.jpg
│   └── ...
└── neutral/        (existing folder)
    ├── neutral_001.jpg
    ├── neutral_002.jpg
    └── ...
```

## Usage

### Run Both Approaches (Recommended)
```powershell
cd c:\Users\laya_\OneDrive\Documents\GitHub\CodeWiseFacialRecognitionCNN

# Default: 20 epochs, batch size 32, learning rate 0.001
python scripts/transfer_learning.py

# Or with custom parameters
python scripts/transfer_learning.py --epochs 30 --batch_size 16 --learning_rate 0.0005
```

### Run Only Frozen Weights
```powershell
python scripts/transfer_learning.py --approach frozen --epochs 20
```

### Run Only Fine-tuning
```powershell
python scripts/transfer_learning.py --approach finetuning --epochs 20
```

### Custom Categories
```powershell
# Classify any two categories
python scripts/transfer_learning.py --categories angry sad
python scripts/transfer_learning.py --categories happy sad --epochs 15
```

### All Available Options
```
--data_dir              Path to dataset directory (default: ../Images)
--categories            Two emotion categories (default: confused neutral)
--approach              Training approach: frozen, finetuning, or both (default: both)
--epochs                Number of epochs (default: 20)
--batch_size            Batch size (default: 32)
--learning_rate         Learning rate (default: 0.001)
--device                Device to use: cuda or cpu (default: cuda)
--checkpoint_dir        Directory for saving checkpoints (default: checkpoints/)
```

## Output Files

After training, you'll find in `checkpoints/`:

### Models
- `frozen_weights_best.pt` - Best model using frozen weights approach
- `frozen_weights_final.pt` - Final model after all epochs (frozen approach)
- `finetuning_best.pt` - Best model using fine-tuning approach
- `finetuning_final.pt` - Final model after all epochs (fine-tuning approach)

### Metrics & Plots
- `frozen_weights_history.png` - Loss & accuracy plots for frozen weights
- `finetuning_history.png` - Loss & accuracy plots for fine-tuning
- `frozen_weights_confusion_matrix.png` - Confusion matrix visualization (frozen)
- `frozen_weights_confusion_matrix.csv` - Confusion matrix data (frozen)
- `finetuning_confusion_matrix.png` - Confusion matrix visualization (fine-tuning)
- `finetuning_confusion_matrix.csv` - Confusion matrix data (fine-tuning)

## Expected Results

### With Frozen Weights
- **Training Speed**: Fast (~1-2 min per epoch on GPU)
- **Convergence**: Quick
- **Typical Accuracy**: 70-85% (depends on data quality)
- **Best for**: Limited data, quick prototyping

### With Fine-tuning
- **Training Speed**: Slower (~5-10 min per epoch on GPU)
- **Convergence**: Gradual
- **Typical Accuracy**: 80-95% (generally better than frozen)
- **Best for**: Maximum performance with sufficient data

## How Transfer Learning Works

### Why It's Effective

1. **ImageNet Pretraining**: ResNet50 learned general visual features
   - Early layers: edges, textures, basic shapes
   - Middle layers: object parts, patterns
   - Late layers: high-level features (faces, objects, etc.)

2. **Domain Adaptation**: Final layer learns emotion-specific features
   - Maps high-level face features → confused/neutral

3. **Data Efficiency**: Much less training data needed than training from scratch
   - From-scratch CNN: 10,000+ images
   - Transfer learning: 500-1,000 images

## Performance Comparison

| Metric | Frozen Weights | Fine-tuning |
|--------|---|---|
| Training Time | Fast | Slower |
| Convergence | Very Fast | Gradual |
| Memory Usage | Low | High |
| Final Accuracy | Good | Better |
| Risk of Overfitting | Lower | Higher |
| Trainable Parameters | 2,050 | 25.5M |

## Tips for Better Results

1. **More Data**: Download more confused images using the download script
2. **Data Augmentation**: The script includes flips, rotations, color jitter
3. **Longer Training**: Increase `--epochs` (try 30-50)
4. **Smaller Learning Rate**: Try `--learning_rate 0.0001` for fine-tuning
5. **Larger Batch Size**: Try `--batch_size 64` if you have GPU memory

## Troubleshooting

### Out of Memory (OOM) Error
```powershell
# Reduce batch size
python scripts/transfer_learning.py --batch_size 16

# Or use CPU
python scripts/transfer_learning.py --device cpu
```

### Poor Accuracy
- Check data quality (remove blurry/incorrect images)
- Train for more epochs
- Increase the amount of training data
- Try different learning rates

### No GPU Available
The script automatically falls back to CPU if CUDA is not available. To use GPU:
- Ensure PyTorch was installed with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

## Next Steps

1. Run the training script
2. Compare results between frozen and fine-tuning approaches
3. Use the best model for inference/deployment
4. Consider ensemble methods (combining both models)

## References

- ResNet50 Paper: https://arxiv.org/abs/1512.03385
- PyTorch Transfer Learning: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- ImageNet: https://www.image-net.org/
