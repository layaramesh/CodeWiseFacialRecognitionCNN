# Confusion Matrix Guide

## What is a Confusion Matrix?

A confusion matrix shows how well your model distinguishes between different classes by displaying:
- **What the model predicted** (columns)
- **What was actually true** (rows)

For binary classification (confused vs neutral), it looks like this:

```
                  Predicted
              Confused  Neutral
Actual 
Confused    [   TP    |   FN   ]
Neutral     [   FP    |   TN   ]
```

## Understanding the Components

### For "Confused vs Neutral" Classification:

- **True Positive (TP)**: Model correctly identified a confused face as confused ✓
- **True Negative (TN)**: Model correctly identified a neutral face as neutral ✓
- **False Positive (FP)**: Model incorrectly said neutral was confused ✗
- **False Negative (FN)**: Model incorrectly said confused was neutral ✗

### Example:
```
Confusion Matrix:
                Predicted
              confused  neutral
True
confused    [    45   |    5   ]
neutral     [     3   |   47   ]
```

This means:
- **45** confused faces correctly identified (TP)
- **47** neutral faces correctly identified (TN)
- **5** confused faces wrongly labeled as neutral (FN) - model missed them
- **3** neutral faces wrongly labeled as confused (FP) - false alarm

## Key Metrics Calculated

### 1. **Accuracy**
Overall correctness = (TP + TN) / Total
```
(45 + 47) / 100 = 92%
```

### 2. **Precision**
When model says "confused", how often is it right? = TP / (TP + FP)
```
45 / (45 + 3) = 93.75%
```

### 3. **Recall (Sensitivity)**
Of all actual confused faces, how many did we find? = TP / (TP + FN)
```
45 / (45 + 5) = 90%
```

### 4. **Specificity**
Of all actual neutral faces, how many did we correctly identify? = TN / (TN + FP)
```
47 / (47 + 3) = 94%
```

### 5. **F1-Score**
Balance between precision and recall = 2 * (Precision * Recall) / (Precision + Recall)
```
2 * (0.9375 * 0.90) / (0.9375 + 0.90) = 91.84%
```

## How to Generate Confusion Matrices

### Method 1: Automatic (during training)

When you run the transfer learning script, confusion matrices are **automatically generated**:

```powershell
python scripts/transfer_learning.py
```

**Output files** in `checkpoints/`:
- `frozen_weights_confusion_matrix.png` - Visual heatmap
- `frozen_weights_confusion_matrix.csv` - Raw numbers
- `finetuning_confusion_matrix.png` - Visual heatmap
- `finetuning_confusion_matrix.csv` - Raw numbers

### Method 2: Standalone Evaluation

To generate confusion matrix for a specific model after training:

```powershell
# Evaluate frozen weights model
python scripts/evaluate_with_confusion_matrix.py \
    --model_path checkpoints/frozen_weights_best.pt \
    --model_name frozen_weights \
    --data_dir Images

# Evaluate fine-tuning model
python scripts/evaluate_with_confusion_matrix.py \
    --model_path checkpoints/finetuning_best.pt \
    --model_name finetuning \
    --data_dir Images
```

**Output files** in `evaluation_results/`:
- `{model_name}_confusion_matrix.png` - Counts visualization
- `{model_name}_confusion_matrix_normalized.png` - Percentages visualization
- `{model_name}_confusion_matrix.csv` - Raw counts
- `{model_name}_confusion_matrix_normalized.csv` - Percentages
- `{model_name}_metrics.json` - All calculated metrics

## Interpreting Results

### Good Performance Signs:
- ✓ High numbers on the **diagonal** (TP and TN)
- ✓ Low numbers **off-diagonal** (FP and FN)
- ✓ Accuracy > 80%
- ✓ Precision and Recall both high

### Warning Signs:
- ✗ High **False Positives**: Too many false alarms
- ✗ High **False Negatives**: Missing too many real cases
- ✗ Imbalanced diagonal: Good at one class, poor at other

### Example Comparison:

**Frozen Weights** (faster training):
```
confused  neutral
[  42  |   8  ]    <- Missed 8 confused faces
[   5  |  45  ]    <- 5 false alarms
Accuracy: 87%
```

**Fine-tuning** (slower but better):
```
confused  neutral
[  48  |   2  ]    <- Only missed 2!
[   1  |  49  ]    <- Only 1 false alarm!
Accuracy: 97%
```

Fine-tuning typically performs better!

## What to Do With Results

1. **Compare approaches**: See which works better (frozen vs fine-tuning)
2. **Identify weaknesses**: Are you missing confused faces? Or getting false alarms?
3. **Improve data**: Add more samples of the class with poor performance
4. **Adjust threshold**: If using probabilities, adjust decision threshold
5. **Use for reporting**: Include in your documentation/papers

## Common Questions

**Q: What's a good accuracy?**
A: For facial emotion recognition:
- 70-80% = Acceptable
- 80-90% = Good
- 90%+ = Excellent

**Q: Why isn't my model 100% accurate?**
A: Some images are genuinely ambiguous. Even humans disagree on subtle expressions!

**Q: Which metric matters most?**
A: Depends on use case:
- Medical diagnosis: High **Recall** (don't miss cases)
- Spam detection: High **Precision** (avoid false alarms)
- General use: Balanced **F1-Score**

**Q: Normalized vs Raw confusion matrix?**
A: 
- **Raw**: Shows actual counts
- **Normalized**: Shows percentages (better when classes have different sizes)

## Visual Examples

After running training, you'll get plots like this:

### Raw Counts Matrix:
Shows exact numbers of predictions in each category.

### Normalized Matrix:
Shows percentages - easier to see model performance at a glance.

Both are saved automatically!
