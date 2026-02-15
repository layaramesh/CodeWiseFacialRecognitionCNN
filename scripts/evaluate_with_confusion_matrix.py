"""
Evaluate trained transfer learning models and generate confusion matrices
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


def load_model(model_path: str, num_classes: int = 2, device: str = "cuda") -> nn.Module:
    """Load a trained ResNet50 model"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Create model architecture
    model = models.resnet50(weights=None)  # Don't load ImageNet weights
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    return model


def get_predictions(model: nn.Module, dataloader: DataLoader, 
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Get all predictions and true labels"""
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         model_name: str, save_path: str) -> None:
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix plot saved to {save_path}")
    plt.close()


def plot_normalized_confusion_matrix(cm: np.ndarray, class_names: List[str],
                                     model_name: str, save_path: str) -> None:
    """Plot and save normalized confusion matrix (percentages)"""
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    
    # Create heatmap with percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    
    plt.title(f'Normalized Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Normalized confusion matrix plot saved to {save_path}")
    plt.close()


def save_confusion_matrix_csv(cm: np.ndarray, class_names: List[str], 
                              model_name: str, save_path: str) -> None:
    """Save confusion matrix to CSV"""
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.index.name = 'True Label'
    df.to_csv(save_path)
    print(f"✓ Confusion matrix CSV saved to {save_path}")


def calculate_metrics(cm: np.ndarray, class_names: List[str]) -> dict:
    """Calculate detailed metrics from confusion matrix"""
    metrics = {}
    
    # For binary classification
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        
        metrics['True Negatives'] = int(tn)
        metrics['False Positives'] = int(fp)
        metrics['False Negatives'] = int(fn)
        metrics['True Positives'] = int(tp)
        
        # Calculate rates
        metrics['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['Recall (Sensitivity)'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['F1-Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / \
                             (metrics['Precision'] + metrics['Recall']) \
                             if (metrics['Precision'] + metrics['Recall']) > 0 else 0
    
    return metrics


def print_metrics(metrics: dict, class_names: List[str]) -> None:
    """Pretty print metrics"""
    print("\n" + "="*60)
    print("CONFUSION MATRIX BREAKDOWN")
    print("="*60)
    
    if len(class_names) == 2:
        print(f"\nPositive class: {class_names[1]}")
        print(f"Negative class: {class_names[0]}")
        print(f"\nTrue Positives (TP):   {metrics['True Positives']:4d}  - Correctly predicted {class_names[1]}")
        print(f"True Negatives (TN):   {metrics['True Negatives']:4d}  - Correctly predicted {class_names[0]}")
        print(f"False Positives (FP):  {metrics['False Positives']:4d}  - Incorrectly predicted {class_names[1]}")
        print(f"False Negatives (FN):  {metrics['False Negatives']:4d}  - Incorrectly predicted {class_names[0]}")
        
        print("\n" + "-"*60)
        print("PERFORMANCE METRICS")
        print("-"*60)
        print(f"Accuracy:    {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
        print(f"Precision:   {metrics['Precision']:.4f} ({metrics['Precision']*100:.2f}%)")
        print(f"Recall:      {metrics['Recall (Sensitivity)']:.4f} ({metrics['Recall (Sensitivity)']*100:.2f}%)")
        print(f"Specificity: {metrics['Specificity']:.4f} ({metrics['Specificity']*100:.2f}%)")
        print(f"F1-Score:    {metrics['F1-Score']:.4f} ({metrics['F1-Score']*100:.2f}%)")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model and generate confusion matrix")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.pt file)")
    parser.add_argument("--data_dir", type=str, default="../Images",
                        help="Path to dataset directory")
    parser.add_argument("--model_name", type=str, default="model",
                        help="Model name for output files")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                        default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save results")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(root=args.data_dir, transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    
    class_names = dataset.classes
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(dataset)}")
    
    # Load model
    model = load_model(args.model_path, num_classes=len(class_names), device=args.device)
    
    # Get predictions
    predictions, true_labels = get_predictions(model, dataloader, device)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(pd.DataFrame(cm, index=class_names, columns=class_names))
    
    # Calculate and print metrics
    metrics = calculate_metrics(cm, class_names)
    print_metrics(metrics, class_names)
    
    # Print classification report
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(true_labels, predictions, 
                               target_names=class_names, digits=4))
    
    # Save confusion matrix (counts)
    cm_csv_path = output_dir / f"{args.model_name}_confusion_matrix.csv"
    save_confusion_matrix_csv(cm, class_names, args.model_name, str(cm_csv_path))
    
    # Save normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_csv_path = output_dir / f"{args.model_name}_confusion_matrix_normalized.csv"
    df_norm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    df_norm.index.name = 'True Label'
    df_norm.to_csv(cm_norm_csv_path)
    print(f"✓ Normalized confusion matrix CSV saved to {cm_norm_csv_path}")
    
    # Plot confusion matrix
    cm_plot_path = output_dir / f"{args.model_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, args.model_name, str(cm_plot_path))
    
    # Plot normalized confusion matrix
    cm_norm_plot_path = output_dir / f"{args.model_name}_confusion_matrix_normalized.png"
    plot_normalized_confusion_matrix(cm, class_names, args.model_name, 
                                    str(cm_norm_plot_path))
    
    # Save metrics to JSON
    import json
    metrics_path = output_dir / f"{args.model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")
    
    print("\n" + "="*60)
    print("✓ Evaluation complete!")
    print(f"Results saved to: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
