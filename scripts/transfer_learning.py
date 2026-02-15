"""
Transfer Learning for Binary Emotion Classification (Confused vs Neutral)

This script implements two training approaches:
A) Frozen weights: Freeze all layers except the final classification layer
B) Fine-tuning: Backprop through the entire model with lower learning rates

Uses ResNet50 pretrained on ImageNet from torchvision.
"""

import os
import sys
import argparse
import pathlib
from pathlib import Path
from typing import Tuple, Dict, List
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent))


class TransferLearningModel:
    """Handles transfer learning training and evaluation"""
    
    def __init__(self, device: str = "cuda", checkpoint_dir: str = "checkpoints"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        print(f"Using device: {self.device}")
        
    def create_model(self, num_classes: int = 2) -> nn.Module:
        """Create ResNet50 model with custom output layer for binary classification"""
        # Load pretrained ResNet50
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace the final fully connected layer
        # ResNet50 has 2048 features in the last layer
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        model = model.to(self.device)
        return model
    
    def freeze_backbone(self, model: nn.Module) -> None:
        """Freeze all layers except the final classification layer"""
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the final classification layer
        for param in model.fc.parameters():
            param.requires_grad = True
        
        print("✓ Frozen all layers except final classification layer")
    
    def unfreeze_backbone(self, model: nn.Module) -> None:
        """Unfreeze all layers for fine-tuning"""
        for param in model.parameters():
            param.requires_grad = True
        
        print("✓ Unfrozen all layers for fine-tuning")
    
    def get_trainable_params(self, model: nn.Module) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                    criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss / (total / len(labels))})
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, model: nn.Module, dataloader: DataLoader, 
                 criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 20, learning_rate: float = 0.001, 
              weight_decay: float = 1e-5, model_name: str = "model") -> Dict:
        """Train the model with validation"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=learning_rate, weight_decay=weight_decay)
        
        # Optional: Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'best_val_acc': 0.0, 'best_epoch': 0
        }
        
        trainable_params = self.get_trainable_params(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
        
        best_val_acc = 0.0
        best_model_path = self.checkpoint_dir / f"{model_name}_best.pt"
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Step the scheduler
            scheduler.step()
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                history['best_val_acc'] = best_val_acc
                history['best_epoch'] = epoch + 1
                torch.save(model.state_dict(), best_model_path)
                print(f"✓ Saved best model (val_acc: {best_val_acc:.2f}%)")
        
        # Save final model
        final_path = self.checkpoint_dir / f"{model_name}_final.pt"
        torch.save(model.state_dict(), final_path)
        print(f"\n✓ Training complete. Final model saved to {final_path}")
        
        return history
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader, 
                 class_names: List[str]) -> Dict:
        """Evaluate model and return detailed metrics"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = np.mean(all_preds == all_labels)
        
        metrics = {'accuracy': accuracy, 'class_names': class_names,
                  'predictions': all_preds, 'true_labels': all_labels}
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                class_acc = np.mean(all_preds[mask] == all_labels[mask])
                metrics[f'{class_name}_accuracy'] = class_acc
                print(f"{class_name} Accuracy: {class_acc:.4f}")
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str], model_name: str) -> None:
        """Generate and save confusion matrix"""
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title(f'{model_name} - Confusion Matrix (Counts)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # Plot 2: Normalized (percentages)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_title(f'{model_name} - Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.checkpoint_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {plot_path}")
        plt.close()
        
        # Save to CSV
        csv_path = self.checkpoint_dir / f"{model_name}_confusion_matrix.csv"
        df = pd.DataFrame(cm, index=class_names, columns=class_names)
        df.index.name = 'True Label'
        df.to_csv(csv_path)
        print(f"✓ Confusion matrix CSV saved to {csv_path}")
        
        # Print confusion matrix
        print(f"\nConfusion Matrix - {model_name}:")
        print(df)
        
        # Print detailed metrics for binary classification
        if len(class_names) == 2:
            tn, fp, fn, tp = cm.ravel()
            print(f"\nDetailed Metrics:")
            print(f"True Positives (TP):  {tp:4d}  - Correctly predicted {class_names[1]}")
            print(f"True Negatives (TN):  {tn:4d}  - Correctly predicted {class_names[0]}")
            print(f"False Positives (FP): {fp:4d}  - Incorrectly predicted {class_names[1]}")
            print(f"False Negatives (FN): {fn:4d}  - Incorrectly predicted {class_names[0]}")
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\nPerformance Metrics:")
            print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Precision:   {precision:.4f} ({precision*100:.2f}%)")
            print(f"Recall:      {recall:.4f} ({recall*100:.2f}%)")
            print(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
            print(f"F1-Score:    {f1:.4f} ({f1*100:.2f}%)")
        
        # Print classification report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    def plot_history(self, history: Dict, model_name: str = "model") -> None:
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{model_name} - Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plot_path = self.checkpoint_dir / f"{model_name}_history.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        print(f"✓ Plot saved to {plot_path}")
        plt.close()


def get_data_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get data augmentation and normalization transforms"""
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def load_data(data_dir: Path, categories: List[str], batch_size: int = 32, 
              val_split: float = 0.2, image_size: int = 224) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Load datasets from directory structure, filtering to only specified categories"""
    from torch.utils.data import Subset, Dataset
    
    train_transforms, val_transforms = get_data_transforms(image_size)
    
    # Load full dataset
    full_dataset = ImageFolder(root=str(data_dir), transform=train_transforms)
    
    # Get indices for only the specified categories
    category_indices = [full_dataset.class_to_idx[cat] for cat in categories]
    
    # Create mapping from original labels to binary labels (0, 1)
    label_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(category_indices)}
    
    # Filter dataset to only include specified categories
    filtered_samples = [(path, label_mapping[label]) for path, label in full_dataset.samples 
                       if label in category_indices]
    
    # Create custom dataset with remapped labels
    class RemappedDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
            self.loader = full_dataset.loader
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            path, label = self.samples[idx]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, label
    
    filtered_dataset = RemappedDataset(filtered_samples, train_transforms)
    
    # Update class names to only show selected categories
    class_names = categories
    
    print(f"Classes: {class_names}")
    for new_idx, cat in enumerate(categories):
        count = sum(1 for _, label in filtered_samples if label == new_idx)
        print(f"  {cat} (label {new_idx}): {count} images")
    print(f"Total images: {len(filtered_dataset)}")
    
    # Split into train and validation
    val_size = int(len(filtered_dataset) * val_split)
    train_size = len(filtered_dataset) - val_size
    train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    return train_loader, val_loader, class_names


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning for Binary Emotion Classification")
    parser.add_argument("--data_dir", type=str, default="../Images", 
                        help="Path to dataset directory (default: ../Images)")
    parser.add_argument("--categories", nargs=2, type=str, default=["confused", "neutral"],
                        help="Two emotion categories to classify (default: confused neutral)")
    parser.add_argument("--approach", type=str, choices=["frozen", "finetuning", "both"], 
                        default="both", help="Training approach: frozen, finetuning, or both")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], 
                        default="cuda", help="Device to use")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    args = parser.parse_args()
    
    # Create temporary directory with only the selected categories
    data_dir = Path(args.data_dir)
    
    # Check if categories exist
    for cat in args.categories:
        cat_path = data_dir / cat
        if not cat_path.exists():
            print(f"Error: Category '{cat}' not found at {cat_path}")
            print(f"Available categories: {[d.name for d in data_dir.iterdir() if d.is_dir()]}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Transfer Learning: Binary Classification")
    print(f"Categories: {' vs '.join(args.categories)}")
    print(f"Approach: {args.approach}")
    print(f"{'='*60}\n")
    
    # Load data
    train_loader, val_loader, class_names = load_data(data_dir, categories=args.categories, 
                                                       batch_size=args.batch_size)
    
    # Initialize trainer
    trainer = TransferLearningModel(device=args.device, checkpoint_dir=args.checkpoint_dir)
    
    # Train Approach A: Frozen weights
    if args.approach in ["frozen", "both"]:
        print(f"\n{'='*60}")
        print("APPROACH A: FROZEN WEIGHTS")
        print("(Only final classification layer is trainable)")
        print(f"{'='*60}\n")
        
        model_a = trainer.create_model(num_classes=2)
        trainer.freeze_backbone(model_a)
        
        history_a = trainer.train(
            model_a, train_loader, val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            model_name="frozen_weights"
        )
        
        trainer.plot_history(history_a, "Frozen Weights")
        print(f"✓ Best validation accuracy: {history_a['best_val_acc']:.2f}% (epoch {history_a['best_epoch']})")
        
        # Evaluate
        model_a.load_state_dict(torch.load(f"{args.checkpoint_dir}/frozen_weights_best.pt"))
        print("\nEvaluation - Frozen Weights:")
        metrics_a = trainer.evaluate(model_a, val_loader, class_names)
        
        # Generate confusion matrix
        trainer.plot_confusion_matrix(metrics_a['true_labels'], metrics_a['predictions'],
                                     class_names, "frozen_weights")
    
    # Train Approach B: Fine-tuning
    if args.approach in ["finetuning", "both"]:
        print(f"\n{'='*60}")
        print("APPROACH B: FINE-TUNING")
        print("(All layers are trainable with lower learning rate)")
        print(f"{'='*60}\n")
        
        model_b = trainer.create_model(num_classes=2)
        trainer.unfreeze_backbone(model_b)
        
        # Use lower learning rate for fine-tuning
        history_b = trainer.train(
            model_b, train_loader, val_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate * 0.1,  # 10x lower learning rate
            model_name="finetuning"
        )
        
        trainer.plot_history(history_b, "Fine-tuning")
        print(f"✓ Best validation accuracy: {history_b['best_val_acc']:.2f}% (epoch {history_b['best_epoch']})")
        
        # Evaluate
        model_b.load_state_dict(torch.load(f"{args.checkpoint_dir}/finetuning_best.pt"))
        print("\nEvaluation - Fine-tuning:")
        metrics_b = trainer.evaluate(model_b, val_loader, class_names)
        
        # Generate confusion matrix
        trainer.plot_confusion_matrix(metrics_b['true_labels'], metrics_b['predictions'],
                                     class_names, "finetuning")
    
    print(f"\n{'='*60}")
    print("✓ Training complete!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
