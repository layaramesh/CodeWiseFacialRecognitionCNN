import csv
from collections import defaultdict
from emotion_utils import (
    extract_classification_from_filename,
    extract_emotion,
    get_label,
    calculate_metrics,
    EMOTION_LABELS,
    LABEL_CATEGORIES
)

def calculate_per_class_metrics(confusion_matrix, emotion_labels):
    """Calculate precision, recall, and F1 for each emotion class"""
    metrics = {}
    
    for label in emotion_labels:
        # True Positives: correctly predicted as this label
        tp = confusion_matrix[label][label]
        
        # False Negatives: actual label but predicted as something else
        fn = sum(confusion_matrix[label][other] for other in emotion_labels if other != label)
        
        # False Positives: predicted as this label but actually something else
        fp = sum(confusion_matrix[other][label] for other in emotion_labels if other != label)
        
        total = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            'total': total,
            'tp': tp,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics

# Read input CSV
input_file = '../output.csv'
emotion_labels = EMOTION_LABELS

# Initialize confusion matrices for each model
model_confusion_matrices = {}
model_names = []

with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    fieldnames = list(reader.fieldnames)
    
    # Model columns (skip filename)
    model_names = fieldnames[1:]
    
    # Initialize confusion matrices
    for model in model_names:
        model_confusion_matrices[model] = defaultdict(lambda: defaultdict(int))
    
    # Process each row
    for row in reader:
        filename = row[fieldnames[0]]
        actual = extract_classification_from_filename(filename)
        
        if not actual:
            continue
        
        # Process each model's prediction
        for model in model_names:
            predicted = extract_emotion(row[model])
            if predicted:
                model_confusion_matrices[model][actual][predicted] += 1

# Print confusion matrices and metrics for each model
for model in model_names:
    print("=" * 100)
    print(f"MODEL: {model}")
    print("=" * 100)
    
    confusion_matrix = model_confusion_matrices[model]
    
    # Print confusion matrix
    print("\nCONFUSION MATRIX")
    print("-" * 100)
    print("\nRows = Actual (Classification), Columns = Predicted\n")
    
    # Print header
    header = "Actual/Predicted".ljust(20)
    for label in emotion_labels:
        header += label[:8].rjust(10)
    header += "Total".rjust(10)
    print(header)
    print("-" * 100)
    
    # Print matrix rows
    total_correct = 0
    total_predictions = 0
    
    for actual_label in emotion_labels:
        row_str = actual_label.ljust(20)
        row_total = 0
        for predicted_label in emotion_labels:
            count = confusion_matrix[actual_label][predicted_label]
            row_str += str(count).rjust(10)
            row_total += count
            if actual_label == predicted_label:
                total_correct += count
            total_predictions += count
        row_str += str(row_total).rjust(10)
        print(row_str)
    
    # Print column totals
    col_totals_str = "Column Total".ljust(20)
    for predicted_label in emotion_labels:
        col_total = sum(confusion_matrix[actual][predicted_label] for actual in emotion_labels)
        col_totals_str += str(col_total).rjust(10)
    col_totals_str += str(total_predictions).rjust(10)
    print("-" * 100)
    print(col_totals_str)
    
    # Calculate overall accuracy
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    
    # Calculate per-class metrics
    metrics = calculate_per_class_metrics(confusion_matrix, emotion_labels)
    
    # Print per-class metrics
    print("\n" + "=" * 100)
    print("PER-CLASS METRICS")
    print("=" * 100)
    print(f"{'Emotion':<15} {'Total':<8} {'Correct':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 100)
    
    for label in emotion_labels:
        m = metrics[label]
        print(f"{label:<15} {m['total']:<8} {m['tp']:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}")
    
    # Calculate macro-averaged metrics
    macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
    macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
    macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)
    
    print("-" * 100)
    print(f"{'Macro Average':<15} {'':<8} {'':<10} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
    print()
    print(f"Overall Accuracy: {accuracy:.4f} ({total_correct}/{total_predictions})")
    print("=" * 100)
    print("\n\n")

# Print comparison summary
print("=" * 100)
print("MODEL COMPARISON SUMMARY")
print("=" * 100)
print(f"{'Model':<40} {'Accuracy':<12} {'Macro Precision':<18} {'Macro Recall':<15} {'Macro F1':<12}")
print("-" * 100)

for model in model_names:
    confusion_matrix = model_confusion_matrices[model]
    
    # Calculate accuracy
    total_correct = sum(confusion_matrix[label][label] for label in emotion_labels)
    total_predictions = sum(confusion_matrix[actual][predicted] 
                           for actual in emotion_labels 
                           for predicted in emotion_labels)
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    
    # Calculate macro metrics
    metrics = calculate_per_class_metrics(confusion_matrix, emotion_labels)
    macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
    macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
    macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)
    
    print(f"{model:<40} {accuracy:<12.4f} {macro_precision:<18.4f} {macro_recall:<15.4f} {macro_f1:<12.4f}")

print("=" * 100)

# Save to CSV
output_csv = '../confusion_matrix.csv'

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Write each model's data
    for model in model_names:
        confusion_matrix = model_confusion_matrices[model]
        
        # Model header
        writer.writerow([f'MODEL: {model}'])
        writer.writerow([])
        
        # Confusion Matrix
        writer.writerow(['CONFUSION MATRIX'])
        
        # Header row
        header = ['Actual/Predicted'] + emotion_labels + ['Total']
        writer.writerow(header)
        
        # Data rows
        for actual_label in emotion_labels:
            row = [actual_label]
            row_total = 0
            for predicted_label in emotion_labels:
                count = confusion_matrix[actual_label][predicted_label]
                row.append(count)
                row_total += count
            row.append(row_total)
            writer.writerow(row)
        
        # Column totals
        col_totals = ['Column Total']
        total_predictions = 0
        for predicted_label in emotion_labels:
            col_total = sum(confusion_matrix[actual][predicted_label] for actual in emotion_labels)
            col_totals.append(col_total)
            total_predictions += col_total
        col_totals.append(total_predictions)
        writer.writerow(col_totals)
        
        writer.writerow([])
        
        # Performance Metrics
        writer.writerow(['PERFORMANCE METRICS'])
        writer.writerow(['Emotion', 'Total', 'Correct', 'Precision', 'Recall', 'F1 Score'])
        
        # Calculate metrics
        metrics = calculate_per_class_metrics(confusion_matrix, emotion_labels)
        total_correct = sum(confusion_matrix[label][label] for label in emotion_labels)
        
        for label in emotion_labels:
            m = metrics[label]
            writer.writerow([label, m['total'], m['tp'], f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"])
        
        # Macro averages
        macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
        macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
        macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        writer.writerow(['Macro Average', '', '', f"{macro_precision:.4f}", f"{macro_recall:.4f}", f"{macro_f1:.4f}"])
        writer.writerow(['Overall Accuracy', f"{accuracy:.4f} ({total_correct}/{total_predictions})"])
        
        writer.writerow([])
        writer.writerow([])
    
    # Model Comparison Summary
    writer.writerow(['MODEL COMPARISON SUMMARY'])
    writer.writerow(['Model', 'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'])
    
    for model in model_names:
        confusion_matrix = model_confusion_matrices[model]
        
        # Calculate accuracy
        total_correct = sum(confusion_matrix[label][label] for label in emotion_labels)
        total_predictions = sum(confusion_matrix[actual][predicted] 
                               for actual in emotion_labels 
                               for predicted in emotion_labels)
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        
        # Calculate macro metrics
        metrics = calculate_per_class_metrics(confusion_matrix, emotion_labels)
        macro_precision = sum(m['precision'] for m in metrics.values()) / len(metrics)
        macro_recall = sum(m['recall'] for m in metrics.values()) / len(metrics)
        macro_f1 = sum(m['f1'] for m in metrics.values()) / len(metrics)
        
        writer.writerow([model, f"{accuracy:.4f}", f"{macro_precision:.4f}", f"{macro_recall:.4f}", f"{macro_f1:.4f}"])
    
    writer.writerow([])
    writer.writerow([])
    
    # Add Custom Label Confusion Matrix (Needs Help vs Understands Material)
    writer.writerow(['CUSTOM LABEL ANALYSIS (Needs Help vs Understands Material)'])
    writer.writerow([])
    
    # Read output_with_classification.csv for hybrid model custom labels
    label_confusion = defaultdict(lambda: defaultdict(int))
    
    with open('../output_with_classification.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            actual = row['Label']
            predicted = row['Predicted Label']
            if actual and predicted:
                label_confusion[actual][predicted] += 1
    
    # Write custom label confusion matrix
    writer.writerow(['Hybrid Model - Label-Based Confusion Matrix'])
    writer.writerow(['Actual/Predicted'] + LABEL_CATEGORIES)
    
    for actual_label in LABEL_CATEGORIES:
        row = [actual_label]
        for predicted_label in LABEL_CATEGORIES:
            count = label_confusion[actual_label][predicted_label]
            row.append(count)
        writer.writerow(row)
    
    writer.writerow([])
    
    # Calculate custom label metrics
    tp = label_confusion['Needs Help']['Needs Help']
    tn = label_confusion['Understands Material']['Understands Material']
    fp = label_confusion['Understands Material']['Needs Help']
    fn = label_confusion['Needs Help']['Understands Material']
    
    metrics = calculate_metrics(tp, tn, fp, fn)
    
    writer.writerow(['CUSTOM LABEL METRICS'])
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['True Positive (TP)', tp])
    writer.writerow(['True Negative (TN)', tn])
    writer.writerow(['False Positive (FP)', fp])
    writer.writerow(['False Negative (FN)', fn])
    writer.writerow([])
    writer.writerow(['Accuracy', f"{metrics['accuracy']:.4f}"])
    writer.writerow(['Precision (Needs Help)', f"{metrics['precision']:.4f}"])
    writer.writerow(['Recall (Needs Help)', f"{metrics['recall']:.4f}"])
    writer.writerow(['F1 Score', f"{metrics['f1']:.4f}"])

print(f"\nResults saved to {output_csv}")
