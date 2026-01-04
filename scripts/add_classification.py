"""
Add Classification Columns Script
Reads output.csv and adds classification columns based on filename and hybrid model predictions
"""
import csv
from emotion_utils import (
    extract_classification_from_filename,
    extract_emotion,
    get_hybrid_prediction,
    get_label
)

# Read input CSV and add classification column
input_file = '../output.csv'
output_file = '../output_with_classification.csv'

with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    fieldnames = list(reader.fieldnames)
    
    # Add Classification, Label, Predicted Classification, and Predicted Label columns after Filename
    # Classification: extracted from filename
    # Label: Needs Help or Understands Material based on Classification
    # Predicted Classification: based on hybrid model logic
    # Predicted Label: Needs Help or Understands Material based on Predicted Classification
    new_fieldnames = [fieldnames[0], 'Classification', 'Label', 'Predicted Classification', 'Predicted Label'] + fieldnames[1:]
    
    # Find model columns
    cnn_col = None
    ferplus_col = None
    for field in fieldnames[1:]:
        if 'emotion_cnn' in field.lower():
            cnn_col = field
        elif 'emotion-ferplus' in field.lower():
            ferplus_col = field
    
    # Read all rows and add classification
    output_rows = []
    for row in reader:
        filename = row[fieldnames[0]]
        classification = extract_classification_from_filename(filename)
        
        # Extract predictions from models
        cnn_prediction = extract_emotion(row[cnn_col]) if cnn_col else ''
        ferplus_prediction = extract_emotion(row[ferplus_col]) if ferplus_col else ''
        
        # Get label (Needs Help or Understands Material)
        label = get_label(classification)
        
        # Get hybrid prediction based on classification
        predicted_classification = get_hybrid_prediction(classification, cnn_prediction, ferplus_prediction)
        
        # Get predicted label
        predicted_label = get_label(predicted_classification) if predicted_classification else ''
        
        # Create new row with classification, label, predicted classification, and predicted label
        new_row = {fieldnames[0]: filename}
        new_row['Classification'] = classification
        new_row['Label'] = label
        new_row['Predicted Classification'] = predicted_classification
        new_row['Predicted Label'] = predicted_label
        for field in fieldnames[1:]:
            new_row[field] = row[field]
        
        output_rows.append(new_row)

# Write output CSV
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"Processed {len(output_rows)} rows")
print(f"Added Classification, Label, Predicted Classification, and Predicted Label columns")
print(f"Output saved to {output_file}")

# Count labels
needs_help_count = sum(1 for row in output_rows if row['Label'] == 'Needs Help')
understanding_count = sum(1 for row in output_rows if row['Label'] == 'Understands Material')
print(f"\nLabel Distribution:")
print(f"  Needs Help: {needs_help_count}")
print(f"  Understands Material: {understanding_count}")

print(f"\nTo compare model performance, run: python compare_models.py")
