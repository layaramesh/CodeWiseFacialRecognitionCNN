"""
Batch runner for emotion recognition models.
Runs all ONNX models in Models/ folder on all images in Images/ folder.
Outputs results to output.csv with:
- Rows: image filenames
- Columns: model names + predictions
"""

import os
import pathlib
import subprocess
import csv
from typing import Dict, List, Tuple
import sys

def find_models(models_dir: str = "Models") -> List[str]:
    """Find all ONNX model files."""
    models_path = pathlib.Path(models_dir)
    if not models_path.exists():
        print(f"Error: {models_dir} directory not found")
        return []
    
    models = list(models_path.glob("*.onnx"))
    return sorted([str(m) for m in models])

def find_images(images_dir: str = "Images") -> List[str]:
    """Find all image files in subdirectories."""
    images_path = pathlib.Path(images_dir)
    if not images_path.exists():
        print(f"Error: {images_dir} directory not found")
        return []
    
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        # Search in all subdirectories
        images.extend(images_path.rglob(ext))
    
    return sorted([str(img) for img in images])

def run_model_on_image(model_path: str, image_path: str, cascade_path: str = "Data/haarcascade_frontalface_default.xml") -> Tuple[str, float]:
    """
    Run a single model on a single image.
    Returns (prediction_label, probability)
    """
    cmd = [
        "dotnet", "run", "--project", "..", "-c", "Release", "--",
        "--model", model_path,
        "--image", image_path,
        "--csv"
    ]
    
    # Add cascade if it exists
    if os.path.exists(cascade_path):
        cmd.extend(["--cascade", cascade_path])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            print(f"Error running model on {os.path.basename(image_path)}: {result.stderr}")
            return "ERROR", 0.0
        
        # Parse CSV output: filename,prediction,probability
        output = result.stdout.strip().split('\n')[-1]  # Get last line
        parts = output.split(',')
        
        if len(parts) >= 3:
            prediction = parts[1]
            probability = float(parts[2])
            return prediction, probability
        else:
            return "PARSE_ERROR", 0.0
            
    except subprocess.TimeoutExpired:
        print(f"Timeout running model on {os.path.basename(image_path)}")
        return "TIMEOUT", 0.0
    except Exception as ex:
        print(f"Exception: {ex}")
        return "EXCEPTION", 0.0

def generate_csv_report(
    models: List[str],
    images: List[str],
    output_file: str = "../output.csv",
    cascade_path: str = "../Data/haarcascade_frontalface_default.xml"
) -> None:
    """Generate CSV report with all predictions."""
    
    if not models:
        print("No models found!")
        return
    
    if not images:
        print("No images found!")
        return
    
    print(f"\nFound {len(models)} models:")
    for m in models:
        print(f"  - {os.path.basename(m)}")
    
    print(f"\nFound {len(images)} images")
    print(f"\nGenerating predictions...")
    
    # Prepare CSV structure
    results: Dict[str, Dict[str, str]] = {}
    
    total_predictions = len(models) * len(images)
    current = 0
    
    # Run each model on each image
    for model_path in models:
        model_name = os.path.basename(model_path)
        print(f"\n{'='*60}")
        print(f"Running model: {model_name}")
        print(f"{'='*60}")
        
        for image_path in images:
            image_name = os.path.basename(image_path)
            
            # Initialize image entry if needed
            if image_name not in results:
                results[image_name] = {"filename": image_name}
            
            # Run prediction
            prediction, probability = run_model_on_image(model_path, image_path, cascade_path)
            
            # Store as "Prediction (XX.XX%)"
            results[image_name][model_name] = f"{prediction} ({probability*100:.2f}%)"
            
            current += 1
            if current % 10 == 0:
                print(f"Progress: {current}/{total_predictions} ({100*current/total_predictions:.1f}%)")
    
    # Write CSV
    model_names = [os.path.basename(m) for m in models]
    fieldnames = ["Filename"] + model_names
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for image_name in sorted(results.keys()):
            row = {"Filename": results[image_name]["filename"]}
            for model_name in model_names:
                row[model_name] = results[image_name].get(model_name, "N/A")
            writer.writerow(row)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total predictions: {total_predictions}")
    print(f"{'='*60}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch emotion recognition runner")
    parser.add_argument("--models", default="../Models", help="Directory containing ONNX models")
    parser.add_argument("--images", default="../Images", help="Directory containing images")
    parser.add_argument("--output", default="../output.csv", help="Output CSV file")
    parser.add_argument("--cascade", default="../Data/haarcascade_frontalface_default.xml", 
                       help="Haar cascade XML path")
    
    args = parser.parse_args()
    
    # Find models and images
    models = find_models(args.models)
    images = find_images(args.images)
    
    if not models or not images:
        sys.exit(1)
    
    # Generate report
    generate_csv_report(models, images, args.output, args.cascade)

if __name__ == "__main__":
    main()
