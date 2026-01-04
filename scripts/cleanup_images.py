"""
Image cleanup script for facial expression dataset.
Removes images that:
- Are corrupted or can't be opened
- Are too small (< 48x48)
- Have wrong format
- Have suspicious aspect ratios or colors

Moves bad images to 'Images_rejected' folder for manual review.
"""

import os
import pathlib
import shutil
from typing import List, Tuple
from PIL import Image
import numpy as np

def validate_image(image_path: str, min_size: int = 48) -> Tuple[bool, str]:
    """
    Validate image file.
    Returns (is_valid, reason_if_invalid)
    """
    try:
        # Try to open image
        with Image.open(image_path) as img:
            # Check if image is valid
            img.verify()
        
        # Re-open for further checks (verify closes the file)
        with Image.open(image_path) as img:
            w, h = img.size
            
            # Check dimensions
            if h < min_size or w < min_size:
                return False, f"Too small ({w}x{h})"
            
            # Check aspect ratio (should be roughly square or portrait for faces)
            aspect = max(w, h) / min(w, h)
            if aspect > 5:
                return False, f"Suspicious aspect ratio ({w}x{h})"
            
            # Convert to array for color checks
            img_array = np.array(img.convert('RGB'))
            
            # Check if mostly black/white (corrupted or not a photo)
            mean_val = img_array.mean()
            if mean_val < 10 or mean_val > 245:
                return False, f"Suspicious brightness (mean={mean_val:.1f})"
            
            # Check variance (solid color images)
            var_val = img_array.var()
            if var_val < 100:
                return False, f"Low variance, possibly solid color (var={var_val:.1f})"
        
        return True, ""
    except Exception as ex:
        return False, f"Error: {ex}"

def cleanup_dataset(
    root_dir: str = "../Images",
    rejected_dir: str = "../Images_rejected"
) -> None:
    """Clean up dataset by removing invalid images."""
    
    root = pathlib.Path(root_dir)
    rejected = pathlib.Path(rejected_dir)
    
    if not root.exists():
        print(f"Error: {root_dir} does not exist")
        return
    
    # Statistics
    stats = {
        "total": 0,
        "valid": 0,
        "corrupted": 0,
        "too_small": 0,
        "aspect_ratio": 0,
        "other": 0
    }
    
    # Process each emotion folder
    for emotion_folder in sorted(root.iterdir()):
        if not emotion_folder.is_dir():
            continue
        
        emotion = emotion_folder.name
        print(f"\n{'='*60}")
        print(f"Processing: {emotion}")
        print(f"{'='*60}")
        
        images = list(emotion_folder.glob("*.jpg")) + list(emotion_folder.glob("*.png"))
        print(f"Found {len(images)} images")
        
        for img_path in images:
            stats["total"] += 1
            
            # Validate basic properties
            is_valid, reason = validate_image(str(img_path))
            
            if not is_valid:
                # Move to rejected folder
                reject_subfolder = rejected / emotion
                reject_subfolder.mkdir(parents=True, exist_ok=True)
                dest = reject_subfolder / img_path.name
                
                shutil.move(str(img_path), str(dest))
                print(f"✗ {img_path.name}: {reason}")
                
                if "small" in reason.lower():
                    stats["too_small"] += 1
                elif "aspect" in reason.lower():
                    stats["aspect_ratio"] += 1
                elif "open" in reason.lower() or "error" in reason.lower():
                    stats["corrupted"] += 1
                else:
                    stats["other"] += 1
                continue
            
            # Image is valid
            stats["valid"] += 1
            # print(f"✓ {img_path.name}")  # Comment out to reduce noise
    
    # Print summary
    print(f"\n{'='*60}")
    print("CLEANUP SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {stats['total']}")
    print(f"Valid images: {stats['valid']} ({100*stats['valid']/stats['total']:.1f}%)")
    print(f"\nRejected:")
    print(f"  Corrupted/unreadable: {stats['corrupted']}")
    print(f"  Too small: {stats['too_small']}")
    print(f"  Bad aspect ratio: {stats['aspect_ratio']}")
    print(f"  Other reasons: {stats['other']}")
    print(f"\nRejected images moved to: {rejected_dir}/")
    print("Review and manually delete or move back as needed.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up facial expression image dataset")
    parser.add_argument("--root", default="../Images", help="Root directory with emotion subfolders")
    parser.add_argument("--rejected", default="../Images_rejected", help="Directory for rejected images")
    
    args = parser.parse_args()
    
    cleanup_dataset(
        root_dir=args.root,
        rejected_dir=args.rejected
    )

if __name__ == "__main__":
    main()
