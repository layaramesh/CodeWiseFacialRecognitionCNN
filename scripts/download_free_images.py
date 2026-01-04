import os
import pathlib
import requests
from typing import List

# Download images from free sources without API keys
# Uses Pexels API (free tier: 200 requests/hour, 20K/month)

def download_from_pexels(query: str, count: int, out_dir: pathlib.Path, label: str) -> None:
    """Download images from Pexels (free API, no credit card required)"""
    # Get free API key from: https://www.pexels.com/api/
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        print(f"Skipping {label}: Set PEXELS_API_KEY environment variable")
        return
    
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": api_key}
    
    page = 1
    downloaded = 0
    
    while downloaded < count:
        per_page = min(80, count - downloaded)
        url = f"https://api.pexels.com/v1/search"
        params = {"query": query, "per_page": per_page, "page": page}
        
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"Pexels API error: {resp.status_code}")
            break
        
        data = resp.json()
        photos = data.get("photos", [])
        if not photos:
            break
        
        for photo in photos:
            if downloaded >= count:
                break
            
            img_url = photo["src"]["medium"]  # or "large", "original"
            fname = f"{label}_{downloaded+1:03d}.jpg"
            dest = out_dir / fname
            
            try:
                img_resp = requests.get(img_url, timeout=20)
                img_resp.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(img_resp.content)
                downloaded += 1
                print(f"Downloaded {label}: {downloaded}/{count}")
            except Exception as ex:
                print(f"Failed to download {img_url}: {ex}")
        
        page += 1

def main():
    queries = {
        "happy": "happy person face emotion",
        "sad": "sad person face emotion",
        "angry": "angry person face emotion",
        "surprise": "surprised person face emotion",
        "fear": "fearful scared person face",
        "disgust": "disgusted person face",
        "neutral": "neutral person face expression",
    }
    
    root = pathlib.Path("../Images2")
    per_class = 50  # Adjust as needed
    
    for label, query in queries.items():
        print(f"\nFetching {per_class} images for '{label}'")
        download_from_pexels(query, per_class, root / label, label)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
