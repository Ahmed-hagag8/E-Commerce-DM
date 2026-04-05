"""
Data Download Script
Downloads all 3 required datasets for the E-Commerce Data Mining project.

Datasets:
1. Online Retail II (UCI ML Repository) - Tabular
2. Amazon Products Dataset (Kaggle) - Images
3. Women's Clothing Reviews (Kaggle) - Hybrid (Text + Tabular)

Prerequisites:
- pip install kaggle openpyxl requests tqdm
- Kaggle API key configured (~/.kaggle/kaggle.json)
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "online_retail": {
        "source": "uci",
        "url": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        "filename": "online_retail_ii.zip",
        "description": "Online Retail II - Tabular transaction data (~1M rows)"
    },
    "amazon_products": {
        "source": "kaggle",
        "dataset": "lokeshparab/amazon-products-dataset",
        "description": "Amazon Products - Product images + metadata"
    },
    "clothing_reviews": {
        "source": "kaggle",
        "dataset": "nicapotato/womens-ecommerce-clothing-reviews",
        "description": "Women's Clothing Reviews - Text + Tabular hybrid"
    }
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f, tqdm(
        desc=desc, total=total_size, unit='iB', unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

    print(f"  ✅ Saved to {dest_path}")


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest_dir)
    print(f"  📦 Extracted to {dest_dir}")


def download_from_kaggle(dataset: str, dest_dir: Path):
    """Download a dataset from Kaggle using the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"  ⬇️  Downloading {dataset} from Kaggle...")
        api.dataset_download_files(dataset, path=str(dest_dir), unzip=True)
        print(f"  ✅ Downloaded to {dest_dir}")
    except Exception as e:
        print(f"  ❌ Kaggle download failed: {e}")
        print(f"  💡 Manual download: https://www.kaggle.com/datasets/{dataset}")
        print(f"     Extract to: {dest_dir}")
        return False
    return True


def main():
    print("=" * 60)
    print("📊 E-Commerce Data Mining — Dataset Downloader")
    print("=" * 60)

    # Create raw data directory
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # 1. Online Retail II (UCI)
    # -------------------------------------------------------
    print(f"\n[1/3] {DATASETS['online_retail']['description']}")
    retail_zip = RAW_DATA_DIR / DATASETS['online_retail']['filename']
    retail_dir = RAW_DATA_DIR / "online_retail"

    if retail_dir.exists() and any(retail_dir.iterdir()):
        print("  ⏭️  Already downloaded, skipping.")
    else:
        retail_dir.mkdir(exist_ok=True)
        download_file(
            DATASETS['online_retail']['url'],
            retail_zip,
            desc="Online Retail II"
        )
        extract_zip(retail_zip, retail_dir)
        retail_zip.unlink()  # Remove zip after extraction

    # -------------------------------------------------------
    # 2. Amazon Products (Kaggle)
    # -------------------------------------------------------
    print(f"\n[2/3] {DATASETS['amazon_products']['description']}")
    amazon_dir = RAW_DATA_DIR / "amazon_products"
    amazon_dir.mkdir(exist_ok=True)

    if any(amazon_dir.iterdir()):
        print("  ⏭️  Already downloaded, skipping.")
    else:
        download_from_kaggle(
            DATASETS['amazon_products']['dataset'],
            amazon_dir
        )

    # -------------------------------------------------------
    # 3. Women's Clothing Reviews (Kaggle)
    # -------------------------------------------------------
    print(f"\n[3/3] {DATASETS['clothing_reviews']['description']}")
    reviews_dir = RAW_DATA_DIR / "clothing_reviews"
    reviews_dir.mkdir(exist_ok=True)

    if any(reviews_dir.iterdir()):
        print("  ⏭️  Already downloaded, skipping.")
    else:
        download_from_kaggle(
            DATASETS['clothing_reviews']['dataset'],
            reviews_dir
        )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("✅ Dataset download complete!")
    print("=" * 60)
    print(f"\n📁 Data directory: {RAW_DATA_DIR}")
    print("\nContents:")
    for item in sorted(RAW_DATA_DIR.rglob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            rel = item.relative_to(RAW_DATA_DIR)
            print(f"  {rel} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
