import argparse
import os
import requests
from zipfile import ZipFile
from tqdm import tqdm
from pathlib import Path

url = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"

def download_file(url, save_path):
    response = requests.head(url)
    file_size = int(response.headers.get('content-length', 0))
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f, tqdm(
            total=file_size, unit='B', unit_scale=True, desc=str(save_path)
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Download finished: {save_path}")

def extract_zip(save_path, extract_dir):
    with ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Data extracted to: {extract_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Download Caltech-101 dataset")
    parser.add_argument("--root", default="./data",
                        help="Directory to store the dataset (default: ./data)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    print(f"⇢ Downloading Caltech-101 to {root} …")

    save_dir = root / "caltech101"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "caltech-101.zip"

    download_file(url, save_path)
    extract_zip(save_path, save_dir)
    print("✓ Caltech-101 is ready!\n"
          f"  e.g. {save_path}")

if __name__ == "__main__":
    main()
