"""
Download model checkpoints from HuggingFace Hub or Google Drive.
Update the URLs below with your actual checkpoint locations.
"""
import os
from pathlib import Path
import requests


def download_from_url(url, dest_path):
    """Download a file from URL to destination path."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {dest_path}")


def main():
    # TODO: Replace with your actual checkpoint URLs
    checkpoints = {
        "admm100": "https://huggingface.co/YOUR_USERNAME/lenless-checkpoints/resolve/main/admm100.pth",
        "le_admm20": "https://huggingface.co/YOUR_USERNAME/lenless-checkpoints/resolve/main/le_admm20.pth",
        "modular_pre_post": "https://huggingface.co/YOUR_USERNAME/lenless-checkpoints/resolve/main/modular_pre_post.pth",
        "modular_pre": "https://huggingface.co/YOUR_USERNAME/lenless-checkpoints/resolve/main/modular_pre.pth",
        "modular_post": "https://huggingface.co/YOUR_USERNAME/lenless-checkpoints/resolve/main/modular_post.pth",
    }

    save_dir = Path("saved")
    save_dir.mkdir(exist_ok=True, parents=True)

    for name, url in checkpoints.items():
        if "YOUR_USERNAME" in url:
            print(f"Skipping {name}: URL not configured")
            continue
        download_from_url(url, save_dir / f"{name}.pth")

    print("\nAll checkpoints downloaded!")


if __name__ == "__main__":
    main()
