"""
AEGIS-ANC Dataset Downloader & Setup Script
Downloads sample or selected datasets for military speech enhancement training.
"""

import os
import argparse
import urllib.request
import tarfile
import zipfile
import shutil

AZURE_DNS_URL = "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

DNS_ESSENTIAL_BLOBS = [
    # Clean speech
    ("clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2", "clean_fullband"),
    # General ambient noise
    ("noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2", "noise_fullband"),
    # Acoustic impulse responses
    ("datasets_fullband.impulse_responses_000.tar.bz2", "rirs"),
]

ESC50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"[-] Already exists: {dest_path}")
        return
    print(f"[+] Downloading: {url}")
    print(f"    -> Destination: {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    print(f"[✓] Downloaded: {dest_path}")

def download_esc50(dest_dir: str = "datasets/esc50"):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "esc50.zip")
    download_file(ESC50_URL, zip_path)
    print("[+] Extracting ESC-50 (battlefield/environmental noise)...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print("[✓] ESC-50 ready.")

def download_dns_essentials(dest_dir: str = "datasets/dns"):
    os.makedirs(dest_dir, exist_ok=True)
    for blob, subfolder in DNS_ESSENTIAL_BLOBS:
        url = f"{AZURE_DNS_URL}/{blob}"
        target_file = os.path.join(dest_dir, subfolder, os.path.basename(blob))
        download_file(url, target_file)

def main():
    parser = argparse.ArgumentParser(description="AEGIS-ANC Dataset Setup Tool")
    parser.add_argument("--dns", action="store_true", help="Download essential DNS Challenge subsets (clean + noise)")
    parser.add_argument("--esc50", action="store_true", help="Download ESC-50 dataset (gunshots, engines, sirens)")
    parser.add_argument("--all", action="store_true", help="Download all essential subsets")
    args = parser.parse_args()

    if not (args.dns or args.esc50 or args.all):
        print("AEGIS-ANC Dataset Downloader")
        print("Usage:")
        print("  python download_datasets.py --esc50       # Download ESC-50 noise (~600MB)")
        print("  python download_datasets.py --dns         # Download essential DNS subsets")
        print("  python download_datasets.py --all         # Download all")
        return

    if args.esc50 or args.all:
        download_esc50()

    if args.dns or args.all:
        download_dns_essentials()

if __name__ == "__main__":
    main()
