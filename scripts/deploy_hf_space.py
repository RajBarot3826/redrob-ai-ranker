#!/usr/bin/env python3
"""
Deploy script — Copies all needed files to hf_space/ directory
for HuggingFace Spaces deployment.

Usage:
    python scripts/deploy_hf_space.py
    
Then push hf_space/ to your HuggingFace Space:
    cd hf_space
    git init
    git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker
    git add .
    git commit -m "Deploy Streamlit app"
    git push -u origin main
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
HF_DIR = ROOT / "hf_space"

def deploy():
    # Ensure hf_space directory exists
    HF_DIR.mkdir(exist_ok=True)

    # Copy Streamlit app
    shutil.copy2(ROOT / "app.py", HF_DIR / "app.py")

    # Copy src/ module
    src_dest = HF_DIR / "src"
    if src_dest.exists():
        shutil.rmtree(src_dest)
    shutil.copytree(ROOT / "src", src_dest)

    # Copy data/jd_query.txt
    data_dest = HF_DIR / "data"
    data_dest.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "data" / "jd_query.txt", data_dest / "jd_query.txt")

    print(f"Deployed to {HF_DIR}/")
    print("Files:")
    for f in sorted(HF_DIR.rglob("*")):
        if f.is_file() and "__pycache__" not in str(f):
            print(f"  {f.relative_to(HF_DIR)}")

    print("\nTo deploy to HuggingFace Spaces:")
    print("  cd hf_space")
    print("  git init")
    print("  git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/redrob-ranker")
    print("  git add .")
    print('  git commit -m "Deploy Streamlit app"')
    print("  git push -u origin main")

if __name__ == "__main__":
    deploy()
