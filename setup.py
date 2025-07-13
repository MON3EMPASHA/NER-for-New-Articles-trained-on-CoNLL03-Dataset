import subprocess
import sys
import os

def download_spacy_models():
    """Download required spaCy models"""
    models = ["en_core_web_sm", "en_core_web_lg"]
    
    for model in models:
        try:
            print(f"Downloading {model}...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
            print(f"Successfully downloaded {model}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {model}: {e}")
            return False
    return True

if __name__ == "__main__":
    print("Setting up spaCy models...")
    if download_spacy_models():
        print("Setup completed successfully!")
    else:
        print("Setup failed!")
        sys.exit(1) 