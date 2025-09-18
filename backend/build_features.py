import os
import numpy as np
import pandas as pd
from PIL import Image
from feature_extractor import FeatureExtractor

# === CONFIG ===
DATASET_FOLDER = "static/images"            # folder where images were downloaded
CSV_FILE = "amazon_products.csv"
        # CSV path (relative to backend)

# === LOAD DATA ===
df = pd.read_csv(CSV_FILE)
fe = FeatureExtractor()

features = []
img_paths = []

for idx, row in df.iterrows():
    img_path = os.path.join(DATASET_FOLDER, f"{idx}.jpg")
    if not os.path.exists(img_path):
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        feat = fe.extract(img)
        features.append(feat)
        img_paths.append(img_path)
        print(f"✅ Processed row {idx}")
    except Exception as e:
        print(f"⚠️ Skipped row {idx}: {e}")

features = np.array(features)
img_paths = np.array(img_paths)

np.save("features.npy", features)
np.save("img_paths.npy", img_paths)

print(f"\n🎉 Done! Extracted {features.shape[0]} features")
