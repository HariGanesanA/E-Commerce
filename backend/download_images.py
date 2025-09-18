import pandas as pd
import requests
import os
from tqdm import tqdm

# === CONFIG ===
csv_file ="amazon_products.csv"
       # path to your CSV (relative to backend)
output_folder = "static/images"            # where images will be saved
os.makedirs(output_folder, exist_ok=True)

# === LOAD CSV ===
df = pd.read_csv(csv_file)
print("📑 Columns found in CSV:", df.columns.tolist())

# === Detect URL column ===
url_col = None
for c in df.columns:
    if df[c].astype(str).str.startswith("http").any():
        url_col = c
        break

if url_col is None:
    raise ValueError("❌ No column with image URLs found in your CSV!")

print(f"✅ Using URL column: {url_col}")

# === Download Images ===
for idx, row in tqdm(df.iterrows(), total=len(df)):
    url = str(row[url_col])
    if not url.startswith("http"):
        continue

    fname = f"{idx}.jpg"  # <-- use row index
    out_path = os.path.join(output_folder, fname)

    if os.path.exists(out_path):
        continue

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        print(f"⚠️ Skipping {url}: {e}")

print("🎉 Done downloading images!")
print(f"📂 Saved to: {output_folder}")
