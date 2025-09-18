import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from feature_extractor import FeatureExtractor
from PIL import Image
from numpy.linalg import norm

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
CSV_FILE = os.path.join(BASE_DIR, "amazon_products.csv")
FEATURES_FILE = os.path.join(BASE_DIR, "features.npy")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------------
df = pd.read_csv(CSV_FILE)

# rename CSV columns -> names we use in the app
df = df.rename(columns={
    "image": "image_path",
    "discount_price": "price",
    "name": "title"
})

features = np.load(FEATURES_FILE, allow_pickle=True)

# -----------------------------------------------------------------------------
# FLASK APP
# -----------------------------------------------------------------------------
app = Flask(__name__)
fe = FeatureExtractor()


# -----------------------------------------------------------------------------
# Helper: find similar images
# -----------------------------------------------------------------------------
def recommend(query_img_path, top_k=5):
    img = Image.open(query_img_path).convert("RGB")
    query_feat = fe.extract(img)
    dists = norm(features - query_feat, axis=1)
    ids = np.argsort(dists)[:top_k]

    results = []
    for idx in ids:
        row = df.iloc[idx]
        results.append({
            "image": row["image_path"],
            "price": row["price"],
            "title": row["title"]
        })
    return results


# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(save_path)

    recs = recommend(save_path, top_k=5)
    return jsonify({
        "uploaded_image": f"/static/uploads/{f.filename}",
        "recommendations": recs
    })


if __name__ == "__main__":
    app.run(debug=True)
