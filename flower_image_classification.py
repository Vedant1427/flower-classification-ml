"""
Flower Classification Using Machine Learning (Image Dataset)
=============================================================
Objective : Classify flower photos into categories (daisy, rose, tulip, ...)
            using classic machine learning on features extracted from images.
Libraries : numpy, pandas, matplotlib, scikit-learn  (+ Pillow to open images)

Dataset   : any Kaggle / TensorFlow flower dataset stored as ONE FOLDER PER FLOWER TYPE:

                flower_photos/
                    daisy/       img1.jpg, img2.jpg ...
                    dandelion/
                    roses/
                    sunflowers/
                    tulips/

Usage
-----
    python flower_image_classification.py path/to/flower_photos
    python flower_image_classification.py path/to/flower_photos --max-images 3000
    python flower_image_classification.py --predict my_flower.jpg      # after training

Outputs (saved in ./outputs):
    01_class_distribution.png, 02_sample_images.png, 03_model_comparison.png,
    04_confusion_matrix.png, 05_per_class_scores.png, 06_misclassified.png,
    07_learning_curve.png, flower_model.joblib, results.txt
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # saves plots to files; remove this line to show pop-up windows
import matplotlib.pyplot as plt
import joblib
from PIL import Image

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_fscore_support,
)

RANDOM_STATE = 42
IMG_SIZE = 64                      # every image is resized to 64 x 64
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
OUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUT_DIR, "flower_model.joblib")
np.random.seed(RANDOM_STATE)

report_lines = []


def log(text=""):
    """Print and also keep the line for results.txt."""
    print(text)
    report_lines.append(str(text))


# ---------------------------------------------------------------------------
# Helpers: finding the dataset, loading images, extracting features
# ---------------------------------------------------------------------------
def find_dataset_root(start):
    """Return (root, class_folders): the first folder that has 2+ sub-folders of images.
    This lets you point at the unzipped download even if it is nested
    (e.g. archive/flowers/flowers/daisy)."""
    for root, dirs, _ in os.walk(start):
        good = []
        for d in sorted(dirs):
            try:
                if any(f.lower().endswith(IMG_EXT) for f in os.listdir(os.path.join(root, d))):
                    good.append(d)
            except OSError:
                pass
        if len(good) >= 2:
            return root, good
    return None, []


def extract_features(img):
    """Turn one 64x64 RGB PIL image into a numeric feature vector (960 values).

    1. Colour histogram (HSV)      -> 48 values   : which colours the flower has
    2. Edge/shape histogram (HOG-lite) -> 144 values : petal shapes and edges
    3. Tiny 16x16 RGB thumbnail    -> 768 values  : overall layout of colour
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # 1. colour histogram in HSV space
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    color = []
    for c in range(3):
        h, _ = np.histogram(hsv[..., c], bins=16, range=(0, 255))
        color.append(h / max(h.sum(), 1))

    # 2. gradient-orientation histogram per 4x4 grid cell
    gray = arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx) % np.pi
    cells, bins = 4, 9
    ch = IMG_SIZE // cells
    hog = []
    for i in range(cells):
        for j in range(cells):
            m = mag[i * ch:(i + 1) * ch, j * ch:(j + 1) * ch].ravel()
            a = ang[i * ch:(i + 1) * ch, j * ch:(j + 1) * ch].ravel()
            h, _ = np.histogram(a, bins=bins, range=(0, np.pi), weights=m)
            hog.append(h / (np.linalg.norm(h) + 1e-6))

    # 3. tiny thumbnail
    tiny = np.asarray(img.resize((16, 16)), dtype=np.float32).ravel() / 255.0

    return np.concatenate([*color, *hog, tiny])


def load_and_prepare(path):
    return Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))


def predict_flower(image_path):
    """Predict the flower type of a single image using the saved model."""
    saved = joblib.load(MODEL_PATH)
    feat = extract_features(load_and_prepare(image_path)).reshape(1, -1)
    model = saved["model"]
    pred = model.predict(feat)[0]
    out = f"Predicted flower: {saved['class_names'][pred]}"
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat)[0]
        out += f"  (confidence {proba[pred]:.1%})"
    return out


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Flower image classification")
parser.add_argument("data_dir", nargs="?", default="flower_photos",
                    help="folder with one sub-folder per flower type")
parser.add_argument("--max-images", type=int, default=3000,
                    help="maximum number of images to use (default 3000)")
parser.add_argument("--predict", metavar="IMAGE",
                    help="predict one image with the already-trained model and exit")
args = parser.parse_args()

if args.predict:
    print(predict_flower(args.predict))
    sys.exit(0)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
log("=" * 60)
log("1. LOADING DATASET")
log("=" * 60)
root, class_names = find_dataset_root(args.data_dir)
if root is None:
    sys.exit(f"Could not find class folders with images inside '{args.data_dir}'.\n"
             "Expected: one sub-folder per flower type, each containing images.")
log(f"Dataset folder : {root}")
log(f"Flower classes : {class_names}")

rng = np.random.RandomState(RANDOM_STATE)
per_class_cap = int(np.ceil(args.max_images / len(class_names)))
paths, labels = [], []
for idx, cname in enumerate(class_names):
    files = sorted(f for f in os.listdir(os.path.join(root, cname))
                   if f.lower().endswith(IMG_EXT))
    rng.shuffle(files)
    for f in files[:per_class_cap]:
        paths.append(os.path.join(root, cname, f))
        labels.append(idx)

log(f"Loading up to {args.max_images} images, resizing to {IMG_SIZE}x{IMG_SIZE} ...")
images, feats, y_list, bad = [], [], [], 0
for p, lab in zip(paths, labels):
    try:
        im = load_and_prepare(p)
    except Exception:
        bad += 1          # skip unreadable / corrupted files
        continue
    images.append(np.asarray(im, dtype=np.uint8))
    feats.append(extract_features(im))
    y_list.append(lab)

images = np.array(images)
X = np.array(feats, dtype=np.float32)
y = np.array(y_list)
log(f"Images loaded  : {len(y)}   (skipped unreadable: {bad})")
log(f"Feature matrix : {X.shape[0]} samples x {X.shape[1]} features")

counts = pd.Series(y).map(dict(enumerate(class_names))).value_counts().reindex(class_names)
log("\nImages per class:")
log(counts.to_string())

# ---------------------------------------------------------------------------
# 2. Exploratory plots
# ---------------------------------------------------------------------------
palette = plt.cm.tab10(np.linspace(0, 1, 10))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(counts.index, counts.values, color=palette[:len(counts)])
ax.set_title("Images per Flower Class")
ax.set_ylabel("Number of images")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=150)
plt.close()

n_show = 6
fig, axes = plt.subplots(len(class_names), n_show, figsize=(2 * n_show, 2 * len(class_names)))
axes = np.atleast_2d(axes)
for r, cname in enumerate(class_names):
    idxs = np.where(y == r)[0][:n_show]
    for c in range(n_show):
        axes[r, c].axis("off")
        if c < len(idxs):
            axes[r, c].imshow(images[idxs[c]])
        if c == 0:
            axes[r, c].set_title(cname, loc="left", fontsize=10)
plt.suptitle("Sample Images from Each Class")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_sample_images.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
idx_all = np.arange(len(y))
idx_train, idx_test = train_test_split(
    idx_all, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

log("\n" + "=" * 60)
log("2. TRAIN / TEST SPLIT")
log("=" * 60)
log(f"Train: {len(idx_train)} images | Test: {len(idx_test)} images (stratified 80/20)")

# ---------------------------------------------------------------------------
# 4. Compare models with cross-validation
# ---------------------------------------------------------------------------
# number of PCA components: up to 100, but never more than the smallest
# training subset used in cross-validation / the learning curve can support
n_pca = max(2, min(100, X_train.shape[1], int(len(X_train) * 0.13)))
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(50, n_pca), random_state=RANDOM_STATE)),
        ("clf", KNeighborsClassifier(n_neighbors=7))]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)),
        ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE))]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
log("\n" + "=" * 60)
log("3. MODEL COMPARISON (3-fold cross-validation on training set)")
log("=" * 60)
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    log(f"{name:<22} mean acc = {scores.mean():.4f}  (+/- {scores.std():.4f})")

fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(cv_results)
means = [cv_results[n].mean() for n in names]
stds = [cv_results[n].std() for n in names]
ax.barh(names, means, xerr=stds, color="#4C72B0", capsize=4)
for i, m in enumerate(means):
    ax.text(m + 0.005, i, f"{m:.1%}", va="center", fontsize=9)
ax.set_xlim(0, 1)
ax.set_xlabel("Cross-validated accuracy")
ax.set_title("Model Comparison")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. Hyperparameter tuning (SVM)
# ---------------------------------------------------------------------------
log("\n" + "=" * 60)
log("4. HYPERPARAMETER TUNING (SVM, GridSearchCV)")
log("=" * 60)
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(random_state=RANDOM_STATE)),
    ("clf", SVC(random_state=RANDOM_STATE)),
])
param_grid = {
    "pca__n_components": sorted({min(50, n_pca), n_pca}),
    "clf__C": [1, 10, 50],
    "clf__gamma": ["scale", 0.001, 0.01],
}
grid = GridSearchCV(svm_pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
log(f"Best params : {grid.best_params_}")
log(f"Best CV acc : {grid.best_score_:.4f}")
best_model = grid.best_estimator_

# ---------------------------------------------------------------------------
# 6. Final evaluation on the test set
# ---------------------------------------------------------------------------
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
log("\n" + "=" * 60)
log("5. TEST SET EVALUATION")
log("=" * 60)
log(f"Test accuracy: {test_acc:.4f}")
log("\nClassification report:")
log(classification_report(y_test, y_pred, target_names=class_names, digits=3))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6.5, 6))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
    ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
ax.set_title(f"Confusion Matrix (accuracy = {test_acc:.2%})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_confusion_matrix.png", dpi=150)
plt.close()

prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
scores_df = pd.DataFrame({"precision": prec, "recall": rec, "f1": f1}, index=class_names)
ax = scores_df.plot(kind="bar", figsize=(8, 4.5), rot=20, ylim=(0, 1.05))
ax.set_title("Per-class Precision / Recall / F1")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_per_class_scores.png", dpi=150)
plt.close()

# Misclassified examples
wrong = np.where(y_pred != y_test)[0]
if len(wrong):
    show = wrong[:10]
    cols = min(5, len(show))
    rows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, w in zip(axes, show):
        ax.imshow(images[idx_test[w]])
        ax.set_title(f"true: {class_names[y_test[w]]}\npred: {class_names[y_pred[w]]}",
                     fontsize=8, color="crimson")
    plt.suptitle(f"Misclassified Test Images ({len(wrong)} of {len(y_test)})")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_misclassified.png", dpi=130)
    plt.close()
log(f"Misclassified test images: {len(wrong)} of {len(y_test)}")

# ---------------------------------------------------------------------------
# 7. Learning curve (does more data help?)
# ---------------------------------------------------------------------------
sizes, tr_scores, va_scores = learning_curve(
    best_model, X_train, y_train, cv=cv, n_jobs=-1,
    train_sizes=np.linspace(0.2, 1.0, 5), scoring="accuracy",
    random_state=RANDOM_STATE, shuffle=True)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(sizes, tr_scores.mean(axis=1), "o-", label="Training accuracy")
ax.plot(sizes, va_scores.mean(axis=1), "o-", label="Validation accuracy")
ax.set_xlabel("Number of training images")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_learning_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. Save model
# ---------------------------------------------------------------------------
joblib.dump({"model": best_model, "class_names": class_names}, MODEL_PATH)

with open(f"{OUT_DIR}/results.txt", "w") as f:
    f.write("\n".join(report_lines))

log("\n" + "=" * 60)
log(f"Done. Plots, model and results saved in ./{OUT_DIR}/")
log("Predict a new photo with:")
log("    python flower_image_classification.py --predict your_flower.jpg")
