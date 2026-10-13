# Flower Classification Using Machine Learning

**Objective:** Build an ML model that classifies flowers into different categories from their images.

**Libraries:** Python, NumPy, Pandas, Matplotlib, scikit-learn (plus Pillow to open image files).

## What is in this folder

```
flower_classification_project/
├── flower_image_classification.py   main project script (train, evaluate, predict)
├── download_dataset.py              downloads the real flower dataset (~3,670 photos)
├── requirements.txt                 libraries to install
├── README.md                        this file
└── data/
    └── demo_flowers/                600 SYNTHETIC drawn "flowers", only to test your setup
```

> **Important:** `data/demo_flowers` is computer-generated test data, not real photos.
> Use it only to check that everything runs. For your actual project and report,
> download the real dataset (Step 3).

## Setup (Windows; on Mac/Linux use `python3` / `pip3`)

1. Install Python 3.9 or newer from python.org (tick **Add Python to PATH**).
2. Open a terminal inside this folder (File Explorer address bar, type `cmd`, Enter).
3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```

## Run

**Step 1: quick check with the demo data (about 15 seconds)**
```
python flower_image_classification.py data/demo_flowers
```

**Step 2: (optional) look at the results**
Open the new `outputs` folder: plots (.png), `results.txt` and the saved model.

**Step 3: download the real dataset and train on it**
```
python download_dataset.py
python flower_image_classification.py data/flower_photos --max-images 3000
```
Training on 3,000 real photos takes a few minutes on a normal laptop.

**Step 4: predict a new flower photo**
```
python flower_image_classification.py --predict path\to\my_flower.jpg
```

## Using a Kaggle dataset instead

Download a flower dataset from kaggle.com, unzip it into the `data` folder, and run:
```
python flower_image_classification.py data\<unzipped folder name>
```
The dataset must have one sub-folder per flower type, each containing that type's images.
The script finds the class folders automatically, even inside nested folders.

## How the project works

1. **Load data**: reads up to 3,000 images (balanced across classes), resizes to 64x64.
2. **Feature extraction**: colour histogram (HSV), edge/shape histogram and a small
   thumbnail turn each photo into 960 numbers.
3. **Model comparison**: Logistic Regression, KNN, SVM and Random Forest, compared with
   3-fold cross-validation.
4. **Tuning**: GridSearchCV tunes the SVM.
5. **Evaluation**: accuracy, classification report, confusion matrix, per-class scores,
   misclassified examples and a learning curve on a held-out 20% test set.
6. **Prediction**: the best model is saved and can classify any new photo.

## Output files (in `outputs/`)

| File | Shows |
|---|---|
| 01_class_distribution.png | images per flower class |
| 02_sample_images.png | example photos from each class |
| 03_model_comparison.png | cross-validated accuracy of each model |
| 04_confusion_matrix.png | which flowers get confused with which |
| 05_per_class_scores.png | precision / recall / F1 per class |
| 06_misclassified.png | test photos the model got wrong (only if there are errors) |
| 07_learning_curve.png | whether more training data would help |
| flower_model.joblib | the saved trained model |
| results.txt | all printed results |

## Expected results

On real photos, classic ML on hand-made features usually reaches roughly 50-70%
accuracy. That is normal: flower photos vary a lot in lighting, angle and background,
and deep learning (CNNs) is the usual way to go higher. Your own accuracy figures will
appear in `outputs/results.txt`; use those numbers in your report.

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python with "Add to PATH" ticked, or try `py` instead of `python` |
| `pip` not found | Use `python -m pip install -r requirements.txt` |
| `Could not find class folders` | Check the path; the folder must contain one sub-folder per flower type |
| Download fails | Check your internet, or download a dataset manually from kaggle.com |
| Training is slow | Lower the image count: `--max-images 1500` |
