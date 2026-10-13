# 🌸 Flower Classification Using Machine Learning

A Python-based Machine Learning project that classifies flower images into different categories using image processing, feature extraction, and supervised learning algorithms.

## 📌 Project Overview

Flower Classification is an image classification project that identifies the category of a flower from its image.

This project extracts visual features from flower images, trains multiple machine learning models, compares their performance, and evaluates the selected model using standard classification metrics.

**Project Type:** Machine Learning / Image Classification  
**Programming Language:** Python  
**Models:** Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Random Forest

## 🎯 Objectives

- Classify flower images into different categories.
- Extract useful colour, edge, and image features.
- Train and compare multiple machine learning algorithms.
- Evaluate performance using accuracy, precision, recall, and F1-score.
- Visualize model performance through graphs.
- Save the trained model for future predictions.

## 🛠️ Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical calculations and array operations |
| Pandas | Organizing and handling tabular results |
| Matplotlib | Creating graphs and visualizations |
| Scikit-learn | Training and evaluating machine learning models |
| Pillow (PIL) | Loading, resizing, and processing images |
| Joblib | Saving and loading the trained model |

## 🌼 Flower Categories

The real flower dataset contains five categories:

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

The project also includes a synthetic demonstration dataset. These generated images are used to test the program and are not real flower photographs.

## ⚙️ Project Workflow

### 1. Dataset Collection

The project loads flower images organized into folders according to their categories. The included dataset downloader retrieves the real flower dataset.

### 2. Image Preprocessing

Images are loaded and resized to 64 × 64 pixels to standardize their dimensions and make feature extraction consistent.

### 3. Feature Extraction

The project converts images into numerical features that machine learning algorithms can understand.

The extracted features include:

- **HSV colour histogram:** Represents the distribution of colours in an image.
- **Edge and shape histogram:** Captures information about edges and visual structure.
- **Image thumbnail features:** Preserve simplified visual and spatial information.

These features are combined into a vector of 960 numerical values per image.

### 4. Model Training

Four supervised machine learning algorithms are compared.

- **Logistic Regression:** Learns relationships between image features and flower categories.
- **K-Nearest Neighbors (KNN):** Predicts a category based on similar training examples.
- **Support Vector Machine (SVM):** Learns decision boundaries to separate flower categories.
- **Random Forest:** Combines multiple decision trees to make predictions.

The project uses three-fold cross-validation to compare models and GridSearchCV to tune the SVM.

### 5. Model Evaluation

The selected model is evaluated on a held-out test dataset. Its performance is measured using accuracy, precision, recall, F1-score, and a confusion matrix.

### 6. Model Saving

The selected trained model and its class names are saved using Joblib in `outputs/flower_model.joblib`.

### 7. Flower Prediction

After training, the saved model can be used to predict the category of a new flower image.

## 📊 Model Outputs and Visualizations

### 1. Model Comparison

**Output:** `outputs/03_model_comparison.png`

![Model Comparison](outputs/03_model_comparison.png)

This graph compares the cross-validation accuracy of Logistic Regression, KNN, SVM, and Random Forest.

**Interpretation:**
- Higher accuracy indicates better average classification performance during cross-validation.
- The comparison helps identify promising algorithms.
- The final test-set evaluation provides a separate measure of model performance.

### 2. Confusion Matrix

**Output:** `outputs/04_confusion_matrix.png`

![Confusion Matrix](outputs/04_confusion_matrix.png)

The confusion matrix compares the actual flower categories with the categories predicted by the selected model.

**Interpretation:**
- The main diagonal represents correct predictions.
- Values outside the diagonal represent incorrect predictions.
- It helps identify which flower categories the model confuses.

### 3. Precision, Recall and F1-Score

**Output:** `outputs/05_per_class_scores.png`

![Precision Recall and F1 Score](outputs/05_per_class_scores.png)

This graph compares precision, recall, and F1-score for each flower category.

- **Precision:** How many predictions for a category were correct.
- **Recall:** How many actual images of a category were correctly identified.
- **F1-score:** The harmonic mean of precision and recall.

Values closer to 1.0 generally indicate better performance.

### 4. Learning Curve

**Output:** `outputs/07_learning_curve.png`

![Learning Curve](outputs/07_learning_curve.png)

The learning curve shows how training and validation performance change as the amount of training data increases.

**Interpretation:**
- A large gap between training and validation performance may indicate overfitting.
- Low performance on both curves may indicate underfitting.
- Improving validation performance as more data is used may suggest that additional training data could help.

## 📈 Evaluation Metrics

| Metric | Explanation |
|---|---|
| Accuracy | Percentage of test images classified correctly |
| Precision | Reliability of predictions for a particular class |
| Recall | Percentage of actual class images correctly identified |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Shows correct and incorrect predictions |
| Cross-validation Accuracy | Average performance across validation folds |

### Understanding the Results

The synthetic demonstration dataset is designed to test the program workflow. Results from synthetic images must not be treated as evidence of real-world flower recognition accuracy.

For meaningful evaluation, run the project on the real flower dataset and use the actual scores recorded in `outputs/results.txt`.

## ⚠️ Limitations

- The project uses handcrafted features rather than a deep learning architecture.
- Lighting, backgrounds, image quality, and similarities between flowers can affect predictions.
- Synthetic demonstration results do not establish real-world performance.
- Model performance can vary depending on the dataset and training configuration.

## 🚀 Future Improvements

- Implement a Convolutional Neural Network (CNN).
- Explore transfer learning using pretrained image classification models.
- Develop a web application for uploading flower images.
- Display prediction confidence and alternative categories.
- Expand the dataset and evaluate performance on unseen real-world photographs.

## 👨‍💻 Author

**Vedant Gothwad**

GitHub: [@Vedant1427](https://github.com/Vedant1427)

**Project Repository:** [Flower Classification Using Machine Learning](https://github.com/Vedant1427/flower-classification-ml)

## 📄 License

No license has been specified yet. Add an appropriate open-source license if you want others to reuse, modify, or distribute this project.

---

*Built with Python and Machine Learning for image classification and practical learning.*