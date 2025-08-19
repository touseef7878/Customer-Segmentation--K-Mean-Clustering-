# 🌲 Forest Cover Type Classification

## 📌 Project Overview

This project predicts **forest cover types** (different kinds of trees) based on **cartographic variables** such as elevation, slope, soil type, and wilderness area.

We apply and compare multiple models:

* **Random Forest**
* **XGBoost (Bonus)**

---

## ⚙️ Tools & Libraries

* Python 🐍
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

---

## 🔧 Requirements

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

---

## ▶️ How to Run

1. Clone the repository

   ```bash
   git clone https://github.com/yourusername/forest-cover-classification.git
   cd forest-cover-classification
   ```

2. Run the script

   ```bash
   python forest_cover_classification.py
   ```

3. Check the `images/` folder for saved evaluation plots.

---

## 📊 Results & Visualizations

### 1. Feature Importance (Random Forest)

![Feature Importance](images/feature_importance_RandomForest.png)

### 2. Confusion Matrix (Random Forest)

![RF Confusion](images/rf_confusion_matrix.png)

### 3. Confusion Matrix (XGBoost - Bonus)

![XGB Confusion](images/xgb_confusion_matrix.png)

---

## 📂 Project Structure

```
├── forest_cover_classification.py   # Main Python script
├── train.csv                        # Dataset
├── images/                          # Saved visualizations
│   ├── rf_feature_importance.png
│   ├── rf_confusion_matrix.png
│   └── xgb_confusion_matrix.png
└── README.md                        # Project documentation
```

---

## 🎯 Covered Topics

* Supervised Learning
* Classification
* Random Forests & XGBoost
* Model Evaluation

---

## 🚀 Bonus Extensions

* ✅ Model comparison (Random Forest vs XGBoost)
* ✅ Hyperparameter tuning

---
