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

### 1. Elbow Method for Optimal K
![Elbow Method](images/kmeans_elbow.png)

### 2. KMeans Clusters (2D: Income vs Spending Score)
![KMeans 2D](images/kmeans_clusters_2d.png)

### 3. KMeans Clusters (3D: Age, Income, Spending Score)
![KMeans 3D](images/kmeans_clusters_3d.png)

### 4. DBSCAN Clusters (2D: Income vs Spending Score)
![DBSCAN Clusters](images/dbscan_clusters.png)

### 5. Average Spending per KMeans Cluster
![Avg Spending](images/avg_spending.png)

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
