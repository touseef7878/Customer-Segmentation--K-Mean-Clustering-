# ===============================================
# Customer Segmentation using KMeans & DBSCAN
# Dataset: Mall Customer Segmentation
# ===============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# ===============================================
# Step 1: Load dataset
# ===============================================
df = pd.read_csv("data/Mall_Customers.csv")

print("Dataset Preview:")
print(df.head())

# ===============================================
# Step 2: Select features for clustering
# Using Age, Annual Income, Spending Score
# ===============================================
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Standardize features for DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================================
# Step 3: Find optimal clusters with Elbow Method
# ===============================================
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K, inertia, "bx-")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.savefig("images/kmeans_elbow.png")
plt.close()

# ===============================================
# Step 4: Apply KMeans Clustering (K=5)
# ===============================================
kmeans = KMeans(n_clusters=5, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X)

# 2D Plot (Income vs Spending Score)
plt.figure(figsize=(6, 5))
sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["KMeans_Cluster"],
    palette="Set2",
    s=70
)
plt.title("KMeans Clusters (2D)")
plt.savefig("images/kmeans_clusters_2d.png")
plt.close()

# 3D Plot (Age, Income, Spending Score)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    df["Age"], df["Annual Income (k$)"], df["Spending Score (1-100)"],
    c=df["KMeans_Cluster"], cmap="Set2", s=60
)
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score")
plt.title("KMeans Clusters (3D)")
plt.savefig("images/kmeans_clusters_3d.png")
plt.close()

# ===============================================
# BONUS PART 1: Apply DBSCAN
# ===============================================
# DBSCAN works better with scaled data
dbscan = DBSCAN(eps=0.7, min_samples=5)  # tune eps for dataset
df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

plt.figure(figsize=(6, 5))
sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["DBSCAN_Cluster"],
    palette="tab10",
    s=70
)
plt.title("DBSCAN Clusters (2D)")
plt.savefig("images/dbscan_clusters.png")
plt.close()

# ===============================================
# BONUS PART 2: Average Spending per Cluster
# ===============================================
avg_spending = df.groupby("KMeans_Cluster")["Spending Score (1-100)"].mean()

plt.figure(figsize=(6, 4))
avg_spending.plot(kind="bar", color="skyblue")
plt.title("Average Spending per KMeans Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Spending Score")
plt.savefig("images/avg_spending.png")
plt.close()

print("\nAnalysis Completed ✅")
print("Cluster averages:")
print(avg_spending)
