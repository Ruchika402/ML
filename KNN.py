# =========================================
# KNN Classification on Iris Dataset
# =========================================

# ---------------------------------
# Step 1: Import Required Libraries
# ---------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------
# Step 2: Load Iris Dataset
# ---------------------------------

iris = load_iris()

X = iris.data
y = iris.target

# Create DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)

# Add target column
df['Target'] = y

print("========== DATASET PREVIEW ==========\n")
print(df.head())

# ---------------------------------
# Step 3: Train-Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# Step 4: Feature Scaling
# ---------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ---------------------------------
# Step 5: Try Multiple K Values
# ---------------------------------

accuracy_list = []

print("\n========== K VALUE ACCURACY ==========\n")

for k in range(1, 11):

    knn = KNeighborsClassifier(n_neighbors=k)

    # Train Model
    knn.fit(X_train_scaled, y_train)

    # Prediction
    y_pred = knn.predict(X_test_scaled)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    accuracy_list.append(acc)

    print(f"K = {k} --> Accuracy = {acc:.4f}")

# ---------------------------------
# Step 6: Select Best K Value
# ---------------------------------

best_k = accuracy_list.index(max(accuracy_list)) + 1

print("\nBest K Value =", best_k)

# ---------------------------------
# Step 7: Train Final KNN Model
# ---------------------------------

knn_final = KNeighborsClassifier(n_neighbors=best_k)

knn_final.fit(X_train_scaled, y_train)

# ---------------------------------
# Step 8: Final Prediction
# ---------------------------------

y_final_pred = knn_final.predict(X_test_scaled)

# ---------------------------------
# Step 9: Final Evaluation
# ---------------------------------

final_accuracy = accuracy_score(y_test, y_final_pred)

print("\n========== FINAL RESULT ==========\n")

print("Final Accuracy =", final_accuracy)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_final_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_final_pred))

# ---------------------------------
# Step 10: Accuracy Graph
# ---------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    accuracy_list,
    marker='o'
)

plt.xlabel("K Value")

plt.ylabel("Accuracy")

plt.title("KNN Accuracy for Different K Values")

plt.grid(True)

plt.show()
# ---------------------------------
# Step 11: Flower Petal Visualization
# ---------------------------------

# Create DataFrame with class names
df = pd.DataFrame(X, columns=iris.feature_names)
df['Target'] = y
df['Flower'] = [iris.target_names[i] for i in y]

plt.figure(figsize=(10, 6))

for flower in iris.target_names:
    subset = df[df['Flower'] == flower]

    plt.scatter(
        subset['petal length (cm)'],
        subset['petal width (cm)'],
        label=flower,
        s=80
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Flower Petal Visualization")
plt.legend()
plt.grid(True)

plt.show()