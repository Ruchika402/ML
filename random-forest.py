# Step 1: Import required libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the Titanic dataset
data = sns.load_dataset('titanic')

print("First 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# Step 3: Select useful columns
data = data[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]

# Remove missing values
data.dropna(inplace=True)

# Step 4: Encode categorical columns
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['embarked'] = data['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Step 5: Split into features (X) and target (y)
X = data.drop('survived', axis=1)
y = data['survived']

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 7: Create and train the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    oob_score=True
)

model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("Model Evaluation Results:")
print("-----------------------------------")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Out-of-Bag Score (OOB):", model.oob_score_)

# Step 10: Feature Importance
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

print("\nFeature Importance:\n")
print(feature_importance.sort_values(ascending=False))

# Step 11: Plot Feature Importance
plt.figure(figsize=(8, 5))
feature_importance.sort_values().plot(kind='barh', color='skyblue')

plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()