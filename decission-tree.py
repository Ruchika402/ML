import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree


# Load dataset
titanic = sns.load_dataset("titanic")

print("First 5 rows:")
print(titanic.head())

print("\nDataset Info:")
print(titanic.info())

print("\nMissing Values:")
print(titanic.isnull().sum())


# Handle missing values
titanic['age'].fillna(titanic['age'].mean(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)


# Convert categorical columns into numeric
titanic = pd.get_dummies(
    titanic,
    columns=['sex', 'embarked', 'class'],
    drop_first=True
)


# Define features (X) and target (y)
X = titanic[
    ['age', 'fare', 'sex_male',
     'embarked_Q', 'embarked_S',
     'class_Second', 'class_Third']
]

y = titanic['survived']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Decision Tree model
dt = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

dt.fit(X_train, y_train)


# Make predictions
y_pred = dt.predict(X_test)


# Evaluate model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Visualize Decision Tree
plt.figure(figsize=(18, 10))

tree.plot_tree(
    dt,
    feature_names=X.columns,
    class_names=['Not Survived', 'Survived'],
    filled=True
)

plt.show()
