import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'Study_Hours': [1,2,3,4,5,6,7,8,9],
    'Marks': [20,30,40,50,60,70,80,90,93]
}

df = pd.DataFrame(data)

print("DATASET")
print(df)

df['Pass'] = np.where(df['Marks'] >= 50, 1, 0)

plt.scatter(df['Study_Hours'], df['Marks'], color='blue')
plt.xlabel('Study Hours')
plt.ylabel('Marks Scored')
plt.title('Study Hours vs Marks')
plt.grid(True)
plt.show()

X = df[['Study_Hours']]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted values:", y_pred)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))





from sklearn.linear_model import LogisticRegression 
import numpy as np
x = np.array(
    [
        [10],[20],[35],[50],[70]
    ]
)

y = np.array(
    [0,0,0,1,1]
)
#craete model
model = LogisticRegression()
#train model
model.fit(x,y)
#new player
new_player_runs = np.array([[45]])
#pedictioniction
prediction = model.predict(new_player_runs)
#probability
probability = model.predict_proba(new_player_runs)
print(new_player_runs[0][0])
print(probability[0][1] * 100,"%")