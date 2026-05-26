import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# 2 Load CSV Dataset

df = pd.read_csv(r"c:\Users\adakr\OneDrive\Desktop\student_scores.csv")
print("Dataset:\n", df)

# 3 Exploratory Data Analysis (EDA)

print("\nStatistical Summary:\n", df.describe())

sns.pairplot(df,
             x_vars=['Hours_Study', 'Attendance', 'Hours_Sleep'],
             y_vars='Score',
             height=4)

plt.show()

# 4 Split Data into Features and Target

X = df[['Hours_Study', 'Attendance', 'Hours_Sleep']]   # independent variables
y = df['Score']                                        # dependent variable