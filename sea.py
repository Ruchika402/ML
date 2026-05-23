import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

tips = sns.load_dataset("tips")
print(tips.head(10))
sns.scatterplot(x="total_bill", y ="tip",data = tips)
plt.title("Scatter plot")
plt.show()