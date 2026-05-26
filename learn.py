import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


data = {
    'area': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    'price': [50, 65, 80, 95, 110, 125, 140, 155, 170]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


plt.scatter(df['area'], df['price'], color='blue', marker='o')
plt.title("House Price vs Area")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (Lakhs)")

plt.show()
# ------------------------------
# Step 3: Prepare and Split Data
# ------------------------------

X = df[['area']]   # Input feature
y = df['price']    # Output target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)

# Step 4: Train Linear Regression Model
# ------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Step 5: Make Predictions
# ------------------------------

y_pred = model.predict(X_test)

# Compare Actual vs Predicted
compare_df = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})

print("\nActual vs Predicted Prices:")
print(compare_df)

# ------------------------------
# Step 6: Visualize Regression Line
# ------------------------------

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

plt.title("Linear Regression Line - House Price Prediction")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (in Lakhs)")
plt.legend()
plt.show()

# ------------------------------
# Step 7: Model Evaluation
# ------------------------------

print("\nModel Evaluation:")

print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")


# ------------------------------
# Step 8: Predict New Value
# ------------------------------

new_area = np.array([[4900]])

predicted_price = model.predict(new_area)

print(f"\nPredicted price for a house with {new_area[0][0]} sq ft area = {predicted_price[0]:.2f} Lakhs")