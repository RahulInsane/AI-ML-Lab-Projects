import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("regression/stock.csv")

# Use Close price
close_prices = df["Close"].values.reshape(-1, 1)

# X = index
X = np.arange(len(close_prices)).reshape(-1, 1)
y = close_prices

# Train (first 80%) Test (last 20%)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Error
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)

print("✅ Linear Regression done")
print("MSE:", mse)
print("RMSE:", rmse)

# Save model
joblib.dump(model, "regression/models/linear_regression.pkl")
print("✅ Saved model: regression/models/linear_regression.pkl")
