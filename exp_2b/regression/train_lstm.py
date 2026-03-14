import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("regression/stock.csv")
data = df[["Close"]].values

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

TIME_STEP = 60
X, y = create_sequences(scaled_data, TIME_STEP)

# Reshape for LSTM: (samples, time_step, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-Test split (80-20)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

print("✅ Training LSTM...")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Predict
y_pred_scaled = model.predict(X_test)

# Convert back to original values
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred_scaled)

# Calculate error
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)

print("✅ LSTM Training Done")
print("MSE:", mse)
print("RMSE:", rmse)

# Save model + scaler
model.save("regression/models/lstm_model.h5")
joblib.dump(scaler, "regression/models/scaler.pkl")

print("✅ Saved model: regression/models/lstm_model.h5")
print("✅ Saved scaler: regression/models/scaler.pkl")
