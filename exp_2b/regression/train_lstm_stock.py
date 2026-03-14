import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATASET_FOLDER = "regression/dataset"
MODELS_FOLDER = "regression/models"
TIME_STEP = 60

os.makedirs(MODELS_FOLDER, exist_ok=True)

# ✅ Change stock file name here every time you want to train
STOCK_FILE = "AAPL.csv"

file_path = os.path.join(DATASET_FOLDER, STOCK_FILE)
df = pd.read_csv(file_path)

if "Close" not in df.columns:
    print(f"❌ Close column not found in {STOCK_FILE}")
    exit()

data = df[["Close"]].values

# Scaling
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Create sequences
X, y = [], []
for i in range(TIME_STEP, len(scaled)):
    X.append(scaled[i - TIME_STEP:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

print(f"\n✅ Training LSTM for {STOCK_FILE} ...")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Predict
y_pred_scaled = model.predict(X_test)

# Inverse scale
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred_scaled)

mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)

print(f"\n✅ LSTM Done for {STOCK_FILE}")
print("MSE:", mse)
print("RMSE:", rmse)

# Save with stock name
stock_name = STOCK_FILE.replace(".csv", "")
model_path = os.path.join(MODELS_FOLDER, f"{stock_name}_lstm.h5")
scaler_path = os.path.join(MODELS_FOLDER, f"{stock_name}_scaler.pkl")

model.save(model_path)
joblib.dump(scaler, scaler_path)

print("\n✅ Saved model:", model_path)
print("✅ Saved scaler:", scaler_path)
