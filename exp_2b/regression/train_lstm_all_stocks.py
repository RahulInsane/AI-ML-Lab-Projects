import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATASET_FOLDER = "regression/dataset"
MODELS_FOLDER = "regression/models"

TIME_STEP = 60
EPOCHS = 5          # You said you have time ✅
BATCH_SIZE = 32

os.makedirs(MODELS_FOLDER, exist_ok=True)

# Load all stock CSV files
stock_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]
stock_files.sort()

print("✅ Total stock files found:", len(stock_files))

trained = 0
skipped = 0

for stock_file in stock_files:
    try:
        file_path = os.path.join(DATASET_FOLDER, stock_file)
        df = pd.read_csv(file_path)

        if "Close" not in df.columns:
            print(f"❌ Skipping {stock_file}: Close column not found")
            skipped += 1
            continue

        data = df[["Close"]].values

        if len(data) < TIME_STEP + 10:
            print(f"⚠️ Skipping {stock_file}: Not enough rows ({len(data)})")
            skipped += 1
            continue

        # Scale data for this stock
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

        # Train-test split (80-20)
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

        print(f"\n✅ Training LSTM for {stock_file}...")
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

        # Save model + scaler
        stock_name = stock_file.replace(".csv", "")
        model_path = os.path.join(MODELS_FOLDER, f"{stock_name}_lstm.h5")
        scaler_path = os.path.join(MODELS_FOLDER, f"{stock_name}_scaler.pkl")

        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        print(f"✅ Saved: {model_path}")
        print(f"✅ Saved: {scaler_path}")

        trained += 1

    except Exception as e:
        print(f"❌ Error training {stock_file}: {e}")
        skipped += 1

print("\n==============================")
print("✅ Batch Training Completed")
print("✅ Trained:", trained)
print("⚠️ Skipped:", skipped)
