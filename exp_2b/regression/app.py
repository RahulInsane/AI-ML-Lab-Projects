from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

DATASET_FOLDER = "regression/dataset"
MODELS_FOLDER = "regression/models"

# Load LR model (common for all)
lr_model = joblib.load("regression/models/linear_regression.pkl")


# -------------------- Helper Functions --------------------
def get_stock_files():
    files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]
    files.sort()
    return files


def load_stock_data(stock_file):
    file_path = os.path.join(DATASET_FOLDER, stock_file)
    df = pd.read_csv(file_path)

    if "Close" not in df.columns:
        raise Exception(f"Close column not found in {stock_file}")

    close_prices = df["Close"].values.reshape(-1, 1)
    return close_prices


def plot_graph(actual, predicted, title):
    plt.figure(figsize=(8, 4))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression/static/plot.png")
    plt.close()


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


@app.route("/", methods=["GET", "POST"])
def home():
    stock_files = get_stock_files()

    selected_stock = stock_files[0] if stock_files else ""
    selected_model = "lr"

    prediction_next = None
    graph_generated = False
    error_msg = ""

    lr_rmse_val = None
    lstm_rmse_val = None

    try:
        if request.method == "POST":
            selected_stock = request.form.get("stock")
            selected_model = request.form.get("model")

        close_prices = load_stock_data(selected_stock)

        # ===================== LR RMSE (for this stock) =====================
        # (We calculate this always so comparison is available)
        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices

        split_lr = int(len(X) * 0.8)
        X_test_lr = X[split_lr:]
        y_test_lr = y[split_lr:]

        y_pred_lr = lr_model.predict(X_test_lr)
        lr_rmse_val = round(rmse(y_test_lr, y_pred_lr), 4)

        # ===================== Linear Regression selected =====================
        if request.method == "POST" and selected_model == "lr":

            # Next day prediction
            X_next = np.array([[len(close_prices)]])
            prediction_next = round(float(lr_model.predict(X_next)[0][0]), 2)

            plot_graph(
                y_test_lr.flatten(),
                y_pred_lr.flatten(),
                f"Actual vs Predicted (Linear Regression) - {selected_stock}"
            )
            graph_generated = True

        # ===================== LSTM RMSE + Prediction =====================
        if request.method == "POST" and selected_model == "lstm":
            stock_name = selected_stock.replace(".csv", "")
            model_path = os.path.join(MODELS_FOLDER, f"{stock_name}_lstm.h5")
            scaler_path = os.path.join(MODELS_FOLDER, f"{stock_name}_scaler.pkl")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                raise Exception(f"LSTM model not trained for {selected_stock}. Train it first!")

            lstm_model = load_model(model_path)
            scaler = joblib.load(scaler_path)

            scaled_data = scaler.transform(close_prices)

            TIME_STEP = 60
            X_seq, y_seq = [], []

            for i in range(TIME_STEP, len(scaled_data)):
                X_seq.append(scaled_data[i - TIME_STEP:i, 0])
                y_seq.append(scaled_data[i, 0])

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], 1)

            split = int(len(X_seq) * 0.8)
            X_test = X_seq[split:]
            y_test = y_seq[split:]

            y_pred_scaled = lstm_model.predict(X_test)

            # inverse scaling
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_actual = scaler.inverse_transform(y_pred_scaled)

            # RMSE for LSTM
            lstm_rmse_val = round(rmse(y_test_actual, y_pred_actual), 4)

            # Next day prediction
            last_60 = scaled_data[-60:].reshape(1, 60, 1)
            pred_scaled = lstm_model.predict(last_60)[0][0]
            pred_next = scaler.inverse_transform([[pred_scaled]])[0][0]
            prediction_next = round(float(pred_next), 2)

            plot_graph(
                y_test_actual.flatten(),
                y_pred_actual.flatten(),
                f"Actual vs Predicted (LSTM) - {selected_stock}"
            )
            graph_generated = True

    except Exception as e:
        error_msg = str(e)

    return render_template(
        "index.html",
        stock_files=stock_files,
        selected_stock=selected_stock,
        selected_model=selected_model,
        prediction=prediction_next,
        graph_generated=graph_generated,
        error_msg=error_msg,
        lr_rmse=lr_rmse_val,
        lstm_rmse=lstm_rmse_val
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
