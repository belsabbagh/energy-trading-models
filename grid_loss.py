import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import time


def load_grid_loss_csv(csv):
    df = pd.read_csv(csv, index_col=0)
    columns = ["grid1-load", "grid1-loss", "grid1-temp"]
    df = df[columns]
    df = df.rename(
        columns={
            "grid1-load": "grid1-load-(MWh)",
            "grid1-loss": "grid1-loss-(MWh)",
            "grid1-temp": "grid1-temp-(K)",
        }
    )
    df["time"] = pd.to_datetime(df.index)
    df = df.set_index("time")
    return df


def preprocess_grid_loss_data(df):
    df["grid1-loss-(MWh)"] = df["grid1-loss-(MWh)"] / 1000
    df["grid1-load-(MWh)"] = df["grid1-load-(MWh)"] / 1000
    df["grid1-temp-(K)"] = df["grid1-temp-(K)"] - 273.15
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df.dropna(inplace=True)
    X = df[["grid1-load-(MWh)", "grid1-temp-(K)", "hour", "day_of_week"]].astype(
        "float64"
    )
    y = df["grid1-loss-(MWh)"]
    return X, y


def create_lstm_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i : (i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


if __name__ == "__main__":
    train = "data/grid-loss/train.csv"
    test = "data/grid-loss/test.csv"

    # Ensure the output directory exists
    os.makedirs("out/eval", exist_ok=True)

    # Preprocess data for both models
    X, y = preprocess_grid_loss_data(load_grid_loss_csv(train))
    X_test, y_test = preprocess_grid_loss_data(load_grid_loss_csv(test))

    # Scale data for neural network
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # Simple neural network model
    model_nn = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(1, input_shape=(4,), activation="linear")]
    )
    model_nn.compile(optimizer="adam", loss="mse", metrics=["mae"])

    start_time = time.time()
    model_nn.fit(X_scaled, y_scaled, epochs=20)
    nn_training_time = time.time() - start_time

    model_nn.save("out/models/grid-loss-nn.keras")
    res_nn = model_nn.evaluate(X_test_scaled, y_test_scaled)
    print(f"NN Model - Loss: {res_nn[0]}, MAE: {res_nn[1]}")

    # Save NN evaluation results to CSV
    nn_eval_df = pd.DataFrame(
        [["Loss", res_nn[0]], ["MAE", res_nn[1]]],
        columns=["Metric", "Value"],
    )
    nn_eval_df.to_csv("out/eval/nn_evaluation.csv", index=False)

    # Preprocess data for the LSTM model
    time_steps = 3
    X_lstm, y_lstm = create_lstm_dataset(X, y, time_steps)
    X_test_lstm, y_test_lstm = create_lstm_dataset(X_test, y_test, time_steps)

    # Scale data for LSTM
    X_lstm_scaled = scaler_X.fit_transform(X_lstm.reshape(-1, 4)).reshape(X_lstm.shape)
    y_lstm_scaled = scaler_y.fit_transform(y_lstm.reshape(-1, 1)).flatten()
    X_test_lstm_scaled = scaler_X.transform(X_test_lstm.reshape(-1, 4)).reshape(
        X_test_lstm.shape
    )
    y_test_lstm_scaled = scaler_y.transform(y_test_lstm.reshape(-1, 1)).flatten()

    # LSTM model
    model_lstm = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(50, activation="relu", input_shape=(time_steps, 4)),
            tf.keras.layers.Dense(1),
        ]
    )
    model_lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])

    start_time = time.time()
    model_lstm.fit(X_lstm_scaled, y_lstm_scaled, epochs=20)
    lstm_training_time = time.time() - start_time

    model_lstm.save("out/models/grid-loss-lstm.keras")
    res_lstm = model_lstm.evaluate(X_test_lstm_scaled, y_test_lstm_scaled)
    print(f"LSTM Model - Loss: {res_lstm[0]}, MAE: {res_lstm[1]}")

    # Save LSTM evaluation results to CSV
    lstm_eval_df = pd.DataFrame(
        [["Loss", res_lstm[0]], ["MAE", res_lstm[1]]],
        columns=["Metric", "Value"],
    )
    lstm_eval_df.to_csv("out/eval/lstm_evaluation.csv", index=False)

    # Save training times to CSV
    training_times_df = pd.DataFrame(
        [["NN", nn_training_time], ["LSTM", lstm_training_time]],
        columns=["Model", "Training Time (s)"],
    )
    training_times_df.to_csv("out/eval/training_times.csv", index=False)
