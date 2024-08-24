import pandas as pd
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt


def load_duration_csv(csv):
    df = pd.read_csv(csv, sep=";")
    columns = [
        "Time",
        "Day",
        "Voltage",
        "Global_intensity",
        "Efficiency",
        "Transaction_amount",
        "Transaction_duration",
    ]
    df = df[columns]
    df = df.set_index(["Time", "Day"])
    return df


def preprocess_duration_data(df):
    df.dropna(inplace=True)
    X = df[["Voltage", "Global_intensity", "Efficiency", "Transaction_amount"]].astype(
        "float64"
    )
    y = df["Transaction_duration"]
    return X, y


def calculate_theoretical_duration(df):
    E = df["Transaction_amount"]
    P = df["Voltage"] * df["Global_intensity"]
    eta = df["Efficiency"]
    return E / (P * eta)


def main():
    train = "data/duration/train.csv"
    test = "data/duration/test.csv"

    # Ensure the output directory exists
    os.makedirs("out/eval", exist_ok=True)

    # Preprocess the training data
    X_train, y_train = preprocess_duration_data(load_duration_csv(train))
    print(X_train.head())
    print(y_train.head())

    # Reshape the input data
    X_train = X_train.values.reshape(-1, 4)

    # Define and compile the model
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(1, input_shape=(4,), activation="linear")]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Train the model and measure training time
    start_time = time.time()
    model.fit(X_train, y_train, epochs=50)
    training_time = time.time() - start_time

    # Save the trained model
    model.save("out/models/duration.keras")

    # Preprocess the test data
    X_test, y_test = preprocess_duration_data(load_duration_csv(test))
    X_test = X_test.values.reshape(-1, 4)

    # Evaluate the model on the test data
    res = model.evaluate(X_test, y_test)
    print(f"Test Loss: {res[0]}, Test MAE: {res[1]}")

    # Save evaluation metrics to CSV
    eval_df = pd.DataFrame(
        [["Loss", res[0]], ["MAE", res[1]]], columns=["Metric", "Value"]
    )
    eval_df.to_csv("out/eval/duration_evaluation.csv", index=False)

    # Save training time to CSV
    training_time_df = pd.DataFrame(
        [["Duration Model", training_time]], columns=["Model", "Training Time (s)"]
    )
    training_time_df.to_csv("out/eval/duration_training_time.csv", index=False)

    # Calculate the theoretical transaction duration
    test_df = load_duration_csv(test)
    y_theoretical = calculate_theoretical_duration(test_df)

    # Predict the transaction duration using the model
    y_pred = model.predict(X_test).flatten()

    # Plot the prediction differences
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values[:20], label="Actual Duration")
    plt.plot(y_pred[:20], label="Predicted Duration")
    plt.plot(y_theoretical.values[:20], label="Theoretical Duration", linestyle="--")
    plt.xlabel("Sample")
    plt.ylabel("Transaction Duration")
    plt.legend()
    plt.title("Transaction Duration: Actual vs Predicted vs Theoretical")
    plt.savefig("out/eval/duration_prediction_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
