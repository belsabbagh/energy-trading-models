import pandas as pd
import tensorflow as tf


def load_grid_loss_data():
    df = pd.read_csv("data/grid-loss/train.csv")
    columns = ["Unnamed: 0", "grid1-load", "grid1-loss", "grid1-temp"]
    df = df[columns]
    df = df.rename(
        columns={
            "Unnamed: 0": "time",
            "grid1-load": "grid1-load-(MWh)",
            "grid1-loss": "grid1-loss-(MWh)",
            "grid1-temp": "grid1-temp-(K)",
        }
    )
    df = df.set_index("time")
    X = df[["grid1-load-(MWh)", "grid1-temp-(K)"]]
    y = df["grid1-loss-(MWh)"]
    return X, y


if __name__ == "__main__":
    X, y = load_grid_loss_data()
    print(X.head())
    print(y.head())
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(1, input_shape=(2,), activation="linear")]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20)
