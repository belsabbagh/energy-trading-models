import pandas as pd
import tensorflow as tf


def load_grid_loss_csv(csv):
    df = pd.read_csv(csv)
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
    return df


def preprocess_grid_loss_data(df):
    df["grid1-loss-(MWh)"] = df["grid1-loss-(MWh)"] / 1000
    df["grid1-load-(MWh)"] = df["grid1-load-(MWh)"] / 1000
    df["grid1-temp-(K)"] = df["grid1-temp-(K)"] - 273.15
    df.dropna(inplace=True)
    X = df[["grid1-load-(MWh)", "grid1-temp-(K)"]].astype("float64")
    y = df["grid1-loss-(MWh)"]
    return X, y


if __name__ == "__main__":
    train = "data/grid-loss/train.csv"
    test = "data/grid-loss/test.csv"
    X, y = preprocess_grid_loss_data(load_grid_loss_csv(train))
    print(X.head())
    print(y.head())
    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(1, input_shape=(2,), activation="linear")]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X, y, epochs=20)
    X_test, y_test = preprocess_grid_loss_data(load_grid_loss_csv(test))
    res = model.evaluate(X_test, y_test)
