import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import tensorflow as tf
from sklearn.metrics import r2_score

def load_duration_csv(csv): 
    df = pd.read_csv(csv, sep=";")
    columns = ["Time", "Day", "Voltage", "Global_intensity", "Efficiency", "Transaction_amount", "Transaction_duration"]
    df = df[columns]
    df = df.set_index(["Time", "Day"])
    return df

def preprocess_duration_data(df):
    df.dropna(inplace=True)
    X = df[["Voltage", "Global_intensity", "Efficiency", "Transaction_amount"]].astype("float64")
    y = df["Transaction_duration"]
    return X, y



# def main(): 
#     train = "../data/duration/train.csv"
#     test = "../data/duration/test.csv"
#     X, y = preprocess_duration_data(load_duration_csv(train))
#     print(X.head())
#     print(y.head())
#     model = LinearRegression()
#     model.fit(X, y)
    
#     X_test, y_test = preprocess_duration_data(load_duration_csv(test))
#     print(model.score(X_test, y_test))
#     # print the most important features
#     print(model.coef_)
#     print(model.intercept_)
    
#     pickle.dump(model, open("../out/models/duration.pkl", 'wb'))

    

def main(): 
    train = "../data/duration/train.csv"
    test = "../data/duration/test.csv"
    X, y = preprocess_duration_data(load_duration_csv(train))
    print(X.head())
    print(y.head())

    # Reshape the input data
    X = X.values.reshape(-1, 4)

    model = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(1, input_shape=(4,), activation="linear")]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X, y, epochs=50)
    model.save("../out/models/duration.keras")
    X_test, y_test = preprocess_duration_data(load_duration_csv(test))
    X_test = X_test.values.reshape(-1, 4)
    res = model.evaluate(X_test, y_test)
    model.summary()
    print(res)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("R-squared:", r2)



if __name__ == "__main__":
    main()