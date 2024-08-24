# load grid-loss.keras and duration.keras
import tensorflow as tf
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# import the models
grid_loss_model = tf.keras.models.load_model("../out/models/grid-loss.keras")
duration_model = tf.keras.models.load_model("../out/models/duration.keras")

# import the data loading functions from src/__init__.py and src/durationModel.py
from __init__ import load_grid_loss_csv, preprocess_grid_loss_data
from durationModel import load_duration_csv, preprocess_duration_data

def get_metrics_grid_loss(): 
    # preprocess the data
    test = "../data/grid-loss/test.csv"
    X_test, y_test = preprocess_grid_loss_data(load_grid_loss_csv(test))
    # get r2 score
    y_pred = grid_loss_model.predict(X_test)

    loss, mae = grid_loss_model.evaluate(X_test, y_test)

    r2 = r2_score(y_test, y_pred)

    return loss, mae, r2

def get_metrics_duration(): 
    test = "../data/duration/test.csv"
    X_test, y_test = preprocess_duration_data(load_duration_csv(test))
    X_test = X_test.values.reshape(-1, 4)
    y_pred = duration_model.predict(X_test)

    loss, mae = duration_model.evaluate(X_test, y_test)

    r2 = r2_score(y_test, y_pred)

    return loss, mae, r2


def main(): 
    grid_loss_loss, grid_loss_mae, grid_loss_r2 = get_metrics_grid_loss()
    duration_loss, duration_mae, duration_r2 = get_metrics_duration()

    # plot the grid_loss mae and duration mae in separate plots as bar charts
    plt.bar(["Grid Loss MAE"], grid_loss_mae)
    # save the plot to ../out/plots/grid_loss_mae.png
    plt.savefig("../out/plots/grid_loss_mae.png")
    plt.clf()
    plt.bar(["Duration MAE"], duration_mae)
    # save the plot to ../out/plots/duration_mae.png
    plt.savefig("../out/plots/duration_mae.png")
    plt.clf()
    
    


    

if __name__ == "__main__":
    main()