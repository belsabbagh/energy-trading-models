# load grid-loss.keras and duration.keras
import tensorflow as tf

grid_loss_model = tf.keras.models.load_model("../out/models/grid-loss.keras")
duration_model = tf.keras.models.load_model("../out/models/duration.keras")

def predict(grid_load, grid_temp, voltage, global_intensity, transaction_amount): 
    grid_loss = grid_loss_model.predict([[grid_load, grid_temp]])
    efficiency = (grid_load - grid_loss[0][0]) / grid_load
    duration = duration_model.predict([[voltage, global_intensity, efficiency, transaction_amount]])
    return grid_loss, duration

def main(): 
    grid_loss, duration = predict(90, 20, 239.696, 2.532, 636.93)
    print(grid_loss, duration)

if __name__ == "__main__":
    main()