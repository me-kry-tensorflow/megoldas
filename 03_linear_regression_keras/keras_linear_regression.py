import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt


def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_data():
    xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    return xs, ys


def get_model(xs, ys):
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(xs, ys, epochs=500, callbacks=[tensorboard_callback], validation_split=0.2)
    plot_loss(history)
    return model


def predict(model, x):
    return model.predict(x)


if __name__ == "__main__":
    xs, ys = get_data()
    my_model = get_model(xs, ys)
    print(predict(my_model, [10.0]))




