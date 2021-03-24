import tensorflow as tf
import numpy as np


def get_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model.fit(xs, ys, epochs=500)
    return model


def predict(model, x):
    return model.predict(x)


if __name__ == "__main__":
    mymodel = get_model()
    print(predict(mymodel, [10.0]))




