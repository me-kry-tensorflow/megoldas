import tensorflow as tf


def calculate_gradient(x):
    # https://docs.python.org/3/library/operator.html
    with tf.GradientTape() as tape:
        y = x**2  # pow(x, 2)

    # dy = (3*3) - 3 = 9 - 3
    return tape.gradient(y, x)


if __name__ == "__main__":
    x = tf.Variable(3.0)
    dy_dx = calculate_gradient(x)
    print(dy_dx.numpy())
