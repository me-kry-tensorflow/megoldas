import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

x = tf.Variable(3.0)

# https://docs.python.org/3/library/operator.html
with tf.GradientTape() as tape:
    y = x**2  # pow(x, 2)

# dy = (3*3) - 3 = 9 - 3
dy_dx = tape.gradient(y, x)

if "__name__" == "__main__":
    print(dy_dx.numpy())
