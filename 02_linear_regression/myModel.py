import tensorflow as tf


class MyModel(tf.Module):
    def __init__(self, **kwargs):
        """
        letrehoz egy `w` _adattagot_, amely tf.Variable, erteke 10
        letrehoz egy `b` _adattagot_, amely tf.Variable, erteke 0
        """
        super().__init__(**kwargs)
        # Initialize the weights to `10.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.w = tf.Variable(10.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        """
        Visszaadja a szamolt erteket w*x + b
        :rtype: object
        """
        return self.w * x + self.b

