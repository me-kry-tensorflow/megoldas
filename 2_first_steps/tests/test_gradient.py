import tensorflow as tf

from src.gradient import dy_dx


class GradientTest(tf.test.TestCase):

    def setUp(self):
        super(tf.test.TestCase, self).setUp()

    def test_gradient(self):
        self.assertEqual(dy_dx, 6)


# tf.test.main()

