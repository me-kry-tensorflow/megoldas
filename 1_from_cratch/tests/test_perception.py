from unittest import TestCase
from src.from_cratch.perceptron import perceptron, predict, weights


class Test(TestCase):

    @classmethod
    def setUpClass(cls):
        for i in range(100000):
            perceptron(1, 1, 1)  # True or true
            perceptron(1, 0, 1)  # True or false
            perceptron(0, 1, 1)  # False or true
            perceptron(0, 0, 0)  # False or false
        print(weights)

    def test_prediction_0_1(self):
        self.assertTrue(predict(0, 1), msg='0 OR 1')

    def test_prediction_1_0(self):
        self.assertTrue(predict(1, 0), msg='1 OR 0')

    def test_prediction_1_1(self):
        self.assertTrue(predict(1, 1), msg='1 OR 1')

    def test_prediction_0_0(self):
        self.assertLess(predict(0, 0), 0.01, msg='0 OR 0')
