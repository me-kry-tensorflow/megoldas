from unittest.case import TestCase
from unittest import mock
import unittest

from linear_regression.globals import generate_data


class LinearRegressionTest(TestCase):

    def test_generate_data(self):
        # given

        # when
        x, y = generate_data()

        # then
        pass


if __name__ == '__main__':
    unittest.main()
