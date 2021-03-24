from unittest.case import TestCase
from unittest import mock
import unittest
from unittest.mock import MagicMock

import numpy as np

import keras_linear_regression


class KerasLinearRegressionTest(TestCase):

    @mock.patch('keras_linear_regression.tf.keras')
    def test_get_model(self, mock_keras):
        # given
        xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
        mock_dense = "mockDense"
        model_mock = MagicMock()
        mock_keras.layers.Dense.return_value = mock_dense
        mock_keras.Sequential.return_value = model_mock

        # when
        keras_linear_regression.get_model()

        # then
        mock_keras.layers.Dense.assert_called_once_with(units=1, input_shape=[1])
        mock_keras.Sequential.assert_called_once_with([mock_dense])
        model_mock.compile.assert_called_once_with(optimizer='sgd', loss='mean_squared_error')
        np.testing.assert_array_equal(xs,
                                      model_mock.fit.call_args[0][0])
        np.testing.assert_array_equal(ys,
                                      model_mock.fit.call_args[0][1])
        self.assertEqual(model_mock.fit.call_args[1], {'epochs': 500})

    def test_predict(self):
        # given
        x = 23
        mock_predict_value = 123
        model_mock = MagicMock()
        model_mock.predict.return_value = mock_predict_value

        # when
        rv = keras_linear_regression.predict(model_mock, x)

        # then
        self.assertEqual(rv, mock_predict_value)
        model_mock.predict.assert_called_once_with(x)


if __name__ == '__main__':
    unittest.main()




