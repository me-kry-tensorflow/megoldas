from unittest.mock import patch, MagicMock
from unittest.case import TestCase
from unittest import mock
from gradient import calculate_gradient


class GradientTest(TestCase):

    @mock.patch('gradient.tf')
    def test_calculate_gradient(self, mock_tf):
        gradient_return_value = 123
        mock_tf.GradientTape = MagicMock()
        mock_tf.GradientTape.gradient = MagicMock()
        mock_tf.GradientTape.gradient.return_value = gradient_return_value
        dy_dx = calculate_gradient()
        self.assertEqual(dy_dx, dy_dx)
        mock_tf.GradientTape.assert_called_once()
        mock_tf.GradientTape.gradient.assert_called_once()

if __name__ == '__main__':
    unittest.main()