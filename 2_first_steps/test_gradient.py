from unittest.mock import patch, MagicMock
from unittest.case import TestCase
from unittest import mock
from gradient import calculate_gradient

class GradientTest(TestCase):

    @mock.patch('gradient.tf.GradientTape')
    def test_calculate_gradient(self, mock_tf):
        gradient_return_value = 123
        mock_tf.return_value.__enter__.return_value.gradient.return_value = gradient_return_value
        # mock_tf.GradientTape.gradient.return_value = gradient_return_value
       # tf.GradientTape().gradient()
        dy_dx = calculate_gradient()
        self.assertEqual(dy_dx, gradient_return_value)
        mock_tf.return_value.__enter__.assert_called_once()
        mock_tf.return_value.__enter__.return_value.gradient.assert_called_once_with(9, 3)

if __name__ == '__main__':
    unittest.main()