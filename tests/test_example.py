import unittest
from src.example import function_to_test  # Replace with actual function name

class TestExample(unittest.TestCase):

    def test_function_to_test(self):
        # Arrange
        input_data = ...  # Define input data
        expected_output = ...  # Define expected output

        # Act
        result = function_to_test(input_data)

        # Assert
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()