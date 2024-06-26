import unittest
from unittest.mock import MagicMock
from Model_Training_Tab.utilities import MLRobotUtils

class TestMLRobotUtils(unittest.TestCase):

    def setUp(self):
        self.utils = MLRobotUtils()
        self.log_text_widget = MagicMock()

    def test_log_message(self):
        self.utils.log_message("Test message", self.log_text_widget, False)
        self.log_text_widget.insert.assert_called_once()

    # Additional tests for other methods...

if __name__ == "__main__":
    unittest.main()
