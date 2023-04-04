from unittest.mock import MagicMock
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(os.getcwd()).parent))


class MockMatlabEngine(MagicMock):
    """
    A mock object for the `matlab.engine` module.
    This mock object can be used in unit tests that involve the `matlab.engine`
    module. It provides a `connect()` method that returns an instance of the
    `MockMatlabEngine` class.
    """
    @classmethod
    def connect(cls, *args, **kwargs):
        """Returns an instance of the `MockMatlabEngine` class.

        This method is used to mock the behavior of the `matlab.engine.connect()`
        function. It returns an instance of the `MockMatlabEngine` class, which
        can be used in unit tests to simulate the behavior of a real MATLAB
        engine.

        Returns:
             An instance of the `MockMatlabEngine` class.

        """
        return cls()


sys.modules['matlab.engine'] = MockMatlabEngine
sys.modules['matlab'] = MockMatlabEngine
sys.modules['matlabengine'] = MockMatlabEngine


