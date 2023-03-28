from unittest.mock import MagicMock
import sys

sys.path.insert(0, str(Path(os.getcwd()).parent))


class MockMatlabEngine(MagicMock):
    @classmethod
    def connect(cls, *args, **kwargs):
        return cls()

sys.modules['matlab.engine'] = MockMatlabEngine
sys.modules['matlab'] = MockMatlabEngine
sys.modules['matlabengine'] = MockMatlabEngine


