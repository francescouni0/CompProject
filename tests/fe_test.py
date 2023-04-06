import unittest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(os.getcwd()).parent))

import pandas as pd
import numpy as np
import matlab.engine
from ML_tools.feature_extractor import feature_extractor
import ML_tools.reading as reading


class TestFeatureExtractor(unittest.TestCase):
    """
    Test the `feature_extractor()` function.
    This test verifies that the output of the `feature_extractor()` function has the expected data types, shapes, and
    values.
    """
    def test_feature_extractor(self):
        # defining input data
        num_regions = 176
       
        paths_masks = reading.data_path("Diffusion_space_segmentations-20230215T134839Z-001", "Diffusion_space_segmentations-20230215T134839Z-001")

        # call function
        df_mean, df_std, group = feature_extractor(paths_masks, paths_masks)

        # check output data types
        self.assertIsInstance(df_mean, pd.DataFrame)
        self.assertIsInstance(df_std, pd.DataFrame)
        self.assertIsInstance(group, pd.Series)

        # check output dataframe shapes
        self.assertEqual(df_mean.shape, (len(paths_masks), num_regions))
        self.assertEqual(df_std.shape, (len(paths_masks), num_regions))

        # check output dataframe values
        self.assertTrue(np.allclose(df_std.values, 0))


if __name__ == '__main__':
    unittest.main()
