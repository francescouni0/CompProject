import unittest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(os.getcwd()).parent))

from ML_tools.reading import data_path
import tempfile


class TestDataPath(unittest.TestCase):
    """
    This class defines the unit tests for the 'data_path' function in 'ML_tools.reading' module.
    """
    
    def setUp(self):
        """
        This method sets up the temporary directory and creates test files for the unit tests.
        """
        
        self.temp_dir = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.temp_dir.name, "Diffusion_space_parameters"))
        os.makedirs(os.path.join(self.temp_dir.name, "Diffusion_space_segmentation"))
        open(os.path.join(self.temp_dir.name, "Diffusion_space_parameters", "1_2_3.nii.gz"), 'a').close()
        open(os.path.join(self.temp_dir.name, "Diffusion_space_parameters", "4_5_6.nii.gz"), 'a').close()
        open(os.path.join(self.temp_dir.name, "Diffusion_space_segmentation", "1_2_abc.nii.gz"), 'a').close()
        open(os.path.join(self.temp_dir.name, "Diffusion_space_segmentation", "4_5_def.nii.gz"), 'a').close()
    
    def test_data_path(self):
        """
        This method tests whether the 'data_path' function returns the correct number of files for a given
        directory and subdirectory.
        """
        img_filepaths = data_path(str(self.temp_dir.name), 'Diffusion_space_parameters')
        seg_filepaths = data_path(str(self.temp_dir.name), 'Diffusion_space_segmentation')
        self.assertEqual(len(img_filepaths), 2)
        self.assertEqual(len(seg_filepaths), 2)


if __name__ == '__main__':
    unittest.main()
