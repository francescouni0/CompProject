import unittest
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(os.getcwd()).parent))

from ML_tools.reading import data_path_mod
import tempfile

paths_masks = data_path_mod("Diffusion_space_segmentations-20230215T134839Z-001", "Diffusion_space_segmentations-20230215T134839Z-001")
        
        
path_subdir = "corrected_MD_image"
        
paths_MD = data_path_mod("Diffusion_parameters_maps-20230215T134959Z-001", path_subdir)

print(paths_MD[0:2])
print(paths_masks[0:2])
class TestDataPath(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.temp_dir.name, "Diffusion_space_parameters"))
        os.makedirs(os.path.join(self.temp_dir.name, "Diffusion_space_segmentation"))
        open(os.path.join(self.temp_dir.name, "Diffusion_space_parameters", "1_2_3.nii.gz"), 'a').close()
        open(os.path.join(self.temp_dir.name, "Diffusion_space_parameters", "4_5_6.nii.gz"), 'a').close()
        open(os.path.join(self.temp_dir.name, "Diffusion_space_segmentation", "1_2_abc.nii.gz"), 'a').close()
        open(os.path.join(self.temp_dir.name, "Diffusion_space_segmentation", "4_5_def.nii.gz"), 'a').close()
    
    def test_data_path(self):
        img_filepaths = data_path_mod(str(self.temp_dir.name), 'Diffusion_space_parameters')
        seg_filepaths = data_path_mod(str(self.temp_dir.name), 'Diffusion_space_segmentation')
        self.assertEqual(len(img_filepaths), 2)
        self.assertEqual(len(seg_filepaths), 2)

if __name__ == '__main__':
    unittest.main()