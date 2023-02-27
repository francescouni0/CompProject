import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from reading import load
from reading import data_path
from nilearn import image
import sys
from nilearn.regions import RegionExtractor
from nilearn import regions

import matlab.engine


#np.set_printoptions(threshold=sys.maxsize)
#QUELLO CHE VOGLIO CERCARE DI FARE ORA È FARE UNA MAPPA STATISTICA DELLE FEATURE

paths_FA= data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_FA_image")
paths_masks=data_path("Diffusion_space_segmentations-20230215T134839Z-001","Diffusion_space_segmentations-20230215T134839Z-001")


#QUESTO MI ORDINA TUTTO IN MODO CHE MASCHERA E FILE CORRISPONDANO
paths_FA.sort(key=lambda x: int(os.path.basename(x).split('_')[3][1:]))
paths_masks.sort(key=lambda x: int(os.path.basename(x).split('_')[2][1:]))

'''
imgs = image.smooth_img(paths_FA,0) 
masks = image.smooth_img(paths_masks,0) 

img70_data=imgs[70].get_fdata()
mask70_data=masks[70].get_fdata()


print(paths_FA[0])
print(paths_masks[0])


plotting.show()

'''


# Start MATLAB engine
matlab_eng = matlab.engine.start_matlab()


# Call a MATLAB function
feature = matlab_eng.feature_extractor('Diffusion_parameters_maps-20230215T134959Z-001/Diffusion_parameters_maps/098_S_4003/corrected_FA_image/2011-03-22_09_23_47.0/I299742/ADNI_098_S_4003_MR_corrected_FA_image_Br_20120421215950180_S102157_I299742.nii', 'Diffusion_space_segmentations-20230215T134839Z-001/Diffusion_space_segmentations/098_S_4003_wmparc_on_MD.nii.gz')

# Print the result
print(feature)

# Stop MATLAB engine
matlab_eng.quit()