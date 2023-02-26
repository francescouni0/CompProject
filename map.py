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