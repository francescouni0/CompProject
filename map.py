import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from reading import load
from reading import data_path
#QUELLO CHE VOGLIO CERCARE DI FARE ORA È FARE UNA MAPPA STATISTICA DELLE FEATURE

paths_FA= data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_FA_image")

imgs=load(paths_FA)
print(imgs[0])

plotting.plot_img(paths_FA[70])



plotting.show()