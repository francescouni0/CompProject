from classifiers import plot_cv_roc,RFPipeline_noPCA
from sklearn.model_selection import train_test_split


import os
import numpy as np
import reading 
from feature_extractor import feature_extractor



paths_FA= reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_FA_image")
paths_masks=reading.data_path("IDiffusion_space_segmentations-20230215T134839Z-001","Diffusion_space_segmentations-20230215T134839Z-001")

a, b ,c=feature_extractor(paths_FA,paths_masks)

X=a.values
y=c.values

plot_cv_roc(X,y,RFPipeline_noPCA,5)