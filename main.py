from classifiers import RFPipeline_noPCA
from sklearn.model_selection import train_test_split


import os
import numpy as np
import reading 
from feature_extractor import feature_extractor



paths_FA= reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_FA_image")
paths_masks=reading.data_path("Diffusion_space_segmentations-20230215T134839Z-001","Diffusion_space_segmentations-20230215T134839Z-001")

a, b ,c=feature_extractor(paths_FA,paths_masks)

X=a.values
y=c.values

X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=.1, random_state=6)

a=RFPipeline_noPCA(X_tr,y_tr,X_tst,y_tst,50,5)