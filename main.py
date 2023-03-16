from classifiers import RFPipeline_noPCA, RFPipeline_PCA , SVMPipeline_feature_reduction
from sklearn.model_selection import train_test_split


import os
import numpy as np
import reading 
from feature_extractor import feature_extractor



paths_FA= reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_FA_image")
paths_masks=reading.data_path("Diffusion_space_segmentations-20230215T134839Z-001","Diffusion_space_segmentations-20230215T134839Z-001")

a, b ,c=feature_extractor(paths_FA,paths_masks)

#RFPipeline_noPCA(a,c,10,5)

SVMPipeline_feature_reduction(a,c,"linear")