import classifiers 
import os
import numpy as np
import reading 
from feature_extractor import feature_extractor
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="ML classifiers analysis in digital mammography."
    )

    parser.add_argument(
        "-dpd",
        "--datapathdiff",
        metavar="",
        help="path of the diffusion parameters map",
        default="Diffusion_parameters_maps-20230215T134959Z-001"
    )
    
    parser.add_argument(
        "-dpm",
        "--datapathmask",
        metavar="",
        help="path of the segmentetion map",
        default="Diffusion_space_segmentations-20230215T134839Z-001"
    )
    parser.add_argument(
        "-dty",
        "--datatype",
        metavar="",
        type=str,
        choices=[
            'MD',
            'FA'    
        ],
        help="Type of datatype from the diffusion tensor",
        default="MD",
    )

    
    parser.add_argument(
        "-mlc",
        "--classifier",
        metavar="",
        type=str,
        choices=[
            'Support Vector Machines',
            'Random Forest'    
        ],
        help="Type of machine learning classifier",
        default="Random Forest",
    )


    args = parser.parse_args()
    
    
    PATH_diff = args.datapathdiff
    PATH_masks = args.datapathmask
    paths_masks=reading.data_path(PATH_masks,"Diffusion_space_segmentations-20230215T134839Z-001")

    if args.datatype=="MD":
        path_subdir="corrected_MD_image"
        paths_MD= reading.data_path(PATH_diff,path_subdir)
        
        mean_md, std_md ,group=feature_extractor(paths_MD,paths_masks)
        if args.classifier=="Random Forest":
            
            classifiers.RFPipeline_noPCA(mean_md,group,10,5)
            
        elif args.classifier=="Support Vector Machines": 

            classifiers.SVMPipeline(mean_md,group,"rbf")
            

    elif args.datatype=="FA":
        path_subdir="corrected_FA_image"
        paths_FA= reading.data_path(PATH_diff,path_subdir)
        
        mean_fa, std_fa ,group=feature_extractor(paths_FA,paths_masks)
        if args.classifier=="Random Forest":
            
            classifiers.RFPipeline_noPCA(mean_fa,group,10,5)
            
        elif args.classifier=="Support Vector Machines": 

            classifiers.SVMPipeline(mean_fa,group,"linear")
            
        
        
        
        
        
