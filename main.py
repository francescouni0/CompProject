import classifiers 
import os
import numpy as np
import reading 
from feature_extractor import feature_extractor
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="ML classifiers analysis for Alzheimer's desease detection on Diffusion Images."
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
        "-clf",
        "--classifier",
        metavar="",
        type=str,
        choices=[
            'Random Forest',
            'Support Vector Machines'    
        ],
        help="Type of ML classifier to use",
        default="RF",
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
            
            option1=input("Principal component analysis? Y or N:    ")
            
            if option1=="n":
            
                classifiers.RFPipeline_noPCA(mean_md,group,10,5)
                
            if option1=="y":
                
                classifiers.RFPipeline_PCA(mean_md,group,10,5)
            else:
                print('Repeat')
            
        elif args.classifier=="Support Vector Machines": 
                
            classifiers.SVMPipeline(mean_md,group,"rbf")
                
            
            

    elif args.datatype=="FA":
        
        path_subdir="corrected_FA_image"
        
        paths_FA= reading.data_path(PATH_diff,path_subdir)
        
        mean_fa, std_fa ,group=feature_extractor(paths_FA,paths_masks)
        if args.classifier=="Random Forest":
            
            option1=input("Principal component analysis? Y or N:    ")
            
            if option1=="n":

                classifiers.RFPipeline_noPCA(mean_fa,group,10,5)
                
            elif option1=="y":
                
                classifiers.RFPipeline_PCA(mean_fa,group,10,5)
            else:
                print('Repeat')
            
        elif args.classifier=="Support Vector Machines": 
    
            option2=input("Feature Reduction? (approx time 6min) Y or N:    ")
            
            if option2=="n":
                
                classifiers.SVMPipeline(mean_fa,group,"rbf")
                
            elif option2=="y":
                
                classifiers.SVMPipeline_feature_red(mean_fa,group)
            else:
                print('Repeat')
            
        
        
        
        
        
