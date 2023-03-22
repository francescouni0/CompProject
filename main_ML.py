
"""CMEPDA Project: Image binary classification using Machine Learning.
This python script evaluates the performance of a RandomForest classifier and a
Support Vector Machine classifier on the feature extracted from the diffusion tensor.
The subjects are divided in two groups: AD and CN. Corresponding to subjects affected
with Alzheimer's disease and control subjects respectively.
As an option, the user can choose to perform a Principal Component Analysis (PCA) on the Random Forest classifier.
and a feature reduction on the Support Vector Machine classifier.
The binary classifiers are evaluate by means the following parameters:
- Accuracy
- Precision
- Recall
- AUC
"""








import classifiers
import reading 
from feature_extractor import feature_extractor, feature_extractor_par
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="ML and DL classifiers analysis for Alzheimer's disease detection on Diffusion Images."
    )

    parser.add_argument(
        "-fe",
        "--featureextractor",
        metavar="",
        type=str,
        choices=[
            'Sequential',
            'Parallel'
        ],
        help="Type of feature extractor to use",
        default="Sequential",
    )

    parser.add_argument(
        "-dpd",
        "--datapathdiff",
        metavar="",
        help="Path of the diffusion parameters map",
        default="Diffusion_parameters_maps-20230215T134959Z-001"
    )
    
    parser.add_argument(
        "-dpm",
        "--datapathmask",
        metavar="",
        help="Path of the segmentation map",
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
    
    #LOAD MASKS DATASET
    
    
    PATH_masks = args.datapathmask
    paths_masks = reading.data_path(PATH_masks, "Diffusion_space_segmentations-20230215T134839Z-001")



    #ANALISYS FOR MD DATASET

    if args.datatype == "MD":
        path_subdir = "corrected_MD_image"
        paths_MD = reading.data_path(PATH_diff, path_subdir)
        
        #SELECTION OF THE FEATURE EXTRACTOR
        
        if args.featureextractor == "Sequential":
            mean_md, std_md, group = feature_extractor(paths_MD, paths_masks)
        elif args.featureextractor == "Parallel":
            mean_md, std_md, group = feature_extractor_par(paths_MD, paths_masks)
        else:
            print('Input not valid, please retry.')
        
        
        #SELECTING THE CLASSIFIER
        
        
        if args.classifier == "Random Forest":
            option1 = input("Principal component analysis? Y or N: ")
            
            
            #OPTION FOR PCA
            
            if option1.lower()[0] == "n":
                classifiers.RFPipeline_noPCA(mean_md, group, 10, 5)
            elif option1.lower()[0] == "y":
                classifiers.RFPipeline_PCA(mean_md, group, 10, 5)
            else:
                print('Input not valid, please retry.')
            
        
        elif args.classifier == "Support Vector Machines":
            classifiers.SVMPipeline(mean_md, group, "rbf")

        else:
            print('Input not valid, please retry.')
    
    
    #ANALISYS FOR FA DATASET
    
    elif args.datatype == "FA":
        path_subdir = "corrected_FA_image"
        paths_FA = reading.data_path(PATH_diff, path_subdir)
        
        
        #SELECTION OF THE FEATURE EXTRACTOR
        
        if args.featureextractor == "Sequential":
            mean_fa, std_fa, group = feature_extractor(paths_FA, paths_masks)
        elif args.featureextractor == "Parallel":
            mean_fa, std_fa, group = feature_extractor_par(paths_FA, paths_masks)
        else:
            print('Input not valid, please retry.')
        
        
        #SELECTING THE CLASSIFIER
        
        
        if args.classifier == "Random Forest":
            option1 = input("Principal component analysis? Y or N: ")
            
            #OPTION FOR PCA
            
            if option1.lower()[0] == "n":
                classifiers.RFPipeline_noPCA(mean_fa, group, 10, 5)
            elif option1.lower()[0] == "y":
                classifiers.RFPipeline_PCA(mean_fa, group, 10, 5)
            else:
                print('Input not valid, please retry.')
            
        elif args.classifier == "Support Vector Machines":
            option2 = input("Feature Reduction? (approx time 6min) Y or N: ")
            
            
            #OPTION FOR FEATURE REDUCTION
            
            
            if option2.lower()[0] == "n":
                classifiers.SVMPipeline(mean_fa, group, "rbf")
            elif option2.lower()[0] == "y":
                classifiers.SVMPipeline_feature_red(mean_fa, group)
            else:
                print('Input not valid, please retry.')

        else:
            print('Input not valid, please retry.')

    else:
        print('Input not valid, please retry.')
            
        
        
        
        
        
