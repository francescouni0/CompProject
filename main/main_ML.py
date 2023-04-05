"""
Image binary classification using Machine Learning.
This python script evaluates the performance of a Random Forest classifier and a Support Vector Machine classifier on
the feature extracted from the Diffusion Tensor Images.
The subjects are divided in two groups: AD and CN, corresponding to subjects suffering from Alzheimer's disease and
control subjects respectively.
As an option, the user can choose to perform a Principal Component Analysis (PCA) on the Random Forest classifier and
a Recursive Feature Elimination (RFE) on the Support Vector Machine classifier.
The binary classifiers are evaluated by means the following parameters: Accuracy, Precision, Recall and AUC.
"""
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(os.getcwd()).parent))

import ML_tools.classifiers as classifiers
import ML_tools.reading as reading
from ML_tools.feature_extractor import feature_extractor, feature_extractor_par
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
        help="Type of feature extractor to use (Sequential/Parallel)",
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
        help="Type of datatype from the diffusion tensor (MD/FA)",
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
        help="Type of ML classifier to use (Random Forest/Support Vector Machines)",
        default="Random Forest",
    )

    args = parser.parse_args()
    
    PATH_diff = args.datapathdiff
    
    # LOAD MASKS DATASET
    PATH_masks = args.datapathmask
    paths_masks = reading.data_path(PATH_masks, "Diffusion_space_segmentations-20230215T134839Z-001")

    # ANALYSIS FOR MD DATASET
    if args.datatype == "MD":
        path_subdir = "corrected_MD_image"
        paths_MD = reading.data_path(PATH_diff, path_subdir)
        
        # SELECTION OF THE FEATURE EXTRACTOR
        if args.featureextractor == "Sequential":
            mean_md, std_md, group = feature_extractor(paths_MD, paths_masks)
        elif args.featureextractor == "Parallel":
            mean_md, std_md, group = feature_extractor_par(paths_MD, paths_masks)
        else:
            print('Input not valid, please retry. (line 106)')

        # SELECTING THE CLASSIFIER
        if args.classifier == "Random Forest":
            option1 = input("Principal component analysis? Y or N: ")

            # OPTION FOR PCA
            if option1.lower()[0] == "n":
                classifiers.RFPipeline_noPCA(mean_md, group, 10, 5)
            elif option1.lower()[0] == "y":
                classifiers.RFPipeline_PCA(mean_md, group, 10, 5)
            else:
                print('Input not valid, please retry. (line 118)')

        elif args.classifier == "Support Vector Machines":
            classifiers.SVM_simple(mean_md, group, "rbf")

        else:
            print('Input not valid, please retry. (line 124)')

    # ANALYSIS FOR FA DATASET
    elif args.datatype == "FA":
        path_subdir = "corrected_FA_image"
        paths_FA = reading.data_path(PATH_diff, path_subdir)

        # SELECTION OF THE FEATURE EXTRACTOR
        if args.featureextractor == "Sequential":
            mean_fa, std_fa, group = feature_extractor(paths_FA, paths_masks)
        elif args.featureextractor == "Parallel":
            mean_fa, std_fa, group = feature_extractor_par(paths_FA, paths_masks)
        else:
            print('Input not valid, please retry. (line 137)')

        # SELECTING THE CLASSIFIER
        if args.classifier == "Random Forest":
            option1 = input("Principal component analysis? Y or N: ")
            
            # OPTION FOR PCA
            if option1.lower()[0] == "n":
                classifiers.RFPipeline_noPCA(mean_fa, group, 5, 5)
            elif option1.lower()[0] == "y":
                classifiers.RFPipeline_PCA(mean_fa, group, 3, 5)
            else:
                print('Input not valid, please retry. (line 149)')
            
        elif args.classifier == "Support Vector Machines":
            option2 = input("Feature Reduction? (approx time 6min) Y or N: ")

            # OPTION FOR FEATURE REDUCTION
            if option2.lower()[0] == "n":
                classifiers.SVM_simple(mean_fa, group, "rbf")
            elif option2.lower()[0] == "y":
                classifiers.SVM_feature_reduction(mean_fa, group)
            else:
                print('Input not valid, please retry. (line 160)')

        else:
            print('Input not valid, please retry. (line 163)')

    else:
        print('Input not valid, please retry. (line 166)')
