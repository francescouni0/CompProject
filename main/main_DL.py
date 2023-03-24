"""
CMEPDA Project: Image binary classification using a custom-built Convolutional Neural Network.
This python script trains and evaluates the performance of a custom-built Convolutional Neural Network.
The Neural Network is trained on the hippocampus region of different types of Diffusion Images.
The data is augmented by applying a random rotation, random zoom to the images and random contrast by means 
of a custom-built function tha uses keras layers.
Using the color channels as a way of avoiding the use of a 3D convolutional layer.
As an option the user can use pre-trained weights for the convolutional layers previously trained on the same dataset.
The subjects are divided in two groups: AD and CN. Corresponding to subjects affected
with Alzheimer's disease and control subjects respectively.
The CNN is evaluated by means the following parameters:
- Accuracy
- Precision
- Recall
- AUC
"""

import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(os.getcwd()).parent))

import CNN_tools.CNN_multi_class as CNN_multi_class
import CNN_tools.CNN_multi_utilities as CNN_multi_utilities
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="CNN classifier for Alzheimer's disease detection on multimodal Diffusion Images."
    )

    parser.add_argument(
        "-dpd",
        "--datapathdiff",
        metavar="",
        help="Path of the diffusion parameters map",
        default="Diffusion_parameters_maps-20230215T134959Z-001"
    )

    parser.add_argument(
        "-ep",
        "--epochs",
        metavar="",
        help="Number of epochs for training",
        default=1000,
    )
    
    parser.add_argument(
        "-bs",
        "--batchsize",
        metavar="",
        help="Batch size for training",
        default=10,
    )

    args = parser.parse_args()

    # IMPORT DATASET OF IMAGES AND LABELS
    images, labels = CNN_multi_utilities.import_dataset()
    
    # FUNCTION THAT PERFORMS DATA AUGMENTATION
    augmented_images, augmented_labels = CNN_multi_utilities.data_augmentation(images, labels)
    
    # TRAIN SPLIT
    x_train, y_train, x_val, y_val, x_test, y_test = CNN_multi_utilities.train_val_test_split(augmented_images,
                                                                                              augmented_labels)

    # CALLING THE MODEL
    shape = (110, 110, 3)
    model = CNN_multi_class.MyModel(shape)
    
    # OPTION TO LOAD PREVIOUS WEIGHTS AND CONTINUE TRAINING
    option = input("Load Previous weights for training? Y or N: ")
    
    if option.lower()[0] == "n":
        model.compile_and_fit(x_train, y_train, x_val, y_val, x_test, y_test, args.epochs, args.batchsize)
    elif option.lower()[0] == "y":
        model.load('model.h5', x_train, y_train, x_val, y_val, x_test, y_test, args.epochs, args.batchsize)
