"""Useful functions for the CNN-2.5D"""

import reading
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomRotation, RandomZoom, RandomCrop, RandomContrast


def import_dataset(k_slice=45):
    """
    Extracts the paths for the FA, MD, and AD diffusion parameter maps and uses them to extract the corresponding NIFTI
    images, which are subsequently converted to a numpy array.

    Parameters
    ----------
    k_slice : int
        Selected slice. Default value 45 is the slice centered on the hippocampus.

    Returns
    -------
    images : ndarray
        Numpy array containing the DTI images of the subjects. Each slice corresponds to a different parameter (FA, MD
        and AD in order).
    labels : ndarray
        Numpy array containing the corresponding label for each DTI image. Value 1 indicates a subject with alzheimer's
        disease.
    """
    paths_FA = reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001", "corrected_FA_image")
    paths_MD = reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001", "corrected_MD_image")
    paths_AD = reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001", "corrected_AD_image")

    dataset = pd.DataFrame(pd.read_csv('ADNI_dataset_diffusion.csv'))
    dataset.sort_values(by=["Subject"], inplace=True, ignore_index=True)
    dataset["Path FA"] = paths_FA
    dataset["Path MD"] = paths_MD
    dataset["Path AD"] = paths_AD

    images_list = []

    for i, datapath in enumerate(dataset["Path FA"]):
        image_FA = np.asarray(nib.load(dataset["Path FA"][i]).get_fdata())
        image_MD = np.asarray(nib.load(dataset["Path MD"][i]).get_fdata())
        image_AD = np.asarray(nib.load(dataset["Path AD"][i]).get_fdata())
        image = np.stack((image_FA[k_slice], image_MD[k_slice], image_AD[k_slice]), axis=-1)
        images_list.append(image)

    images = np.array(images_list, dtype='float64')
    labels = np.asarray(dataset["Group"], dtype='float64')

    return images, labels


def data_augmentation(images, labels):
    """
    Applies random rotations, zooms, crops and contrast enhancements to increase the size of the input dataset.

    Parameters
    ----------
    images : ndarray
        Numpy array containing the DTI images of the subjects.
    labels : ndarray
        Numpy array containing the corresponding label for each DTI image.

    Returns
    -------
    augmented_images : ndarray
        Numpy array containing the original DTI images and the augmented ones.
    augmented_labels : ndarray
        Numpy array containing the corresponding label for each image (original and augmented).
    """
    aug_rotation = Sequential([RandomRotation((-0.5, 0.5))])
    aug_zoom_1 = Sequential([RandomZoom(0.5)])
    aug_zoom_2 = Sequential([RandomZoom(0.6)])
    aug_zoom_3 = Sequential([RandomZoom(0.65)])
    aug_zoom_4 = Sequential([RandomZoom(0.7)])
    aug_crop = Sequential([RandomCrop(110, 110, seed=3)])
    aug_contrast_1 = Sequential([RandomContrast(1, seed=5)])
    aug_contrast_2 = Sequential([RandomContrast(0.8, seed=8)])

    augmented_images = np.concatenate(
        (
            images,
            aug_rotation(images),
            aug_zoom_1(images),
            aug_zoom_2(images),
            aug_zoom_3(images),
            aug_zoom_4(images),
            aug_crop(images),
            aug_contrast_1(images),
            aug_contrast_2(images),
        ),
        axis=0
    )

    augmented_labels = np.concatenate(
        (
            labels,
            labels,
            labels,
            labels,
            labels,
            labels,
            labels,
            labels,
            labels,
        )
    )

    return augmented_images, augmented_labels


def train_val_test_split(images, labels):
    """
    Splits arrays into random train, validation and test subsets.

    Parameters
    ----------
    images : ndarray
        Numpy array containing the DTI images of the subjects.
    labels : ndarray
        Numpy array containing the corresponding label for each DTI image.

    Returns
    -------
    x_train : ndarray
        Numpy array containing the train subset of DTI images.
    y_train : ndarray
        Numpy array containing the labels corresponding to x_train.
    x_val : ndarray
        Numpy array containing the validation subset of DTI images.
    y_val : ndarray
        Numpy array containing the labels corresponding to x_val.
    x_test : ndarray
        Numpy array containing the test subset of DTI images.
    y_test : ndarray
        Numpy array containing the labels corresponding to x_test.
    """

    X_train, x_test, Y_train, y_test = train_test_split(images[:, :, :], labels, test_size=0.2, random_state=10)
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=20)

    return x_train, y_train, x_val, y_val, x_test, y_test
