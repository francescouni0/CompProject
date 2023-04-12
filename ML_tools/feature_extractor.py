import numpy as np
import pandas as pd
import matlab.engine
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(os.getcwd()).parent))


def feature_extractor(image_filepaths, masks_filepaths):
    """
    Uses the MATLAB Engine API to run the feature_extractor.m function. From the outputs of that function, it defines 2
    dataframes containing the extracted features and a series containing the labels of the respective subjects.

    Parameters
    ----------
        image_filepaths : list
            Paths to the diffusion parameters maps.
        masks_filepaths : list
            Paths to the diffusion space segmentations.

    Returns
    -------
        df_mean : pandas.DataFrame
            Mean of pixel values for each region (columns) and each subject (rows).
        df_std : pandas.DataFrame
            Standard deviation of pixel values for each region (columns) and each subject (rows).
        group : pandas.Series
            Subject labels.
    """

# Start MATLAB engine
    eng = matlab.engine.start_matlab()
    
    eng.addpath('./ML_tools')
    
    current_folder=(eng.pwd())

# Call a MATLAB function
    [region, mean, std] = eng.feature_extractor(image_filepaths, masks_filepaths, nargout=3)
# Stop MATLAB engine
    eng.quit()
# Create Pd dataframe
    n_regxsub = np.shape(mean[:][1])
    mean_t = np.transpose(np.asarray(mean))
    std_t = np.transpose(np.asarray(std))
    df_mean = pd.DataFrame(mean_t[1:, 1:(n_regxsub[0]-1)],
                           index=mean[0][1:(n_regxsub[1])],
                           columns=region[1:(n_regxsub[0]-1)])
    df_std = pd.DataFrame(std_t[1:, 1:(n_regxsub[0]-1)],
                          index=std[0][1:(n_regxsub[1])],
                          columns=region[1:(n_regxsub[0]-1)])
    df_group = pd.read_csv('ADNI_dataset_diffusion.csv')
    df_group.sort_values(by=["Subject"], inplace=True)
    group = df_group["Group"]

    return df_mean, df_std, group


def feature_extractor_par(image_filepaths, masks_filepaths):
    """
    Uses the MATLAB Engine API to run the feature_extractor_par.m function (parallelized version of feature_extractor.m).
    From the outputs of that function, it defines 2 dataframes containing the extracted features and an array containing
    the labels of the respective subjects.

    Parameters
    ----------
        image_filepaths : list
            Paths to the diffusion parameters maps.
        masks_filepaths : list
            Paths to the diffusion space segmentations.

    Returns
    -------
        df_mean : pandas.DataFrame
            Mean of pixel values for each region (columns) and each subject (rows).
        df_std : pandas.DataFrame
            Standard deviation of pixel values for each region (columns) and each subject (rows).
        group : pandas.Series
            Subject labels.
    """
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    eng.addpath('./ML_tools')

    current_folder = (eng.pwd())

    # Call a MATLAB function
    [region, mean, std] = eng.feature_extractor_par(image_filepaths, masks_filepaths, nargout=3)
    # Stop MATLAB engine
    eng.quit()
    # Create Pd dataframe
    n_regxsub = np.shape(mean[:][1])
    mean_t = np.transpose(np.asarray(mean))
    std_t = np.transpose(np.asarray(std))
    df_mean = pd.DataFrame(mean_t[1:, 1:(n_regxsub[0] - 1)],
                           index=mean[0][1:(n_regxsub[1])],
                           columns=region[1:(n_regxsub[0] - 1)])
    df_std = pd.DataFrame(std_t[1:, 1:(n_regxsub[0] - 1)],
                          index=std[0][1:(n_regxsub[1])],
                          columns=region[1:(n_regxsub[0] - 1)])
    df_group = pd.DataFrame(pd.read_csv('ADNI_dataset_diffusion.csv'))
    df_group.sort_values(by=["Subject"], inplace=True)
    group = df_group["Group"]

    return df_mean, df_std, group
