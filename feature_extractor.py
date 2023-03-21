import numpy as np
import pandas as pd
import matlab.engine


def feature_extractor(image_filepaths, masks_filepaths):
    """_summary_

    Args:
        image_filepaths (_type_): _description_
        masks_filepaths (_type_): _description_
    """
# Start MATLAB engine
    eng = matlab.engine.start_matlab()

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
    df_group = pd.DataFrame(pd.read_csv('ADNI_dataset_diffusion.csv'))
    df_group.sort_values(by=["Subject"], inplace=True)
    group = df_group["Group"]

    return df_mean, df_std, group


def feature_extractor_par(image_filepaths, masks_filepaths):
    """_summary_

    Args:
        image_filepaths (_type_): _description_
        masks_filepaths (_type_): _description_
    """
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    # Call a MATLAB function
    [region, mean, std] = eng.feature_extractor(image_filepaths, masks_filepaths, nargout=3)
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
