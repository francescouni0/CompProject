import os
import numpy as np
from reading import load
from reading import data_path
import pandas as pd
import time

import matlab.engine
#QUELLO CHE VOGLIO CERCARE DI FARE ORA È FARE UNA MAPPA STATISTICA DELLE FEATURE







#DEFINISCO FUNZIONE DI ESTRAZIONE FEATURE

def feature_extractor(image_filepaths, masks_filepaths):
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

# Call a MATLAB function
    [region, mean, std] = eng.feature_extractor(image_filepaths, masks_filepaths, nargout=3)


# Stop MATLAB engine
    eng.quit()
# Create Pd dataframe
    n_regxsub=np.shape(mean[:][1])
    mean_t=np.transpose(np.asarray(mean))
    std_t=np.transpose(np.asarray(std))
    df_mean = pd.DataFrame(mean_t[1:,1:(n_regxsub[0]-1)],index=mean[0][1:(n_regxsub[1])],columns=region[1:(n_regxsub[0]-1)])
    df_std=pd.DataFrame(std_t[1:,1:(n_regxsub[0]-1)],index=std[0][1:(n_regxsub[1])],columns=region[1:(n_regxsub[0]-1)])

    df_group=pd.DataFrame(pd.read_csv('ADNI_dataset_diffusion.csv'))
    df_group.sort_values(by=["Subject"],inplace=True)
    group=df_group["Group"]
    
    
    
    return df_mean, df_std, group





    #print(s[:,1])
