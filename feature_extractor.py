import os
import numpy as np
from reading import load
from reading import data_path

import time

import matlab.engine
#QUELLO CHE VOGLIO CERCARE DI FARE ORA È FARE UNA MAPPA STATISTICA DELLE FEATURE

paths_FA= data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_FA_image")
paths_masks=data_path("Diffusion_space_segmentations-20230215T134839Z-001","Diffusion_space_segmentations-20230215T134839Z-001")


#QUESTO MI ORDINA TUTTO IN MODO CHE MASCHERA E FILE CORRISPONDANO DA AGGIUNGERE ALLA DEFINIZIONE DI DATAPATH
#SOLO CHE MASCHERE E MAPPE VANNO SORTATE IN MANIERA DIVERSA
paths_FA.sort(key=lambda x: int(os.path.basename(x).split('_')[3][1:]))
paths_masks.sort(key=lambda x: int(os.path.basename(x).split('_')[2][1:]))





#DEFINISCO FUNZIONE DI ESTRAZIONE FEATURE

def feature_extractor(image_filepaths, masks_filepaths):
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()






# Call a MATLAB function
    [region, mean, std] = eng.feature_extractor(image_filepaths, masks_filepaths, nargout=3)


# Stop MATLAB engine
    eng.quit()
# Return the result
    return region, np.asarray(mean), np.asarray(std)


if __name__ == "__main__":
    start = time.time()
    [r, m, s] = feature_extractor(paths_FA, paths_masks)
    end = time.time()
    print('Elapsed time: ', end-start)
    print(paths_FA[0])
    print(paths_masks[0])
    print(m[:,1])
    #print(s[:,1])
