import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

# La funzione scorre tutte le cartelle e seleziona i file nelle cartelle e sottocartelle
# selezionate, restituisce un array di paths per i file della sottocartella
def data_path(dir,subdir):
    r = []
    a = []

    for root, dirs, files in os.walk(dir):    
            for name in files:
                r.append(os.path.join(root, name))
            
    
    for i, word in enumerate(r):
        
        if subdir in word:
            
            a.append(r[i])
            
                    
                    
    return a
#CERCARE DI FARE UNA FUNZIONE CHE DEFINISCE QUESTE VARIABILI

paths_AD= data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_AD_image")
paths_MD= data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_MD_image")
paths_RD= data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_RD_image")
paths_FA= data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_FA_image")

#Prende un array di paths e fa il caricamento delle immagini, restituisce un array
#di oggetti nib
def load(a):
    f=[]
    for i,word in enumerate(a):
        
        f.append(nib.load(a[i]))
        
    return  f
    
