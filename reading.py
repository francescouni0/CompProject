import os
import nibabel as nib

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
            
    
    if "segmentation" in subdir:
        a.sort(key=lambda x: int(os.path.basename(x).split('_')[2]))
    
    else:
        a.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))         
                    
                    
    return a

#






#Prende un array di paths e fa il caricamento delle immagini, restituisce un array
#di oggetti nib
def load(a):
    f=[]
    for i,word in enumerate(a):
        
        f.append(nib.load(a[i]))
        
    return  f
    
