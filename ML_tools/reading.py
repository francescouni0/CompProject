import os
import nibabel as nib
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(os.getcwd()).parent))
os.chdir('..')



# La funzione scorre tutte le cartelle e seleziona i file nelle cartelle e sottocartelle
# selezionate, restituisce un array di paths per i file della sottocartella
def data_path(dir,subdir):
    """_summary_

    Args:
        dir (_type_): _description_
        subdir (_type_): _description_

    Returns:
        _type_: _description_
    """
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


