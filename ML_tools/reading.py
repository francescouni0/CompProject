import os
import nibabel as nib
import sys
from pathlib import Path
import os
import re

sys.path.insert(0, str(Path(os.getcwd()).parent))
os.chdir('..')


def data_path(dir, subdir):
    """
    Creates a list collecting absolute paths to the files contained in a sub-folder of a parent folder.

    Parameters
    ----------
    dir : str
        Name of the parent folder.
    subdir : str
        Name of the parent folder.

    Returns
    -------
    filepaths : list
        Paths to the files contained in the specified sub-folder.
    """
    r = []
    filepaths = []

    for root, dirs, files in os.walk(dir):    
        for name in files:
            r.append(os.path.join(root, name))

    for i, word in enumerate(r):
        if subdir in word:
            filepaths.append(r[i])

    if "segmentation" in subdir:
        filepaths.sort(key=lambda x: int(os.path.basename(x).split('_')[2]))
    else:
        filepaths.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))

    return filepaths




def data_path_mod(dir, subdir):
    """
    Creates a list collecting absolute paths to the files contained in a sub-folder of a parent folder.

    Parameters
    ----------
    dir : str
        Name of the parent folder.
    subdir : str
        Name of the parent folder.

    Returns
    -------
    filepaths : list
        Paths to the files contained in the specified sub-folder.
    """
    r = []
    filepaths = []

    for root, dirs, files in os.walk(dir):    
        for name in files:
            r.append(os.path.join(root, name))

    for i, word in enumerate(r):
        if subdir in word:
            filepaths.append(r[i])

    filepaths.sort(key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[1]))

    return filepaths



