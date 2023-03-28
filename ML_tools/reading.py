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
    roots = []
    filepaths = []

    for root, dirs, files in os.walk(dir):
        for name in files:
            roots.append(os.path.join(root, name))

    for i, word in enumerate(roots):
        if subdir in word:
            filepaths.append(roots[i])

    if "segmentation" in subdir:
        filepaths.sort(key=lambda x: int(os.path.basename(x).split('_')[2]))
    else:
        filepaths.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))

    return filepaths


def data_path_general(dir, subdir):
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
    roots = []
    filepaths = []

    for root, dirs, files in os.walk(dir):
        for name in files:
            roots.append(os.path.join(root, name))

    for i, word in enumerate(roots):
        if subdir in word:
            filepaths.append(roots[i])

    filepaths.sort(key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[1]))

    return filepaths
