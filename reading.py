import os


def data_path(dir, subdir):
    """
    Creates a list collecting absolute paths to the files contained in a sub-folder of a parent folder.

    Parameters
    ----------
        dir : str
            Name of the parent folder.
        subdir : str
            Name of the sub-folder.

    Returns
    -------
        file_paths : list
            Paths to the files contained in the specified sub-folder.
    """

    roots = []
    file_paths = []

    for root, dirs, files in os.walk(dir):    
        for name in files:
            roots.append(os.path.join(root, name))
            
    for i, word in enumerate(roots):
        if subdir in word:
            file_paths.append(roots[i])

    if "segmentation" in subdir:
        file_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[2]))
    else:
        file_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))

    return file_paths
