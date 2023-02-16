import os
import numpy as np

file=[]

for root, subfolders, filenames in os.walk("Diffusion_parameters_maps-20230215T134959Z-001"):
    file=np.append(file,filenames)
    #print(filenames)


print(len(file))
print(file[0:2])

