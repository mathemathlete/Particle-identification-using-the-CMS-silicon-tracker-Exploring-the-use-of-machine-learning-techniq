import uproot
import pandas as pd
import numpy as np

data = pd.DataFrame()
with uproot.open("slim_nt_mc_aod_1.root") as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    # tree.show()
    data_tree = tree.arrays(library="pd") # open data with array from numpy 
    for array_name, array in data_tree.items(): # loop over the arrays in the tree
        if array_name != "nstrips":
            data[array_name] = array
        if array_name == "dedx_shape":
            B = []
            for i in range(len(array)):
                B.append(len(array[i]))
    data["Number of elements"]= B 

array_dx=data['dedx_pathlength']
array_de=data['dedx_charge']

print(array_dx[52].sum())