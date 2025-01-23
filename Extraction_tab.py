import uproot
import pandas as pd
import numpy as np

data = pd.DataFrame()
with uproot.open("slim_nt_mc_aod_1.root") as file:
    key = file.keys()[0]
    tree = file[key]
    # tree.show()
    data_tree = tree.arrays(library="pd")
    for array_name, array in data_tree.items():
        if array_name != "nstrips":
            data[array_name] = array
        if array_name == "dedx_shape":
            B = []
            for i in range(len(array)):
                B.append(len(array[i]))
    data["Number of elements"]= B 
print(data)