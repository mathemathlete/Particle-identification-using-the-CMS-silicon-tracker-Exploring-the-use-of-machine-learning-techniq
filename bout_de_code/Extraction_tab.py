import uproot
import pandas as pd
import numpy as np

def import_data(data):
    file_name = "Root_Files/data_GRU_V3.root"
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        # tree.show()
        data_tree = tree.arrays(library="pd")  # open data with panda
        for array_name, array in data_tree.items():
            print(array_name)
            data[array_name] = array
    return(data)

data = pd.DataFrame()
print(import_data(data))