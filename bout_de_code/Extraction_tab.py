import uproot
import pandas as pd
import numpy as np

def import_data(data):
    file_name = "tree.root"
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        # tree.show()
        data_tree = tree.arrays(["dedx_charge", "dedx_pathlength", "dedx_isstrip"],library="pd",entry_stop=10)  # open data with panda
        for array_name, array in data_tree.items():
            data[array_name] = array
            print(array_name)
    return(data)

data = pd.DataFrame()
print(import_data(data))
