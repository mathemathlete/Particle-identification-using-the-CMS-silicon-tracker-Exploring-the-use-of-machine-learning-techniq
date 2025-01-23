import uproot
import pandas as pd
import numpy as np

def import_data(data):
    file_name = "slim_nt_mc_aod_1.root"
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        # tree.show()
        data_tree = tree.arrays(library="pd")  # open data with panda
        for array_name, array in data_tree.items():
            if array_name != "nstrips":
                data[array_name] = array
            if array_name == "dedx_shape":
                Nbre = []
                for i in range(len(array)):
                    Nbre.append(len(array[i]))
        data["Number of elements"] = Nbre
    return(data)
    