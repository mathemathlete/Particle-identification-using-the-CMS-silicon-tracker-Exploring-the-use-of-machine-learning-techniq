import uproot
import pandas as pd

with uproot.open("slim_nt_mc_aod_1.root") as file:
    key = file.keys()[0]
    tree = file[key]
    data = tree.arrays(library="np")
    for array_name, array in data.items():
        #print("\n \n \n \n")
        print(f"Array {array_name}:")
        print(array)