import uproot
import pandas as pd

with uproot.open("slim_nt_mc_aod_1.root") as file:
    key = file.keys()[0] # open the first Ttree
    tree = file[key]    
    data = tree.arrays(library="np") # open data with array from numpy 
    for array_name, array in data.items(): # loop over the arrays in the tree
        #print("\n \n \n \n")
        print(f"Array {array_name}:")
        print(array)