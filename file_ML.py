import uproot
import pandas as pd
import numpy as np
import awkward as ak

branch_of_interest = ["dedx_charge", "dedx_pathlength", "track_p"]

# Open the ROOT file
file = uproot.open("root/tree.root")
data = pd.DataFrame()
with uproot.open("root/tree.root") as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy 

data_filtered = data[data['track_p'] <= 5 ].reset_index(drop=True) #take only particle with momentum less than 5 GeV

data_filtered['dedx_cluster']=data_filtered['dedx_charge']/data_filtered['dedx_pathlength'] #calculate dedx and create a new column
# Save the manipulated DataFrame to a new ROOT file

with uproot.recreate("de_dx_cluster.root") as new_file:
    new_file["tree_name"] = { "dedx_cluster": data_filtered['dedx_cluster'], "track_p": data_filtered['track_p'] }