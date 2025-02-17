import uproot
import pandas as pd
import numpy as np
import awkward as ak

branch_of_interest = ["dedx_charge", "dedx_pathlength", "track_p"]

# Open the ROOT file
file_name = "Root_files/signal.root"
data = pd.DataFrame()
with uproot.open(file_name) as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy 

data_filtered = data[data['track_p'] <= 5 ].reset_index(drop=True) #take only particle with momentum less than 5 GeV

data_filtered['dedx_cluster']=data_filtered['dedx_charge']/data_filtered['dedx_pathlength'] #calculate dedx and create a new column
data_filtered['Ih'] = np.sqrt(ak.sum(data_filtered['dedx_cluster']**2, axis=-1) / ak.count(data_filtered['dedx_cluster'], axis=-1)) #calculate quadratique mean of dedx along a track

# Save the manipulated DataFrame to a new ROOT file

with uproot.recreate("signal_filtré.root") as new_file:
    new_file["tree_name"] = { "track_p": data_filtered['track_p'], "dedx_cluster" : data_filtered['dedx_cluster'], "Ih": data_filtered['Ih']}