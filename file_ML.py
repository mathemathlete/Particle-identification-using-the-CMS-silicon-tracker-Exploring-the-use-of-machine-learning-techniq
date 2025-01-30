import uproot
import pandas as pd
import numpy as np
import Identification as id
import awkward as ak


mass_limit = 0.789 # determined empirically
scaling = 1e3 # scaling factor for the Bethe-Bloch curve determined empirically
branch_of_interest = ["dedx_charge", "dedx_pathlength", "track_p"]

# Open the ROOT file

def preparation_data(file_in,file_out):
    data = pd.DataFrame()
    with uproot.open(file_in) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy 

    data_filtered = data[data['track_p'] <= 2 ].reset_index(drop=True) #take only particle with momentum less than 5 GeV
    data_filtered['dedx_cluster']=data_filtered['dedx_charge']/data_filtered['dedx_pathlength'] #calculate dedx and create a new column
    data_filtered['Ih'] = np.sqrt(ak.sum(data_filtered['dedx_cluster']**2, axis=-1) / ak.count(data_filtered['dedx_cluster'], axis=-1)) #calculate quadratique mean of dedx along a track
    data_filtered=data_filtered[data_filtered['Ih'] <= 12000].reset_index(drop=True) #Premier filtrage sur les données dedx
    data_filtered = data[(data['Ih'] >= id.bethe_bloch(mass_limit, data['track_p']) * scaling)] #Filtrage du bruit 

# Save the manipulated DataFrame to a new ROOT file
    with uproot.recreate(file_out) as new_file:
        new_file["tree_name"] = { "dedx_cluster": data_filtered['dedx_cluster'], "track_p": data_filtered['track_p'] }



