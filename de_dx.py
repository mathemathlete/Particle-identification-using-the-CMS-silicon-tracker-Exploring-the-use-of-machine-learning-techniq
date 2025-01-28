import uproot
import pandas as pd
import awkward as ak

branch_of_interest = ["dedx_charge", "dedx_pathlength", "track_p"]

# Open the ROOT file
file = uproot.open("slim_nt_mc_aod_992.root")
data = pd.DataFrame()
with uproot.open("slim_nt_mc_aod_992.root") as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy 

data_filtered = data[data['track_p'] <= 5 ]

dedx=data_filtered['dedx_charge']/data_filtered['dedx_pathlength']
It=pd.DataFrame()
# for i in range (len(dedx)):
#     for j in range (len(dedx[i])):
#         It[i]=((dedx[i][j])**2)
#     It[i]=(It[i]/len(dedx[i]))**0.5

print(len(dedx[5][:]))

# Save the manipulated DataFrame to a new ROOT file
with uproot.recreate("clean_p.root") as new_file:
    new_file["tree_name"] = data_filtered