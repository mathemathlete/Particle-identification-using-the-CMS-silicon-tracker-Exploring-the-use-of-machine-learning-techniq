import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak     
import uproot

def import_data(file_name, branch_of_interest):
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy
    return data

def affichage (filename):
    #branch_of_interest=["Ih","track_p"]
    branch_of_interest = ["dedx_charge", "dedx_pathlength", "track_p"]

    data=pd.DataFrame()
    data=import_data(filename,branch_of_interest)
    data['dedx_cluster']=data['dedx_charge']/data['dedx_pathlength'] 
    data['Ih'] = np.sqrt(ak.sum(data['dedx_cluster']**2, axis=-1) / ak.count(data['dedx_cluster'], axis=-1))
    # data = data[data['track_p'] <= 5000 ].reset_index(drop=True)
    # data = data[data['track_p'] >= 500 ].reset_index(drop=True)

    data = data[data['Ih'] <= 15000 ].reset_index(drop=True)
    data = data[data['track_p'] >= 5 ].reset_index(drop=True)

    plt.figure(1)
    
    plt.hist2d(data["track_p"], data["Ih"], bins=500, cmap='viridis', label='Data')
    plt.colorbar(label='Counts')
    plt.xlabel(r'p')
    plt.ylabel(r'$-(\frac{dE}{dx}$)')
    plt.title('Bethe-Bloch fit data')
    plt.grid(True)
    plt.legend()

    plt.show()  

if __name__ == "__main__":
    filename = "Root_files\signal.root"
    affichage(filename)
