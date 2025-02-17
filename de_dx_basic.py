import Creation_plus_filtrage as cpf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak     

def affichage ():
    branch_of_interest=["Ih","track_p"]
    #branch_of_interest = ["dedx_charge", "dedx_pathlength", "track_p"]

    data=cpf.import_data("Root_files/data_real.root",branch_of_interest)
    # data['dedx_cluster']=data['dedx_charge']/data['dedx_pathlength'] 
    # data['Ih'] = np.sqrt(ak.sum(data['dedx_cluster']**2, axis=-1) / ak.count(data['dedx_cluster'], axis=-1))
    # data = data[data['track_p'] <= 5000 ].reset_index(drop=True)
    # data = data[data['track_p'] >= 500 ].reset_index(drop=True)

    data = data[data['Ih'] <= 100000 ].reset_index(drop=True)
    data = data[data['track_p'] <= 100 ].reset_index(drop=True)

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
    affichage()
