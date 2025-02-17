import uproot
import pandas as pd
import matplotlib.pyplot as plt
import sys
import Identification as id
import numpy as np
import awkward as ak

m_p = 938e-3# proton mass in eV
m_deut = 1875e-3 # deuteron mass in eV
m_pion = 139e-3 # pion mass in eV
m_kaon = 493e-3 # kaon mass in eV

mass_limit = 0.789 # determined empirically
scaling = 1e3 # scaling factor for the Bethe-Bloch curve determined empirically

def preparation_data(file_name, affichage=False):
<<<<<<< HEAD
    branch_of_interest = ["track_p","dedx_pathlength","dedx_charge"]
=======
    branch_of_interest = ["Ih", "track_p"]
    file_name="Root_files/signal_filtré.root"
>>>>>>> ebcae38ee43b596fd8d9f03ad6b72c09c5ac20e1
    data = pd.DataFrame()
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy 
<<<<<<< HEAD
    data['dedx_cluster']=data['dedx_charge']/data['dedx_pathlength'] #calculate dedx and create a new column
    data['Ih'] = np.sqrt(ak.sum(data['dedx_cluster']**2, axis=-1) / ak.count(data['dedx_cluster'], axis=-1)) #calculate quadratique mean of dedx along a track
=======
    print(data)
>>>>>>> ebcae38ee43b596fd8d9f03ad6b72c09c5ac20e1
    data=data[data['Ih'] <= 12000].reset_index(drop=True) #Premier filtrage sur les données dedx
    
    filtred_data = data[(data['Ih'] >= id.bethe_bloch(mass_limit, data['track_p']) * scaling) & (data['track_p'] < 2)].reset_index(drop=True) #Filtrage du bruit 



    filtred_p = filtred_data['track_p']
    filtred_dedx = filtred_data['Ih']

    if (affichage==True):
        unfiltred_p = data['track_p']
        unfiltred_dedx = data['Ih']
        return unfiltred_p, unfiltred_dedx, filtred_p, filtred_dedx
    else :
        return filtred_p, filtred_dedx

    
    

def affichage ():
<<<<<<< HEAD
    p, dedx, p2, dedx2 = preparation_data("Root_Files/signal.root", True)
=======
    p, dedx, p2, dedx2 = preparation_data("Root_files/signal_filtré.root", True)
>>>>>>> ebcae38ee43b596fd8d9f03ad6b72c09c5ac20e1

    plt.figure(1)
    p_values = np.logspace(np.log10(0.0001), np.log10(5), 5000)
    beth_bloch_curve_theory_kaon = id.bethe_bloch(m_kaon, p_values) * scaling
    beth_bloch_curve_theory_proton = id.bethe_bloch(m_p, p_values) * scaling
    plt.hist2d(p, dedx, bins=500, cmap='viridis', label='Data')
    plt.plot(p_values, beth_bloch_curve_theory_kaon, color='red', label='Beth-Bloch theory for Kaon')
    plt.plot(p_values, beth_bloch_curve_theory_proton, color='green', label='Beth-Bloch theory for proton')
    Separation = id.bethe_bloch(mass_limit, p_values) * scaling
    plt.plot(p_values, Separation, color='black', label='Separation')
    plt.colorbar(label='Counts')
    plt.xlabel(r'p')
    plt.ylabel(r'$-(\frac{dE}{dx}$)')
    plt.title('Bethe-Bloch fit of scale between data and theory')
    plt.grid(True)
    plt.legend()

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.hist2d(p, dedx, bins=500, cmap='viridis')
    plt.colorbar(label='Counts')
    plt.xlabel(r'p')
    plt.ylabel(r'$-(\frac{dE}{dx}$)')
    plt.title('Unfiltred Data')
    plt.xlim(0,5)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.hist2d(p2, dedx2, bins=500, cmap='viridis')
    plt.colorbar(label='Counts')
    plt.xlabel(r'p')
    plt.ylabel(r'$-(\frac{dE}{dx}$)')
    plt.title('Filtred Data')
    plt.grid(True)
    plt.xlim(0,5)

    plt.show()  

if __name__ == "__main__":
    affichage()
