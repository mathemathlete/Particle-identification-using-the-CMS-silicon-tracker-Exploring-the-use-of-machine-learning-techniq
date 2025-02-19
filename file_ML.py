import uproot
import pandas as pd
import numpy as np
import Identification as id
import awkward as ak
import Creation_plus_filtrage as cpf

# Open the ROOT file

def preparation_data(file_in,file_out,branch_of_interest):
    """Préparation du fichier sans filtrage préalable"""
    data = pd.DataFrame()
    with uproot.open(file_in) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy 

    data_filtered = data[data['track_p'] <= 1.2 ].reset_index(drop=True) #take only particle with momentum less than 5 GeV
    data_filtered['dedx_cluster']=data_filtered['dedx_charge']/data_filtered['dedx_pathlength'] #calculate dedx and create a new column
    data_filtered['Ih'] = np.sqrt(ak.sum(data_filtered['dedx_cluster']**2, axis=-1) / ak.count(data_filtered['dedx_cluster'], axis=-1)) #calculate quadratique mean of dedx along a track
    data_filtered=data_filtered[data_filtered['Ih'] <= 14000].reset_index(drop=True) #Premier filtrage sur les données dedx
    data_filtered = data_filtered[(data_filtered['Ih'] >= id.bethe_bloch(mass_limit, data_filtered['track_p']) * scaling)] #Filtrage du bruit 
    
# Save the manipulated DataFrame to a new ROOT file
    with uproot.recreate(file_out) as new_file:
        new_file["tree_name"] = { "dedx_cluster": data_filtered['dedx_cluster'], "track_p": data_filtered['track_p'],'Ih':data_filtered['Ih'], 'track_eta': data_filtered['track_eta']  }


def preparation_data2(data,file_out,branch_of_interest_out):
    """Preparer avec Filtrage"""
    data['dedx_charge'] = data['dedx_charge'].apply(lambda x: np.asarray(x))
    data['dedx_pathlength'] = data['dedx_pathlength'].apply(lambda x: np.asarray(x))
    data['dedx_cluster'] = data['dedx_charge'] / data['dedx_pathlength']

    data_filtered = data[data['track_p'] <= 1.2].reset_index(drop=True) #take only particle with momentum less than 5 GeV
    data_filtered['dedx_cluster']=data_filtered['dedx_charge']/data_filtered['dedx_pathlength'] #calculate dedx and create a new column
    data_filtered['Ih'] = np.sqrt(ak.sum(data_filtered['dedx_cluster']**2, axis=-1) / ak.count(data_filtered['dedx_cluster'], axis=-1)) #calculate quadratique mean of dedx along a track
    data_filtered=data_filtered[data_filtered['Ih'] <= 15000].reset_index(drop=True) #Premier filtrage sur les données dedx
    data_filtered = data_filtered[(data_filtered['Ih'] >= id.bethe_bloch(mass_limit, data_filtered['track_p']) * scaling)] #Filtrage du bruit 
    data_filtered=data_filtered[data_filtered['Ih'] <= 14000].reset_index(drop=True) #Premier filtrage sur les données dedx
    data_filtered = data_filtered[(data_filtered['Ih'] <= id.bethe_bloch(mass_limit, data_filtered['track_p']) * scaling)] #Filtrage du bruit 
# Save the manipulated DataFrame to a new ROOT file
    with uproot.recreate(file_out) as new_file:
        new_file["tree_name"] = {branch: data_filtered[branch] for branch in branch_of_interest_out if branch in data_filtered}




if __name__ == "__main__" :  
    mass_limit = 0.789 # determined empirically
    scaling = 1e3 # scaling factor for the Bethe-Bloch curve determined empirically
    branch_of_interest = ["dedx_charge", "dedx_pathlength", "track_p"]
    branch_of_interest_LSTM_in = ["dedx_charge", "dedx_pathlength", "track_p","track_eta","dedx_modulegeom"]
    branch_of_interest_LSTM_out = ["dedx_pathlength","dedx_cluster", "track_p","track_eta", "Ih","ndedx"]

    data = cpf.filtrage_dedx("Root_files/tree.root",["dedx_charge", "dedx_pathlength", "track_p","track_eta"],False,False,False)
    preparation_data2(data,"ML_training_LSTM_non_filtré.root",branch_of_interest_LSTM_out)

    # filtred_data = cpf.filtrage_dedx("Root_files/data.root",branch_of_interest_LSTM_in,True,True,True)
    # preparation_data2(filtred_data,"data_real_kaon.root",branch_of_interest_LSTM_out)

    # data_plot = cpf.filtrage_dedx("Root_Files/signal.root",["dedx_charge", "dedx_pathlength", "track_p","track_eta"],True,True,True)
    # preparation_data2(data_plot,"Signal_Plot.root",branch_of_interest_LSTM)

    # preparation_data("Root_files/tree.root","Root_files/ML_training_1.2.root",branch_of_interest_LSTM)