import uproot
import pandas as pd
import numpy as np
import Identification as id
import awkward as ak
import Creation_plus_filtred as cpf

# Open the ROOT file

def preparation_data(file_in,file_out,branch_of_interest,p_min,p_max,Ih_max):
    """Preparation of the file for the ML process
    Input : file_in : name of the file to open (type : string)
            file_out : name of the file to save (type : string)
            branch_of_interest : list of the branches to extract (type : list)
    Output : None
    Effect : Create a new file with the branches of interest and the new branches dedx_cluster and Ih that has the name file_out"""
    mass_limit = 0.789 # determined empirically
    scaling = 1e3 # scaling factor for the Bethe-Bloch curve determined empirically
    data = pd.DataFrame()
    with uproot.open(file_in) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy 

    data_filtered = data[data['track_p'] <= p_min ].reset_index(drop=True) #take only particle with momentum less than pmin GeV
    data_filtered = data_filtered[data_filtered['track_p'] >= p_max].reset_index(drop=True) #take only particle with momentum more than pmax GeV
    data_filtered['dedx_cluster']=data_filtered['dedx_charge']/data_filtered['dedx_pathlength'] #calculate dedx and create a new column
    data_filtered['Ih'] = np.sqrt(ak.sum(data_filtered['dedx_cluster']**2, axis=-1) / ak.count(data_filtered['dedx_cluster'], axis=-1)) #calculate quadratique mean of dedx along a track
    data_filtered=data_filtered[data_filtered['Ih'] <= Ih_max].reset_index(drop=True) #Premier filtrage sur les données dedx
    data_filtered = data_filtered[(data_filtered['Ih'] >= id.bethe_bloch(mass_limit, data_filtered['track_p']) * scaling)] #Filtrage du bruit 
    
# Save the manipulated DataFrame to a new ROOT file
    with uproot.recreate(file_out) as new_file:
        new_file["tree_name"] = { "dedx_cluster": data_filtered['dedx_cluster'], "track_p": data_filtered['track_p'],'Ih':data_filtered['Ih'], 'track_eta': data_filtered['track_eta']  }


def preparation_data2(data,file_out,branch_of_interest_out,p_min,p_max,Ih_max,particle_type):
    """Preparation of the file for the ML process after the use of Creation_plsu_filtred.py
    Input : data : dataframe (optimaly in output of cpf.filtrage_dedx) (type : string)
            file_out : name of the file to save (type : string)
            branch_of_interest : list of the branches to extract (type : list)
    Output : None
    Effect : Create a new file with the branches of interest and the new branches dedx_cluster and Ih that has the name file_out"""
    mass_limit = 0.789 # determined empirically
    scaling = 1e3 # scaling factor for the Bethe-Bloch curve determined empirically
    data['dedx_charge'] = data['dedx_charge'].apply(lambda x: np.asarray(x))
    data['dedx_pathlength'] = data['dedx_pathlength'].apply(lambda x: np.asarray(x))
    data['dedx_cluster'] = data['dedx_charge'] / data['dedx_pathlength']

    data_filtered = data[data['track_p'] <= p_min].reset_index(drop=True) #take only particle with momentum less than p_min GeV
    data_filtered = data_filtered[data_filtered['track_p'] >= p_max].reset_index(drop=True) #take only particle with momentum more than pmax GeV
    data_filtered['dedx_cluster']=data_filtered['dedx_charge']/data_filtered['dedx_pathlength'] #calculate dedx and create a new column
    data_filtered['Ih'] = np.sqrt(ak.sum(data_filtered['dedx_cluster']**2, axis=-1) / ak.count(data_filtered['dedx_cluster'], axis=-1)) #calculate quadratique mean of dedx along a track
    data_filtered=data_filtered[data_filtered['Ih'] <= Ih_max].reset_index(drop=True) # First filtering on the dedx data
    if particle_type == "proton":
        data_filtered = data_filtered[(data_filtered['Ih'] <= id.bethe_bloch(mass_limit, data_filtered['track_p']) * scaling)] # filtering of other particles than proton
    if particle_type == "kaon":
        data_filtered = data_filtered[(data_filtered['Ih'] <= id.bethe_bloch(mass_limit, data_filtered['track_p']) * scaling)] #Flitering of other particles than kaon
   
    # Save the manipulated DataFrame to a new ROOT file
    with uproot.recreate(file_out) as new_file: # create a new file to save the filtered data 
        new_file["tree_name"] = {branch: data_filtered[branch] for branch in branch_of_interest_out if branch in data_filtered}




if __name__ == "__main__" :  
    branch_of_interest = ["dedx_charge", "dedx_pathlength", "track_p"]
    branch_of_interest_LSTM_in = ["dedx_charge", "dedx_pathlength", "track_p","track_eta"]
    branch_of_interest_LSTM_out = ["dedx_pathlength","dedx_cluster", "track_p","track_eta", "Ih","ndedx"]
    branch_of_interest_LSTM_V3_in = ["dedx_charge", "dedx_pathlength", "track_p","track_eta","dedx_modulegeom"]
    branch_of_interest_LSTM_V3_out = ["ndedx","dedx_pathlength","dedx_cluster","dedx_modulegeom", "track_p","track_eta", "Ih",]

    # data = cpf.filtrage_dedx("Root_files/tree.root",["dedx_charge", "dedx_pathlength", "track_p","track_eta"],False,False,False)
    # preparation_data2(data,"ML_training_LSTM_non_filtré.root",branch_of_interest_LSTM_out,0,1.2,15000)

    # filtred_data = cpf.filtrage_dedx("Root_files/data.root",branch_of_interest_LSTM_in,True,True,True)
    # preparation_data2(filtred_data,"data_real_kaon.root",branch_of_interest_LSTM_out,0,1.2,15000)

    # data_plot = cpf.filtrage_dedx("Root_Files/signal.root",["dedx_charge", "dedx_pathlength", "track_p","track_eta"],True,True,True)
    # preparation_data2(data_plot,"Signal_Plot.root",branch_of_interest_LSTM,0,1.2,15000)

    # preparation_data("Root_files/tree.root","Root_files/ML_training_1.2.root",branch_of_interest_LSTM,0,1.2,15000)

    data_V3 = cpf.filtrage_dedx("Root_files/data.root",branch_of_interest_LSTM_V3_in,True,True,True)
    preparation_data2(data_V3,"data_real_filtred.root",branch_of_interest_LSTM_V3_out,0,1.2,15000,"proton")