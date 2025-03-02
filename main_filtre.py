from Core import ML_plot as ML
import subprocess 
from Core import Creation_plus_filtred as cpf
from Core import Identification as id
from Core import file_ML as fml
import sys
import os

######################################### Part where we filter the data ###############################################

# If we want to have a lighter file, we can filter the data with LSTM_in and LSTM_out that won't use modulegeom branch
# Else if we want to use the V3 ML model, we have to use LSTM_V3_in and LSTM_V3_out
branch_of_interest_LSTM_in = ["dedx_charge", "dedx_pathlength", "track_p","track_eta"]
branch_of_interest_LSTM_out = ["dedx_pathlength","dedx_cluster", "track_p","track_eta", "Ih","ndedx"]
branch_of_interest_LSTM_V3_in = ["dedx_charge", "dedx_pathlength", "track_p","track_eta","dedx_modulegeom"]
branch_of_interest_LSTM_V3_out = ["ndedx","dedx_pathlength","dedx_cluster","dedx_modulegeom", "track_p","track_eta", "Ih",]
# Choose the filters
isstrip_ON = True
insideTkmod_ON = True
dedx_clusclean_ON = True

# Choose the p range 
p_min = 0
p_max = 1.2
Ih_max = 15000
# Choose the input file and the output name of the file
file_name_in="Root_files/tree.root"
file_out="Root_files/tree_filtred.root"

data = cpf.filtrage_dedx(file_name_in,branch_of_interest_LSTM_V3_in,isstrip_ON,insideTkmod_ON,dedx_clusclean_ON)
fml.preparation_data2(data,"data_V3.root",branch_of_interest_LSTM_V3_out,0,1.2,15000)

