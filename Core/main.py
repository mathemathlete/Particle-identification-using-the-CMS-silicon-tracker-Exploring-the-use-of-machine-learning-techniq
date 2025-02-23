import ML_plot as ML
import subprocess 
import Creation_plus_filtred as cpf
import Identification as id
import file_ML as fml

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
# Choose the input file and the output file
file_name_in="Root_files/tree.root"
file_out="Root_files/tree_filtred.root"
data = cpf.filtrage_dedx(file_name_in,branch_of_interest_LSTM_V3_in,isstrip_ON,insideTkmod_ON,dedx_clusclean_ON)
fml.preparation_data2(data,"data_V3.root",branch_of_interest_LSTM_V3_out,0,1.2,15000)

######################################### Part where we train the ML model ###############################################
# execute Machine learning train
ML_train=False 
# execute plot ML_output and choose the parameter
plot_1=False # 1D plot the prediction of ML 

hist= True # Histogram 1D of Ecart between theory and prediction
hist_2=True # Histogram 2D of Ecart between theory and prediction and 
path_ML="ML_out.root"
# execute plot of the difference between Ih formula and ML prediction
Test_ML=True

path_in="ML_in.root"
path_out="ML_out.root"
Hist=True # Histogram 2D of de/dx with ML, with Ih and the difference between the two

if ML_train==True:
    subprocess.run(["python", "ML_functions_first.py"])

######################################### Part where we test the ML model ###############################################
if plot_1==True:
    branch_of_interest = ["dedx","track_p"]
    ML.plot_ML(path_ML, branch_of_interest, True ,hist,hist_2)

if Test_ML==True:
    ML.plot_diff_Ih(path_out,path_in, True,Hist)