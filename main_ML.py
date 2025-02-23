from Core import ML_plot as ML
import subprocess 
from Core import Creation_plus_filtred as cpf
from Core import Identification as id
from Core import file_ML as fml
from Architecture_RNN import GRU_plus_LSTM_V1 as rnn
from Architecture_RNN import GRU_plus_LSTM_V2a as rnn2
from Architecture_RNN import GRU_plus_LSTM_V2b as rnn3
from Architecture_RNN import GRU_plus_LSTM_V3 as rnn4
from Architecture_RNN import GRU_plus_MLP_V1 as rnn5
from Architecture_RNN import GRU_plus_MLP_V2a as rnn6
from Architecture_RNN import GRU_plus_MLP_V2b as rnn7
from Architecture_RNN import GRU_plus_MLP_V3 as rnn8
import torch.nn as nn
import torch.optim as optim
import pandas as pd

#########################################  training/testing ML  ###############################################
ML_train=True 
ML_test=True

######################################### Choose the ML model  ###############################################
# for use the model you want you need to modify the file model and the import name line 6
time_start = rnn.timeit.default_timer()

file_name = "Root_Files/ML_training_LSTM.root" # choose your data file
branch_of_interest = ["ndedx_cluster","dedx_cluster","track_p","track_eta","Ih"]
file_model = "Models/best_model_GRU_LSTM_200epoch_V1.pth"

data=cpf.import_data(file_name,branch_of_interest)
train_data, test_data = rnn.train_test_split(data, test_size=0.25, random_state=42)

# --- prepare the training data  ---
ndedx_values_train = train_data["ndedx_cluster"].to_list()
dedx_values = train_data["dedx_cluster"].to_list()
data_th_values = id.bethe_bloch(938e-3, train_data["track_p"]).to_list()  # Targets (valeurs théoriques)
p_values_train = train_data["track_p"].to_list()
eta_values_train =  train_data["track_eta"].to_list()
Ih_values_train = train_data["Ih"].to_list()
dataset_rnn = rnn.ParticleDataset(ndedx_values_train, dedx_values, data_th_values,p_values_train,eta_values_train,Ih_values_train)
dataloader_rnn = rnn.DataLoader(dataset_rnn, batch_size=32, shuffle=True, collate_fn=rnn.collate_fn)

# --- prepare the test data ---
ndedx_values_test = test_data["ndedx_cluster"].to_list()
dedx_values_test = test_data["dedx_cluster"].to_list()
data_th_values_test = id.bethe_bloch(938e-3, test_data["track_p"]).to_list()
p_values_test = test_data["track_p"].to_list()
eta_values_test =  test_data["track_eta"].to_list()
Ih_values_test = test_data["Ih"].to_list()
test_dataset_rnn = rnn.ParticleDataset(ndedx_values_test,dedx_values_test, data_th_values_test,p_values_test,eta_values_test,Ih_values_test)
test_dataloader_rnn = rnn.DataLoader(test_dataset_rnn, batch_size=32, collate_fn=rnn.collate_fn)

# --- Initialisation du modèle, fonction de perte et optimiseur ---
dedx_hidden_size = 256
dedx_num_layers = 2   # With one layer, GRU dropout is not applied.
lstm_hidden_size = 64
lstm_num_layers = 2
adjustement_scale = 0.64
dropout_GRU = 0.18
dropout_dedx = 0.1
dropout_LSTM = 0.29
epoch = 20

model = rnn.LSTMModel(dedx_hidden_size, dedx_num_layers, lstm_hidden_size, lstm_num_layers, dropout_GRU, dropout_dedx, dropout_LSTM, adjustement_scale)


criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1.97e-6)

# Learning rate scheduler: reduce LR on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

######################################### Part where we test the ML model ###############################################
# --- Testing model ---
if ML_train==True:
    print("Training model...")

    ML.loss_epoch(rnn.start_ML(model,file_model,dataloader_rnn,criterion,epoch, True, False,False))

if ML_test==True:
    print("Testing model...")
    predictions, test_loss = rnn.start_ML(model,file_model,test_dataloader_rnn,criterion, False, True,False)


time_end = rnn.timeit.default_timer()
elapsed_time = time_end - time_start
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Execution time: {elapsed_time:.2f} seconds ({int(hours)} h {int(minutes)} min {seconds:.2f} sec)")

######################################### Part where we plot data from ML ###############################################

data_plot=pd.DataFrame()
data_plot['track_p']=test_data["track_p"].to_list()
data_plot['dedx']=predictions
data_plot['Ih']=Ih_values_test
data_plot['Ih']=data_plot['Ih']*1e-3
data_plot['track_eta']=test_data['track_eta']

ylim_plot=[2,9]
ML.plot_ML(data_plot,ylim_plot, True,True, True)
#ML.plot_ratio(data_plot,id.m_p)  
ML.density(data_plot,15,ylim_plot)
ML.std(data_plot,15,True)
ML.biais(data_plot,"track_eta",15)
ML.biais(data_plot,"track_p",15)

