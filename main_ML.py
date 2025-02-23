from Core import ML_plot as ML
import subprocess 
from Core import Creation_plus_filtred as cpf
from Core import Identification as id
from Core import file_ML as fml
from Architecture_RNN import GRU_plus_LSTM_V1 as rnn
from Architecture_RNN import GRU_plus_LSTM_GPU_V2a as rnn2
from Architecture_RNN import GRU_plus_LSTM_GPU_V2b as rnn3
from Architecture_RNN import GRU_plus_LSTM_GPU_V3 as rnn4
from Architecture_RNN import GRU_plus_MLP_GPU_V1 as rnn5
from Architecture_RNN import GRU_plus_MLP_GPU_V2a as rnn6
from Architecture_RNN import GRU_plus_MLP_GPU_V2b as rnn7
from Architecture_RNN import GRU_plus_MLP_GPU_V3 as rnn8
import torch.nn as nn
import torch.optim as optim
import pandas as pd

#########################################  training/testing ML  ###############################################
ML_train=False 
ML_test=True

######################################### Choose the ML model  ###############################################
# for use the model you want you need to modify the file model and the import name line 6
time_start = rnn.timeit.default_timer()

file_name = "Root_Files/data_GRU_V3.root" # choose your data file
branch_of_interest = ["ndedx_cluster","dedx_cluster","track_p","track_eta","Ih","dedx_pathlength","dedx_modulegeom"] # choose the branches you want to use
file_model_save = "Models/bidon.pth"

data=cpf.import_data(file_name,branch_of_interest)
train_data, test_data = rnn.train_test_split(data, test_size=0.25, random_state=42)

# --- prepare the training data  ---
ndedx_values_train = train_data["ndedx_cluster"].to_list()
dedx_value_train = train_data["dedx_cluster"].to_list()
dx_values_train = train_data["dedx_pathlength"].to_list()
modulegeom_values_train = train_data["dedx_modulegeom"].to_list()
data_th_values = id.bethe_bloch(938e-3, train_data["track_p"]).to_list()  # Targets (valeurs théoriques)
p_values_train = train_data["track_p"].to_list()
eta_values_train =  train_data["track_eta"].to_list()
Ih_values_train = train_data["Ih"].to_list()

# --- prepare all the possible Particle Dataset ---
dataset_rnn = rnn.ParticleDataset(ndedx_values_train, dedx_value_train, data_th_values,p_values_train,eta_values_train,Ih_values_train)
dataset_rnn2 = rnn2.ParticleDataset_V2a(ndedx_values_train, dedx_value_train, data_th_values,eta_values_train,Ih_values_train)
dataset_rnn3 = rnn3.ParticleDataset_V2b(ndedx_values_train, dedx_value_train, data_th_values,eta_values_train)
dataset_rnn4 = rnn4.ParticleDataset_V3(ndedx_values_train, dedx_value_train, dx_values_train,modulegeom_values_train, data_th_values, eta_values_train, Ih_values_train)
dataset_rnn5 = rnn5.ParticleDataset_V1(ndedx_values_train, dedx_value_train, data_th_values,p_values_train,eta_values_train,Ih_values_train)
dataset_rnn6 = rnn6.ParticleDataset_V2a(ndedx_values_train, dedx_value_train, data_th_values,eta_values_train,Ih_values_train)
dataset_rnn7 = rnn7.ParticleDataset_V2b(ndedx_values_train, dedx_value_train, data_th_values,eta_values_train)
dataset_rnn8 = rnn8.ParticleDataset_V3(ndedx_values_train, dedx_value_train, dx_values_train,modulegeom_values_train, data_th_values, eta_values_train, Ih_values_train)

# --- prepare the dataloader ---
dataloader_rnn = rnn.DataLoader(dataset_rnn, batch_size=32, collate_fn=rnn.collate_fn)
dataloader_rnn2 = rnn2.DataLoader(dataset_rnn2, batch_size=32, collate_fn=rnn2.collate_fn)
dataloader_rnn3 = rnn3.DataLoader(dataset_rnn3, batch_size=32, collate_fn=rnn3.collate_fn)
dataloader_rnn4 = rnn4.DataLoader(dataset_rnn4, batch_size=32, collate_fn=rnn4.collate_fn)
dataloader_rnn5 = rnn5.DataLoader(dataset_rnn5, batch_size=32, collate_fn=rnn5.collate_fn)
dataloader_rnn6 = rnn6.DataLoader(dataset_rnn6, batch_size=32, collate_fn=rnn6.collate_fn)
dataloader_rnn7 = rnn7.DataLoader(dataset_rnn7, batch_size=32, collate_fn=rnn7.collate_fn)
dataloader_rnn8 = rnn8.DataLoader(dataset_rnn8, batch_size=32, collate_fn=rnn8.collate_fn)

# --- prepare the test data ---
ndedx_values_test = test_data["ndedx_cluster"].to_list()
dedx_values_test = test_data["dedx_cluster"].to_list()
data_th_values_test = id.bethe_bloch(938e-3, test_data["track_p"]).to_list()
p_values_test = test_data["track_p"].to_list()
eta_values_test =  test_data["track_eta"].to_list()
Ih_values_test = test_data["Ih"].to_list()

# --- prepare the test Particle Dataset ---
test_dataset_rnn = rnn.ParticleDataset(ndedx_values_test, dedx_values_test, data_th_values_test,p_values_test,eta_values_test,Ih_values_test)
test_dataset_rnn2 = rnn2.ParticleDataset_V2a(ndedx_values_test, dedx_values_test, data_th_values_test,eta_values_test,Ih_values_test)
test_dataset_rnn3 = rnn3.ParticleDataset_V2b(ndedx_values_test, dedx_values_test, data_th_values_test,eta_values_test)
test_dataset_rnn4 = rnn4.ParticleDataset_V3(ndedx_values_test, dedx_values_test, dx_values_train,modulegeom_values_train, data_th_values_test, eta_values_test, Ih_values_test)
test_dataset_rnn5 = rnn5.ParticleDataset_V1(ndedx_values_test, dedx_values_test, data_th_values_test,p_values_test,eta_values_test,Ih_values_test)
test_dataset_rnn6 = rnn6.ParticleDataset_V2a(ndedx_values_test, dedx_values_test, data_th_values_test,eta_values_test,Ih_values_test)
test_dataset_rnn7 = rnn7.ParticleDataset_V2b(ndedx_values_test, dedx_values_test, data_th_values_test,eta_values_test)
test_dataset_rnn8 = rnn8.ParticleDataset_V3(ndedx_values_test, dedx_values_test, dx_values_train,modulegeom_values_train, data_th_values_test, eta_values_test, Ih_values_test)

test_dataloader_rnn = rnn.DataLoader(test_dataset_rnn, batch_size=32, collate_fn=rnn.collate_fn)
test_dataloader_rnn2 = rnn2.DataLoader(test_dataset_rnn2, batch_size=32, collate_fn=rnn2.collate_fn)
test_dataloader_rnn3 = rnn3.DataLoader(test_dataset_rnn3, batch_size=32, collate_fn=rnn3.collate_fn)
test_dataloader_rnn4 = rnn4.DataLoader(test_dataset_rnn4, batch_size=32, collate_fn=rnn4.collate_fn)
test_dataloader_rnn5 = rnn5.DataLoader(test_dataset_rnn5, batch_size=32, collate_fn=rnn5.collate_fn)
test_dataloader_rnn6 = rnn6.DataLoader(test_dataset_rnn6, batch_size=32, collate_fn=rnn6.collate_fn)
test_dataloader_rnn7 = rnn7.DataLoader(test_dataset_rnn7, batch_size=32, collate_fn=rnn7.collate_fn)
test_dataloader_rnn8 = rnn8.DataLoader(test_dataset_rnn8, batch_size=32, collate_fn=rnn8.collate_fn)





# --- LSTM Initialisation, 
dedx_hidden_size = 256
dedx_num_layers = 2   # With one layer, GRU dropout is not applied.
lstm_hidden_size = 64
lstm_num_layers = 2
adjustement_scale = 0.64
dropout_GRU = 0.18
dropout_dedx = 0.1
dropout_LSTM = 0.29

# --- MLP Initialisation, 
dedx_hidden_size = 256
dedx_num_layers = 2   # With one layer, GRU dropout is not applied.
mlp_hidden_size1 = 500
mlp_hidden_size2 = 200
mlp_hidden_size3 = 100  
adjustement_scale = 0.5
dropout_GRU = 0.1
dropout_MLP = 0.1
dropout_dedx = 0.1

# Number of epochs
epoch = 1

model = rnn.LSTMModel(dedx_hidden_size, dedx_num_layers, lstm_hidden_size, lstm_num_layers, dropout_GRU, dropout_dedx, dropout_LSTM, adjustement_scale)
model2 = rnn2.LSTM_V2a(dedx_hidden_size, dedx_num_layers, lstm_hidden_size, lstm_num_layers, dropout_GRU, dropout_dedx, dropout_LSTM, adjustement_scale)
model3 = rnn3.LSTM_V2b(dedx_hidden_size, dedx_num_layers, lstm_hidden_size, lstm_num_layers, dropout_GRU, dropout_dedx, dropout_LSTM, adjustement_scale)
model4 = rnn4.LSTM_V3(dedx_hidden_size, dedx_num_layers, lstm_hidden_size, lstm_num_layers, dropout_GRU, dropout_dedx, dropout_LSTM, adjustement_scale)
model5 = rnn5.MLP_V1(dedx_hidden_size, dedx_num_layers, mlp_hidden_size1, mlp_hidden_size2, mlp_hidden_size3, dropout_GRU, dropout_MLP, dropout_dedx, adjustement_scale)
model6 = rnn6.MLP_V2a(dedx_hidden_size, dedx_num_layers, mlp_hidden_size1, mlp_hidden_size2, mlp_hidden_size3, dropout_GRU, dropout_MLP, dropout_dedx, adjustement_scale)
model7 = rnn7.MLP_V2b(dedx_hidden_size, dedx_num_layers, mlp_hidden_size1, mlp_hidden_size2, mlp_hidden_size3, dropout_GRU, dropout_MLP, dropout_dedx, adjustement_scale)
model8 = rnn8.MLP_V3(dedx_hidden_size, dedx_num_layers, mlp_hidden_size1, mlp_hidden_size2, mlp_hidden_size3, dropout_GRU, dropout_MLP, dropout_dedx, adjustement_scale)

# Loss function and optimizer
criterion = nn.MSELoss()
# Other optimizers to consider with momentum possibly
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1.97e-6)

# Learning rate scheduler: reduce LR on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

######################################### Part where we test the ML model ###############################################
# --- Testing model ---
if ML_train==True:
    print("Training model...")
    ML.loss_epoch(rnn.start_ML(model,file_model_save,dataloader_rnn,criterion,epoch, True, False))

if ML_test==True:
    print("Testing model...")
    predictions, test_loss = rnn.start_ML(model,file_model_save,test_dataloader_rnn,criterion,epoch, False, True)

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


##################################### Part where can eventually run the tuning ############################################
