import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import uproot
from Core import Identification as id
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import timeit
from Core import ML_plot as ML
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from Core import Creation_plus_filtred as cpf
import numpy as np
import matplotlib.pyplot as plt

class ParticleDataset(Dataset):
    def __init__(self, ndedx_cluster, dedx_values, target_values, p_values,eta_values,Ih_values):
        self.ndedx_cluster = ndedx_cluster # int
        self.dedx_values = dedx_values # dedx values is an array of a variable size
        self.target_values = target_values # int
        self.p_values = p_values # float
        self.eta_values = eta_values # float
        self.Ih_values = Ih_values # float 

    def __len__(self):
        return len(self.dedx_values)

    def __getitem__(self, idx):
        x = torch.tensor(self.ndedx_cluster[idx],dtype=torch.float32)
        y = torch.tensor(self.dedx_values[idx], dtype=torch.float32)
        z = torch.tensor(self.target_values[idx], dtype=torch.float32)
        t = torch.tensor(self.p_values[idx], dtype=torch.float32)
        u = torch.tensor(self.eta_values[idx], dtype=torch.float32)
        o = torch.tensor(self.Ih_values[idx], dtype=torch.float32)
        return x, y, z , t , u, o 

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, hidden_size_lstm):
        """
        Parameters:
          - input_size: size of the dedx array (e.g. 60)
          - hidden_size: hidden size for the classical branch
          - num_layers: number of layers in the LSTM
          - hidden_size_lstm: hidden size for the LSTM branch
        """
        super(LSTMModel, self).__init__()
        # Classical branch for dedx array (input_size should be 60)
        self.predictdedx = nn.Linear(input_size, hidden_size)
        self.predictdedx2 = nn.Linear(hidden_size, 1)
        self.dropout_predictdedx = nn.Dropout(0.4)
        
        # LSTM branch: input is dedx prediction (1 dim) concatenated with extras (4 dims) => 5 dims.
        self.lstm = nn.LSTM(5, hidden_size_lstm, num_layers, batch_first=True, dropout=0.1)
        # Map LSTM hidden state to a scalar adjustment.
        self.lstm_fc = nn.Linear(hidden_size_lstm, 1)
        
        self.relu = nn.ReLU()
        # Scaling factor so that the extra parameters have only a slight influence.
        self.adjustment_scale = 0.3
        
    def forward(self, dedx_array, extras):
        """
        Parameters:
          - dedx_array: Tensor of shape (batch, 60) 
          - extras: Tensor of shape (batch, 4)
        Returns:
          - output: refined scalar prediction (batch, 1)
        """
        # --- Classical Branch ---
        x = self.relu(self.predictdedx(dedx_array))        # (batch, hidden_size)
        dedx_pred = self.predictdedx2(self.dropout_predictdedx(x))  # (batch, 1)
        
        # --- LSTM Branch ---
        # Form a 5-dim vector by concatenating the classical prediction and extras.
        lstm_input = torch.cat([dedx_pred, extras], dim=1)   # (batch, 5)
        # We need to add a sequence dimension; here we use a sequence length of 1.
        lstm_input_seq = lstm_input.unsqueeze(1)             # (batch, 1, 5)
        
        # Run through LSTM. (Since the sequence length is 1, the LSTM’s output and hidden state
        # will correspond to that single “time” step.)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input_seq)
        # h_n has shape (num_layers, batch, hidden_size_lstm); use the last layer’s hidden state.
        lstm_hidden = h_n[-1]                                # (batch, hidden_size_lstm)
        # Map to a scalar adjustment.
        adjustment = self.lstm_fc(lstm_hidden)               # (batch, 1)
        
        # --- Final Output ---
        # Add a scaled adjustment to the classical dedx prediction.
        output = dedx_pred + self.adjustment_scale * adjustment  # (batch, 1)
        return output
     
def collate_fn(batch):
    """
    Expects each sample in batch to be a tuple of:
      (ndedx, dedx, target, p, eta, Ih)
    We only use dedx (the dedx array), target, and extras (ndedx, p, eta, Ih) here.
    """
    ndedx_list, dedx_list, target_list, p_list, eta_list, Ih_list = zip(*batch)
    
    max_length = 60  # Fixed length for dedx arrays.
    dedx_tensors = []
    for d in dedx_list:
        d = d.clone().detach().float()  # Ensure tensor is detached and float.
        if d.size(0) < max_length:
            padding = torch.zeros(max_length - d.size(0))
            d_padded = torch.cat([d, padding])
        else:
            d_padded = d[:max_length]
        dedx_tensors.append(d_padded)
    
    # Stack into a tensor of shape (batch, 60)
    sequences_padded = torch.stack(dedx_tensors)
    
    # Process extra features: we expect 4 features per sample.
    extras = torch.stack([
        torch.tensor([ndedx, p, eta, Ih], dtype=torch.float32)
        for ndedx, p, eta, Ih in zip(ndedx_list, p_list, eta_list, Ih_list)
    ])
    
    # Process targets into a tensor.
    targets = torch.tensor(target_list, dtype=torch.float32)
    
    return sequences_padded, targets, extras


def train_model(model, dataloader, criterion, optimizer, scheduler, epochs=20):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
        model.train()
        epoch_loss = 0 
        for batch, (dedx_array, targets, extras) in enumerate(dataloader):
            # Forward pass: note that lengths are no longer needed.
            outputs = model(dedx_array, extras)  # outputs shape: (batch, 1)
            outputs = outputs.squeeze()         # shape: (batch,)
            targets = targets.squeeze()         # shape: (batch,)
            loss = criterion(outputs, targets)
            
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            if batch % 100 == 0:
                current = batch * batch_size + dedx_array.size(0)
                percentage = (current / size) * 100
                print(f"loss: {loss.item():>7f} ({percentage:.2f}%)")
            
        scheduler.step(epoch_loss)
        print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")

def test_model(model, dataloader, criterion):
    predictions = []
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for dedx_array, targets, extras in dataloader:
            outputs = model(dedx_array, extras)
            outputs = outputs.squeeze()
            targets = targets.squeeze()
            loss = criterion(outputs, targets)
            test_loss += loss.item()       
            if outputs.dim() == 0:
                predictions.append(outputs.item())
            else:
                predictions.extend(outputs.tolist())
    print("Mean Test Loss: {:.4f}".format(test_loss/len(dataloader)))
    return predictions, targets, test_loss

if __name__ == "__main__":
    # --- Importation des données ( à remplacer par la fonction d'importation du X)---
    time_start = timeit.default
    file_name = "ML_training_LSTM.root"
    data = pd.DataFrame()
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(["ndedx_cluster","dedx_cluster","track_p","track_eta","Ih"], library="pd") # open data with array from numpy
        train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    # --- Préparer les données de l'entrainement ---
    ndedx_values_train = train_data["ndedx_cluster"].to_list()
    dedx_values = train_data["dedx_cluster"].to_list()
    data_th_values = id.bethe_bloch(938e-3, train_data["track_p"]).to_list()  # Targets (valeurs théoriques)
    p_values_train = train_data["track_p"].to_list()
    eta_values_train =  train_data["track_eta"].to_list()
    Ih_values_train = train_data["Ih"].to_list()
    dataset = ParticleDataset(ndedx_values_train, dedx_values, data_th_values,p_values_train,eta_values_train,Ih_values_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # --- Préparer les données de tests ---
    ndedx_values_test = test_data["ndedx_cluster"].to_list()
    dedx_values_test = test_data["dedx_cluster"].to_list()
    data_th_values_test = id.bethe_bloch(938e-3, test_data["track_p"]).to_list()
    p_values_test = test_data["track_p"].to_list()
    eta_values_test =  test_data["track_eta"].to_list()
    Ih_values_test = test_data["Ih"].to_list()
    test_dataset = ParticleDataset(ndedx_values_test,dedx_values_test, data_th_values_test,p_values_test,eta_values_test,Ih_values_test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # --- Initialisation du modèle, fonction de perte et optimiseur ---
    input_size = 60          # dedx max array length
    hidden_size = 128        # hidden size for the classical branch
    num_layers = 2           # number of LSTM layers
    hidden_size_lstm = 64    # LSTM hidden size
    
    model = LSTMModel(input_size, hidden_size, num_layers, hidden_size_lstm) 
    criterion = nn.MSELoss() # Si pas une grosse influence des outliers
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler: reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',  factor=0.1, patience = 3, verbose=True)
    

    # --- Entraînement du modèle ---
    train_model(model, dataloader, criterion, optimizer, scheduler, epochs=40)
    # torch.save(model.state_dict(), "model.pth")

    # --- Sauvegarde et Chargement du modèle ---
    # model=MLP(input_size=100)
    # state_dict = torch.load('model.pth',weights_only=True)  


    # --- Évaluation du modèle ---
    print("Evaluation du modèle...")
    predictions ,targets, test_loss = test_model(model, test_dataloader, criterion)


    # --- Création des histogrammes ---
    plt.figure(figsize=(12, 6))

    # Histogramme des prédictions
    plt.subplot(1, 2, 1)
    plt.hist(predictions, bins=50, alpha=0.7, label='Prédictions')
    plt.xlabel('Valeur')
    plt.ylabel('N')
    plt.title('Histogramme des Prédictions')
    plt.legend()

    # Histogramme des valeurs théoriques
    plt.subplot(1, 2, 2)
    plt.hist(data_th_values_test, bins=50, alpha=0.7, label='Valeurs Théoriques')
    plt.xlabel('Valeur')
    plt.ylabel('N')
    plt.title('Histogramme des Valeurs Théoriques')
    plt.legend()
    plt.tight_layout()

    np_th= np.array(targets)
    np_pr = np.array(predictions)

    # --- Comparaison des prédictions et des valeurs théoriques ---
    plt.figure(figsize=(8, 8))
    plt.hist2d(p_values_test, np_pr-np_th, bins=500, cmap='viridis', label='Data')
    plt.xlabel('Valeur')
    plt.ylabel('th-exp')
    plt.title('Ecart entre théorique et prédite')
    plt.legend()

    p_axis = np.logspace(np.log10(0.0001), np.log10(2), 500)
    plt.figure(figsize=(8, 8))
    plt.hist2d(p_values_test,np_pr,bins=500, cmap='viridis', label='Data')
    plt.plot(p_axis,id.bethe_bloch(938e-3,np.array(p_axis)),color='red')
    plt.xscale('log')
    plt.show()