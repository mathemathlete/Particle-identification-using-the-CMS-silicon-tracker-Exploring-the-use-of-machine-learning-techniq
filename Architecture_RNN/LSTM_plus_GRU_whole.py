import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import uproot
import Identification as id
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import timeit
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

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
    def __init__(self, dedx_hidden_size, dedx_num_layers, lstm_hidden_size, lstm_num_layers):
        super(LSTMModel, self).__init__()
        # GRU for the dedx branch.
        self.dedx_rnn = nn.GRU(
            input_size=40,             # Max Size of the trace deposit 
            hidden_size=dedx_hidden_size,
            num_layers=dedx_num_layers,
            batch_first=True,
            dropout=0.1  # Dropout applies only if num_layers > 1
        )
        self.dedx_fc = nn.Linear(dedx_hidden_size, 1)
        self.dropout_dedx = nn.Dropout(0.4)
        
        # LSTM for the adjustment branch.
        # It will process a vector of dimension 5: dedx prediction (1) concatenated with extras (4).
        self.adjust_lstm = nn.LSTM(
            input_size=5,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.adjust_fc = nn.Linear(lstm_hidden_size, 1)
        
        self.relu = nn.ReLU()
        # Scaling factor to give a slight influence to the adjustment.
        self.adjustment_scale = 0.1

    def forward(self, dedx_seq, lengths, extras):
        """
        Parameters:
          - dedx_seq: Tensor of shape (batch, 1, 40)
          - lengths: Tensor of sequence lengths
          - extras: Tensor of extra features of shape (batch, 4)
        Returns:
          - output: Scalar prediction for each sample, shape (batch, 1)
        """
        # --- dedx Branch ---
        # GRU expects (batch, seq_len, input_size), but our input is already one full vector per sample.
        # So we pass it directly without packing.
        packed_out, hidden = self.dedx_rnn(dedx_seq)  # No need for pack_padded_sequence

        # hidden is of shape (num_layers, batch, dedx_hidden_size); we use the last layer.
        hidden_last = hidden[-1]  # (batch, dedx_hidden_size)
        dedx_pred = self.dedx_fc(self.dropout_dedx(hidden_last))  # (batch, 1)
        
        # --- Adjustment Branch ---
        # Concatenate dedx prediction with extras: shape becomes (batch, 5)
        lstm_input = torch.cat([dedx_pred, extras], dim=1)
        # Unsqueeze to add a time dimension: (batch, 1, 5)
        lstm_input_seq = lstm_input.unsqueeze(1)
        lstm_out, (h_n, c_n) = self.adjust_lstm(lstm_input_seq)
        # Use the last layer's hidden state: shape (batch, lstm_hidden_size)
        lstm_hidden = h_n[-1]
        adjustment = self.adjust_fc(lstm_hidden)  # (batch, 1)
        
        # Final output: dedx prediction plus a scaled adjustment.
        output = dedx_pred + self.adjustment_scale * adjustment  # (batch, 1)
        return output


def collate_fn(batch):
    """
    Processes dedx_values as a full vector, rather than a time sequence.
    """
    ndedx_list, dedx_list, target_list, p_list, eta_list, Ih_list = zip(*batch)
    
    max_length = 40  # Fixed dedx array length.
    processed = []
    for d in dedx_list:
        d = d.clone().detach().float()
        if d.size(0) < max_length:
            # Pad with zeros.
            padding = torch.zeros(max_length - d.size(0))
            d_padded = torch.cat([d, padding])
        else:
            # Truncate if necessary.
            d_padded = d[:max_length]
        processed.append(d_padded)
    
    # Stack and then unsqueeze to get shape (batch, 1, 40)
    sequences = torch.stack(processed)         # (batch, 40)
    sequences = sequences.unsqueeze(1)           # (batch, 1, 40)
    # Since each sample is now one full vector, lengths are no longer needed.
    
    # Process extra features: we combine ndedx, p, eta, Ih to form a vector of 4 features per sample.
    extras = torch.stack([
        torch.tensor([ndedx, p, eta, Ih], dtype=torch.float32)
        for ndedx, p, eta, Ih in zip(ndedx_list, p_list, eta_list, Ih_list)
    ])
    
    # Targets as a tensor.
    targets = torch.tensor(target_list, dtype=torch.float32)
    
    return sequences, None, targets, extras  # No need for lengths anymore.


def train_model(model, dataloader, criterion, optimizer, scheduler, epochs=20):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    for epoch in range(epochs):
        epoch_loss = 0 
        print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
        model.train()
        for batch, (inputs, lengths, targets, extras) in enumerate(dataloader):  # Expect 3 values
            outputs = model(inputs, lengths, extras)  # Pass both inputs and lengths to the model
            outputs = outputs.squeeze()
            targets = targets.squeeze()
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(inputs)
                percentage = (current / size) * 100
                print(f"loss: {loss:>7f} ({percentage:.2f}%)")
        scheduler.step(epoch_loss)
        print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")
        
def test_model(model, dataloader, criterion):
    predictions = []
    model.eval()  # Mettre le modèle en mode évaluation
    test_loss = 0.0
    with torch.no_grad():  # Désactiver la grad pour l'évaluation
        for inputs, lengths, targets, extras in dataloader:  # Expecting 3 values from the dataloader
            outputs = model(inputs, lengths, extras)  # Pass both inputs and lengths to the model
            outputs = outputs.squeeze()  # Ensure outputs are 1-dimensional
            targets = targets.squeeze()  # Ensure targets are 1-dimensional
            loss = criterion(outputs, targets)
            test_loss += loss.item()       
            if outputs.dim() == 0:
                predictions.append(outputs.item())
            else:
                predictions.extend(outputs.tolist())
            # Affichage des prédictions
    print("Prédictions sur le jeu de données de test :")
    print(f"Test Loss: {test_loss/len(dataloader):.4f}")
    return predictions, targets, test_loss

if __name__ == "__main__":
    # --- Importation des données ( à remplacer par la fonction d'importation du X)---
    time_start = timeit.default_timer()
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
    dedx_hidden_size = 128
    dedx_num_layers = 2   # With one layer, GRU dropout is not applied.
    lstm_hidden_size = 64
    lstm_num_layers = 2

    model = LSTMModel(dedx_hidden_size, dedx_num_layers, lstm_hidden_size, lstm_num_layers)
    criterion = nn.HuberLoss() # Si pas une grosse influence des outliers
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler: reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3,  factor=0.1)
    

    # --- Entraînement du modèle ---
    train_model(model, dataloader, criterion, optimizer, scheduler, epochs=40)
    torch.save(model.state_dict(), "model_LSTM_plus_GRU_whole.pth")

    # --- Sauvegarde et Chargement du modèle ---
    
    # state_dict = torch.load('model.pth',weights_only=True)


    # --- Évaluation du modèle ---
    print("Evaluation du modèle...")
    predictions ,targets, test_loss = test_model(model, test_dataloader, criterion)

    time_end = timeit.default_timer()
    print(f"Temps d'execution : {time_end - time_start}")

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