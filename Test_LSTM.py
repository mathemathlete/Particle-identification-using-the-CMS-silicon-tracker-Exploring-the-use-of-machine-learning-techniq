import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import uproot
import Identification as id
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class ParticleDataset(Dataset):
    def __init__(self, ndedx_cluster, dedx_values, target_values, p_values,eta_values,I_h_values):
        self.ndedx_cluster = ndedx_cluster # int
        self.dedx_values = dedx_values # dedx values is an array of a variable size
        self.target_values = target_values # int
        self.p_values = p_values # float
        self.eta_values = eta_values # float
        self.I_h_values = I_h_values # float 

    def __len__(self):
        return len(self.dedx_values)

    def __getitem__(self, idx):
        x = torch.tensor(self.ndedx_cluster[idx],dtype=torch.float32)
        y = torch.tensor(self.dedx_values[idx], dtype=torch.float32)
        z = torch.tensor(self.target_values[idx], dtype=torch.float32)
        t = torch.tensor(self.p_values[idx], dtype=torch.float32)
        u = torch.tensor(self.eta_values[idx], dtype=torch.float32)
        o = torch.tensor(self.I_h_values[idx], dtype=torch.float32)
        return x, y, z , t , u, o 
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        # Trier les séquences en fonction de la longueur (obligatoire pour pack_padded_sequence)
        lengths, perm_idx = lengths.sort(descending=True)
        x = x[perm_idx]

        # Pack des séquences avant passage dans le LSTM
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # Passage dans le LSTM
        packed_out, (h_n, c_n) = self.lstm(packed_x)

        # Dépacker la séquence
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Restaurer l'ordre initial
        _, inv_perm_idx = perm_idx.sort()
        out = out[inv_perm_idx]

        # Récupération des derniers états utiles (avec les longueurs)
        last_outputs = out[torch.arange(out.size(0)), lengths - 1]

        # Passage dans la couche fully connected
        output = self.fc(self.relu(last_outputs))
        return output
    
def collate_fn(batch):
    # Unpack all items from each sample
    ndedx_list, dedx_list, target_list, p_list, eta_list, I_h_list = zip(*batch)
    
    # For each sample, create the augmented sequence
    augmented_sequences = []
    for ndedx, dedx_seq, p, eta, I_h in zip(ndedx_list, dedx_list, p_list, eta_list, I_h_list):
        # Ensure dedx_seq is a tensor of shape (L, 1)
        seq_tensor = torch.tensor(dedx_seq, dtype=torch.float32).unsqueeze(-1)
        # Create extra features tensor of shape (L, 4)
        extras_tensor = torch.tensor([ndedx, p, eta, I_h], dtype=torch.float32).unsqueeze(0).repeat(seq_tensor.size(0), 1)
        # Concatenate along feature dimension -> shape: (L, 1+4)
        augmented_seq = torch.cat([seq_tensor, extras_tensor], dim=1)
        augmented_sequences.append(augmented_seq)
    
    lengths = torch.tensor([seq.size(0) for seq in augmented_sequences], dtype=torch.int64)
    # Pad the augmented sequences (each of shape (L, 4)) to shape (batch, max_seq_length, 5)
    sequences_padded = pad_sequence(augmented_sequences, batch_first=True, padding_value=0)
    
    targets = torch.stack([torch.tensor(t, dtype=torch.float32) for t in target_list])
    return sequences_padded, lengths, targets

def train_model(model, dataloader, criterion, optimizer, epochs=20):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}\n-------------------------------")
        model.train()
        for batch, (inputs, lengths, targets) in enumerate(dataloader):  # Expect 3 values
            outputs = model(inputs, lengths)  # Pass both inputs and lengths to the model
            loss = criterion(outputs.squeeze(), targets)
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(input)
                percentage = (current / size) * 100
                print(f"loss: {loss:>7f} ({percentage:.2f}%)")

def test_model(model, dataloader, criterion):
    predictions = []
    model.eval()  # Mettre le modèle en mode évaluation
    test_loss = 0.0
    with torch.no_grad():  # Désactiver la grad pour l'évaluation
        for inputs, lengths, targets in dataloader:  # Expecting 3 values from the dataloader
            outputs = model(inputs, lengths)  # Pass both inputs and lengths to the model
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
    file_name = "Root_files/ML_training_1.2.root"
    data = pd.DataFrame()
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(["ndedx_cluster","dedx_cluster","track_p","track_eta","I_h"], library="pd") # open data with array from numpy
        train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    # --- Préparer les données de l'entrainement ---
    ndedx_values_train = train_data["ndedx_cluster"].to_list()
    dedx_values = train_data["dedx_cluster"].to_list()
    data_th_values = id.bethe_bloch(938e-3, train_data["track_p"]).to_list()  # Targets (valeurs théoriques)
    p_values_train = train_data["track_p"].to_list()
    eta_values_train =  train_data["track_eta"].to_list()
    I_h_values_train = train_data["I_h"].to_list()
    dataset = ParticleDataset(ndedx_values_train, dedx_values, data_th_values,p_values_train,eta_values_train,I_h_values_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # --- Préparer les données de tests ---
    ndedx_values_test = test_data["ndedx_cluster"].to_list()
    dedx_values_test = test_data["dedx_cluster"].to_list()
    data_th_values_test = id.bethe_bloch(938e-3, test_data["track_p"]).to_list()
    p_values_test = test_data["track_p"].to_list()
    eta_values_test =  test_data["track_eta"].to_list()
    I_h_values_test = test_data["I_h"].to_list()
    test_dataset = ParticleDataset(ndedx_values_test,dedx_values_test, data_th_values_test,p_values_test,eta_values_test,I_h_values_test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # --- Initialisation du modèle, fonction de perte et optimiseur ---
    input_size = 5
    hidden_size = 128
    num_layers = 3
    model = LSTMModel(input_size, hidden_size, num_layers,) 
    criterion = nn.MSELoss() # Grosse influence des outliers
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Entraînement du modèle ---
    train_model(model, dataloader, criterion, optimizer, epochs=5)
    # torch.save(model.state_dict(), "model.pth")

    # --- Sauvegarde du modèle ---
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