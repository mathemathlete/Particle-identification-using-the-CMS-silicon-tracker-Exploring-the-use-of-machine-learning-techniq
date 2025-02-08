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
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class ParticleDataset(Dataset):
    def __init__(self, dedx_values, target_values, p_values,eta_values,I_h_values,):
        self.dedx_values = dedx_values
        self.target_values = target_values
        # self.p_values = p_values
        # self.eta_values = eta_values
        # self.I_h_values = I_h_values

    def __len__(self):
        return len(self.dedx_values)

    def __getitem__(self, idx):
        x = torch.tensor(self.dedx_values[idx], dtype=torch.float32)
        y = torch.tensor(self.target_values[idx], dtype=torch.float32)
        # z = torch.tensor(self.p_values[idx], dtype=torch.float32)
        # t = torch.tensor(self.eta_values[idx], dtype=torch.float32)
        # o = torch.tensor(self.I_h_values[idx], dtype=torch.float32)
        # return x, y, z , t , o 
        return x, y
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        # Pack the padded sequences (ignores padding in computations)
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # LSTM processing
        packed_out, (h_n, c_n) = self.lstm(packed_x)

        # Unpack the sequence
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Take the last valid output (using lengths)
        last_outputs = out[torch.arange(out.size(0)), lengths - 1]

        # Pass through the fully connected layer
        output = self.fc(self.relu(last_outputs))
        return output

def collate_fn(batch):
    inputs, targets = zip(*batch)

    # Convert input lists to tensors
    inputs = [torch.tensor(x, dtype=torch.float32) for x in inputs]
    
    # Compute sequence lengths before padding
    lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.int64)

    # Pad sequences to the same length
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)

    # Convert targets to tensor
    targets = torch.tensor(targets, dtype=torch.float32)

    return inputs_padded, lengths, targets

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

def test_model(model, dataloader, criterion,max_len):
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
    file_name = "Root_files/ML_training.root"
    max_len=100
    data = pd.DataFrame()
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(["dedx_cluster","track_p"], library="pd") # open data with array from numpy
        train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    # --- Préparer les données de l'entrainement ---
    dedx_values = train_data["dedx_cluster"].to_list()
    data_th_values = id.bethe_bloch(938e-3, train_data["track_p"]).to_list()  # Targets (valeurs théoriques)
    p_values = test_data["track_p"].to_list()
    dataset = ParticleDataset(dedx_values, data_th_values,None,None,None)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    # --- Préparer les données de tests ---
    dedx_values_test = test_data["dedx_cluster"].to_list()
    data_th_values_test = id.bethe_bloch(938e-3, test_data["track_p"]).to_list()
    test_dataset = ParticleDataset(dedx_values_test, data_th_values_test,None,None,None)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # --- Initialisation du modèle, fonction de perte et optimiseur ---
    input_size = 50
    hidden_size = 128
    num_layers = 3
    model = LSTMModel(input_size, hidden_size, num_layers,) 
    criterion = nn.MSELoss()
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
    predictions ,targets, test_loss = test_model(model, test_dataloader, criterion,max_len)


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
    plt.hist2d(p_values, np_pr-np_th, bins=500, cmap='viridis', label='Data')
    plt.xlabel('Valeur')
    plt.ylabel('th-exp')
    plt.title('Ecart entre théorique et prédite')
    plt.legend()

    p_axis = np.logspace(np.log10(0.0001), np.log10(2), 500)
    plt.figure(figsize=(8, 8))
    plt.hist2d(p_values,np_pr,bins=500, cmap='viridis', label='Data')
    plt.plot(p_axis,id.bethe_bloch(938e-3,np.array(p_axis)),color='red')
    plt.xscale('log')
    plt.show()