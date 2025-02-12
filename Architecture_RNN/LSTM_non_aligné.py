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
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.extra_fc = nn.Linear(4, hidden_size)
        self.fusion_fc = nn.Linear(hidden_size * 2, 64)
        self.out_fc = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, lengths, extras):
        """
        x: Tensor of shape (batch_size, seq_length, 1) - dedx values sequence
        lengths: Tensor of shape (batch_size,) - actual sequence lengths
        extras: Tensor of shape (batch_size, 4) - extra parameters (ndedx, p, eta, Ih)
        """
        # Sort sequences by length for packing
        lengths, perm_idx = lengths.sort(descending=True)
        x = x[perm_idx]
        extras = extras[perm_idx]

        # Pack the padded sequence and pass it through the LSTM
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, (h_n, c_n) = self.lstm(packed_x)
        # Unpack the output
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        # Extract the last valid hidden state for each sample
        last_outputs = out[torch.arange(out.size(0)), lengths - 1]

        # Process extra features
        extras_processed = self.relu(self.extra_fc(extras))
        # Concatenate LSTM output with processed extras
        fusion = torch.cat([last_outputs, extras_processed], dim=1)
        # Pass through the fusion layer
        fusion = self.relu(self.fusion_fc(fusion))
        fusion = self.dropout(fusion)
        # Final output layer to predict the target
        output = self.out_fc(fusion)
        # Restore original order
        _, inv_perm_idx = perm_idx.sort()
        output = output[inv_perm_idx]
        return output.squeeze(-1) 

    
def collate_fn(batch):
    # Unpack all items from each sample
    ndedx_list, dedx_list, target_list, p_list, eta_list, Ih_list = zip(*batch)

    # Convert dedx_values sequences to tensors (variable lengths)
    dedx_tensors = [d.clone().detach().unsqueeze(-1) for d in dedx_list]

    # Compute sequence lengths
    lengths = torch.tensor([seq.size(0) for seq in dedx_tensors], dtype=torch.int64)

    # Pad sequences to max length in batch
    sequences_padded = pad_sequence(dedx_tensors, batch_first=True, padding_value=0)

    # Convert extra features to a tensor (batch_size, 4)
    extras = torch.stack([torch.tensor([ndedx, p, eta, Ih], dtype=torch.float32) 
                          for ndedx, p, eta, Ih in zip(ndedx_list, p_list, eta_list, Ih_list)])

    # Convert targets to tensor
    targets = torch.tensor(target_list, dtype=torch.float32)

    return sequences_padded, lengths, targets, extras

def train_model(model, dataloader, criterion, optimizer,scheduler, epochs=20):
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

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(inputs)
                percentage = (current / size) * 100
                print(f"loss: {loss:>7f} ({percentage:.2f}%)")

        scheduler.step(epoch_loss)


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
    input_size = 1
    hidden_size = 64
    num_layers = 2
    model = LSTMModel(input_size, hidden_size, num_layers) 
    criterion = nn.MSELoss() # Si pas une grosse influence des outliers
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Learning rate scheduler: reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',  factor=0.5, verbose=True)
    

    # --- Entraînement du modèle ---
    train_model(model, dataloader, criterion, optimizer, scheduler, epochs=20)
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