import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import uproot
import Identification as id
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- Importation des données ---
file_name = "ML_training.root"
data = pd.DataFrame()
with uproot.open(file_name) as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    data = tree.arrays(["dedx_cluster","track_p"], library="pd") # open data with array from numpy

class ParticleDataset(Dataset):
    def __init__(self, dedx_values, target_values, max_len=30):
        self.dedx_values = dedx_values
        self.target_values = target_values
        self.max_len = max_len

    def __len__(self):
        return len(self.dedx_values)

    def __getitem__(self, idx):
        x = torch.tensor(self.dedx_values[idx], dtype=torch.float32)
        y = torch.tensor(self.target_values[idx], dtype=torch.float32)

        return x, y

# --- Définir le modèle de réseau de neurones ---
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Première couche cachée
        self.fc2 = nn.Linear(hidden_size, 1)  # Cible scalaire (somme)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def collate_fn(batch):
    inputs, targets = zip(*batch)  # Sépare les entrées et les cibles

    # Convertir en tensors
    inputs = [x.clone().detach().float() for x in inputs]
    targets = torch.tensor(targets, dtype=torch.float32)

    # Padding des entrées pour qu'elles aient toutes une taille de `max_len`
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)

    return inputs_padded, targets

# --- Préparer les données ---

dedx_values = data["dedx_cluster"].to_list()
data_th_values = id.bethe_bloch(938e-3, data["track_p"]).to_list()  # Targets (valeurs théoriques)
dataset = ParticleDataset(dedx_values, data_th_values)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)


# --- Initialisation du modèle, fonction de perte et optimiseur ---
model = MLP(input_size=100)  # Taille fixe (max_len)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Fonction d'entraînement ---
def train_model(model, dataloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # Remplir (padding) les entrées pour qu'elles aient la même taille (20)
            inputs_padded = torch.zeros((inputs.size(0), 100))  # Padded to max_len=20
            for i in range(inputs.size(0)):
                inputs_padded[i, :inputs[i].size(0)] = inputs[i]  # Remplir avec les données réelles
            
            # Calculer les prédictions
            outputs = model(inputs_padded)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backpropagation et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

# --- Entraînement du modèle ---
#train_model(model, dataloader, criterion, optimizer, epochs=10)

# --- Sauvegarde du modèle ---
torch.save(model.state_dict(), "model.pth")