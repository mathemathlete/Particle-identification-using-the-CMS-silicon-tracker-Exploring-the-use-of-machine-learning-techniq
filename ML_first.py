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


# --- Importation des données ---
file_name = "ML_training.root"
data = pd.DataFrame()
with uproot.open(file_name) as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    data = tree.arrays(["dedx_cluster","track_p"], library="pd") # open data with array from numpy
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
    # print(len(data))
    # print(len(train_data))
    # print(len(test_data))
    # print(len(train_data) + len(test_data))

class ParticleDataset(Dataset):
    def __init__(self, dedx_values, target_values, max_len=100):
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

dedx_values = train_data["dedx_cluster"].to_list()
data_th_values = id.bethe_bloch(938e-3, train_data["track_p"]).to_list()  # Targets (valeurs théoriques)
p_values = test_data["track_p"].to_list()
dataset = ParticleDataset(dedx_values, data_th_values)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)


# --- Initialisation du modèle, fonction de perte et optimiseur ---
model = MLP(input_size=100)  # Taille fixe (max_len)
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Fonction d'entraînement ---
def train_model(model, dataloader, criterion, optimizer, epochs):
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


def train_model2(model, dataloader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):
        print(f"\n Epoch {epoch+1}\n-------------------------------")
        size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        for batch, (input, targets) in enumerate(dataloader):
            inputs_padded = torch.zeros((input.size(0), 100))  # Padded to max_len=100
            for i in range(input.size(0)):
                inputs_padded[i, :input[i].size(0)] = input[i]
            # Compute prediction and loss
            outputs = model(inputs_padded)
            loss = criterion(outputs.squeeze(), targets)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(inputs_padded)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# --- Entraînement du modèle ---
train_model2(model, dataloader, criterion, optimizer, epochs=10)

# --- Sauvegarde du modèle ---
torch.save(model.state_dict(), "model.pth")


# model=MLP(input_size=100)
# state_dict = torch.load('model.pth',weights_only=True)  

# --- Évaluation du modèle ---
dedx_values_test = test_data["dedx_cluster"].to_list()
data_th_values_test = id.bethe_bloch(938e-3, test_data["track_p"]).to_list()
test_dataset = ParticleDataset(dedx_values_test, data_th_values_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

print("Evaluation du modèle...")

predictions = []
model.eval()  # Mettre le modèle en mode évaluation
test_loss = 0.0
with torch.no_grad():  # Désactiver la grad pour l'évaluation
    for inputs, targets in test_dataloader:
        inputs_padded = torch.zeros((inputs.size(0), 100))  # Padded to max_len=100
        for i in range(inputs.size(0)):
            inputs_padded[i, :inputs[i].size(0)] = inputs[i]  # Remplir avec les données réelles
        
        outputs = model(inputs_padded)
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
print(f"Test Loss: {test_loss/len(test_dataloader):.4f}")

# print(predictions)
# print(len(predictions))
# print(len(data_th_values_test))

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
