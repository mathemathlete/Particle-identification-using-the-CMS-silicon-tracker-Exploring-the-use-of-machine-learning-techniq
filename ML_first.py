import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# --- Génération de données synthétiques ---
class SumDataset(Dataset):
    def __init__(self, num_samples=1000, min_len=3, max_len=20):
        self.num_samples = num_samples
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Génération d'un vecteur de taille aléatoire
        vec_len = np.random.randint(self.min_len, self.max_len + 1)
        vec = np.random.randn(vec_len).astype(np.float32)
        
        # Valeur cible : somme des valeurs du vecteur
        target = np.sum(vec)
        
        # On retourne le vecteur et sa cible
        return torch.tensor(vec), torch.tensor(target)

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

# --- Préparer les données ---
dataset = SumDataset(num_samples=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- Initialisation du modèle, fonction de perte et optimiseur ---
# Le plus grand nombre d'éléments dans un vecteur est 20, donc on fixe la taille d'entrée
model = MLP(input_size=20)  # Taille fixe (max_len)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Fonction d'entraînement ---
def train_model(model, dataloader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            # Remplir (padding) les entrées pour qu'elles aient la même taille (20)
            inputs_padded = torch.zeros((inputs.size(0), 20))  # Padded to max_len=20
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
train_model(model, dataloader, criterion, optimizer, epochs=5)

# --- Tester le modèle avec une entrée aléatoire ---
sample_input = torch.randn(5)  # Un vecteur de taille 5
sample_input_padded = torch.zeros(1, 20)
sample_input_padded[0, :sample_input.size(0)] = sample_input

model.eval()
with torch.no_grad():
    predicted = model(sample_input_padded)
    print(f"Entrée : {sample_input}")
    print(f"Prédiction : {predicted.item()}")