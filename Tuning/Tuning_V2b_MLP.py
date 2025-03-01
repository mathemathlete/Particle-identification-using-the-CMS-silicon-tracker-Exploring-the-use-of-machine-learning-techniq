import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import uproot
from Core import Identification as id
from Core import ML_plot as ml
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import timeit
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
from ray.tune import ExperimentAnalysis
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from Core import Identification as id
from Core import ML_plot as ml

def collate_fn(batch):
    ndedx_list, dedx_list, target_list, eta_list = zip(*batch)
    lengths = torch.tensor([len(d) for d in dedx_list], dtype=torch.int64)
    padded_sequences = pad_sequence([d.clone().detach().unsqueeze(-1) if isinstance(d, torch.Tensor) else torch.tensor(d).unsqueeze(-1) for d in dedx_list], batch_first=True)
    extras = torch.stack([torch.tensor([ndedx, eta], dtype=torch.float32) for ndedx, eta in zip(ndedx_list, eta_list)])
    targets = torch.tensor(target_list, dtype=torch.float32)
    return padded_sequences, lengths, targets, extras

class ParticleDataset(Dataset):
    def __init__(self, ndedx_cluster, dedx_values, target_values,eta_values):
        self.ndedx_cluster = ndedx_cluster # int
        self.dedx_values = dedx_values # dedx values is an array of a variable size
        self.target_values = target_values # float
        self.eta_values = eta_values # float

    def __len__(self):
        return len(self.dedx_values)

    def __getitem__(self, idx):
        x = torch.tensor(self.ndedx_cluster[idx],dtype=torch.float32)
        y = torch.tensor(self.dedx_values[idx], dtype=torch.float32)
        z = torch.tensor(self.target_values[idx], dtype=torch.float32)
        u = torch.tensor(self.eta_values[idx], dtype=torch.float32)
        return x, y, z ,u

class MLP_V2b(nn.Module):
    def __init__(self, dedx_hidden_size, dedx_num_layers, mlp_hidden_size1,mlp_hidden_size2,mlp_hidden_size3, dropout_GRU, dropout_dedx,dropout_MLP, adjustment_scale):
        super(MLP_V2b, self).__init__()
        self.dedx_rnn = nn.GRU(
            input_size=1, 
            hidden_size=dedx_hidden_size,
            num_layers=dedx_num_layers,
            batch_first=True,
            dropout=dropout_GRU if dedx_num_layers > 1 else 0.0
        )
        self.dedx_fc = nn.Linear(dedx_hidden_size, 1)
        self.dropout_dedx = nn.Dropout(dropout_dedx)

        # Instead of an LSTM branch, use a simple MLP
        self.adjust_mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden_size1),  # Input: 3 (dedx_pred + extras),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size1, mlp_hidden_size2),  # Output: mlp_hidden_size2
            nn.ReLU(),
            nn.Linear(mlp_hidden_size2, mlp_hidden_size3),  # Output: mlp_hidden_size3
            nn.ReLU(),
            nn.Linear(mlp_hidden_size3, 1)  # Final output
        )
        self.dropout_MLP = nn.Dropout(dropout_MLP)
        self.relu = nn.ReLU()
        
        # Adjustment scale remains as a multiplier
        self.adjustment_scale = adjustment_scale

    def forward(self, dedx_seq, lengths, extras):
        # Process dedx_seq with GRU
        packed_seq = pack_padded_sequence(dedx_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.dedx_rnn(packed_seq)
        hidden_last = hidden[-1]
        dedx_pred = self.relu(self.dropout_dedx(self.dedx_fc(hidden_last)))
        
        # Concatenate dedx_pred (shape: [batch_size, 1]) with extras ([batch_size, 2]) → [batch_size, 43
        combined = torch.cat([dedx_pred, extras], dim=1)
        x = self.adjust_mlp(combined)
        adjustment = self.dropout_MLP(x)
        
        final_value = dedx_pred + self.adjustment_scale * adjustment
        return final_value
    
def train_model(model, dataloader, criterion, optimizer, scheduler, epochs, device):
    model.to(device)  # Ensure model is on GPU
    loss_array = []
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    start_global = timeit.default_timer()
    for epoch in range(epochs):
        start_epoch = timeit.default_timer()
        epoch_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
        model.train()

        for batch, (inputs, lengths, targets, extras) in enumerate(dataloader):
            # Move data to GPU
            inputs, lengths, targets, extras = inputs.to(device), lengths.to(device), targets.to(device), extras.to(device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs, lengths, extras).squeeze()
            targets = targets.squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch % 100 == 0:
                loss_value = loss.item()
                current = batch * batch_size + len(inputs)
                percentage = (current / size) * 100
                print(f"Loss: {loss_value:>7f} ({percentage:.2f}%)")
        mean_epoch_loss = epoch_loss / len(dataloader)
        scheduler.step(mean_epoch_loss)
        print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")
        loss_array.append(mean_epoch_loss)
        print(f"Mean Epoch Loss : {mean_epoch_loss}")
        end = timeit.default_timer()
        elapsed_time_epoch = end - start_epoch
        elapsed_time_global = end - start_global
        hours_epoch, remainder_epoch = divmod(elapsed_time_epoch, 3600)
        minutes_epoch, seconds_epoch = divmod(remainder_epoch, 60)
        hours_global, remainder_global = divmod(elapsed_time_global, 3600)
        minutes_global, seconds_global = divmod(remainder_global, 60)
        print(f"Execution time for epoch {epoch+1}: {int(hours_epoch)} hr {int(minutes_epoch)} min {seconds_epoch:.2f} sec")
        print(f"Total execution time: {int(hours_global)} hr {int(minutes_global)} min {seconds_global:.2f} sec")
    return loss_array

        
def test_model(model, dataloader, criterion,device):
    predictions = []
    model.eval()  # Mettre le modèle en mode évaluation
    test_loss = 0.0
    with torch.no_grad():  # Désactiver la grad pour l'évaluation
        for inputs, lengths, targets, extras in dataloader:  # Expecting 3 values from the dataloader
            inputs, lengths, targets, extras = inputs.to(device), lengths.to(device), targets.to(device), extras.to(device)
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
    return predictions, test_loss

def train_model_ray(config, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLP_V2b(
        dedx_hidden_size=config["dedx_hidden_size"],
        dedx_num_layers=config["dedx_num_layers"],
        mlp_hidden_size1=config["mlp_hidden_size1"],
        mlp_hidden_size2 =config["mlp_hidden_size2"],
        mlp_hidden_size3 = config["mlp_hidden_size3"],
        adjustment_scale=config["adjustment_scale"],
        dropout_GRU=config["dropout_GRU"],
        dropout_dedx=config["dropout_dedx"],
        dropout_MLP=config["dropout_MLP"]
    ).to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)
    
    dataset = ParticleDataset(ndedx_values_train, dedx_values_train, data_th_values,eta_values_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    for epoch in range(30):
        model.train()
        total_loss = 0.0

        for inputs, lengths, targets, extras in dataloader:
            inputs, lengths, targets, extras = inputs.to(device), lengths.to(device), targets.to(device), extras.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, lengths, extras)
            outputs = outputs.squeeze()  
            targets = targets.squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        session.report({"loss": avg_loss})


if __name__ == "__main__":
    # --- Data Import ---
    time_start = timeit.default_timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose GPU if available, otherwise CPU

    file_name = "Root_Files/ML_training_LSTM.root"
    data = pd.DataFrame()
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(["ndedx_cluster","dedx_cluster","track_p","track_eta","Ih"], library="pd") # open data with array from numpy
        train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    # --- Préparer les données de l'entrainement ---
    ndedx_values_train = train_data["ndedx_cluster"].to_list()
    dedx_values_train = train_data["dedx_cluster"].to_list()
    data_th_values = id.bethe_bloch(938e-3, train_data["track_p"]).to_list()  # Targets (valeurs théoriques)
    eta_values_train =  train_data["track_eta"].to_list()
    dataset = ParticleDataset(ndedx_values_train, dedx_values_train, data_th_values,eta_values_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # --- Préparer les données de tests ---
    ndedx_values_test = test_data["ndedx_cluster"].to_list()
    dedx_values_test = test_data["dedx_cluster"].to_list()
    data_th_values_test = id.bethe_bloch(938e-3, test_data["track_p"]).to_list()
    eta_values_test =  test_data["track_eta"].to_list()
    p_values_test = test_data["track_p"].to_list()
    Ih_values_test = test_data["Ih"].to_list()
    test_dataset = ParticleDataset(ndedx_values_test,dedx_values_test, data_th_values_test,eta_values_test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # --- Hyperparameter Initialization ---
    search_space = {
        "dedx_hidden_size": tune.choice([256, 512, 1024]),
        "dedx_num_layers": tune.choice([2, 3]),
        "mlp_hidden_size1" :tune.choice([500, 750, 1000, 1250]),
        "mlp_hidden_size2" :tune.choice([250,500, 750, 1000]),
        "mlp_hidden_size3" : tune.choice([100, 200, 300]),
        "dropout_GRU": tune.uniform(0.1, 0.5),
        "dropout_dedx" : tune.uniform(0.1,0.5),
        "dropout_MLP": tune.uniform(0.1, 0.5),
        "adjustment_scale": tune.uniform(0.1, 1.0),
        "learning_rate": tune.loguniform(1e-4, 1e-2),   
        "weight_decay": tune.loguniform(1e-6, 1e-3),    
        "batch_size" : tune.choice([16,32,64]),
    }

    ray.init(ignore_reinit_error=True)

    analysis = tune.run(
        train_model_ray,
        config=search_space,
        num_samples=20,
        scheduler=ASHAScheduler(metric="loss", mode="min"),
        search_alg=OptunaSearch(metric="loss", mode="min"),
        resources_per_trial={"cpu": 10, "gpu": 0.8},
    )
    
    best_config = analysis.get_best_config(metric="loss", mode="min")

    best_model = MLP_V2b(
        dedx_hidden_size=best_config["dedx_hidden_size"],
        dedx_num_layers=best_config["dedx_num_layers"],
        mlp_hidden_size1=best_config["mlp_hidden_size1"],
        mlp_hidden_size2 =best_config["mlp_hidden_size2"],
        mlp_hidden_size3 = best_config["mlp_hidden_size3"],
        adjustment_scale=best_config["adjustment_scale"],
        dropout_GRU=best_config["dropout_GRU"],
        dropout_dedx=best_config["dropout_dedx"],
        dropout_MLP=best_config["dropout_MLP"]
    ).to(device)

    optimizer = optim.Adam(best_model.parameters(), lr=best_config["learning_rate"], weight_decay = best_config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
    criterion = nn.MSELoss()

    loss_array = train_model(best_model, dataloader, criterion, optimizer, scheduler, epochs=60)
    torch.save(best_model.state_dict(), "GRU_plus_MLP_V2b_tuned_60epoch.pth")

    # model.load_state_dict(torch.load("GRU_plus_MLP_V2b_tuned_60epoch.pth", weights_only=True,map_location=torch.device('cpu')))

    predictions, test_loss = test_model(best_model, test_dataloader, criterion)
    print(f"Final Test Loss: {test_loss}")

    time_end = timeit.default_timer()
    print(f"Execution Time: {time_end - time_start}")

    # Plotting

    data_plot=pd.DataFrame()
    data_plot['track_p']=test_data["track_p"].to_list()
    data_plot['dedx']=predictions
    data_plot['Ih']=Ih_values_test
    data_plot['Ih']=data_plot['Ih']*1e-3
    data_plot['track_eta']=test_data['track_eta']

    ylim_plot=[2,9]
    ml.loss_epoch(loss_array)
    ml.plot_ML(data_plot,ylim_plot, True,True, True)
    #ML.plot_ratio(data_plot,id.m_p)  
    ml.density(data_plot,15,ylim_plot)
    ml.std(data_plot,15,True)
    ml.biais(data_plot,"track_eta",15)
    ml.biais(data_plot,"track_p",15)