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
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
from ray.tune import ExperimentAnalysis
import ML_plot as ml

def collate_fn(batch):
    ndedx_list, dedx_list,dx_list,modulegeom_list, target_list, eta_list, Ih_list = zip(*batch)
    lengths = torch.tensor([len(d) for d in dedx_list], dtype=torch.int64)
    padded_sequences_dedx = pad_sequence([d.clone().detach().unsqueeze(-1) if isinstance(d, torch.Tensor) else torch.tensor(d).unsqueeze(-1) for d in dedx_list], batch_first=True)
    padded_sequences_dx = pad_sequence([d.clone().detach().unsqueeze(-1) if isinstance(d, torch.Tensor) else torch.tensor(d).unsqueeze(-1) for d in dx_list], batch_first=True)
    padded_sequences_modulegeom = pad_sequence([d.clone().detach().unsqueeze(-1) if isinstance(d, torch.Tensor) else torch.tensor(d).unsqueeze(-1) for d in modulegeom_list], batch_first=True)
    extras = torch.stack([torch.tensor([ndedx, eta, Ih], dtype=torch.float32) for ndedx, eta, Ih in zip(ndedx_list, eta_list, Ih_list)])
    targets = torch.tensor(target_list, dtype=torch.float32)
    return padded_sequences_dedx,padded_sequences_dx,padded_sequences_modulegeom, lengths, targets, extras

class ParticleDataset(Dataset):
    def __init__(self, ndedx_cluster, dedx_values, dx_values,modulegeom_values, target_values, eta_values,Ih_values):
        self.ndedx_cluster = ndedx_cluster # int
        self.dedx_values = dedx_values # dedx values is an array of a variable size
        self.dx_values = dx_values # dx values is an array of a variable size
        self.modulegeom_values = modulegeom_values
        self.target_values = target_values # float
        self.eta_values = eta_values # float
        self.Ih_values = Ih_values # float 

    def __len__(self):
        return len(self.dedx_values)

    def __getitem__(self, idx):
        x = torch.tensor(self.ndedx_cluster[idx],dtype=torch.float32)
        y = torch.tensor(self.dedx_values[idx], dtype=torch.float32)
        z = torch.tensor(self.dx_values[idx], dtype=torch.float32)
        t = torch.tensor(self.modulegeom_values[idx], dtype=torch.float32)
        u = torch.tensor(self.target_values[idx], dtype=torch.float32)
        o = torch.tensor(self.eta_values[idx], dtype=torch.float32)
        p = torch.tensor(self.Ih_values[idx], dtype=torch.float32)
        return x, y, z , t , u, o ,p

class MLP_V3(nn.Module):
    def __init__(self, dedx_hidden_size, dedx_num_layers, mlp_hidden_size1, mlp_hidden_size2,mlp_hidden_size3,
                 adjustment_scale, dropout_GRU,dropout_dedx, dropout_MLP):
        super(MLP_V3, self).__init__()
        self.dedx_rnn = nn.GRU(
            input_size=3,
            hidden_size=dedx_hidden_size,
            num_layers=dedx_num_layers,
            batch_first=True,
            dropout=dropout_GRU if dedx_num_layers > 1 else 0.0
        )

        self.dedx_fc = nn.Linear(dedx_hidden_size, 1)
        self.dropout_dedx= nn.Dropout(dropout_dedx)
        
        self.adjust_mlp = nn.Sequential(
            nn.Linear(4, mlp_hidden_size1),  # Input: 3 (dedx_pred + extras),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size1, mlp_hidden_size2),  # Output: mlp_hidden_size2
            nn.ReLU(),
            nn.Linear(mlp_hidden_size2, mlp_hidden_size3),  # Output: mlp_hidden_size3
            nn.ReLU(),
            nn.Linear(mlp_hidden_size3, 1)  # Final output
        )
        self.dropout_MLP = nn.Dropout(dropout_MLP)
        self.adjustment_scale = adjustment_scale

    def forward(self, dedx_seq, dx_seq,geom_seq, lengths, extras):
        # Process dedx_seq with GRU
        packed_seq_dedx = pack_padded_sequence(dedx_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_seq_dx = pack_padded_sequence(dx_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_seq_geom = pack_padded_sequence(geom_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)

        combined_dedx_dx = torch.cat([pad_packed_sequence(packed_seq_dedx, batch_first=True)[0], 
                                    pad_packed_sequence(packed_seq_dx, batch_first=True)[0],
                                     pad_packed_sequence(packed_seq_geom, batch_first=True)[0]], dim=2)
        packed_combined = pack_padded_sequence(combined_dedx_dx, lengths.cpu(), batch_first=True, enforce_sorted=False)


        _, hidden = self.dedx_rnn(packed_combined)
        hidden_last = hidden[-1]
        dedx_pred = self.dropout_dedx(self.dedx_fc(hidden_last))
        
        # Concatenate dedx_pred (shape: [batch_size, 1]) with extras ([batch_size, 3]) → [batch_size, 4]
        combined = torch.cat([dedx_pred, extras], dim=1)
        x = self.adjust_mlp(combined)
        adjustment = self.dropout_MLP(x)

        final_value = dedx_pred + self.adjustment_scale * adjustment
        return final_value
    
def train_model(model, dataloader, criterion, optimizer, scheduler, epochs,device):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    loss_array = []
    start_global = timeit.default_timer()
    for epoch in range(epochs):
        start_epoch = timeit.default_timer()
        epoch_loss = 0 
        print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
        model.train()
        for batch, (dedx_seq, dx_seq,geom_seq, lengths, targets, extras) in enumerate(dataloader):
            dedx_seq,dx_seq,geom_seq, lengths, targets, extras = dedx_seq.to(device),dx_seq.to(device),geom_seq.to(device),lengths.to(device), targets.to(device), extras.to(device)
            outputs = model(dedx_seq,dx_seq,geom_seq, lengths, extras)
            outputs = outputs.squeeze()
            targets = targets.squeeze()
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * batch_size + len(dedx_seq)
                percentage = (current / size) * 100
                print(f"loss: {loss_val:>7f} ({percentage:.2f}%)")

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

def test_model(model, dataloader, criterion):
    predictions = []
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        for dedx_seq, dx_seq,geom_seq, lengths, targets, extras in dataloader:
            outputs = model(dedx_seq, dx_seq,geom_seq, lengths, extras)
            outputs = outputs.squeeze()
            targets = targets.squeeze()
            loss = criterion(outputs, targets)
            test_loss += loss.item()       
            if outputs.dim() == 0:
                predictions.append(outputs.item())
            else:
                predictions.extend(outputs.tolist())
    print("Predictions on test data:")
    print(f"Test Loss: {test_loss/len(dataloader):.4f}")
    return predictions, targets, test_loss

def train_model_ray(config, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLP_V3(
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
    
    dataset = ParticleDataset(ndedx_values_train, dedx_values_train, dx_values_train,modulegeom_values_train, data_th_values, eta_values_train, Ih_values_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for dedx_seq, dx_seq,geom_seq, lengths, targets, extras in dataloader:
            dedx_seq,dx_seq,geom_seq, lengths, targets, extras = dedx_seq.to(device),dx_seq.to(device),geom_seq.to(device), lengths.to(device), targets.to(device), extras.to(device)
            optimizer.zero_grad()
            outputs = model(dedx_seq,dx_seq,geom_seq, lengths, extras)
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
    file_name = "Root_Files/data_GRU_V3.root"
    data = pd.DataFrame()
    with uproot.open(file_name) as file:
        key = file.keys()[0]
        tree = file[key]
        data = tree.arrays(["ndedx_cluster", "dedx_cluster","dedx_modulegeom","dedx_pathlength", "track_p", "track_eta", "Ih"], library="pd")
        train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    # --- Prepare Training Data ---
    ndedx_values_train = train_data["ndedx_cluster"].to_list()
    dedx_values_train = train_data["dedx_cluster"].to_list()
    dx_values_train = train_data["dedx_pathlength"].to_list()
    modulegeom_values_train = train_data["dedx_modulegeom"].to_list()
    data_th_values = id.bethe_bloch(id.m_p, train_data["track_p"]).to_list()
    eta_values_train = train_data["track_eta"].to_list()
    Ih_values_train = train_data["Ih"].to_list()
    dataset = ParticleDataset(ndedx_values_train, dedx_values_train, dx_values_train,modulegeom_values_train, data_th_values, eta_values_train, Ih_values_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # --- Prepare Test Data ---
    ndedx_values_test = test_data["ndedx_cluster"].to_list()
    dedx_values_test = test_data["dedx_cluster"].to_list()
    dx_values_test = test_data["dedx_pathlength"].to_list()
    modulegeom_values_test = test_data["dedx_modulegeom"].to_list()
    data_th_values_test = id.bethe_bloch(id.m_p, test_data["track_p"]).to_list()
    p_values_test = test_data["track_p"].to_list()
    eta_values_test = test_data["track_eta"].to_list()
    Ih_values_test = test_data["Ih"].to_list()
    test_dataset = ParticleDataset(ndedx_values_test, dedx_values_test,dx_values_test,modulegeom_values_test, data_th_values_test, eta_values_test, Ih_values_test)
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
        num_samples=10,
        scheduler=ASHAScheduler(metric="loss", mode="min"),
        search_alg=OptunaSearch(metric="loss", mode="min"),
        resources_per_trial={"cpu": 10, "gpu": 1},
    )
    
    best_config = analysis.get_best_config(metric="loss", mode="min")

    best_model = MLP_V3(
        dedx_hidden_size=best_config["dedx_hidden_size"],
        dedx_num_layers=best_config["dedx_num_layers"],
        mlp_hidden_size1=best_config["mlp_hidden_size1"],
        mlp_hidden_size2 =best_config["mlp_hidden_size2"],
        mlp_hidden_size3 = best_config["mlp_hidden_size3"],
        adjustment_scale=best_config["adjustment_scale"],
        dropout_GRU=best_config["dropout_GRU"],
        dropout_dedx=best_config["dropout_dedx"],
        dropout_MLP=best_config["dropout_MLP"]
    )

    optimizer = optim.Adam(best_model.parameters(), lr=best_config["learning_rate"], weight_decay = best_config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
    criterion = nn.MSELoss()

    loss_array = train_model(best_model, dataloader, criterion, optimizer, scheduler, epochs=200)
    torch.save(best_model.state_dict(), "GRU_plus_MLP_V3_tuned_200epoch.pth")

    # model.load_state_dict(torch.load("GRU_plus_MLP_V3.pth", weights_only=True,map_location=torch.device('cpu')))

    predictions, targets, test_loss = test_model(best_model, test_dataloader, criterion)
    print(f"Final Test Loss: {test_loss}")

    time_end = timeit.default_timer()
    print(f"Execution Time: {time_end - time_start}")

    # --- Création des histogrammes ---
    plt.figure(figsize=(12, 6))

    # Histogramme des prédictions
    plt.subplot(1, 2, 1)
    plt.hist(predictions, bins=50, alpha=0.7, label='Prédictions')
    plt.xlabel('Valeur')
    plt.ylabel('N')
    plt.xlim(4,10)
    plt.ylim(0, 2000)
    plt.title('Histogramme des Prédictions')
    plt.legend()

    # Histogramme des valeurs théoriques
    plt.subplot(1, 2, 2)
    plt.hist(data_th_values_test, bins=50, alpha=0.7, label='Valeurs Théoriques')
    plt.xlabel('Valeur')
    plt.ylabel('N')
    plt.title('Histogramme des Valeurs Théoriques')
    plt.xlim(4,10)
    plt.ylim(0, 2000)
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

    ml.loss_epoch(loss_array)