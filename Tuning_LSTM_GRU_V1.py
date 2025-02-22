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
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
from ray.tune import ExperimentAnalysis
import ML_plot as ml

def collate_fn(batch):
    ndedx_list, dedx_list, target_list, p_list, eta_list, Ih_list = zip(*batch)
    lengths = torch.tensor([len(d) for d in dedx_list], dtype=torch.float32)
    padded_sequences = pad_sequence(
        [d.clone().detach().unsqueeze(-1) if isinstance(d, torch.Tensor) else torch.tensor(d).unsqueeze(-1) 
         for d in dedx_list],
        batch_first=True
    )
    extras = torch.stack([
        torch.tensor([ndedx, p, eta, Ih], dtype=torch.float32)
        for ndedx, p, eta, Ih in zip(ndedx_list, p_list, eta_list, Ih_list)
    ])
    targets = torch.tensor(target_list, dtype=torch.float32)
    return padded_sequences, lengths, targets, extras

class ParticleDataset(Dataset):
    def __init__(self, ndedx_cluster, dedx_values, target_values, p_values, eta_values, Ih_values):
        self.ndedx_cluster = ndedx_cluster  # int
        self.dedx_values = dedx_values      # dedx values is an array of variable size
        self.target_values = target_values  # int        
        self.p_values = p_values            # float
        self.eta_values = eta_values        # float
        self.Ih_values = Ih_values          # float 

    def __len__(self):
        return len(self.dedx_values)

    def __getitem__(self, idx):
        x = torch.tensor(self.ndedx_cluster[idx], dtype=torch.float32)
        y = torch.tensor(self.dedx_values[idx], dtype=torch.float32)
        z = torch.tensor(self.target_values[idx], dtype=torch.float32)
        t = torch.tensor(self.p_values[idx], dtype=torch.float32)
        u = torch.tensor(self.eta_values[idx], dtype=torch.float32)
        o = torch.tensor(self.Ih_values[idx], dtype=torch.float32)
        return x, y, z, t, u, o 

class LSTMModel(nn.Module):
    def __init__(self, dedx_hidden_size, dedx_num_layers, lstm_hidden_size, lstm_num_layers,
                 adjustement_scale, dropout_GRU,dropout_dedx, dropout_LSTM,):
        super(LSTMModel, self).__init__()
        self.dedx_rnn = nn.GRU(
            input_size=1, 
            hidden_size=dedx_hidden_size,
            num_layers=dedx_num_layers,
            batch_first=True,
            dropout=dropout_GRU if dedx_num_layers > 1 else 0.0
        )
        self.dedx_fc = nn.Linear(dedx_hidden_size, 1)
        self.dropout_dedx= nn.Dropout(dropout_dedx)
        
        self.adjust_lstm = nn.LSTM(
            input_size=5,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_LSTM if lstm_num_layers > 1 else 0.0
        )
        self.adjust_fc = nn.Linear(lstm_hidden_size, 1)
        self.relu = nn.ReLU()
        self.adjustment_scale = adjustement_scale

    def forward(self, dedx_seq, lengths, extras):
        packed_seq = pack_padded_sequence(dedx_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.dedx_rnn(packed_seq)
        hidden_last = hidden[-1]
        dedx_pred = self.dedx_fc(hidden_last)
        
        lstm_input = torch.cat([dedx_pred, extras], dim=1).unsqueeze(1)
        _, (h_n, _) = self.adjust_lstm(lstm_input)
        lstm_hidden = h_n[-1]
        adjustment = self.adjust_fc(lstm_hidden)
        
        return dedx_pred + self.adjustment_scale * adjustment



def train_model(model, dataloader, criterion, optimizer, scheduler, epochs=20):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    start_global = timeit.default_timer()
    losses_array = []
    for epoch in range(epochs):
        start_epoch = timeit.default_timer()
        epoch_loss = 0 
        print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
        model.train()
        for batch, (inputs, lengths, targets, extras) in enumerate(dataloader):
            outputs = model(inputs, lengths, extras)
            outputs = outputs.squeeze()
            targets = targets.squeeze()
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if batch % 100 == 0:
                loss_val, current = loss.item(), batch * batch_size + len(inputs)
                percentage = (current / size) * 100
                print(f"loss: {loss_val:>7f} ({percentage:.2f}%)")
        mean_epoch_loss = epoch_loss/len(dataloader)
        scheduler.step(mean_epoch_loss)
        print(f"Mean Epoch Loss : {mean_epoch_loss}")
        losses_array.append(mean_epoch_loss)
        print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")
        end = timeit.default_timer()
        elapsed_time_epoch = end - start_epoch
        elapsed_time_global = end - start_global
        hours_epoch, remainder_epoch = divmod(elapsed_time_epoch, 3600)
        minutes_epoch, seconds_epoch = divmod(remainder_epoch, 60)
        hours_global, remainder_global = divmod(elapsed_time_global, 3600)
        minutes_global, seconds_global = divmod(remainder_global, 60)
        print(f"Execution time for epoch {epoch+1}: {int(hours_epoch)} hr {int(minutes_epoch)} min {seconds_epoch:.2f} sec")
        print(f"Total execution time: {int(hours_global)} hr {int(minutes_global)} min {seconds_global:.2f} sec")
    return losses_array

def test_model(model, dataloader, criterion):
    predictions = []
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    with torch.no_grad():
        for inputs, lengths, targets, extras in dataloader:
            outputs = model(inputs, lengths, extras)
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
    return predictions, test_loss

def train_model_ray(config, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMModel(
        dedx_hidden_size=config["dedx_hidden_size"],
        dedx_num_layers=config["dedx_num_layers"],
        lstm_hidden_size=config["lstm_hidden_size"],
        lstm_num_layers=config["lstm_num_layers"],
        adjustement_scale=config["adjustment_scale"],
        dropout_dedx=config["dropout_dedx"],
        dropout_GRU=config["dropout_GRU"],
        dropout_LSTM=config["dropout_LSTM"]
    ).to(device)
    
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config["decrease_factor_scheduler"])
    
    # Create a fresh DataLoader for this trial
    dataset = ParticleDataset(ndedx_values_train, dedx_values, data_th_values, p_values_train, eta_values_train, Ih_values_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    for epoch in range(20):
        model.train()
        total_loss = 0.0

        for inputs, lengths, targets, extras in dataloader:
            inputs, lengths, targets, extras = inputs.to(device), lengths.to(device), targets.to(device), extras.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, lengths, extras).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        # Use Ray AIR's session reporting API
        session.report({"loss": avg_loss})

if __name__ == "__main__":
    # --- Data Import ---
    time_start = timeit.default_timer()
    file_name = "Root_Files/ML_training_LSTM.root"
    data = pd.DataFrame()
    with uproot.open(file_name) as file:
        key = file.keys()[0]
        tree = file[key]
        data = tree.arrays(["ndedx_cluster", "dedx_cluster", "track_p", "track_eta", "Ih"], library="pd")
        train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

    # --- Prepare Training Data ---
    ndedx_values_train = train_data["ndedx_cluster"].to_list()
    dedx_values = train_data["dedx_cluster"].to_list()
    data_th_values = id.bethe_bloch(938e-3, train_data["track_p"]).to_list()
    p_values_train = train_data["track_p"].to_list()
    eta_values_train = train_data["track_eta"].to_list()
    Ih_values_train = train_data["Ih"].to_list()
    dataset = ParticleDataset(ndedx_values_train, dedx_values, data_th_values, p_values_train, eta_values_train, Ih_values_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # --- Prepare Test Data ---
    ndedx_values_test = test_data["ndedx_cluster"].to_list()
    dedx_values_test = test_data["dedx_cluster"].to_list()
    data_th_values_test = id.bethe_bloch(938e-3, test_data["track_p"]).to_list()
    p_values_test = test_data["track_p"].to_list()
    eta_values_test = test_data["track_eta"].to_list()
    Ih_values_test = test_data["Ih"].to_list()
    test_dataset = ParticleDataset(ndedx_values_test, dedx_values_test, data_th_values_test, p_values_test, eta_values_test, Ih_values_test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # --- Hyperparameter Initialization ---
    search_space = {
        "dedx_hidden_size": tune.choice([64, 128, 256, 512, 1024]),
        "dedx_num_layers": tune.choice([1, 2, 3]),
        "lstm_hidden_size": tune.choice([32, 64, 128]),
        "lstm_num_layers": tune.choice([1, 2, 3]),
        "adjustment_scale": tune.uniform(0.1, 1.0),
        "dropout_dedx" : tune.uniform(0.1,0.5),
        "dropout_GRU": tune.uniform(0.1, 0.5),
        "dropout_LSTM": tune.uniform(0.1, 0.5),
        "learning_rate": tune.loguniform(1e-4, 1e-2),   
        "weight_decay": tune.loguniform(1e-6, 1e-3),    
        "decrease_factor_scheduler": tune.choice([0.5, 0.1])
    }
    # Le tuning avait été lancé sans évaluation du dropout_dedx, vu qu'on a pas eu le tps de le relancer on le pose à l'extérieur 
    dropout_dedx = 0.1
    ray.init(ignore_reinit_error=True)

    # analysis = tune.run(
    #     train_model_ray,
    #     config=search_space,
    #     num_samples=20,
    #     scheduler=ASHAScheduler(metric="loss", mode="min"),
    #     search_alg=OptunaSearch(metric="loss", mode="min"),
    #     resources_per_trial={"cpu": 10, "gpu": 0.8},
    # )
    
    # best_config = analysis.get_best_config(metric="loss", mode="min")
    
    analysis = ExperimentAnalysis("C:/Users/Kamil/ray_results/Tuning_GRU_LSTM_1per1")  # Load experiment data
    # Get the best trial based on a metric (e.g., lowest loss)
    best_trial = analysis.get_best_trial(metric="loss", mode="min")  
    best_config = best_trial.config  # Best hyperparameters

    best_model = LSTMModel(
        dedx_hidden_size=best_config["dedx_hidden_size"],
        dedx_num_layers=best_config["dedx_num_layers"],
        lstm_hidden_size=best_config["lstm_hidden_size"],
        lstm_num_layers=best_config["lstm_num_layers"],
        adjustement_scale=best_config["adjustment_scale"],
        dropout_GRU=best_config["dropout_GRU"],
        dropout_dedx=0.1,
        dropout_LSTM=best_config["dropout_LSTM"]
    )
    
    optimizer = optim.Adam(best_model.parameters(), lr=best_config["learning_rate"], weight_decay=best_config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
    criterion = nn.MSELoss()

    losses_array = train_model(best_model, dataloader, criterion, optimizer, scheduler, epochs=200)
    torch.save(best_model.state_dict(), "best_model_GRU_LSTM_200epoch.pth")

    predictions, test_loss = test_model(best_model, test_dataloader, criterion)
    print(f"Final Test Loss: {test_loss}")

    time_end = timeit.default_timer()
    print(f"Execution Time: {time_end - time_start}")

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(predictions, bins=50, alpha=0.7, label='Predictions')
    plt.xlabel('Value')
    plt.ylabel('N')
    plt.xlim(4, 10)
    plt.ylim(0, 2000)
    plt.title('Histogram of Predictions')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(data_th_values_test, bins=50, alpha=0.7, label='Theoretical Values')
    plt.xlabel('Value')
    plt.ylabel('N')
    plt.title('Histogram of Theoretical Values')
    plt.xlim(4, 10)
    plt.ylim(0, 2000)
    plt.legend()
    plt.tight_layout()

    np_th = np.array(targets)
    np_pr = np.array(predictions)

    plt.figure(figsize=(8, 8))
    plt.hist2d(p_values_test, np_pr - np_th, bins=500, cmap='viridis', label='Data')
    plt.xlabel('Value')
    plt.ylabel('th-exp')
    plt.title('Difference between theoretical and predicted')
    plt.legend()

    p_axis = np.logspace(np.log10(0.0001), np.log10(2), 500)
    plt.figure(figsize=(8, 8))
    plt.hist2d(p_values_test, np_pr, bins=500, cmap='viridis', label='Data')
    plt.plot(p_axis, id.bethe_bloch(938e-3, np.array(p_axis)), color='red')
    plt.xscale('log')
    plt.show()

    ml.loss_epoch(losses_array)