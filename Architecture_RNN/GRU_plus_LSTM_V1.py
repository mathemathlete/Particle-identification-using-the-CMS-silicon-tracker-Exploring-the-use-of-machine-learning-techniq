import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import uproot
from Core import Identification as id
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import timeit
from Core import ML_plot as ML
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from Core import Creation_plus_filtred as cpf

def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable-length sequences.
    
    Args:
        batch (list): List of tuples containing dedx sequences, targets, and ndedx, p, eta, Ih as extras parameters.
    
    Returns:
        tuple: Padded sequences, sequence lengths, targets, and extra parameters.
    """
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
    """
    Dataset class for particle data.

    Attributes:
        ndedx_cluster (list): List of ndedx values.
        dedx_values (list): List of dedx sequences (each sequence may have variable length).
        target_values (list): List of target values.
        p_values (list): List of p values.
        eta_values (list): List of eta values.
        Ih_values (list): List of Ih values.
    """
    def __init__(self, ndedx_cluster, dedx_values, target_values, p_values, eta_values, Ih_values):
        
        self.ndedx_cluster = ndedx_cluster  # int
        self.dedx_values = dedx_values      # dedx values is an array of variable size
        self.target_values = target_values  # int        
        self.p_values = p_values            # float
        self.eta_values = eta_values        # float
        self.Ih_values = Ih_values          # float 

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.dedx_values)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing:
                - ndedx (Tensor): ndedx value as a tensor.
                - dedx (Tensor): dedx sequence as a tensor.
                - target (Tensor): target value as a tensor.
                - p (Tensor): p value as a tensor.
                - eta (Tensor): eta value as a tensor.
                - Ih (Tensor): Ih value as a tensor.
        """
        x = torch.tensor(self.ndedx_cluster[idx], dtype=torch.float32)
        y = torch.tensor(self.dedx_values[idx], dtype=torch.float32)
        z = torch.tensor(self.target_values[idx], dtype=torch.float32)
        t = torch.tensor(self.p_values[idx], dtype=torch.float32)
        u = torch.tensor(self.eta_values[idx], dtype=torch.float32)
        o = torch.tensor(self.Ih_values[idx], dtype=torch.float32)
        return x, y, z, t, u, o 

class LSTMModel(nn.Module):
    """
    LSTM model that processes dedx sequences with a GRU and applies an adjustment with an LSTM.

    The model uses a GRU to process the dedx sequence and a fully connected layer to predict an initial value.
    It then concatenates this prediction with additional features and passes it through an LSTM to compute an adjustment.
    The final prediction is the sum of the initial prediction and a scaled adjustment.

    Args:
        dedx_hidden_size (int): Hidden size for the GRU processing dedx sequences.
        dedx_num_layers (int): Number of GRU layers.
        lstm_hidden_size (int): Hidden size for the adjustment LSTM.
        lstm_num_layers (int): Number of LSTM layers.
        adjustment_scale (float): Scaling factor for the adjustment.
        dropout_GRU (float): Dropout probability for GRU (applied if dedx_num_layers > 1).
        dropout_dedx (float): Dropout probability for the dedx fully connected layer.
        dropout_LSTM (float): Dropout probability for LSTM (applied if lstm_num_layers > 1).
    """
    def __init__(self, dedx_hidden_size, dedx_num_layers, lstm_hidden_size, lstm_num_layers,
                 adjustment_scale, dropout_GRU,dropout_dedx, dropout_LSTM,):
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
        self.adjustment_scale = adjustment_scale

    def forward(self, dedx_seq, lengths, extras):
        """
        Forward pass of the LSTM model.

        Args:
            dedx_seq (Tensor): Padded dedx sequences of shape [batch_size, seq_len, 1].
            lengths (Tensor): Actual lengths of each dedx sequence.
            extras (Tensor): Extra features of shape [batch_size, 3] (ndedx, p, eta, Ih)

        Returns:
            Tensor: Final prediction combining dedx prediction and the scaled adjustment.
        """
        packed_seq = pack_padded_sequence(dedx_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.dedx_rnn(packed_seq)
        hidden_last = hidden[-1]
        dedx_pred = self.dedx_fc(hidden_last)
        
        lstm_input = torch.cat([dedx_pred, extras], dim=1).unsqueeze(1)
        _, (h_n, _) = self.adjust_lstm(lstm_input)
        lstm_hidden = h_n[-1]
        adjustment = self.adjust_fc(lstm_hidden)
        
        return dedx_pred + self.adjustment_scale * adjustment

# As we launch a subprocess in main , we need to define train model & test model for every code model
# Could be optimized
def train_model(model, dataloader, criterion, optimizer, scheduler, epochs, device):
    """
    Train the model for a specified number of epochs.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader providing training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for updating model parameters.
        scheduler (Scheduler): Learning rate scheduler.
        epochs (int): Number of epochs to train.
        device (torch.device): Device (CPU/GPU) on which to run training.

    Returns:
        list: A list containing the mean loss for each epoch.
    """
    model.to(device)  # Ensure model is on GPU
    loss_array = [] # Store loss values for each epoch to plot the convergence
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
    """
    Evaluate the model on a test dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader providing test data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device (CPU/GPU) on which to run evaluation.

    Returns:
        tuple: A tuple containing:
            - predictions (list): List of predictions for each test sample.
            - test_loss (float): Total loss over the test set.
    """
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

def start_ML(model,file_model, train,test,tuned_test):
    """
    Entry point for starting the machine learning process for training or testing.

    Args:
        model (nn.Module): The model instance.
        file_model (str): Path to the saved model file.
        train (bool): If True, the model will be trained.
        test (bool): If True, the model will be evaluated.
        tuned_test (bool): If True, the model will be evaluated with tuned hyperparameters.

    Returns:
        If training:
            list: Loss history over epochs.
            float : test_loss under criterion
        If testing (either normal test or tuned test):
            tuple: (predictions, test_loss) from the test dataset.
    """
    if train==True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose GPU if available, otherwise CPU
        losses_epoch = train_model(model, dataloader, criterion, optimizer, scheduler,epoch , device)
        torch.save(model.state_dict(), model)
        return losses_epoch
   
    if test==True:
        model.load_state_dict(torch.load(file_model, weights_only=True)) 
        print("Evaluation du modèle...")
        predictions, test_loss = test_model(model, test_dataloader, criterion)
        return predictions, test_loss
    
    if tuned_test==True:
        model = torch.load(file_model)
        print("Evaluation du modèle...")
        predictions, test_loss = test_model(model, test_dataloader, criterion)
        return predictions, test_loss

if __name__ == "__main__":
    # --- Importation des données ( à remplacer par la fonction d'importation du X)---
    time_start = timeit.default_timer()

    file_name = "Root_Files/ML_training_LSTM.root"
    branch_of_interest = ["ndedx_cluster","dedx_cluster","track_p","track_eta","Ih"]
    file_model = "model_LSTM_40_epoch_15000_V2a.pth"

    data=cpf.import_data(file_name,branch_of_interest)
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
    dedx_hidden_size = 256
    dedx_num_layers = 2   # With one layer, GRU dropout is not applied.
    lstm_hidden_size = 128
    lstm_num_layers = 2
    epoch = 200

    model = LSTMModel(dedx_hidden_size, dedx_num_layers, lstm_hidden_size, lstm_num_layers)
    criterion = nn.MSELoss() # Si pas une grosse influence des outliers
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    
    # Learning rate scheduler: reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # --- Entraînement du modèle ---
    losses_epoch = train_model(model, dataloader, criterion, optimizer, scheduler, epoch)
    torch.save(model.state_dict(), "model_LSTM_40_epoch_15000.pth")

    # --- Sauvegarde et Chargement du modèle ---
    # model.load_state_dict(torch.load("model_LSTM_plus_GRU_1per1.pth", weights_only=True)) 

    # --- Évaluation du modèle ---
    print("Evaluation du modèle...")
    predictions, test_loss = start_ML(model,file_model, False, True)


    time_end = timeit.default_timer()
    elapsed_time = time_end - time_start
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Execution time: {elapsed_time:.2f} seconds ({int(hours)} h {int(minutes)} min {seconds:.2f} sec)")


    data_plot=pd.DataFrame()
    data_plot['track_p']=test_data["track_p"].to_list()
    data_plot['dedx']=predictions
    data_plot['Ih']=Ih_values_test
    data_plot['Ih']=data_plot['Ih']*1e-3
    data_plot['track_eta']=test_data['track_eta']

    # ML.plot_ML_inside(data_plot, False,True , False)
    ylim_plot=[2,9]
    #ML.plot_ML(data_plot,ylim_plot, True,False, False)
    ML.plot_ratio(data_plot,id.m_p)  
    #ML.density(data_plot,15,ylim_plot)
    #ML.std(data_plot,15,True)
    #ML.loss_epoch(start_ML(model,file_model, True, False))
