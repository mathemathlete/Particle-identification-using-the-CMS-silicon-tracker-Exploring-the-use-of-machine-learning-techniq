import torch
import uproot
import pandas as pd 
import Identification as id
import ML_first as ml 
from torch.nn.utils.rnn import pad_sequence


file_name = "ML_training.root"
data = pd.DataFrame()
with uproot.open(file_name) as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    data = tree.arrays(["dedx_cluster","track_p"], library="pd") # open data with array from numpy
    

def prediction(model, dataloader):
    predictions = []
    for batch in dataloader:
            batch_padded = pad_sequence(batch, batch_first=True, padding_value=0)
            outputs = model(batch_padded).squeeze()
            predictions.extend(outputs.tolist())

    return predictions


dedx_values= data["dedx_cluster"].to_list()
data_th_values = id.bethe_bloch(938e-3, data["track_p"]).to_list()

# Assuming the model is saved as 'model.pth'

model=ml.MLP(len(dedx_values))
state_dict = torch.load('model.pth',weights_only=True)  
model.load_state_dict(state_dict)
model.eval()  # Met le modèle en mode évaluation

# Get predictions
predictions = prediction(model, dedx_values)

    # Print or process predictions as needed
print(predictions)
