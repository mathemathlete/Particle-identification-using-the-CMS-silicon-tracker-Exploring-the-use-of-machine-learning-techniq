import uproot
import pandas as pd
import numpy as np

def import_data(data):
    file_name = "ML_training_LSTM_filtré_Max_Ih_20000.root"
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        # tree.show()
        data_tree = tree.arrays(library="pd",entry_stop=100)  # open data with panda
        for array_name, array in data_tree.items():
            data[array_name] = array
    return(data)

data = pd.DataFrame()
print(import_data(data))