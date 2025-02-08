import Creation_plus_filtrage as cpf
import pandas as pd
import uproot

branch=['dedx_cluster', 'track_p']


print(cpf.import_data("Root_files/ML_training_1.2.root",branch))