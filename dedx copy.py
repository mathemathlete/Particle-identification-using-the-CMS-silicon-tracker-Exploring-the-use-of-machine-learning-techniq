import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from concurrent.futures import ProcessPoolExecutor


data = pd.DataFrame()
n_thread = 4

# Open the ROOT file
def import_data(file,data):
    with uproot.open(file) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(library="pd") # open data with array from numpy 
    return(data)  

def de_dx(data):
    de=ak.sum(data['dedx_charge'],axis=-1)
    dx=ak.sum(data['dedx_pathlength'],axis=-1)
    dedx=de/dx
    return dedx


simu=import_data("clean_p.root",data)
p=simu['track_p']
chunks = np.array_split(simu,n_thread)

with ProcessPoolExecutor() as executor:
    results = list(executor.map(de_dx, chunks))

dedx_final = pd.concat(results, ignore_index=True)

# plot
plt.scatter(p, de_dx(dedx_final))
plt.xlabel(r'p')
plt.ylabel(r'$-\frac{dE}{dx}$)')
plt.title('Bethe-Bloch Formula')
plt.grid(True)
plt.show()
