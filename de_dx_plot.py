import uproot
import pandas as pd
import matplotlib.pyplot as plt
import sys
import Identification as bb
import numpy as np

m_p = 938e-3 # proton mass in GeV

branch_of_interest = ["It", "track_p"]
data = pd.DataFrame()
with uproot.open("de_dx.root") as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy 
    

data=data[data['It'] <= 12000].reset_index(drop=True) 

p=data['track_p']
dedx=data['It']*1e-9

# 2D histogram
p_values = np.logspace(np.log10(0.000001), np.log10(5), 500)
beth_bloch_curve = bb.bethe_bloch(p_values, m_p)
# print(beth_bloch_curve)
plt.hist2d(p, dedx, bins=500, cmap='viridis')
plt.plot(p_values, beth_bloch_curve, color='red', label='Beth-Bloch theory')
plt.colorbar(label='Counts')
plt.xlabel(r'p')
plt.ylabel(r'$-\frac{dE}{dx}$)')
plt.title('Bethe-Bloch Formula')
plt.grid(True)
plt.show()
plt.legend()
