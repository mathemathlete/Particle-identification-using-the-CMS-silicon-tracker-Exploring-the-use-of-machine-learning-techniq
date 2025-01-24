import uproot
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak

branch_of_interest = ["dedx_charge", "dedx_pathlength", "track_p"]
data = pd.DataFrame()
with uproot.open("clean_p.root") as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy 
   
array_de=ak.sum(data['dedx_charge'],axis=-1)
array_dx=ak.sum(data['dedx_pathlength'],axis=-1)
array_p=data['track_p']
dedx=array_de/array_dx
print(array_p)  

# plot
plt.scatter(array_p, dedx)
plt.xlabel(r'p')
plt.ylabel(r'$-\frac{dE}{dx}$)')
plt.title('Bethe-Bloch Formula')
plt.grid(True)
plt.show()
