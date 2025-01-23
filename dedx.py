import uproot
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak


data = pd.DataFrame()
with uproot.open("slim_nt_mc_aod_1.root") as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    # tree.show()
    data_tree = tree.arrays(library="pd") # open data with array from numpy 
    for array_name, array in data_tree.items(): # loop over the arrays in the tree
        if array_name != "nstrips":
            data[array_name] = array
        if array_name == "dedx_shape":
            B = []
            for i in range(len(array)):
                B.append(len(array[i]))
    data["Number of elements"]= B 

array_dx=ak.sum(data['dedx_pathlength'],axis=-1)
array_de=ak.sum(data['dedx_charge'],axis=-1)
dedx=array_de/array_dx

array_p=data['track_p']


plt.plot(dedx, array_p)
plt.xlabel(r'p (GeV/c)')
plt.ylabel(r'$-\frac{dE}{dx}$ (GeV cm$^2$/g)')
plt.title('Bethe-Bloch Formula')
plt.grid(True)
plt.show()