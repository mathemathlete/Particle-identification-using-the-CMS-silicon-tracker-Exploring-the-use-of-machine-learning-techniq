import uproot
import numpy as np
import matplotlib.pyplot as plt

with uproot.open("slim_nt_mc_aod_1.root") as file:
    key = file.keys()[0] # open the first Ttree
    tree = file[key]    
    data = tree.arrays(library="np") # open data with array from numpy 
    for array_name, array in data.items(): # loop over the arrays in the tree    
        array_de  = np.array(data['dedx_charge'])
        array_dx = np.array(data['dedx_pathlength'])
        array_p = np.array(data['track_p'])
    # Perform element-wise division
    print(array_de[:,0])
    #de=[sum(array_de[i][:]) for i in len(array_de[:][0])]
    #de_dx = np.divide(mean_de, mean_dx)
# Plot the data
# plt.plot(de_dx, array_p)
# plt.xlabel(r'p (GeV/c)')
# plt.ylabel(r'$-\frac{dE}{dx}$ (MeV cm$^2$/g)')
# plt.title('Bethe-Bloch Formula')
# plt.grid(True)
# plt.show()