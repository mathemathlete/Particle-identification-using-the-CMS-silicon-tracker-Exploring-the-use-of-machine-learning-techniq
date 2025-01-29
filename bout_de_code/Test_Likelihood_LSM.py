import numpy as np
from scipy.optimize import minimize
from scipy.stats import landau, norm
from scipy.signal import convolve
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import awkward as ak

class Likelihood_landau:
    # Distribution de Landau de paramètre µ/mu (location) et c (scale)
    def __init__(self, data):
        self.data = np.array(data)

    # Les fonctions qui suivent sont les fonctions que l'on va chercher à estimer avec le LikeLihood
    def landau(self,*args):
        return landau.pdf(self.data, *args)
    
    def gauss(self,*args):
        return norm.pdf(self.data,*args)
    
    def landau_plus_gauss(self,*args):
        return (self.landau(*args[0:2]) + self.gauss(*args[2:4]))
    
    def landau_plus_deux_gauss(self,*args):
        return (self.landau(*args[0:2]) + self.gauss(*args[2:4]) + self.gauss(*args[4:6]))
    
    def landau_conv_gauss(self,*args):
        return (convolve( self.landau(*args[0:2]),self.gauss(*args[2:4])))
    
    def log_likelihood(self,function, *args):
        # We ask for the function that we want to estimate with the likelihood method
        return -(np.sum(np.log(function(self.data,*args))))
    
    def estimate_parameter(self,*args):
        # Minimize the negative log-likelihood
        result = minimize(self.log_likelihood,*args, bounds=[(1e-10, None)])
        if result.success:
            return result.x[0], result.x[1] # Return estimated parameter
        else:
            raise RuntimeError("Echec")

if __name__ == "__main__": 
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
    
    ## Pour voir la tête qu'on les données
    # print(array_p) 
    # print("\n")
    # print(dedx) 
    # print(ak.max(dedx))
    # print(ak.min(dedx))
    # print(ak.mean(dedx))
    # print("\n")
    # print(array_p.max())
    # print(array_p.min())
    # print(array_p.mean())

    # Partie à decommenter après avoir fait les tests
    # Number of bins rule (here Freedman-Diaconis (pue sa mère comme règle en fait))
    # taille_bin = 2*(ak.max(dedx)-ak.min(dedx))/(len(dedx)**(1/3))
    # A revoir , en comptant le nbre ça avait pas l'air de marcher
    taille_bin = 439
    bins = np.arange(300, 43956800,taille_bin)
    data= ak.to_numpy(dedx)
    print(dedx)
    plt.hist(dedx, bins, density=True, histtype='step', color='r', label='Histogramme')
    lando_dist = Likelihood_landau(data)
    estimated_param = lando_dist.estimate_parameter(1,1) # initialisation at value loc=1 and scale=1
    print(f"Paramètres estimés Likelihood (mu,c): {estimated_param}") # Print estimated parameters
    loc_est, scale_est = estimated_param

    courbe_est = landau.pdf(bins, loc=loc_est, scale=scale_est)

    # bin_edges = np.linspace(xmin, xmax, 100)  
    # hist, edges = np.histogram(data, bins=bin_edges, density=True)  # Density = True => Normalisation or not
    # bin_centers = (edges[:-1] + edges[1:]) / 2
    
    # plt.plot(x, courbe_th, 'r-', lw=2, label='Courbe théorique')
    plt.plot(bins, courbe_est, 'k--', lw=2, label='Courbe estimée')
    # plt.plot(bin_centers, hist, drawstyle='steps-mid', label='Histogramme à partir des données simulées')
    plt.title('Comparaison entre Simulation et Estimateur')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.show()  
