import numpy as np
from scipy.optimize import minimize
from scipy.stats import landau, norm
from scipy.signal import convolve
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import awkward as ak

class Likelihood_landau:
    def __init__(self, data):
        self.data = np.array(data)  # Store data as numpy array

    def landau(self, loc, scale):
        """Landau distribution PDF for given location and scale"""
        return landau.pdf(self.data, loc, scale)
    
    def gauss(self, mean, stddev):
        """Gaussian distribution PDF for given mean and standard deviation"""
        return norm.pdf(self.data, mean, stddev)
    
    def landau_plus_gauss(self, loc, scale, mean, stddev):
        """Sum of Landau and Gaussian PDFs"""
        return self.landau(loc, scale) + self.gauss(mean, stddev)
    
    def landau_plus_deux_gauss(self, loc, scale, mean1, stddev1, mean2, stddev2):
        """Sum of Landau and two Gaussian PDFs"""
        return self.landau(loc, scale) + self.gauss(mean1, stddev1) + self.gauss(mean2, stddev2)
    
    # Pas verifier le fonctionnement pour l'instant
    # def landau_conv_gauss(self, loc, scale, mean, stddev):
    #     """Convolution of Landau distribution with Gaussian distribution"""
    #     return convolve(self.landau(loc, scale), self.gauss(mean, stddev))
    
    def log_likelihood(self, params, function):
        epsilon = 1e-10
        # print(-np.sum(np.log(function(*params))))
        """Negative log-likelihood function for estimation"""
        return -np.sum(np.log(function(*params) + epsilon))

    def estimate_parameter(self, function, initial_guess):
        """Estimate parameters using maximum likelihood"""
        result = minimize(self.log_likelihood, initial_guess, args=(function,), bounds=[(1e-10, None)] * len(initial_guess))
        if result.success:
            return result.x  # Return the estimated parameters
        else:
            raise RuntimeError("Optimization failed")

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
    
    # Regarder potentiellement des théories
    param_initial = [1, 1, -1, 1]
    land_dic = Likelihood_landau(dedx)
    estimated_param = land_dic.estimate_parameter(land_dic.landau_plus_gauss, [1, 1, -1, 1])
    
    nbre_bin = 100
    xmin = -10
    xmax = 20
    x = np.linspace(xmin,xmax,len(dedx))

    bin_edges = np.linspace(xmin, xmax, 100)  
    hist, edges = np.histogram(dedx, bins=bin_edges, density=True)  # Density = True => Normalisation or not
    bin_centers = (edges[:-1] + edges[1:]) / 2

    courbe_est = landau.pdf(x, loc=estimated_param[0], scale=estimated_param[1]) + norm.pdf(x,estimated_param[2],estimated_param[3])

    plt.plot(x, courbe_est, 'k--', lw=2, label='Courbe estimée')
    plt.plot(bin_centers, hist, drawstyle='steps-mid', label='Histogramme à partir des données simulées')
    plt.title('Comparaison entre Simulation et Estimateur')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.show() 