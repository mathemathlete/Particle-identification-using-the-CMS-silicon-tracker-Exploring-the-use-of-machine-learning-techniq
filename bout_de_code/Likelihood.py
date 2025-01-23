import numpy as np
from scipy.optimize import minimize
from scipy.stats import landau
import matplotlib.pyplot as plt

class Likelihood_landau:
    # Distribution de Landau de paramètre µ/mu (location) et c (scale)
    def __init__(self, data):
        self.data = np.array(data)
    
    def log_likelihood(self, param):
        mu,c = param
        # Assuming Landau distribution
        return -(np.sum(np.log(landau.pdf(self.data, loc=mu, scale=c))))
    
    def estimate_parameter(self,init_mu, init_c):
        # Minimize the negative log-likelihood
        param =[init_mu,init_c]
        result = minimize(self.log_likelihood,param, bounds=[(1e-10, None)])
        if result.success:
            return result.x[0], result.x[1] # Return estimated parameter
        else:
            raise RuntimeError("Echec")
        

if __name__ == "__main__":
    loc_th = 5
    scale_th = 3
    Nbre = 100000
    data = []
    for i in range(Nbre):
        data.append(landau.rvs(loc_th, scale_th) )
    param = landau.fit(data)
    
    loc_est, scale_est = param
    print(f"Paramètres estimés fit: loc = {loc_est:.2f}, scale = {scale_est:.2f}")
    
    lando_dist = Likelihood_landau(data)
    estimated_param = lando_dist.estimate_parameter(1,1) # initialisation at value loc=1 and scale=1
    print(f"Paramètres estimés Likelihood (mu,c): {estimated_param}") # Print estimated parameters

    xmin = loc_th - 2*scale_th
    xmax = loc_th + 10*scale_th
    x = np.linspace(xmin,xmax, 1000)
    xx = np.linspace(xmin,xmax, Nbre)
    courbe_th = landau.pdf(x, loc=loc_th, scale=scale_th)
    courbe_est = landau.pdf(x, loc=loc_est, scale=scale_est)

    bin_edges = np.linspace(xmin, xmax, 100)  
    hist, edges = np.histogram(data, bins=bin_edges, density=True)  # Density = True => Normalisation or not
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    plt.plot(x, courbe_th, 'r-', lw=2, label='Courbe théorique')
    plt.plot(x, courbe_est, 'k--', lw=2, label='Courbe estimée')
    plt.plot(bin_centers, hist, drawstyle='steps-mid', label='Histogramme à partir des données simulées')
    plt.title('Comparaison entre Simulation et Estimateur')
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.grid(True)
    plt.show()  
