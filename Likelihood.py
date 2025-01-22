import numpy as np
from scipy.optimize import minimize
from scipy.stats import landau

class Likelihood_landau:
    # Distribution de Landau de paramètre µ/mu (location) et c (scale)
    def __init__(self, data):
        self.data = np.array(data)
    
    def log_likelihood(self, param):
        mu,c = param
        # Assuming Lando distribution
        return -(np.sum(np.log(landau.pdf(self.data, loc=mu, scale=c))))
    
    def estimate_parameter(self,init_mu, init_c):
        # Minimize the negative log-likelihood
        param =[init_mu,init_c]
        result = minimize(self.log_likelihood,param, bounds=[(1e-10, None)])
        if result.success:
            return result.x[0], result.x[1] # Return estimated parameter
        else:
            raise RuntimeError("Echec")

# Example usage
if __name__ == "__main__":
    data = []
    for i in range(100000):
        data.append(landau.rvs(loc=5, scale=3) )
    param = landau.fit(data)
    
    loc_est, scale_est = param
    print(f"Paramètres estimés fit: loc = {loc_est:.2f}, scale = {scale_est:.2f}")
    
    lando_dist = likelihood_landau(data)
    estimated_param = lando_dist.estimate_parameter(1,1)
    print(f"Paramètres estimés Likelihood (mu,c): {estimated_param:.2f}")