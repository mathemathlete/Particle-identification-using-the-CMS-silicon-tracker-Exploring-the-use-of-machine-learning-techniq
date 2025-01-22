import numpy as np
from scipy.optimize import minimize

class LandoDistribution:
    def __init__(self, data):
        self.data = np.array(data)
    
    def log_likelihood(self, theta):
        # Assuming Lando distribution has a PDF of the form f(x|theta) = theta * exp(-theta * x)
        # Log-likelihood function: sum(log(f(x|theta))) = sum(log(theta) - theta * x)
        return np.sum(np.log(theta) - theta * self.data)
    
    def negative_log_likelihood(self, theta):
        # Negative log-likelihood for minimization
        return -self.log_likelihood(theta)
    
    def estimate_parameter(self):
        # Initial guess for theta
        initial_theta = 1.0
        # Minimize the negative log-likelihood
        result = minimize(self.negative_log_likelihood, initial_theta, bounds=[(1e-10, None)])
        if result.success:
            return result.x[0]
        else:
            raise RuntimeError("Optimization failed")

# Example usage
if __name__ == "__main__":
    data = [0.5, 1.2, 0.3, 0.8, 1.5]  # Example data
    lando_dist = LandoDistribution(data)
    estimated_theta = lando_dist.estimate_parameter()
    print(f"Estimated parameter (theta): {estimated_theta}")