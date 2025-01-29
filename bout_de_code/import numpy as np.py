import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import landau, norm
from scipy.signal import convolve
from scipy.optimize import minimize

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
    
    def landau_conv_gauss(self, loc, scale, mean, stddev):
        """Convolution of Landau distribution with Gaussian distribution"""
        return convolve(self.landau(loc, scale), self.gauss(mean, stddev))
    
    # Pr illustrer l'utilisation de convolve
    # def landau_conv_two_diracs(self, *args):
    #     # Create Dirac delta spikes as spikes in the array
    #     # We simulate Dirac delta functions by putting high values at specific points
    #     dirac1 = np.zeros_like(self.data)
    #     dirac2 = np.zeros_like(self.data)
        
    #     # Setting spikes for Dirac delta functions at specific positions
    #     dirac1[np.argmin(np.abs(self.data - args[4]))] = 1e5  # First Dirac spike
    #     dirac2[np.argmin(np.abs(self.data - args[5]))] = 1e5  # Second Dirac spike
        
    #     # Convolve Landau with Diracs (shifting Landau distribution)
    #     landau_pdf = self.landau(*args[0:2])
    #     conv_dirac1 = convolve(landau_pdf, dirac1, mode='same')
    #     conv_dirac2 = convolve(landau_pdf, dirac2, mode='same')
        
    #     return conv_dirac1 + conv_dirac2

    def log_likelihood(self, params, function):
        # print(-np.sum(np.log(function(*params))))
        """Negative log-likelihood function for estimation"""
        return -np.sum(np.log(function(*params)))

    def estimate_parameter(self, function, initial_guess):
        """Estimate parameters using maximum likelihood"""
        result = minimize(self.log_likelihood, initial_guess, args=(function,), bounds=[(1e-10, None)] * len(initial_guess))
        if result.success:
            return result.x  # Return the estimated parameters
        else:
            raise RuntimeError("Optimization failed")
    


# Function to generate data convolved with Landau and Gaussian
def generate_convolved_data(loc, scale, gauss_mean, gauss_stddev, Nbre):
    # Generate Landau-distributed data
    landau_data = np.array([landau.rvs(loc, scale) for _ in range(Nbre)])
    
    # Create Gaussian distribution PDF over the same range
    gauss_data = np.array([np.random.normal(loc, scale) for _ in range(Nbre)])
    
    # Convolve the Landau data with the Gaussian distribution
    convolved_data = convolve(landau_data, gauss_data, mode = 'full') 
    
    return landau_data, gauss_data, convolved_data

# Example usage
loc_th = 3
scale_th = 3
gauss_mean = -2
gauss_stddev = 1
Nbre = 100000
xmin = loc_th - 2*scale_th
xmax = loc_th + 10*scale_th

# Generate the data
landau_data, gauss_data, convolved_data = generate_convolved_data(loc_th, scale_th, gauss_mean, gauss_stddev, Nbre)
counts, bin_edges = np.histogram(convolved_data, bins=10, range=(xmin,xmax))
print(landau_data)
print(gauss_data)
print(counts)
plt.bar(bin_edges[:-1],counts)
plt.show()
# land_dic = Likelihood_landau(filtered_convolved_data)
# estimated_param_conv = land_dic.estimate_parameter(land_dic.landau_conv_gauss, [1, 1, 0, 1])
# print("Estimated Parameters for landau_conv_gauss:", estimated_param_conv)


# # Plot the individual Landau and Gaussian distributions
# x_vals = np.linspace(-10, 10, Nbre)

# plt.figure(figsize=(12, 8))

# # Plot Landau Distribution
# plt.subplot(2, 2, 1)
# plt.plot(x_vals, landau.pdf(x_vals, loc_th, scale_th), label="Landau PDF")
# plt.title("Landau Distribution")
# plt.xlabel("x")
# plt.ylabel("Density")
# plt.grid(True)

# # Plot Gaussian Distribution
# plt.subplot(2, 2, 2)
# plt.plot(x_vals, gauss_data, label="Gaussian PDF", color='orange')
# plt.title("Gaussian Distribution")
# plt.xlabel("x")
# plt.ylabel("Density")
# plt.grid(True)

# # Plot Convolved Data
# plt.subplot(2, 2, 3)
# plt.hist(filtered_convolved_data, bins=30, density=True, alpha=0.6, color='g', range=(-10, 10))
# plt.title("Histogram of Convolved Data (Clipped to [-10, 10])")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.grid(True)

# # Plot the convolved data PDF for comparison
# plt.subplot(2, 2, 4)
# plt.plot(x_vals, convolve(landau.pdf(x_vals, loc_th, scale_th), gauss_data, mode='same'), label="Convolved PDF")
# plt.title("Convolution of Landau and Gaussian")
# plt.xlabel("x")
# plt.ylabel("Density")
# plt.grid(True)

# plt.tight_layout()
# plt.show()
