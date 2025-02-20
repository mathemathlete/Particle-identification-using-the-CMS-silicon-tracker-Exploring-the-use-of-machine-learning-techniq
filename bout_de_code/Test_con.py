# Fichier pour les tests à la con
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import landau, norm
from scipy.signal import convolve

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
        return convolve(self.landau(loc, scale), self.gauss(mean, stddev) )
    
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
        """Negative log-likelihood function for estimation"""
        return -np.sum(np.log(function(*params)))

    def estimate_parameter(self, function, initial_guess):
        """Estimate parameters using maximum likelihood"""
        result = minimize(self.log_likelihood, initial_guess, args=(function,), bounds=[(1e-10, None)] * len(initial_guess))
        if result.success:
            return result.x  # Return the estimated parameters
        else:
            raise RuntimeError("Optimization failed")
    


if __name__ == "__main__":
    
    
    # A= np.linspace(0,10,11)
    # print(A)
    # print(A[0:2])
    # print(A[2:3])
    # print(A[4:5])

    # loc_th = 5
    # scale_th = 3
    # Nbre = 1000
    # data = []
    # x=np.linspace(-10,10,10000)
    # for i in range(Nbre):
    #     data.append(landau.rvs(loc_th, scale_th))
    # land_dic = Likelihood_landau(data)
    # estimated_param = land_dic.estimate_parameter(land_dic.landau, [1,1])  # Using landau
    # print("Estimated Parameters for landau:", estimated_param)

    # estimated_param_gauss = land_dic.estimate_parameter(land_dic.landau_plus_gauss, [1, 1, 0, 1])
    # print("Estimated Parameters for landau_plus_gauss:", estimated_param_gauss)

    # # Using landau_plus_deux_gauss
    # estimated_param_2gauss = land_dic.estimate_parameter(land_dic.landau_plus_deux_gauss, [1, 1, 0, 1, 0, 1])
    # print("Estimated Parameters for landau_plus_deux_gauss:", estimated_param_2gauss)
    
    # Using landau_conv_gauss
    # estimated_param_conv = land_dic.estimate_parameter(land_dic.landau_conv_gauss, [1, 1, 0, 1])
    # print("Estimated Parameters for landau_conv_gauss:", estimated_param_conv)
    
    # land_dic2= Likelihood_landau(x)
    # land_dic2.landau(loc_th,scale_th)
    # A=land_dic2.landau_plus_gauss(0,1,3,1)
    # # result_conv_two_diracs = land_dic.landau_conv_two_diracs(0, 1, 0, 1, -7, 7)

    # plt.plot(x,A)
    # plt.show()


    # Exemple de données
    x = np.random.randn(10000)
    y = np.random.randn(10000)

    # Création de l'histogramme 2D
    plt.hist2d(x, y, bins=100, cmap='viridis')

    # Ajouter une courbe (par exemple une courbe gaussienne)
    x_curve = np.linspace(-4, 4, 100)
    y_curve = np.exp(-x_curve**2)  # Fonction gaussienne
    plt.plot(x_curve, y_curve, color='red', label='Courbe gaussienne')

    # Ajouter une légende
    plt.legend()

    # Ajouter des labels et un titre
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Histogramme 2D et Courbe sur le même plot')

    # Ajouter une barre de couleurs
    plt.colorbar(label='Fréquence')

    # Affichage du graphique
    plt.show()


















































def density(data,num_splits):

    density_Ih = np.sort(data['Ih'])
    cumulative_density = np.cumsum(density_Ih) / np.sum(density_Ih)
    threshold_Ih = density_Ih[np.searchsorted(cumulative_density, 0.9)]  # Trouver la densité qui couvre 90%
    contour_level = np.mean(density_Ih >= threshold_Ih)

    density_ML = np.sort(data['dedx'])
    cumulative_density_2 = np.cumsum(density_ML) / np.sum(density_ML)
    threshold_ML = density_ML[np.searchsorted(cumulative_density_2, 0.9)]  # Trouver la densité qui couvre 90%
    contour_level_2 = np.mean(density_ML >= threshold_ML)

    split_size = len(data) // num_splits
    data=data.sort_values(by='track_p')
    sub_data = [data.iloc[i * split_size:(i + 1) * split_size] for i in range(num_splits)] # split in n sample
    std_pred = [sub_df['dedx'].std() for sub_df in sub_data]
    mpv_pred = [stats.mode(sub_df['dedx'], keepdims=True)[0][0] for sub_df in sub_data]

    std_Ih=[sub_df['Ih'].std() for sub_df in sub_data] # Calculate standard deviation for each sample
    mpv_Ih = [stats.mode(sub_df['Ih'], keepdims=True)[0][0] for sub_df in sub_data]


    mean_p= [sub_df['track_p'].mean() for sub_df in sub_data]

    # std_data = pd.DataFrame()
    # std_data['std']=std_pred
    # std_data['std_Ih']=std_Ih
    # std_data['track_p']=mean_p
  

    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.hist2d(data['track_p'],data['Ih'],bins=500,  cmap='viridis', label='Data')
    sns.kdeplot(x=data['track_p'], y=data['Ih'], levels=[contour_level], colors="red", linewidths=2)
    plt.errorbar(mean_p, mpv_Ih, yerr=std_Ih,  label='standard déviation', fmt='o', capsize=3, color='b')
    plt.xlabel('p in GeV/c')
    plt.ylabel(r'$-(\frac{dE}{dx}$)')
    plt.title('Beth-Bloch recontruction with Ih formula')
    plt.legend()



    plt.subplot(1,2,2)
    plt.hist2d(data['track_p'],data['dedx'],bins=500,  cmap='viridis', label='Data')
    sns.kdeplot(x=data['track_p'], y=data['dedx'], levels=[contour_level_2], colors="red", linewidths=2)
    plt.errorbar(mean_p, mpv_pred, yerr=std_pred,  label='standard déviation', fmt='o', capsize=3, color='b')
    plt.xlabel('p in GeV/c')
    plt.ylabel(r'$-(\frac{dE}{dx}$)')
    plt.title('Beth-Bloch recontruction with Machine Learning')
    plt.legend()

    plt.show()
