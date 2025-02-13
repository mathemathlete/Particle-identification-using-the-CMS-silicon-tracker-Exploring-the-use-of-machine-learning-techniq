import matplotlib.pyplot as plt
import numpy as np
import Creation_plus_filtrage as cpf
import Identification as id
import awkward as ak
import seaborn as sns
from scipy.spatial.distance import mahalanobis, cdist

def plot_ML(path_ML,branch_of_interest, hist,hist_2, dev):
    
    data=cpf.import_data(path_ML, branch_of_interest)
    np_th= np.array(id.bethe_bloch(938e-3,data['track_p']))
    np_pr = np.array(data['dedx'])

    if hist==True:
       
        plt.figure(figsize=(12, 6))

        # Histogramme des prédictions
        plt.subplot(1, 2, 1)
        plt.hist(np_pr, bins=50, alpha=0.7, label='Prédictions')
        plt.xlabel('momentum')
        plt.ylabel('N')
        plt.title('Histogramme des Prédictions')
        plt.legend()

        # Histogramme des momentums théoriques
        plt.subplot(1, 2, 2)
        plt.hist(np_th, bins=50, alpha=0.7, label='momentums Théoriques')
        plt.xlabel('momentum')
        plt.ylabel('N')
        plt.title('Histogramme des momentums Théoriques')
        plt.legend()

        plt.tight_layout()

    if hist_2==True:   

            std_dev = np.std(np_pr - np_th)
            plt.figure(figsize=(8, 8))
            plt.hist(np_pr - np_th, bins=200,range=[-7.5,7.5], color='blue', alpha=0.7, label='1D Histogram')
            plt.axvline(std_dev, color='red', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_dev:.2f}')
            plt.axvline(-std_dev, color='red', linestyle='dashed', linewidth=1)
            plt.xlabel('th-exp')
            plt.ylabel('Counts')
            plt.title('1D Histogram of Ecart between theory and prediction')
            plt.legend()
            plt.show()

    if dev==True:
    # --- Comparaison des prédictions et des momentums théoriques ---
        plt.figure(figsize=(8, 8))
        plt.subplot(1,2,1)
        plt.hist2d(data['track_p'], np_pr-np_th, bins=500, cmap='viridis', label='Data')
        plt.xlabel('momentum')
        plt.ylabel('th-exp')
        plt.title('Ecart entre théorique et prédite')
        plt.legend()

        p_axis = np.logspace(np.log10(0.0001), np.log10(2), 500)
        plt.subplot(1,2,2)
        plt.hist2d(data['track_p'],np_pr,bins=500, cmap='viridis', label='Data')
        plt.plot(p_axis,id.bethe_bloch(938e-3,np.array(p_axis)),color='red')
        plt.xscale('log')
        plt.show()


    



def plot_diff_Ih(path_test,path_Ih, hist, hist_2):

    data_Ih=cpf.import_data(path_Ih, branch_of_interest_1)
    data_test=cpf.import_data(path_test, branch_of_interest_2)
    data_Ih['Ih'] = np.sqrt(ak.sum(data_Ih['dedx_cluster']**2, axis=-1) / ak.count(data_Ih['dedx_cluster'], axis=-1) ) * 1e-3 #calculate quadratique mean of dedx along a track
    if hist==True:
        plt.figure(1,figsize=(12, 6))
        plt.subplot(1,2,1)
        plt.hist(data_Ih['Ih']-data_test['dedx'], bins=100, alpha=0.7, label='Ih-dedx')
        plt.xlabel('difference between Ih and dedx')    
        plt.ylabel('N')
        plt.title('Histogram of difference between Ih and dedx')
        plt.legend()
        plt.subplot(1,2,2)
        plt.hist(data_Ih['Ih']/data_test['dedx'], bins=100, alpha=0.7, label='Ih/dedx')
        plt.xlabel('ratio between Ih and dedx')
        plt.ylabel('N')
        plt.title('Histogram of ratio between Ih and dedx')
        plt.legend()
        plt.tight_layout()

    if hist_2==True:

        density_Ih = np.sort(data_Ih['Ih'])
        cumulative_density = np.cumsum(density_Ih) / np.sum(density_Ih)
        threshold_Ih = density_Ih[np.searchsorted(cumulative_density, 0.9)]  # Trouver la densité qui couvre 90%
        contour_level = np.mean(density_Ih >= threshold_Ih)

        density_ML = np.sort(data_test['dedx'])
        cumulative_density_2 = np.cumsum(density_ML) / np.sum(density_ML)
        threshold_ML = density_ML[np.searchsorted(cumulative_density_2, 0.9)]  # Trouver la densité qui couvre 90%
        contour_level_2 = np.mean(density_ML >= threshold_ML)

        plt.figure(2, figsize=(12, 6))
        plt.subplot(2,2,1)
        plt.hist2d(data_Ih['track_p'],data_Ih['Ih'],bins=500,  cmap='viridis', label='Data')
        sns.kdeplot(x=data_Ih['track_p'], y=data_Ih['Ih'], levels=[contour_level], colors="red", linewidths=2)
        plt.xlabel('p in GeV/c')
        plt.ylabel(r'$-(\frac{dE}{dx}$)')
        plt.title('Beth-Bloch recontruction with Ih formula')
        plt.legend()
        plt.subplot(2,2,2)
        plt.hist2d(data_test['track_p'],data_test['dedx'],bins=500,  cmap='viridis', label='Data')
        sns.kdeplot(x=data_test['track_p'], y=data_test['dedx'], levels=[contour_level_2], colors="red", linewidths=2)
        plt.xlabel('p in GeV/c')
        plt.ylabel(r'$-(\frac{dE}{dx}$)')
        plt.title('Beth-Bloch recontruction with Machine Learning')
        plt.legend()
        plt.subplot(2,2,3)
        plt.hist2d(data_test['track_p'],data_Ih['Ih']-data_test['dedx'],bins=500,  cmap='viridis', label='Data')
        plt.xlabel('p in GeV/c')
        plt.ylabel(r'$\Delta\frac{dE}{dx}$')
        plt.title('Difference between two Beth-Bloch reconstruction')
        plt.tight_layout()
        plt.show()

def dist_Mahalanobis (path,branch_of_interest):   
    data_brut=cpf.import_data(path, branch_of_interest)
    data = np.column_stack((data_brut['track_p'], data_brut['dedx']))
    mean_vec = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    # distance calculation 
    distances = np.array([mahalanobis(x, mean_vec, inv_cov_matrix) for x in data])
    threshold = np.percentile(distances, 97)
    outliers = data[distances > threshold]
    # plot data and outelier
    plt.figure(figsize=(8, 6))
    plt.hist2d(data_brut['track_p'], data_brut['dedx'], bins=500, cmap="viridis")
    plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers détectés', marker='X', edgecolors='black')
    plt.axhline(y=mean_vec[1], color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=mean_vec[0], color='k', linestyle='--', alpha=0.5)
    plt.colorbar(label='Density of point')
    plt.title("Détection des Outliers avec la Distance de Mahalanobis (Hist2D)")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.show()



def dispertion_indication(path,branch_of_interest):

    data_brut=cpf.import_data(path, branch_of_interest)
    data = np.column_stack((data_brut['track_p'], data_brut['dedx']))
    mean_vec = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    det_cov = np.linalg.det(cov_matrix)
    mean_point = np.mean(data, axis=0)
   

    distances = np.array([mahalanobis(x, mean_vec, inv_cov_matrix) for x in data])
    threshold = np.percentile(distances, 97)
    outliers = data[distances > threshold]

    distances_to_mean = cdist(outliers, mean_point.reshape(1, -1), metric='euclidean')
    mean_distance = np.mean(distances_to_mean)

    return det_cov, mean_distance


if __name__ == "__main__":
    # parameter for ML_plot
    branch_of_interest = ["dedx","track_p"]
    path_ML='ML_out.root'
    # plot_ML(path_ML, branch_of_interest, True, False, True)



    # parameter for plot_diff_Ih
    branch_of_interest_1 = ['dedx_cluster', 'track_p']
    branch_of_interest_2 = ['dedx','track_p']
    path_Ih="ML_in.root"
    path_test='ML_out.root'
    #plot_diff_Ih(path_test,path_Ih,True,True)
    
    print(dispertion_indication(path_test,branch_of_interest))