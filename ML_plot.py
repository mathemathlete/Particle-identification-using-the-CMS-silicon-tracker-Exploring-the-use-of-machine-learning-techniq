import matplotlib.pyplot as plt
import numpy as np
import Creation_plus_filtred as cpf
import Identification as id
import awkward as ak
import pandas as pd
from scipy.spatial.distance import mahalanobis, cdist
from scipy import stats





def plot_ML(data,ylim, hist,hist_2, dev):
    """
    Plots various comparisons between theoretical Bethe-Bloch predictions 
    and experimental/Machine Learning (ML) reconstructed values.

    Args:
    - data (dict or DataFrame): Contains track momentum (`track_p`), 
      ML predicted `dedx`, and Ih computed values.
    - ylim (tuple): Y-axis limits for plots.
    - hist (bool, optional): If True, plots histograms of predictions and theoretical momenta.
    - hist_2 (bool, optional): If True, plots deviations between experimental/ML data and theory.
    - dev (bool, optional): If True, plots 2D histograms of deviations.

    """
    np_th= np.array(id.bethe_bloch(938e-3,data['track_p']))
    np_pr = np.array(data['dedx'])
    np_Ih=np.array(data['Ih'])

    plt.figure()
    plt.subplot(1,2,1)
    plt.hist2d(data["track_p"], data["Ih"], bins=500, cmap='viridis', label='Data')
    plt.colorbar(label='Counts')
    plt.xlabel(r'p')
    plt.ylabel(r'$-(\frac{dE}{dx}$)')
    plt.ylim(ylim)
    plt.title('Beth-Bloch reconstruction with Ih formula')
    plt.grid(True)
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist2d(data["track_p"], data["dedx"], bins=500, cmap='viridis', label='Data')
    plt.colorbar(label='Counts')
    plt.xlabel(r'p')
    plt.ylabel(r'$-(\frac{dE}{dx}$)')
    plt.ylim(ylim)
    plt.title('Beth-Bloch reconstruction with Machine Learning')
    plt.grid(True)
    plt.legend()
    plt.show() 


    if hist==True:
       
        plt.figure(figsize=(12, 6))

        # Histogram of Prediction
        plt.subplot(1, 2, 1)
        plt.hist(np_pr, bins=50, alpha=0.7, label='Prediction')
        plt.xlabel('momentum')
        plt.ylabel('N')
        plt.title('Histogram of Prediction')
        plt.legend()

        # Histogram of Theoretical Momentums
        plt.subplot(1, 2, 2)
        plt.hist(np_th, bins=50, alpha=0.7, label='Theoretical Momentums')
        plt.xlabel('Momentums')
        plt.ylabel('N')
        plt.title('Histogram of Theoretical Momentums')
        plt.legend()

        plt.tight_layout()

    if hist_2==True:   
        std_dev_pr = np.std(np_pr - np_th)
        std_dev_Ih=np.std(np_Ih - np_th)
          
          
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.hist(np_pr - np_th, bins=200,range=[-7.5,7.5], color='blue', alpha=0.7, label='1D Histogram')
        plt.axvline(std_dev_pr, color='red', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_dev_pr:.2f}')
        plt.axvline(-std_dev_pr, color='red', linestyle='dashed', linewidth=1)
        plt.xlabel('exp-th')
        plt.ylabel('Counts')
        plt.ylim([0,1100])
        plt.title('1D Histogram of Ecart between theory and prediction')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.hist(np_Ih - np_th, bins=200, range=[-7.5,7.5], color='blue', alpha=0.7, label='1D Histogram')
        plt.axvline(std_dev_Ih, color='red', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_dev_Ih:.2f}')
        plt.axvline(-std_dev_Ih, color='red', linestyle='dashed', linewidth=1)
        plt.xlabel('Ih-th')
        plt.ylabel('Counts')
        plt.ylim([0,1100])
        plt.title('1D Histogram of Ecart between Ih formula and theorie')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.hist(np_Ih - np_th, bins=200, range=[-7.5,7.5], color='blue', alpha=0.7, label='1D Histogram')
        plt.axvline(std_dev_Ih, color='blue', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_dev_Ih:.2f}')
        plt.axvline(-std_dev_Ih, color='blue', linestyle='dashed', linewidth=1)
        plt.hist(np_pr - np_th, bins=200, range=[-7.5,7.5], color='red', alpha=0.7, label='1D Histogram')
        plt.axvline(std_dev_pr, color='red', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_dev_pr:.2f}')
        plt.axvline(-std_dev_pr, color='red', linestyle='dashed', linewidth=1)
        plt.xlabel('deviation')
        plt.ylabel('Counts')
        plt.ylim([0,1100])
        plt.title('1D Histogram of deviation with theorie')
        plt.legend()
        plt.show()

    if dev==True:
    # --- Comparison between theoretical and experimental values
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


def plot_ratio(data,m_part,y_lim):
    """
    Plots histograms of the ratio between computed energy loss (dE/dx) 
    and theoretical Bethe-Bloch predictions for both Ih formula and ML predictions.

    Args:
    - data (DataFrame): Contains 'track_p', 'Ih', and 'dedx' values.
    - m_part (float): Mass of the particle for Bethe-Bloch calculation.
    - y_lim (tuple): Y-axis limits for histograms.

    """
    data['ratio_Ih']=data['Ih']/id.bethe_bloch(m_part,np.array(data['track_p']))
    data['ratio_pred']=data['dedx']/id.bethe_bloch(m_part,np.array(data['track_p']))
   

    plt.figure(1,figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.hist(data['ratio_Ih'], bins=100, alpha=0.7, label='Ih/th')
    plt.xlabel('ratio between Ih and theorie')    
    plt.ylabel('N')
    plt.ylim(y_lim)
    plt.title('Histogram of ratio between Ih and theorie')
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(data['ratio_pred'], bins=100, alpha=0.7, label='pred/th')
    plt.xlabel('ratio between prediction and theorie')
    plt.ylabel('N')
    plt.ylim(y_lim)
    plt.title('Histogram of ratio between prediction and theorie')
    plt.legend()
    plt.show()

def density(data,num_splits,ylim):
    """
    Analyzes the energy loss per unit length (dE/dx) distribution 
    by splitting data into bins of momentum and computing mean & standard deviation.

    Args:
    - data (DataFrame): Input dataset containing 'track_p', 'Ih', and 'dedx'.
    - num_splits (int): Number of momentum bins.
    - ylim (tuple): Y-axis limits for visualization.

    Returns:
    - Displays histograms with error bars for both Ih formula and ML predictions.
    """
 
    split_size = len(data) // num_splits
    data=data.sort_values(by='track_p')
    sub_data = [data.iloc[i * split_size:(i + 1) * split_size] for i in range(num_splits)] # split in n sample
    std_pred = [sub_df['dedx'].std() for sub_df in sub_data]
    mean_pred = [sub_df['dedx'].mean() for sub_df in sub_data]

    std_Ih=[sub_df['Ih'].std() for sub_df in sub_data] # Calculate standard deviation for each sample
    mean_Ih = [sub_df['Ih'].mean() for sub_df in sub_data]  # Calculate mean value of Ih for each sample
    print(mean_Ih)

    mean_p= [sub_df['track_p'].mean() for sub_df in sub_data] # Calculate mean value of p  for each sample

    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.hist2d(data['track_p'],data['Ih'],bins=500,  cmap='viridis', label='Data')
    plt.errorbar(mean_p, mean_Ih, yerr=std_Ih,  label='standard déviation', fmt='o', capsize=3, color='r')
    plt.xlabel('p in GeV/c')
    plt.ylabel(r'$-(\frac{dE}{dx}$)')
    plt.ylim(ylim)

    plt.title('Beth-Bloch recontruction with Ih formula')
    plt.legend()



    plt.subplot(1,2,2)
    plt.hist2d(data['track_p'],data['dedx'],bins=500,  cmap='viridis', label='Data')
    plt.errorbar(mean_p, mean_pred, yerr=std_pred,  label='standard déviation', fmt='o', capsize=3, color='r')
    plt.xlabel('p in GeV/c')
    plt.ylabel(r'$-(\frac{dE}{dx}$)')
    plt.ylim(ylim)
    plt.title('Beth-Bloch recontruction with Machine Learning')
    plt.legend()

    plt.show()



def dist_Mahalanobis (path,branch_of_interest):   
    """
    Detects outliers in a dataset using the Mahalanobis distance.

    Parameters:
    - path (str): Path to the dataset.
    - branch_of_interest (list): List of features to extract.

    Returns:
    - Displays a 2D histogram with detected outliers

    Never Used
    """
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
    plt.title("Outlier Detection using Mahalanobis Distance (2D Histogram)")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.legend()
    plt.show()



def dispertion_indication(path,branch_of_interest):
    """
    Computes the dispersion of outliers in a dataset based on the Mahalanobis distance.

    Parameters:
    - path (str): Path to the dataset.
    - branch_of_interest (list): List of features to extract.

    Returns:
    - mean_distance (float): Mean Euclidean distance of detected outliers from the data mean.
    """
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

    return mean_distance


def std(data,num_splits,plot):

    """
    Compute standard deviations and their errors for predicted and Ih values across momentum bins.

    Parameters:
    - data (DataFrame): Input data containing 'track_p', 'dedx', and 'Ih'.
    - num_splits (int): Number of bins to split the data.
    - plot (bool): If True, plot the standard deviation as a function of track momentum.

    Returns:
    - std_data (DataFrame): DataFrame containing standard deviations, means, and errors.
    """
    split_size = len(data) // num_splits
    data=data.sort_values(by='track_p')
    sub_data = [data.iloc[i * split_size:(i + 1) * split_size] for i in range(num_splits)] # split in n sample
    std_pred = [sub_df['dedx'].std() for sub_df in sub_data] # Calculate standard deviation for each sample
    std_Ih=[sub_df['Ih'].std() for sub_df in sub_data] # Calculate standard deviation for each sample
    mean_pred = [sub_df['dedx'].mean() for sub_df in sub_data] # Calculate standard deviation for each sample
    mean_Ih=[sub_df['Ih'].mean() for sub_df in sub_data] # Calculate standard deviation for each sample

    mean_p= [sub_df['track_p'].mean() for sub_df in sub_data]
    error = [std / np.sqrt(2 * (split_size - 1)) for std in std_pred]  # calculate the error of the std
    error_Ih = [std / np.sqrt(2 * (split_size - 1)) for std in std_Ih] 
    std_data = pd.DataFrame()
    std_data['std_pred']=std_pred
    std_data['std_Ih']=std_Ih
    std_data['track_p']=mean_p
    std_data['error']=error
    std_data['error_Ih']=error_Ih
    std_data['mean_pred']=mean_pred
    std_data['mean_Ih']=mean_Ih
    

    if plot==True:
        plt.subplot(1, 2, 1)
        plt.scatter(std_data['track_p'],std_data['std_pred'])
        plt.errorbar(std_data['track_p'], std_data['std_pred'], yerr=std_data['error'], label='standard déviation', fmt='o', capsize=3, color='b')
        plt.ylabel("standard deviation of the predicted values")
        plt.xlabel("P in GeV/c")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(std_data['track_p'] ,std_data['std_Ih'])
        plt.errorbar(std_data['track_p'], std_data['std_Ih'], yerr=std_data['error_Ih'], label='standard déviation', fmt='o', capsize=3, color='b')
        plt.ylabel("standard deviation of Ih")
        plt.xlabel("P in GeV/c")
        plt.legend()
        
        plt.show()

    return std_data



def loss_epoch(losses_epoch):
    plt.figure()
    epoch_count = [i+1 for i in range(len(losses_epoch))]
    plt.plot(epoch_count, losses_epoch)
    plt.title("Loss function Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def biais(data,biais,num_splits):
    """
    Analyze the impact of a bias variable on standard deviation and mean values.

    Parameters:
    - data (DataFrame): Input data containing 'track_p', 'dedx', and 'Ih'.
    - biais (str): Column name representing the bias variable.
    - num_splits (int): Number of bins to split the data.

    Returns:
    - std_data (DataFrame): DataFrame containing computed statistics.
    """
    split_size = len(data) // num_splits
    data=data.sort_values(by=biais)
    

    sub_data = np.array_split(data, num_splits)

    std_pred = [sub_df['dedx'].std() for sub_df in sub_data] # Calculate standard deviation for each sample
    std_Ih=[sub_df['Ih'].std() for sub_df in sub_data] # Calculate standard deviation for each sample
    mean_pred = [sub_df['dedx'].mean() for sub_df in sub_data] # Calculate standard deviation for each sample
    mean_Ih=[sub_df['Ih'].mean() for sub_df in sub_data] # Calculate standard deviation for each sample
    std_data=pd.DataFrame()
    std_data['std_pred']=std_pred
    std_data['mean_pred']=mean_pred
    std_data["sigma_mu_pred"]= std_data['std_pred']/ std_data['mean_pred']
    std_data['std_Ih']=std_Ih
    std_data['mean_Ih']=mean_Ih
    std_data["sigma_mu_Ih"]=std_data['std_Ih']/std_data['mean_Ih']
    mean_biais= [sub_df[biais].mean() for sub_df in sub_data]
    std_data[biais]=mean_biais

    error_std_pred = [std / np.sqrt(2 * (split_size - 1)) for std in std_pred]  # calculate the error of the std
    error_std_Ih =  [std / np.sqrt(2 * (split_size - 1)) for std in std_Ih] 
    error_mean_pred=[std / np.sqrt(split_size) for std in std_pred]
    error_mean_Ih=[std / np.sqrt(split_size) for std in std_Ih]
   
    error_pred= [np.sqrt((error_std_pred[i]/std_pred[i])**2+(error_mean_pred[i]/mean_pred[i])**2) for i in range(len(std_Ih))] 
    error_Ih= [np.sqrt((error_std_Ih[i]/std_Ih[i])**2+(error_mean_Ih[i]/mean_Ih[i])**2) for i in range(len(std_Ih))] 

    std_data['error_pred']=error_pred
    std_data['error_Ih']=error_Ih

    plt.subplot(1, 2, 1)
    plt.scatter( std_data[biais] , std_data["sigma_mu_pred"])
    plt.errorbar( std_data[biais] , std_data["sigma_mu_pred"], yerr= std_data['error_pred'], label='standard déviation', fmt='o', capsize=3, color='b')
    plt.ylabel(r'$\frac{\sigma}{\mu}$')
    plt.title("impact of the bias on the prediction")
    plt.xlabel(biais)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter( std_data[biais] ,std_data["sigma_mu_Ih"])
    plt.errorbar( std_data[biais], std_data["sigma_mu_Ih"], yerr=std_data['error_Ih'], label='standard déviation', fmt='o', capsize=3, color='b')
    plt.ylabel(r"$\frac{\sigma}{\mu}$")
    plt.title("impact of the bias on the Ih formula")
    plt.xlabel(biais)
    plt.legend()
    
    plt.show()

def correlation(data,branch_1,branch_2):
    """
    Compute and visualize the correlation between two variables.

    Parameters:
    - data (DataFrame): Input dataset containing numerical columns.
    - branch_1 (str): Name of the first column.
    - branch_2 (str): Name of the second column.

    Returns:
    - coef_correlation (float): Pearson correlation coefficient.
    """
    covariance = np.cov(data[branch_1], data[branch_2], bias=True)  # `bias=True` pour la version normale
    print(covariance)
    coef_correlation = np.corrcoef(data[branch_1], data[branch_2])
    print(coef_correlation)

if __name__ == "__main__":
    # parameter for ML_plot
    branch_of_interest = ["dedx","track_p","track_eta"]
    path_ML='ML_out.root'
    #plot_ML(path_ML, branch_of_interest, True, True, True)

    # parameter for plot_diff_Ih
    #plot_diff_Ih(path_test,path_Ih,True,True)
    branch_of_interest_1 = ['track_p','Ih','track_eta']
    data=cpf.import_data("Root_files/data_real_kaon.root",branch_of_interest_1)
    # data['Ih']=data['Ih']*1e-3
    # data["dedx"]=data['Ih']*1.25
    correlation(data,'Ih','track_eta')