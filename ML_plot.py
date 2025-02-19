import matplotlib.pyplot as plt
import numpy as np
import Creation_plus_filtrage as cpf
import Identification as id
import awkward as ak
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import mahalanobis, cdist
from scipy import stats


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


def plot_ML_inside(data, hist,hist_2, dev):
    
    np_th= np.array(id.bethe_bloch(938e-3,data['track_p']))
    np_pr = np.array(data['dedx'])
    np_Ih=np.array(data['Ih'])*1e-3

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



def plot_ratio(data):

    data['Ih']=data['Ih']*1e-3
    data['ratio_Ih']=data['Ih']/id.bethe_bloch(938e-3,np.array(data))
    mpv_Ih = stats.mode(data['ratio_Ih'], keepdims=True)[0][0]
    std_1=data['ratio'].std()

    data['ratio_pred']=data['dedx']/id.bethe_bloch(938e-3,np.array(data))
    mpv_pred = stats.mode(data['ratio_pred'], keepdims=True)[0][0]
    std_2=data['ratio_pred'].std()

    plt.figure(1,figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.hist(data['ratio_Ih'], bins=100, alpha=0.7, label='Ih/th')
    plt.axvline(mpv_Ih+std_1, color='blue', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_1:.2f}\n mean={mpv_Ih:.2f}')
    plt.axvline(mpv_Ih-std_1, color='blue', linestyle='dashed', linewidth=1)
    plt.xlabel('ratio between Ih and theorie')    
    plt.ylabel('N')
    plt.ylim([0,850])
    plt.title('Histogram of ratio between prediction and theorie')
    plt.legend()

    plt.subplot(1,2,2)
    plt.hist(data['ratio_pred'], bins=100, alpha=0.7, label='Ih/th')
    plt.axvline(mpv_pred+std_2, color='blue', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_2:.2f}\n mean={mpv_pred:.2f}')
    plt.axvline(mpv_pred-std_2, color='blue', linestyle='dashed', linewidth=1)
    plt.xlabel('ratio between Ih and dedx')
    plt.ylabel('N')
    plt.ylim([0,850])
    plt.title('Histogram of ratio between Ih and theorie')
    plt.legend()
    plt.tight_layout()

def density(data,num_splits):
    
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

    return mean_distance


def std(data,num_splits,plot):
    # data=cpf.import_data(path, branch_of_interest)
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

    split_size = len(data) // num_splits
    data=data.sort_values(by=biais)
    
    # sub_data = [data.iloc[i * split_size:(i + 1) * split_size] for i in range(num_splits - 1)] # split in n sample
    # sub_data.append(data.iloc[(num_splits - 1) * split_size:])
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



if __name__ == "__main__":
    # parameter for ML_plot
    branch_of_interest = ["dedx","track_p","track_eta"]
    path_ML='ML_out.root'
    #plot_ML(path_ML, branch_of_interest, True, True, True)

    # parameter for plot_diff_Ih
    #plot_diff_Ih(path_test,path_Ih,True,True)
    branch_of_interest_1 = ['track_p','Ih','track_eta']
    data=cpf.import_data("Root_files/data_real_kaon.root",branch_of_interest_1)
    print(data)
    data['Ih']=data['Ih']*1e-3
    data["dedx"]=data['Ih']*1.25
    density(data,10)