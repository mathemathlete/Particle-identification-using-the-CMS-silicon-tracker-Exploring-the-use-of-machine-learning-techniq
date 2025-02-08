import numpy as np
import Creation_plus_filtrage as cpf
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt

branch_of_interest_1 = ['dedx_cluster', 'track_p']
branch_of_interest_2 = ['dedx','track_p']
path_Ih="Root_files/ML_training_1.2.root"
path_test='ML_out.root'


data_Ih=cpf.import_data(path_Ih, branch_of_interest_1)
data_test=cpf.import_data(path_test, branch_of_interest_2)
data_Ih['Ih'] = np.sqrt(ak.sum(data_Ih['dedx_cluster']**2, axis=-1) / ak.count(data_Ih['dedx_cluster'], axis=-1)) #calculate quadratique mean of dedx along a track

def plot(data_test,data_Ih, hist, hist_2):
    if hist==True:
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        plt.hist(data_Ih['Ih']-data_test['dedx'], bins=100, alpha=0.7, label='Ih-dedx')
        plt.xlabel('difference between Ih and dedx')
        plt.ylabel('N')
        plt.title('Histogramme des différences entre Ih et dedx')
        plt.legend()
        plt.tight_layout()
        plt.subplot(1,2,2)
        plt.hist(data_Ih['Ih']/data_test['dedx'], bins=100, alpha=0.7, label='Ih/dedx')
        plt.xlabel('ratio between Ih and dedx')
        plt.ylabel('N')
        plt.title('Histogramme des ratio entre Ih et dedx')
        plt.legend()
        plt.show()

    if hist_2==True:
        plt.figure(figsize=(12, 6))
        plt.subplot(2,2,1)
        plt.hist2d(data_Ih['track_p'],data_Ih['Ih'],bins=500,  cmap='viridis', label='Data')
        plt.subplot(2,2,2)
        plt.hist2d(data_test['track_p'],data_test['dedx'],bins=500,  cmap='viridis', label='Data')
        plt.subplot(2,2,3)
        plt.hist2d(data_test['track_p'],data_Ih['Ih']-data_test['dedx'],bins=500,  cmap='viridis', label='Data')
        plt.show()


print(len(data_test['dedx']),len(data_Ih['Ih']),len(data_test['track_p']))
plot(data_test,data_Ih,False,False)