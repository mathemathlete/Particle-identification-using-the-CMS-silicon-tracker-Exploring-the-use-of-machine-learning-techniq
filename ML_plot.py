import uproot
import matplotlib.pyplot as plt
import numpy as np
import Creation_plus_filtrage as cpf


def plot(data, hist, dev):
    
    
    np_th= np.array(targets)
    np_pr = np.array(predictions)

    if hist==True
       
        plt.figure(figsize=(12, 6))

        # Histogramme des prédictions
        plt.subplot(1, 2, 1)
        plt.hist(predictions, bins=50, alpha=0.7, label='Prédictions')
        plt.xlabel('Valeur')
        plt.ylabel('N')
        plt.title('Histogramme des Prédictions')
        plt.legend()

        # Histogramme des valeurs théoriques
        plt.subplot(1, 2, 2)
        plt.hist(data_th_values_test, bins=50, alpha=0.7, label='Valeurs Théoriques')
        plt.xlabel('Valeur')
        plt.ylabel('N')
        plt.title('Histogramme des Valeurs Théoriques')
        plt.legend()

        plt.tight_layout()

   
    if dev==True
    # --- Comparaison des prédictions et des valeurs théoriques ---
        plt.figure(figsize=(8, 8))
        plt.hist2d(p_values, np_pr-np_th, bins=500, cmap='viridis', label='Data')
        plt.xlabel('Valeur')
        plt.ylabel('th-exp')
        plt.title('Ecart entre théorique et prédite')
        plt.legend()

        p_axis = np.logspace(np.log10(0.0001), np.log10(2), 500)
        plt.figure(figsize=(8, 8))
        plt.hist2d(p_values,np_pr,bins=500, cmap='viridis', label='Data')
        plt.plot(p_axis,id.bethe_bloch(938e-3,np.array(p_axis)),color='red')
        plt.xscale('log')
        plt.show()


