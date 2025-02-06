import matplotlib.pyplot as plt
import numpy as np
import Creation_plus_filtrage as cpf
import Identification as id


def plot(data, hist, dev):
    
    data=cpf.import_data(data, branch_of_interest)
    np_th= np.array(id.bethe_bloch(938e-3,data['track_p']))
    np_pr = np.array(data['dedx'])

    if hist==True:
       
        plt.figure(figsize=(12, 6))

        # Histogramme des prédictions
        plt.subplot(1, 2, 1)
        plt.hist(np_pr, bins=50, alpha=0.7, label='Prédictions')
        plt.xlabel('Valeur')
        plt.ylabel('N')
        plt.title('Histogramme des Prédictions')
        plt.legend()

        # Histogramme des valeurs théoriques
        plt.subplot(1, 2, 2)
        plt.hist(np_th, bins=50, alpha=0.7, label='Valeurs Théoriques')
        plt.xlabel('Valeur')
        plt.ylabel('N')
        plt.title('Histogramme des Valeurs Théoriques')
        plt.legend()

        plt.tight_layout()



    if dev==True:
    # --- Comparaison des prédictions et des valeurs théoriques ---
        plt.figure(figsize=(8, 8))
        plt.hist2d(data['track_p'], np_pr-np_th, bins=500, cmap='viridis', label='Data')
        plt.xlabel('Valeur')
        plt.ylabel('th-exp')
        plt.title('Ecart entre théorique et prédite')
        plt.legend()

        p_axis = np.logspace(np.log10(0.0001), np.log10(2), 500)
        plt.figure(figsize=(8, 8))
        plt.hist2d(data['track_p'],np_pr,bins=500, cmap='viridis', label='Data')
        plt.plot(p_axis,id.bethe_bloch(938e-3,np.array(p_axis)),color='red')
        plt.xscale('log')
        plt.show()


branch_of_interest = ["dedx","track_p"]
plot("ML_out.root",True,True)