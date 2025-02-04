import uproot
import pandas as pd
import numpy as np
import awkward as ak

def creation_fichier_root(file_name, branch_of_interest, isstrip=False, insideTkmod=False, clusterclean=False):
    """
    Crée un fichier root en extrayant les branches d'intérêt et en appliquant des filtres optionnels.

    Conditions :
    - Le premier élément de `branch_of_interest` doit être un `dedx`.
    - Si `isstrip`, `insideTkmod` ou `clusterclean` sont activés, les colonnes correspondantes doivent être présentes dans `branch_of_interest`.
    - L'ordre des colonnes doit respecter l'organisation attendue (Instlumi, npx/I, dedx_charge, ...).

    Paramètres :
    - file_name : Nom du fichier ROOT.
    - branch_of_interest : Liste des branches à extraire.
    - isstrip, insideTkmod, clusterclean : booléens activant les filtres.
    
    Retourne :
    - Un DataFrame Pandas contenant les données filtrées.
    """

    with uproot.open(file_name) as file:
        key = file.keys()[0]  # Ouvrir le premier TTree
        tree = file[key]

        # Chargement des données sous forme de DataFrame Pandas
        data = tree.arrays(branch_of_interest, library="pd", entrystart=0, entrystop=10)

    # Déterminer les indices des colonnes utilisées pour le filtrage
    filter_columns = {
        "dedx_isstrip": isstrip,
        "dedx_insideTkmod": insideTkmod,
        "dedx_clusclean": clusterclean
    }
    
    active_filters = [col for col, flag in filter_columns.items() if flag and col in branch_of_interest]

    if active_filters:
        # Création du masque de filtrage (indices des lignes à supprimer)
        mask = np.zeros(len(data), dtype=bool)
        
        for col in active_filters:
            mask |= data[col]  # Appliquer un filtre booléen

        # Appliquer le masque pour supprimer les lignes qui remplissent une condition de filtrage
        data = data[~mask]

    # Sélectionner les colonnes finales
    final_df = data[["dedx_charge", "dedx_pathlength"]].reset_index(drop=True)

    return final_df


branch_of_interest = ["dedx_charge", "dedx_pathlength", "track_p"]
# creation_fichier_root(True,branch_of_interest,False,False,False,False)

# Open the ROOT file
file_name = "tree.root"
data = pd.DataFrame()
with uproot.open(file_name) as file:
    key = file.keys()[0]  # open the first Ttree
    tree = file[key]
    data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy 