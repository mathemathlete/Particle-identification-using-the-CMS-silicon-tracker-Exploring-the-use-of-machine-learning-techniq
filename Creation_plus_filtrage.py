import uproot
import pandas as pd
import awkward as ak
import numpy as np


def import_data(file_name, branch_of_interest):
    with uproot.open(file_name) as file:
        key = file.keys()[0]  # open the first Ttree
        tree = file[key]
        data = tree.arrays(branch_of_interest, library="pd") # open data with array from numpy
    return data


def filtrage_dedx(file_name, branch_of_interest, isstrip=False, insideTkmod=False, dedx_clusclean=False):
    """
    Crée un fichier root en extrayant les branches d'intérêt et en appliquant des filtres optionnels.

    Conditions :
    - Le premier élément de `branch_of_interest` doit être un `dedx`.
    - Si `isstrip`, `insideTkmod` ou `clusterclean` sont activés, les colonnes correspondantes doivent être présentes dans `branch_of_interest`.
    - L'ordre des colonnes doit respecter l'organisation attendue (Instlumi, npx/I, dedx_charge, dedx_pathlength, dedx_isstrip, 47
    dedx_insideTkmod, dedx_clusclean    ).

    Paramètres :
    - file_name : Nom du fichier ROOT.
    - branch_of_interest : Liste des branches à extraire.
    - isstrip, insideTkmod, clusterclean : booléens activant les filtres.
    
    Retourne :
    - Un DataFrame Pandas contenant uniquement les colonnes `dedx_charge` et `dedx_pathlength`, avec les lignes vides supprimées.
    """
    active_filters = []
    if isstrip:
        active_filters.append("dedx_isstrip")
    if insideTkmod:
        active_filters.append("dedx_insideTkmod")
    if dedx_clusclean:
        active_filters.append("dedx_clusclean")
    branch_of_interest_extract = np.copy(branch_of_interest)
    branch_of_interest.extend(active_filters)

    data=import_data(file_name, branch_of_interest)
    # Déterminer les indices des colonnes utilisées pour le filtrage

    if active_filters:
        # Convert the DataFrame columns to Awkward Arrays for filtering
        data_charge = ak.Array(data["dedx_charge"].tolist())
        data_pathlength = ak.Array(data["dedx_pathlength"].tolist())
        data_geom=ak.Array(data["dedx_modulegeom"].tolist())
        masks = [ak.Array(data[col].tolist()) for col in active_filters]

        # Combine masks using logical AND
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = combined_mask & mask

        # Apply the combined mask to filter the data
        filtered_charge = data_charge[combined_mask]
        filtered_pathlength = data_pathlength[combined_mask]
        filtered_geom = data_geom[combined_mask]

        # Update the DataFrame with the filtered data
        data["dedx_charge"] = [np.asarray(x) for x in filtered_charge.tolist()]
        data["dedx_pathlength"] = [np.asarray(x) for x in filtered_pathlength.tolist()]
        data["dedx_modulegeom"] = [np.asarray(x) for x in filtered_geom.tolist()]

        # Remove rows where either `dedx_charge` or `dedx_pathlength` is an empty list
        data = data[data["dedx_charge"].apply(len) > 0]
        data = data[data["dedx_pathlength"].apply(len) > 0]
        data = data[data["dedx_modulegeom"].apply(len) > 0]

    # Keep only the branch_of_interest
    data = data[branch_of_interest_extract]
    return data


def ecriture_root(data,file_out):
    with uproot.recreate(file_out) as new_file:
        new_file["tree_name"] = { "dedx_charge": data['dedx_charge'],"dedx_pathlength" : data['dedx_pathlength'] , 'track_eta': data['track_eta'],"track_p" : data["track_p"],   }


if __name__ == "__main__":
    branch_of_interest = ["dedx_charge", "dedx_pathlength","track_p","track_eta"]

    # Example usage 2 (avec filtre)
    data = filtrage_dedx("Root_files/tree.root",branch_of_interest,True,True,True)
    ecriture_root(data,"Root_Files/a")
    print(data)

    data2 = filtrage_dedx("Root_files/tree.root",branch_of_interest,False,False,False)
    ecriture_root(data,"Root_Files/aa")
    print(data2)