import uproot
import pandas as pd
import awkward as ak

def filtrage_dedx(file_name, isstrip=False, insideTkmod=False, dedx_clusclean=False):
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
    branch_of_interest = ["dedx_charge", "dedx_pathlength","track_p"]
    active_filters = []
    if isstrip:
        active_filters.append("dedx_isstrip")
    if insideTkmod:
        active_filters.append("dedx_insideTkmod")
    if dedx_clusclean:
        active_filters.append("dedx_clusclean")
    branch_of_interest.extend(active_filters)

    with uproot.open(file_name) as file:
        key = file.keys()[0] 
        tree = file[key]

        # Chargement des données sous forme de DataFrame Pandas
        data = tree.arrays(branch_of_interest, library="pd")

    # Déterminer les indices des colonnes utilisées pour le filtrage

    if active_filters:
        # Convert the DataFrame columns to Awkward Arrays for filtering
        data_charge = ak.Array(data["dedx_charge"].tolist())
        data_pathlength = ak.Array(data["dedx_pathlength"].tolist())
        masks = [ak.Array(data[col].tolist()) for col in active_filters]

        # Combine masks using logical AND
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = combined_mask & mask

        # Apply the combined mask to filter the data
        filtered_charge = data_charge[combined_mask]
        filtered_pathlength = data_pathlength[combined_mask]

        # Update the DataFrame with the filtered data
        data["dedx_charge"] = filtered_charge.tolist()
        data["dedx_pathlength"] = filtered_pathlength.tolist()

        # Remove rows where either `dedx_charge` or `dedx_pathlength` is an empty list
        data = data[data["dedx_charge"].apply(len) > 0]
        data = data[data["dedx_pathlength"].apply(len) > 0]

    # Keep only the `dedx_charge` and `dedx_pathlength` columns
    data = data[["dedx_charge", "dedx_pathlength","track_p"]]

    return data

if __name__ == "__main__":
    ## Test to debug (reminder : reduce the amount of data with the empty_stop parameter line )
    # Example usage (sans filtre)
    A = filtrage_dedx("tree.root")
    print(A)

    # Example usage 2 (avec filtre)
    A = filtrage_dedx("tree.root",True,False,True)
    print(A)