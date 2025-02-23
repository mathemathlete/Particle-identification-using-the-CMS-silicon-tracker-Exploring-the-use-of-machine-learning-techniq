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
    Creates a ROOT file by extracting selected branches and applying optional filters.

    Conditions:
    - The first element of `branch_of_interest` must be `dedx`.
    - If `isstrip`, `insideTkmod`, or `clusterclean` are enabled, their corresponding columns must be included in `branch_of_interest`.
    - The column order must follow the expected structure:  
    (Instlumi, npx/I, dedx_charge, dedx_pathlength, dedx_isstrip,  
    dedx_insideTkmod, dedx_clusclean).

    Parameters:
    - file_name : str  
    Name of the ROOT file.  
    - branch_of_interest : list  
    List of branches to extract.  
    - isstrip, insideTkmod, clusterclean : bool  
    Boolean flags to enable specific filtering conditions.  

    Returns:
    - pd.DataFrame  
    A Pandas DataFrame containing only `dedx_charge` and `dedx_pathlength` columns,  
    with empty rows removed.

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
    # Determoine the active filters

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