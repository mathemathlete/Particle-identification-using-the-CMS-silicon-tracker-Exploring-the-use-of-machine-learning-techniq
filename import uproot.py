import uproot
with uproot.open("slim_nt_mc_aod_1.root") as file:
    A = file.values() 
    B = file.keys()
    print(A)
    print(B)
    for key in file.keys():
        tree = file[key]
        data = tree.arrays(library="np")
        first_key = file.keys()[0]
        first_tree = file[first_key]
        first_data = first_tree.arrays(library="np")
        for array_name, array in first_data.items():
            print(f"Array {array_name}:")
            print(array)