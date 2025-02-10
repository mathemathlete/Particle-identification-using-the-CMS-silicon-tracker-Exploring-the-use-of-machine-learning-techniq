import ML_plot as ML
import subprocess 

# execute Machine learning train
ML_train=False 
# execute plot ML_output and choose the parameter
plot_1=False # 1D plot the prediction of ML 

hist= True # Histogram 1D of Ecart between theory and prediction
hist_2=True # Histogram 2D of Ecart between theory and prediction and 
path_ML="ML_out.root"
# execute plot of the difference between Ih formula and ML prediction
Test_ML=True

path_in="ML_in.root"
path_out="ML_out.root"
Hist=True # Histogram 2D of de/dx with ML, with Ih and the difference between the two

if ML_train==True:
    subprocess.run(["python", "ML_functions_first.py"])

if plot_1==True:
    branch_of_interest = ["dedx","track_p"]
    ML.plot_ML(path_ML, branch_of_interest, True ,hist,hist_2)

if Test_ML==True:
    ML.plot_diff_Ih(path_out,path_in, True,Hist)