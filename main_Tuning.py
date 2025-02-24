import subprocess 

##################################### Part where can eventually run the tuning ############################################
# We run the tuning as a subprocess because of some casualties that were encountered during the adaptation of the code with the main
# An upgrade would be to integrate the tuning in the main code
file_tuning = "Tuning/Tuning_LSTM_GRU_V1.py" # modify this variable to change the tuning file among the one present in directory
subprocess.run(["python", file_tuning])