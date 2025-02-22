# Particle identification using the CMS silicon tracker : Exploring the use of machine learning techniques

The goal of this project is to explore alternative methods for correcting observed dE/dx biases
by incorporating additional information, such as the path length of particles within each layer.

## Structure of the data

The simulated and real acquired data are contained in several root files that are converted in Dataframe with uproot , that have the following structure :
    - *nstrips* contains a column in order to check eventually bugs during the treatment ( the different lines are empty in our cases)
    - *InstLumi* contains a 'float' describing the instant luminosity of the collision.
    - *npv* contains a 'int' of the number of primary vertex of the collision.
    - *dedx_charge* contains for each lines a variable size 'int' array that contains the energy (in *eV*) collected in the different detectors
    - *dedx_pathlength* contains for each lines a variable size 'float' array that contains the spacial distance (in *cm*) travelled in the different detectors.
    - *dedx_detid* contains a variable size 'int' array  about the id of the detector.
    - *dedx_isstrip* contains a variable size 'bool' array. If the value is *False*, the detection was made by a pixel. If the value is *True* the detection was made by a strip. The reliable informations considered here will be data collected by strip.
    - *dedx_insideTkmod* contains a variable size 'bool' array. If the value is *False*, the detection was made in a detector located in the border, which can imply some undetected energy. If the value is *True*, this problem isn't expected to occur, meaning the energy detection is likely more reliable.
    - *dedx_modulegeom* contains a variable size 'int' array about the informations about the geometry of the detector.
    - *dedx_shape* isn't considered in our entire project
    - *dedx_clusclean* contains a variable size 'bool' array. If the value is *False*, the cluster informations may not be related to a particle but some noise that could be irrelevant. 
    - *track_p* contains a 'float32' that gives the impulsion of the collision
    - *track_eta* contains a 'float32' that gives the rapidity of the collision

## Structure of the project 

Here is the following structure of the code :
* ./
    + Root_files directory contain the different core datas that can be used. As they are too massive to be directly upload in github, here is a seafile link that gives access to the following files :
    The files in the seafile are :
        - *tree.root* contains simulated data of proton & kaon at medium/low *p* (1Go)
        - *data.root* contains real data of proton & kaon of proton & kaon at medium/low *p* 
        - *data_GRU_simulated_V3.root* is an already filtred data on tree.root to be used as input in the Machine Learning (ML) files. 
        - *data_GRU_real_V3.root* is an already filtred data on data.root to be used as input in the Machine Learning (ML) files.
    + Architecture_RNN contain all the code related to the different Machine Learning. The variable size array called input are treated in a Gated Recurrent Unit (GRU) to make a __dedx__ value prediction which is then corrected by a neural network that can be a Long Short Term Memory or LSTM (RNN type of Network) or sequential Linear Layers (MLP), that contains the __dedx__ value in addition of __extras__ global parameters. The version of these files changes the amount of input that are treated by the different neural network : 
        - V1 :
            - Input : [dedx]
            - Extras : *N_hit*, *p*, *eta*, *I_h* 
            with I_h being the value of the estimator *I_h* and *N_hit* the number of hits by the track.
        - V2a : 
            - Input : [dedx]
            - Extras : *N_hit*, *eta*, *I_h* 
        - V2b : 
            - Input : [dedx]
            - Extras : *N_hit*, *eta*
        - V3 : 
            - Input : [dedx,modulegeom,pathlengh]
            - Extras : *N_hit*, *eta*,*I_h* 
    NB : dedx contains the values of __dedx_charge__ divided by the values of __dedx_pathlengh__ 
    : ML_V0 contains an old ML with only two Linear Layers
    When a training is ran, at the end of a run , the model is stored in a .pth file that has this template:
    *model_GRU_[LSTM/MLP]_V[1/2a/2b/3].pth*
    + Tuning contains the Tuning of the different tuning algorithm that are used in order to calculate the optimal hyperparameters in order to have the best precision with the models. When a tuning is ran, a directory is created with this template : 
     *"C:/Users/UserName/ray_results/train_model_ray_yyyy-mm-_hh-mm-ss*
     All the different trials are stored in this directory using ray[tune] & after the end of tuning, the best configuration with the best hyperparameters is launched.
     There is one tuning programm for each different algorithm except for both V2a ML (as their structure is very similar to V2a, we suppose that the hyperparameters for V2b could work for V2a). The model is then stored in a .pth file with this template :
     *best_model_GRU_[LSTM/MLP]_V[1/2a/2b/3].pth*
    For the user, we suppose that we should run the tuning only once and then either launch the model by loading the model in the "classic file" in Architecture_RNN, or if the number of epoch wasn't satisfying to observe a convergence, we can choose to recover the hyperparameters with the *Recup_Tuning.py* to change them in the "classic file" to be able to choose the number of epoch.
    + Core contains the core files that are either used to process the initial data, or to plot the results
        - *Creation_plus_filtred.py* filtrate the initial data under the 3 boolean conditions in order to filtrate the relevant data. 
        - *file_ML.py* filtrate the data in output of *creation_plus_filtred.py* by selecting the area that we want (here we study at low *p* , especially at *p* <1.2 GeV) and we preparate the data by ajusting the Dataframe in input of the Machine Learning.
        - *ML_plot.py* is used to plot the result of the prediction made by the ML.
        - *Identification.py* generate the theoretical Bethe-Bloch formula. There were also a function to identify a particle based on the (dedx,p) value but the data used wasn't allowing us to use it. It could be used for further development.
        - *main.py* can be used to call every function that was used , either to train or print predictions of a ML, but also to filtrate the data.
    + Utilities contains the files that can be used to eventually debug
        - *dedx_basic.py* print the graph of the unfiltred data in order to see what the data look like
        - *Extraction_tab.py* print the first value of a root file 
        - Recup_Tuning print the best hyperparameters of a done Tuning
    + venv contains the directory in order to install the correct environment
    + requirement.txt contains all the different library in order to be able to run the different programms.
    
## Creation of the environment 
Create the environment by running : 
```
pip install -r requirements.txt
```

## Use of the code :
Every part of the '''main.py''' is cutted in part in order to run the relative function that are used in order to :
    - Preprocess the file in order to formate it to be used for the ML
    - Run either :
        _ the training of the basic ML files
        _ The tuning of the ML files and run the best found configuration 
        _ the result of the different predictions made by the ML
The changes of the variable in main allows to change the different parameters (like the filters, the range of p , etc...)



