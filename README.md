# Particle identification using the CMS silicon tracker : Exploring the use of machine learning techniques

The goal of this project is to explore alternative methods for correcting observed __dE/dx__ biases
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
    + architecture_RNN contains all the code related to the different Machine Learning. The variable size array called input are treated in a Gated Recurrent Unit (GRU) to make a __dedx__ value prediction which is then corrected by a neural network that can be a Long Short Term Memory or LSTM (RNN type of Network) or sequential Linear Layers (MLP), that contains the __dedx__ value in addition to __extras__ global parameters. The version of these files changes the amount of input that is treated by the different neural networks: 
        - V1 :
            - Input : [ __dedx__ ]
            - Extras : __N_hit__, __p__, __eta__, __I_h__ 
            with I_h being the value of the estimator __I_h__ and __N_hit__ the number of hits by the track.
        - V2a : 
            - Input : [ __dedx__ ]
            - Extras : __N_hit__, __eta__, __I_h__ 
        - V2b : 
            - Input : [ __dedx__ ]
            - Extras : __N_hit__, __eta__ 
        - V3 : 
            - Input : [ __dedx__ ,__modulegeom__,__pathlength__]
            - Extras : __N_hit__, __eta__, __I_h__ <br>
    NB: dedx contains the values of __dedx_charge__ divided by the values of __dedx_pathlength__. ML_V0 contains an old ML with only two Linear Layers.<br>
    When training is run, at the end of a run, the model is stored in a .pth file that has this template: *model_GRU_[LSTM/MLP]_V[1/2a/2b/3].pth*
    
    + core contains the core files that are either used to process the initial data or to plot the results:
        - *Creation_plus_filtred.py* filters the initial data under the 3 boolean conditions to extract the relevant data. 
        - *file_ML.py* filters the output data of *creation_plus_filtred.py* by selecting the area that we want (here we study at low __p__ , especially at __p__  < pmin) and prepares the data by adjusting the *Dataframe* for input into Machine Learning.
        - *ML_plot.py* is used to plot the results of the ML predictions.
        - *Identification.py* generates the theoretical Bethe-Bloch formula. There was also a function to identify a particle based on the (__dedx__ ,__p__ ) value, but the data used wasn't allowing us to use it. It could be used for further development.
        - *main.py* can be used to call every function that was used, either to train or print ML predictions, but also to filter the data.

    + models store all the .pth files to load them for plotting different results, either the *best_model[..].pth* or the *model[..].pth* as explained further for *best_model*

    + requirement.txt contains all the different libraries needed to run the different programs.

    + root_files directory contains the different core datasets that can be used. As they are too massive to be directly uploaded to GitHub, here is a Seafile link that provides access to the following files https://seafile.unistra.fr/d/ff1385710b2f42548108/:
        The files in Seafile are:
        - *tree.root* contains simulated data of proton & kaon at medium/low __p__ (1GB)
        - *data.root* contains real data of proton & kaon at medium/low __p__ (2GB)
        - *data_GRU_simulated_V3.root* is a pre-filtered dataset from *tree.root* to be used as input in the Machine Learning (ML) files.
        - *data_GRU_real_V3.root* is a pre-filtered dataset from *data.root* to be used as input in the Machine Learning (ML) files.

    + tuning contains the tuning of different algorithms used to calculate the optimal hyperparameters for achieving the best model precision. When tuning is run, a directory is created with this template:  
      *"C:/Users/UserName/ray_results/train_model_ray_yyyy-mm-dd_hh-mm-ss"*.  
      All the different trials are stored in this directory using ray[tune]. After tuning ends, the best configuration with the best hyperparameters is selected and launched.<br>
      There is one tuning program for each different algorithm except for both V2a ML (as their structure is very similar to V2a, we assume that the hyperparameters for V2b could work for V2a). The model is then stored in a .pth file with this template: *best_model_GRU_[LSTM/MLP]_V[1/2a/2b/3].pth*<br>
      For the user, we assume that tuning should be run only once. Afterward, either:<br>
      - The model can be launched by loading it in the "classic file" in Architecture_RNN.
      - If the number of epochs was insufficient for convergence, the hyperparameters can be retrieved with *Recup_Tuning.py* and adjusted in the "classic file" to modify the number of epochs.

    + utilities contains files that can be used for debugging:
        - *dedx_basic.py* prints the graph of the unfiltered data to visualize its distribution.
        - *Extraction_tab.py* prints the first value of a root file.
        - *Recup_Tuning.py* prints the best hyperparameters from a completed tuning session.

    + venv contains the directory for installing the correct environment.

    
## Creation of the environment 
Create the environment by running : <br>
```
pip install -r requirements.txt
```

## Use of the code :
Every part of the '''main.py''' is cutted in part in order to run the relative function that are used in order to :<br>
- Preprocess the file in order to formate it to be used for the ML <br>
- Run either :<br>
    * the training of the basic ML files<br>
    * The tuning of the ML files and run the best found configuration <br>
    * the result of the different predictions made by the ML<br>
The changes of the variable in main allows to change the different parameters (like the filters, the range of __p__  , etc...)<br>



