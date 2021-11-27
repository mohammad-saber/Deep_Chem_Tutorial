# Deep_Chem_Tutorial
Sample notebooks for Deep Chem library (Application of Graph Neural Networks in Material Informatics)

This repository provides several notebooks as tutorial for Deep Chem, including:

- Basics
- Datasets
- How to get graph features
- How to use PyTorch & Keras models
- Cross-validation & hyperparameter tuning


I have used several graph models including:
- GraphConvModel
- WeaveModel
- GATModel
- GCNModel
- AttentiveFPModel


# Deep Chem

DeepChem is an open source library for use of deep-learning in drug discovery, materials science, quantum chemistry, and biology.

Deep Chem sorce code ([Link](https://github.com/deepchem/deepchem)) 


# Dependencies:

- Python: 3.7.11
- DeepChem: 2.5.0
- TensorFlow: 2.6.0
- tensorflow_probability: 0.14.0
- PyTorch: 1.9.1
- RDKit: 2021.03.5
- scikit-learn: 0.23.2
- pyGPGO: 0.5.1
- DGL: 0.7.1
- DGL-LifeSci: 0.2.8


## Yaml environment

I created a Conda virtual environment named **test_GNN**, and installed main libraries. In order to make the same Conda environment, download the yaml ([yaml](https://github.com/mohammad-saber/Deep_Chem_Tutorial/blob/main/GNN_environment.yaml)) file. Then, run the following command in Terminal:

    conda env create --file GNN_environment.yaml

Note: The current directory in Terminal should be the directory that the yaml file is stored. 




