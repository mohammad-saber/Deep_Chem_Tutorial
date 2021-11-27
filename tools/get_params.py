
import json

def get_search_space(n_tasks):
    
    ''' 
    search space for hyper-parameter tuning  
    '''
    
    search_space_GraphConvModel = {
    'n_tasks': [n_tasks],
    'mode': ['regression'],
    'batch_size': [64, 100], # I added 100 because it is default value
    'learning_rate': [0.01, 0.001],
    'dropout': [0.2, 0.4],   # Dropout must be included in every layer to predict uncertainty , if dropout is zero ---> ERROR
    'graph_conv_layers': [[64], [64, 64]],
    'dense_layer_size': [64, 128]
    }

    search_space_WeaveModel = {
        'n_tasks': [n_tasks],
        'mode': ['regression'],   
        'batch_size': [64, 100],
        'learning_rate': [0.01, 0.001],
        'dropout': [0, 0.25, 0.4],   # Default value is 0.25. Name of parameter is 'dropouts'. For other models is called 'dropout'. If you use 'dropout' here, still works.  
        'n_weave': [2, 3],
        'batch_normalize': [False]         # Use of batch normalization can cause issues with NaNs. If you’re having trouble with NaNs while using this model, consider setting batch_normalize=False. Default value is True. 
        }
       
    search_space_GATModel = {
        'n_tasks': [n_tasks],
        'mode': ['regression'],   
        'batch_size': [64, 100],
        'learning_rate': [0.01, 0.001],
        'dropout': [0, 0.2, 0.4]
        }
       
    search_space_GCNModel = {
        'n_tasks': [n_tasks],
        'mode': ['regression'],   
        'batch_size': [64, 100],
        'learning_rate': [0.01, 0.001],
        'dropout': [0, 0.2, 0.4],
        'graph_conv_layers': [[64], [64, 64]]
        }
       
    search_space_AttentiveFPModel = {
        'n_tasks': [n_tasks],
        'mode': ['regression'],   
        'batch_size': [64, 100],
        'learning_rate': [0.01, 0.001],
        'dropout': [0, 0.2, 0.4],
        'num_layers': [2, 3]
        }
       
    search_space_models = {'GraphConvModel': search_space_GraphConvModel, 'WeaveModel': search_space_WeaveModel,
                   'GATModel': search_space_GATModel, 'GCNModel': search_space_GCNModel, 'AttentiveFPModel': search_space_AttentiveFPModel}
    
    return search_space_models


def get_hyperparams_models(n_tasks):
    
    '''
    Hyperparameters for different models, usually used in CV  
    '''
    
    params_GraphConvModel = {'n_tasks': n_tasks,             # No. of tasks 
                            'mode': 'regression',            # Either “classification” or “regression”
                            'batch_size':100,                # Batch size for training and evaluating
                            'learning_rate': 0.001,
                            'dropout': 0.2,                  # Dropout probablity to use for each layer. The length of this list should equal len(graph_conv_layers)+1 (one value for each convolution layer, and one for the dense layer). Alternatively this may be a single value instead of a list, in which case the same value is used for every layer.
                            'graph_conv_layers': [64, 64],   # Width of channels for the Graph Convolution Layers
                            'dense_layer_size': 128          # Width of channels for Atom Level Dense Layer before GraphPool
                            }
    
    params_WeaveModel = {'n_tasks': n_tasks,              # No. of tasks 
                         'mode': 'regression',            # Either “classification” or “regression”
                         'batch_size':100,                # Batch size for training and evaluating
                         'learning_rate': 0.001,
                         'dropout': 0.25,                # Dropout probablity to use for each fully connected layer
                         'n_hidden': 50,                  # No. of units(convolution depths) in corresponding hidden layer. Default value is 0.25. Name of parameter is 'dropouts'. For other models is called 'dropout'. If you use 'dropout' here, still works.       
                         'n_weave': 2,                    # No. of weave layers
                         'fully_connected_layer_sizes': [2000, 100],   # Size of each dense layer in the network. The length of this list determines the number of layers.                                 
                         'batch_normalize': False         # Use of batch normalization can cause issues with NaNs. If you’re having trouble with NaNs while using this model, consider setting batch_normalize=False. Default value is True.
                         }
    
    params_GATModel = {'n_tasks': n_tasks,              # No. of tasks 
                       'mode': 'regression',            # Either “classification” or “regression”
                       'batch_size':100,                # Batch size for training and evaluating
                       'learning_rate': 0.001,
                       'dropout': 0.2,                  # Dropout probability within each GAT layer
                       'predictor_dropout': 0           # Dropout probability in the output MLP predictor.  
                        }
    
    params_GCNModel = {'n_tasks': n_tasks,              # No. of tasks 
                       'mode': 'regression',            # Either “classification” or “regression”
                       'batch_size':100,                # Batch size for training and evaluating
                       'learning_rate': 0.001,
                       'graph_conv_layers': [64, 64],   # Width of channels for GCN layers
                       'dropout': 0.2,                  # Dropout probability within each GAT layer
                       'predictor_dropout': 0,          # Dropout probability in the output MLP predictor
                       'batchnorm': False               # Whether to apply batch normalization to the output of each GCN layer
                        }
    
    params_AttentiveFPModel = {'n_tasks': n_tasks,              # No. of tasks 
                               'mode': 'regression',            # Either “classification” or “regression”
                               'batch_size':100,                # Batch size for training and evaluating
                               'learning_rate': 0.001,
                               'num_layers': 2,                 # No. of graph neural network layers
                               'dropout': 0.2,                  # Dropout probability within each GAT layer
                               }
    
    hyperparams_models = {'GraphConvModel': params_GraphConvModel, 'WeaveModel': params_WeaveModel,
                   'GATModel': params_GATModel, 'GCNModel': params_GCNModel, 'AttentiveFPModel':params_AttentiveFPModel}

    return hyperparams_models


def get_best_hyperparams_models(best_hyperparams_filepath):

    '''
    Load hyperparameters already saved in a JSON file
    '''
    # load feature selection results from JSON file
    with open(best_hyperparams_filepath, 'r') as f:
        jso = json.load(f)

    return jso