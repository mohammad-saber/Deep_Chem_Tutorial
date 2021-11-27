
import numpy as np
import json
import os, copy

import deepchem as dc

from sklearn.model_selection import KFold, LeaveOneOut, LeaveOneGroupOut, StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from tools.visualization import plot_prediction


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def standard_scaling(X):
    """
    Input Parameters:
    ----------
    X : pandas series or numpy array

    Returns:
    -------
    X : pandas series or numpy array
    """
    ss = StandardScaler()
    ss.fit(X)
    X = ss.transform(X)
    return X


def standard_scaling_inv(X, X_inverse):
    """
    Inverse transform of standard scaling
    
    Input Parameters:
    ----------
    X : DataFrame
        original data, we use it to fit.transform scaler object
        then, scaler is used to inverse transform X_inverse
    
    X_inverse : DataFrame
        data that is inverse transformed

    Returns:
    -------
    X_inverse : DataFrame
        data after inverse transform
    """
    ss = StandardScaler()
    ss.fit(np.array(X))
    #X = ss.transform(np.array(X))
    X_inverse = ss.inverse_transform(X_inverse)
    return X_inverse


# -----------------------------------------------------------------------------
# Main Functions
# -----------------------------------------------------------------------------

def get_GraphConvModel(model_params):
    return dc.models.GraphConvModel(**model_params)


def hyperparams_tuning(x, y, model_name, search_space, save_hyperparams_folder, nb_epoch=100, frac_train=0.8, seed=0):    
    """
    Make dataset from graph features, fit model and predict on test set
    Models use different featurizers. 
    
    Input Parameters:
    ----------------
    x : 1D numpy array
        smiles or mols
        
    y : numpy array (1D or 2D)
        Some PyTorch models only accepts 2D y. It is better to pass 2D array. 

    model_name : string

    search_space : dictionary
        It is different for every model 
 
    save_hyperparams_folder : string
         folder to save models during hyperparameter optimization
 
    nb_epoch : positive integer

    frac_train : numeric [0, 1]
        fraction of data used as train subset when splitting dataset 
        
    seed : integer
        seed for data split
      
    Returns:
    -------
    best_hyperparams_dict : dictionary

    R2, RMSE, MAE : numeric
    
    all_results : dictionary
        results for all search space
    """
    metric_r2 = dc.metrics.Metric(dc.metrics.r2_score)
    metric_mse = dc.metrics.Metric(dc.metrics.mean_squared_error)
    metric_mae = dc.metrics.Metric(dc.metrics.mean_absolute_error)

    # make folder to save checkpoints
    if not os.path.exists(f'{save_hyperparams_folder}/{model_name}'):
        os.mkdir(f'{save_hyperparams_folder}/{model_name}')

    if model_name == 'GraphConvModel':  
        featurizer = dc.feat.ConvMolFeaturizer(use_chirality=False, per_atom_fragmentation=False)
    elif model_name == 'WeaveModel':
        featurizer = dc.feat.WeaveFeaturizer(graph_distance=True)
    elif model_name in ['GATModel', 'GCNModel', 'AttentiveFPModel']:
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)   # some of the PyTorch models require "use_edges=True"
    else:
        raise Exception('model name is not valid')

    # Make dataset
    features = featurizer.featurize(x)   
    dataset = dc.data.NumpyDataset(X=features, y=y)

    # Split Dataset
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset = splitter.train_test_split(dataset=dataset, frac_train=frac_train, seed=seed)
            

    # Grid Search
    # https://deepchem.readthedocs.io/en/latest/api_reference/hyper.html#grid-hyperparameter-optimization
    class_model = getattr(dc.models, model_name)   # example: dc.models.GraphConvModel
    optimizer = dc.hyper.GridHyperparamOpt(class_model)
    
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
                search_space,      # dictionary that maps hyperparameter names (strings) to lists of possible parameter values
                train_dataset, 
                valid_dataset,     # dataset used for validation (optimization on valid scores) 
                metric_r2,         # metric used for evaluation
                nb_epoch=nb_epoch,
                use_max=True,      # If True, return the model with the highest score. Else return model with the minimum score.
                logdir=f'{save_hyperparams_folder}/{model_name}'   # directory in which to store created models. If not set, will use a temporary directory.
                )
    
    # best hyperparameters is a tuple with the same order of "search_space" keys
    best_hyperparams_dict = copy.deepcopy(search_space)
    for i, key in enumerate(best_hyperparams_dict.keys()):
        best_hyperparams_dict[key] = best_hyperparams[i]
    print(f'best hyperparameters : {best_hyperparams_dict}')

    '''
    # Restore model, test model metric
    best_model.get_checkpoints()   # filename of best model saved in the folder 
    best_model_checkpoints = best_model.get_checkpoints()[0]
    best_model_folder = os.path.dirname(best_model_checkpoints)   # name of folder where best model is saved        
    best_hyperparams_dict['model_dir'] = best_model_folder    
    model_l = class_model(**best_hyperparams_dict)
    model_l.restore()
    print("Loaded model R2:", model_l.evaluate(valid_dataset, metric_r2))
    '''
    
    R2 = best_model.evaluate(valid_dataset, metric_r2)['r2_score']
    RMSE = np.sqrt( best_model.evaluate(valid_dataset, metric_mse)['mean_squared_error'] )
    MAE = best_model.evaluate(valid_dataset, metric_mae)['mean_absolute_error']
    print("Validation set (R2):", R2, '\n')
    #print(all_results)

    return best_hyperparams_dict, R2, RMSE, MAE, all_results


def hyperparams_tuning_models(search_space_models, x_train, y_train, metric_hyperparams_filepath, save_hyperparams_folder,
                  nb_epoch=100, frac_train=0.8, seed=0):
    """
    Apply CV on graph models
    Every model has its own parameters. This function doesn't apply hyper-parameters tuning.
    Input X is fixed. loop is only for models
    
    Input Parameters:
    ----------------
    search_space_models : dictionary 
        {'model_name1': search_space1, 'model_name2': search_space2}
   
    x_train, y_train : numpy array
        x is smiles : 1D array
        y is dependent variable : 1D or 2D array (Some PyTorch models only accepts 2D y. It is better to pass 2D array) 

    metric_hyperparams_filepath : string  
           
    save_hyperparams_folder : string
         folder to save models during hyperparameter optimization
 
    nb_epoch : positive integer

    frac_train : numeric [0, 1]
        fraction of data used as train subset when splitting dataset 
        
    seed : integer
        seed for data split
    
    Returns:
    -------
    metric_hyperparams, best_hyperparams_all : dictionary
    """    
    
    R2_all, RMSE_all, MAE_all = {}, {}, {}
    best_hyperparams_all = {}
    
    for model_name, search_space in search_space_models.items():
        print(" -----  Current Model : " , model_name, " -----")
        try: 
            best_hyperparams_dict, R2, RMSE, MAE, _ = hyperparams_tuning(x_train, y_train, model_name, 
                            search_space, save_hyperparams_folder, nb_epoch=nb_epoch, frac_train=frac_train, seed=seed)
    
            R2_all[model_name] = R2
            RMSE_all[model_name] = RMSE
            MAE_all[model_name] = MAE
            best_hyperparams_all[model_name] = best_hyperparams_dict
                   
        except Exception as e:   # in case of failure  
            print(e)    
            # In case of failure, we save very small value for R2 and very large value for RMSE & MAE (worst results)
            R2_all[model_name] = -1e+10
            RMSE_all[model_name] = 1e+10
            MAE_all[model_name] = 1e+10 

        # This code is place here to save results in every iteration. If you stop code execution, current results are already saved. 
        metric_hyperparams = {'R2': R2_all, 'RMSE': RMSE_all, 'MAE': MAE_all}
        
        # save results in JSON file
        with open(metric_hyperparams_filepath, 'w') as f:
            json.dump(metric_hyperparams, f, indent=2, ensure_ascii=False)

        # save results in JSON file
        best_hyperparams_filepath = metric_hyperparams_filepath.replace("metric_hyperparams", "best_hyperparams")
        with open(best_hyperparams_filepath, 'w') as f:
            json.dump(best_hyperparams_all, f, indent=2, ensure_ascii=False)

    return metric_hyperparams, best_hyperparams_all


def make_data_fit_model(X_train, X_test, y_train, y_test, model_name, model_params, nb_epoch, get_uncertainty=True):    
    """
    Make dataset from graph features, fit model and predict on test set
    Models use different featurizers. 
    
    Input Parameters:
    ----------------
    X_train, X_test : 1D numpy array
        smiles
        
    y_train, y_test : numpy array (1D or 2D)
        Some PyTorch models only accepts 2D y. It is better to pass 2D array. 

    model_name : string

    model_params : dictionary
        It is different for every model 
        
    nb_epoch : positive integer

    get_uncertainty : boolean
        If True, uncertainty (std) will be calculated
        Only 'GraphConvModel' has this feature. Also, "dropout" must have value more than zero otherwise you will get error :  "dropout must be included in every layer to predict uncertainty"

    Returns:
    -------
    y_true, y_pred, y_std : numpy array
        Dimension is similar to y_train or y_test
    """

    if model_name == 'GraphConvModel':  
        featurizer = dc.feat.ConvMolFeaturizer(use_chirality=False, per_atom_fragmentation=False)
        if get_uncertainty: model_params['uncertainty'] = True   # Only 'GraphConvModel' can predict uncertainty, dropout must be included in every layer to predict uncertainty , if dropout is zero ---> ERROR
    elif model_name == 'WeaveModel':
        featurizer = dc.feat.WeaveFeaturizer(graph_distance=True)
    elif model_name in ['GATModel', 'GCNModel', 'AttentiveFPModel']:
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)   # some of the PyTorch models require "use_edges=True"
    else:
        raise Exception('model name is not valid')
        
    features_train = featurizer.featurize(X_train)   # numpy array, it returns a graph object for every molecule
    features_test = featurizer.featurize(X_test)   

    dataset_train = dc.data.NumpyDataset(X=features_train, y=y_train)
    dataset_test = dc.data.NumpyDataset(X=features_test, y=y_test)

    class_model = getattr(dc.models, model_name)   # example: dc.models.GraphConvModel
    model = class_model(**model_params)
    
    model.fit(dataset_train, nb_epoch=nb_epoch)
        
    y_true_ = dataset_test.y
    
    if model_name == 'GraphConvModel' and get_uncertainty==True:
        y_pred_, y_std_ = model.predict_uncertainty(dataset_test)   # numpy array
    else:
        y_pred_ = model.predict(dataset_test)
        y_std_ = np.array([])

    return y_true_, y_pred_, y_std_


def CV_graph(X, y, model_name, model_params, nb_epoch, CV_method, k = 10, seed = 0, groups = None, 
             apply_inverse_scaling=False, y_original=[], get_uncertainty=True):
    """
    Apply CV on a graph model 
    
    Input Parameters:
    ----------------
    X, y : numpy array
        X is smiles : 1D array
        y is dependent variable : 1D or 2D array (Some PyTorch models only accepts 2D y. It is better to pass 2D array)

    model_name : string

    model_params : dictionary
        It is different for every model 
        
    nb_epoch : positive integer
    
    CV_method : string
        cross validation method : ["loo", "logo", "k-fold", "sk-fold", "gk-fold"]
  
    k : positive integer
        No. of splits (folds) in k-fold cross-validation
    
    seed : integer or Noone
        random_state = seed (for data split)

    gropus : numpy array
        groups used in LOGO & GroupKFold

    apply_inverse_scaling : boolean
        If True, inverse scaling will be applied on prediction results (y_pred, y_std)
        It is used when you have used standard scaling on "y" before CV and you want to convert it back into original scale.
        For this operation, "y_original" is required. 
        If you don't want to use "inverse_scaling", you can pass "y_original" as an empty list because it will not be used anymore.         

    y_original : numpy array
        original y_train without scaling, it's used to inverse transform of scaled "y".
        only used when apply_inverse_scaling=True

    get_uncertainty : boolean
        If True, uncertainty (std) will be calculated
        Only 'GraphConvModel' has this feature. Also, "dropout" must have value more than zero otherwise you will get error :  "dropout must be included in every layer to predict uncertainty"
    
    Returns:
    -------
    y_true, y_pred, y_std : numpy array 
        Dimension is similar to y_train or y_test
    """
    assert type(X)==np.ndarray, 'X data must be numpy array'      
    assert type(y)==np.ndarray, 'y data must be numpy array'      

    y_true_list = []   # we use List since we can append 1D or 2D numpy array
    y_pred_list = []
    y_std_list = []
    index_list = []
    
    # LOOCV
    if CV_method == 'loo':
        loo = LeaveOneOut()
        cv = loo.split(X)
    # LOGO
    elif CV_method == 'logo':
        logo = LeaveOneGroupOut() 
        cv = logo.split(X, y, groups)
    # k-fold
    elif CV_method == 'k-fold':
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        cv = kf.split(X)
    # sk-fold
    elif CV_method == 'sk-fold':   # folds are made by preserving the percentage of samples for each class (used for classification)
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        cv = kf.split(X, y)
    # Group k-fold
    elif CV_method == 'gk-fold':   # The same group will not appear in two different folds (the number of distinct groups has to be at least equal to the number of folds)
        gkf = GroupKFold(n_splits=k)
        cv = gkf.split(X, y, groups)
    else:
        raise Exception('cv_method value is not valid. acceptable values : ["loo", "logo", "k-fold", "sk-fold", "gk-fold"]')
             
    for train_index, test_index in cv:        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        y_true_, y_pred_, y_std_ = make_data_fit_model(X_train, X_test, y_train, y_test, model_name, 
                                       model_params, nb_epoch, get_uncertainty=get_uncertainty)
        
        y_true_list.append(y_true_)
        y_pred_list.append(y_pred_) 
        y_std_list.append(y_std_)             
        index_list.append(test_index)

    # Convert to numpy array   ,   dimension is similar to original input data
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_std = np.concatenate(y_std_list)
    index = np.concatenate(index_list)

    # sort
    i = np.argsort(index)
    index = index[i]
    y_true = y_true[i]
    y_pred = y_pred[i]
    if model_name == 'GraphConvModel' and get_uncertainty==True: y_std = y_std[i]
    
    # inverse transform 
    if apply_inverse_scaling:
        y_true = standard_scaling_inv(y_original, y_true)   # inverse transform
        y_pred = standard_scaling_inv(y_original, y_pred)   # inverse transform
        if model_name == 'GraphConvModel' and get_uncertainty==True: y_std = y_std*np.array(y_original.std())   # get std of prediction
        
    return y_true, y_pred, y_std 


def CV_graph_models(hyperparams_models, x_train, y_train, nb_epoch, metrics_filepath, fig_save_folder, fig_name, plot_title,
                  CV_method, show_plot = True, k = 10, seed = 0, groups = None, 
                  apply_inverse_scaling=False, y_train_original= [], get_uncertainty=True):
    """
    Apply CV on graph models
    Every model has its own parameters. This function doesn't apply hyper-parameters tuning.
    Input X is fixed. loop is only for models
    
    Input Parameters:
    ----------------
    hyperparams_models : dictionary of model hyperparameters
        {'model_name1': params1, 'model_name2': params2}
   
    x_train, y_train : numpy array
        x is smiles : 1D array
        y is dependent variable : 1D or 2D array (Some PyTorch models only accepts 2D y. It is better to pass 2D array) 
            
    nb_epoch : positive integer

    metrics_filepath, fig_save_folder, fig_name, plot_title : string
    
    CV_method : string
        cross validation method : ["loo", "logo", "k-fold", "sk-fold", "gk-fold"]

    show_plot : boolean
      
    k : positive integer
        No. of splits (folds) in k-fold cross-validation
    
    seed : integer or Noone
        random_state = seed (some models don't have random_state)

    gropus : numpy array
        groups used in LOGO & GroupKFold

    apply_inverse_scaling : boolean
        If True, inverse scaling will be applied on prediction results (y_pred, y_std)
        It is used when you have used standard scaling on "y" before CV and you want to convert it back into original scale.
        For this operation, "y_original" is required. 
        If you don't want to use "inverse_scaling", you can pass "y_original" as an empty list because it will not be used anymore.         

    y_train_original : numpy array
        original y_train without scaling, it's used to inverse transform of scaled "y".
        only used when apply_inverse_scaling=True

    get_uncertainty : boolean
        If True, uncertainty (std) will be calculated
        Only 'GraphConvModel' has this feature. Also, "dropout" must have value more than zero otherwise you will get error :  "dropout must be included in every layer to predict uncertainty"
    
    Returns:
    -------
    results_metrics : dictionary

    results_y : dictionary
        y_true, y_pred, y_std for all models
    """    
    
    R2_all, RMSE_all, MAE_all = {}, {}, {}
    y_true_k, y_pred_k, y_std_k = {}, {}, {}   # save y_true, y_pred, y_std for all kernels

    for model_name, model_params in hyperparams_models.items():
        print(" -----  Current Model : " , model_name, " -----")
        try: 
            y_true, y_pred, y_std = CV_graph(x_train, y_train, model_name, model_params, nb_epoch, 
                         CV_method, k = k, seed = seed, groups = groups,  
                         apply_inverse_scaling=apply_inverse_scaling, y_original=y_train_original, get_uncertainty=get_uncertainty)

            if show_plot:
                plot_prediction(y_true, y_pred, f'{plot_title} ({model_name})', fig_save_folder, f'{fig_name}_{model_name}')
            
            R2 = r2_score(y_true, y_pred)
            print('R2 : %.3f' % R2, '\n')
            RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
            MAE = mean_absolute_error(y_true, y_pred)
    
            R2_all[model_name] = R2
            RMSE_all[model_name] = RMSE
            MAE_all[model_name] = MAE
            
            y_true_k[model_name] = y_true.tolist()   # convert to List to be saved in Json file , ".tolist()" keeps the original dimension of numpy array into List (shape of List is based on numpy array)
            y_pred_k[model_name] = y_pred.tolist()
            y_std_k[model_name] = y_std.tolist()
                   
        except Exception as e:   # in case of failure  
            print(e)    
            # In case of failure, we save very small value for R2 and very large value for RMSE & MAE (worst results)
            R2_all[model_name] = -1e+10
            RMSE_all[model_name] = 1e+10
            MAE_all[model_name] = 1e+10 

        # This code is place here to save results in every iteration. If you stop code execution, current results are already saved. 
        results_metrics = {'R2': R2_all, 'RMSE': RMSE_all, 'MAE': MAE_all}
        results_y = {'true_value': y_true_k, 'predicted_value': y_pred_k, 'predicted_std_value': y_std_k}
        
        # save results in JSON file
        with open(metrics_filepath, 'w') as f:
            json.dump(results_metrics, f, indent=2, ensure_ascii=False)

        # save results in JSON file
        results_y_filepath = metrics_filepath.replace("metrics", "results_y")
        with open(results_y_filepath, 'w') as f:
            json.dump(results_y, f, indent=2, ensure_ascii=False)

    return results_metrics, results_y

