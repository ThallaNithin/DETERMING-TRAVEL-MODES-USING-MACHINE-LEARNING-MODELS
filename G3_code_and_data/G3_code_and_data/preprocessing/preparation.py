import numpy as np
import sklearn
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit

def train_test_split(data,test_size):
    """
    Input parameters: 
    data: original dataset
    test_size: test size
    Output: Return dataset split
    """
    return sklearn.model_selection.train_test_split(data, test_size=test_size, random_state=42)

def read_csv(filePath):
    """
    Input parameters: 
    filePath: file path to retrieve csv
    Output: Return file reference object
    """
    df = pd.read_csv(filePath,encoding='unicode_escape')
    return df

def mode_field_proportions(data,field):
    """
    Input parameters: 
    data: original dataset
    field: field to get its proportion
    Output: Return proportion of the field
    """
    return data[field].value_counts() / len(data)

def get_stratified_split(data, field, splits,test_size,random_state):
    """
    Input parameters: 
    data: original dataset
    splits: number of splits
    Output: Return stratified split
    """
    split = StratifiedShuffleSplit(n_splits=splits, test_size = test_size, random_state=random_state)
    for train_index, test_index in split.split(data, data[field]):
        strat_train = data.loc[train_index]
        strat_test = data.loc[test_index]
    return strat_train, strat_test
    
 # link of the code of reference : https://github.com/ageron/handson-ml/blob/master/02_end_to_end_machine_learning_project.ipynb   
def get_proportions_comparison(data, strat_data, random_data, target):
    """
    Input parameters: 
    data: original dataset
    splits: number of splits
    Output: Return proportions of the dataset
    """
    compare_proportions = pd.DataFrame({
        "Overall": mode_field_proportions(data, target),
        "Stratified": mode_field_proportions(strat_data, target),
        "Random": mode_field_proportions(random_data, target),
    }).sort_index()
    compare_proportions["Rand. %error"] = 100 * compare_proportions["Random"] / compare_proportions["Overall"] - 100
    compare_proportions["Strat. %error"] = 100 * compare_proportions["Stratified"] / compare_proportions["Overall"] - 100
    return compare_proportions
    
def convert_data_numpy(data):
    """
    Input parameters: 
    data: original dataset
    Output: numpy array
    """
    data_matrix = data.to_numpy()
    X = data_matrix[ :,1:50].astype('int32') # X's features (matrix)
    y = data_matrix[ :,0].astype('int32') # y's target

    n_classes = np.unique(y).shape[0] #number of classes
    n_features = X.shape[1] #  #number of features
    return X, y
    
def get_training_test_data(strat_train,strat_test):
    """
    Input parameters: 
    strat_train: stratified train
    strat_test: stratified test
    Output: Return training features and target
    """
    # Convert the train DataFrame to Numpy array
    train_data = strat_train.to_numpy()
    test_data = strat_test.to_numpy()

    ## X_train and y_train
    y_train = train_data[ :,0].astype('int32')
    X_train = train_data[ :,1:50].astype('int32')

    ## X_test and y_test
    y_test = test_data[ :,0].astype('int32')
    X_test = test_data[ :,1:50].astype('int32')
    return X_train, y_train, X_test, y_test;
    