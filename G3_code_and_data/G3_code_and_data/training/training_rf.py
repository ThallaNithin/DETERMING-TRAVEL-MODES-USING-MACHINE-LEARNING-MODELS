import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble
import sklearn.metrics
import util.storage
import training.analysis

def get_classifier_with_hyperparameters():
    #1 Define Hyperparameters to use Random hyperparameter search

    #explore the number of trees
    n_estimators =  [int(x) for x in np.linspace(1, 100, 60)] 

    # explore the number of features : If “auto”, then max_features=sqrt(n_features), 
    #If “log2”, then max_features=log2(n_features).
    max_features = ['auto','sqrt', 'log2'] 

    # explore tree depth 
    max_depth = [int(x) for x in np.linspace(5, 55, 11)] 
    max_depth.append(None) # Add "None" (the default) as a possible value

    # The minimum number of samples required to split an internal node
    min_samples_split = [int(x) for x in np.linspace(2, 10, 9)]

    # The minimum number of samples required to be at a leaf node.
    min_samples_leaf = [3,4]

    # Method of selecting samples for training each tree: If False, the whole dataset is used to build each tree.
    bootstrap = [True, False]

    # Create the random grid
    random_param_grid = {'n_estimators': n_estimators,
                         'max_features': max_features,
                         'max_depth': max_depth,
                         'min_samples_split': min_samples_split,
                         'min_samples_leaf': min_samples_leaf,
                         'bootstrap': bootstrap}
    
    rf = sklearn.ensemble.RandomForestClassifier(random_state=0)
    param_dist = random_param_grid
    cv = 8
    n_jobs = -1
    n_iter = 1000
    
    return rf, param_dist, cv, n_jobs, n_iter

def plot_sensitivity_analysis(cv_accuracy):
    K_values = range(1, 11)

    max_acc_held_out = np.amax(cv_accuracy) #Return the max of held-out accuracy
    max_index = np.argmax(cv_accuracy)

    plt.scatter(max_index+1, max_acc_held_out, s=80, edgecolors='r', facecolors='none', label='Optimum k-fold')
    plt.plot(K_values, cv_accuracy ,"bx-")
    plt.xticks(K_values)
    plt.title('Held-out accuracy using CV- Random Forest')
    plt.xlabel('k_values')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    util.storage.save_fig("Held-out-acc_Random_Forest")
    plt.show();
    
def get_cross_validation(folds, X_train, y_train,X_test, y_test):
    
    rnd_clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, random_state=0)
    rnd_clf.fit(X_train,y_train)
    
    accuracy_train, accuracy_test, cv_accuracy = training.analysis.get_cross_validation(rnd_clf,folds, X_train, y_train, X_test, y_test)
    
    return rnd_clf, accuracy_train, accuracy_test, cv_accuracy
    
     