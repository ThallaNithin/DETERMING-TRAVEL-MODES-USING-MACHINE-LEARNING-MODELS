import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm

import util.storage
import training.analysis

def get_classifier_with_hyperparameters():
    #1 Define Hyperparameters for to use Random hyperparameter search

    # Create the random grid
    parameter_space={'C': [0.01, .1, 1, 10, 100, 1000],
                 'gamma': [ 0.01,  0.1 ,  1.  , 10.  ]}
    svm = sklearn.svm.SVC(kernel='rbf',random_state=0,probability=True)
    param_dist = parameter_space
    cv = 8
    n_jobs = -1
    n_iter = 24
    
    return svm, param_dist, cv, n_jobs, n_iter

def plot_sensitivity_analysis(cv_accuracy):
    K_values = range(1, 11)

    max_acc_held_out = np.amax(cv_accuracy) #Return the max of held-out accuracy
    max_index = np.argmax(cv_accuracy)

    plt.scatter(max_index+1, max_acc_held_out, s=80, edgecolors='r', facecolors='none', label='Optimum k-fold')
    plt.plot(K_values, cv_accuracy ,"bx-")
    plt.xticks(K_values)
    plt.title('Held-out accuracy using CV- Support Vector Machines')
    plt.xlabel('k_values')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    util.storage.save_fig("Held-out-acc_SVM")
    plt.show();
    
def get_cross_validation(folds, X_train, y_train,X_test, y_test):
    
    svm = sklearn.svm.SVC(random_state=0, probability = True)
    svm.fit(X_train,y_train)
    
    accuracy_train, accuracy_test, cv_accuracy = training.analysis.get_cross_validation(svm,folds, X_train, y_train, X_test, y_test)
    
    return svm, accuracy_train, accuracy_test, cv_accuracy
    
     