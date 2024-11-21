import time
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.metrics
import util.storage
import training.analysis

def get_classifiers_config():
    """
    Get various experiments for logistic regression to obtain the best among them.
    
    Input Parameters:
    
    Output:    
    return: array of classifier configurations
    
    """
    C =10
    max_iter = 10

    # Create a dictionary with different hyperparameter for Multinomial Logistic Regression
    # Key: solver name
    # Values: custom hyperparamters such as C, penalty, max_iter

    # In this case by default fit_intercept is True, we will keep it because,
    # it will add an additional intercept column of all 1's to our X's matrix


    classifiers = {
        # Newton’s Method It’s computationally expensive because of the Hessian Matrix
        'Multinomial + L2 penalty (Solver: Newton-cg)': sklearn.linear_model.LogisticRegression(C=C, penalty='l2',
                                                                  solver='newton-cg',
                                                                  multi_class='multinomial',
                                                                  max_iter=max_iter),
        'Multinomial + L2 penalty (Solver: lbfgs)': sklearn.linear_model.LogisticRegression(C=C, penalty='l2',
                                                              solver='lbfgs',
                                                              multi_class='multinomial',
                                                              max_iter=max_iter),
        'Multinomial + L2 penalty (Solver: sag)': sklearn.linear_model.LogisticRegression(C=C, penalty='l2',
                                                            solver='sag',
                                                            multi_class='multinomial',
                                                            max_iter=max_iter),
        'Multinomial + L2 penalty (Solver: saga)': sklearn.linear_model.LogisticRegression(C=C, penalty='l2',
                                                            solver='saga',
                                                            multi_class='multinomial',
                                                            max_iter=max_iter),
        'Multinomial + L1 penalty (Solver: saga)': sklearn.linear_model.LogisticRegression(C=C, penalty='l1',
                                                            solver='saga',
                                                            multi_class='multinomial',
                                                            max_iter=max_iter),
        'OVR + L2 penalty (Solver:lbfgs)': sklearn.linear_model.LogisticRegression(C=C, penalty='l2',
                                                                           solver='lbfgs',
                                                                           multi_class='ovr',
                                                                           max_iter=max_iter),
        'OVR + L2 penalty (Solver:Newton-cg)': sklearn.linear_model.LogisticRegression(C=C, penalty='l2',
                                                                           solver='newton-cg',
                                                                           multi_class='ovr',
                                                                           max_iter=max_iter),
        'OVR + L2 penalty (Solver:sag)': sklearn.linear_model.LogisticRegression(C=C, penalty='l2',
                                                                           solver='sag',
                                                                           multi_class='ovr',
                                                                           max_iter=max_iter),
        'OVR + L2 penalty (Solver:saga)': sklearn.linear_model.LogisticRegression(C=C, penalty='l2',
                                                                           solver='saga',
                                                                           multi_class='ovr',
                                                                           max_iter=max_iter),
        'OVR + L1 penalty (Solver:liblinear)': sklearn.linear_model.LogisticRegression(C=C, penalty='l1',
                                                                           solver='liblinear',
                                                                           multi_class='ovr',
                                                                           max_iter=max_iter),     
        'OVR + L1 penalty (Solver:saga)': sklearn.linear_model.LogisticRegression(C=C, penalty='l1',
                                                                           solver='saga',
                                                                           multi_class='ovr',
                                                                           max_iter=max_iter)
}
    print(classifiers)
    return classifiers


def get_classifier_with_hyperparameters():
    """
    Get classifier with hyperparameters to be passed to the cross validation random search
    
    Input Parameters:
    
    Output:    
    return: classifier config
    
    """
    # Create the random grid
    param_dist = {
                    'C': [100, 10, 1.0, 0.1, 0.01],
                    'tol' : [0.001, 0.0001, 0.005]
                 }
    lr = sklearn.linear_model.LogisticRegression(random_state=0,solver='newton-cg',multi_class='multinomial',penalty='l2')
    
    cv = 8
    n_jobs = -1
    n_iter = 100
    
    return lr, param_dist, cv, n_jobs, n_iter

def plot_sensitivity_analysis(cv_accuracy):
    K_values = range(1, 11)

    max_acc_held_out = np.amax(cv_accuracy) #Return the max of held-out accuracy
    max_index = np.argmax(cv_accuracy)

    plt.scatter(max_index+1, max_acc_held_out, s=80, edgecolors='r', facecolors='none', label='Optimum k-fold')
    plt.plot(K_values, cv_accuracy ,"bx-")
    plt.xticks(K_values)
    plt.title('Held-out accuracy using CV- Multinomial Logistic Regression')
    plt.xlabel('k_values')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    util.storage.save_fig("Held-out-acc Multinomial Logistic Regression")
    plt.show();
    
def print_accuracies_by_solver(classifiers, X_train, y_train, X_test, y_test):
    t0 = time.time()

    n_classifiers = len(classifiers)

    trn_accuracies = np.zeros(n_classifiers)
    test_accuracies = np.zeros(n_classifiers)
    cv_accuracies = np.zeros((n_classifiers,9))
    run_time_solvers = np.zeros(n_classifiers)

    for index, (name, classifier) in enumerate(classifiers.items()):
        
        ts = time.time()
        
        classifier.fit(X_train, y_train)

        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)
        
        train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
        trn_accuracies[index]=train_accuracy
        test_accuracy = sklearn.metrics.accuracy_score(y_test, y_test_pred)
        test_accuracies[index] = test_accuracy 
        cv_accuracy = sklearn.model_selection.cross_val_score(classifier, 
                                                          X_train, y_train, cv=9)
        for i in range(9):
            cv_accuracies[index,i] = cv_accuracy[i]
        
        print("Training accuracy for %s: %0.1f%% " % (name, train_accuracy * 100))
        print("held-out accuracy (testing) for %s: %0.1f%% " % (name, test_accuracy * 100))
        [print("held-out accuracy for %s (%d-fold): %.1f%%" % (name, i+1, cv_accuracy[i]*100)) 
         for i in range(1, len(cv_accuracy))]
        
        run_time_solver = time.time() - ts
        run_time_solvers[index] = run_time_solver
        
    run_time = time.time() - t0
    print("Example run in %.3f s" % run_time)
    return trn_accuracies, test_accuracies, cv_accuracies, run_time_solvers
    
def get_cross_validation(X_train, y_train,X_test, y_test):
    
    lr_clf = sklearn.linear_model.LogisticRegression(random_state=0,solver='newton-cg',multi_class='multinomial',penalty='l2')
    lr_clf.fit(X_train,y_train)
    
    accuracy_train, accuracy_test, cv_accuracy = training.analysis.get_cross_validation(lr_clf,10, X_train, y_train, X_test, y_test)
    
    return lr_clf, accuracy_train, accuracy_test, cv_accuracy
    
     