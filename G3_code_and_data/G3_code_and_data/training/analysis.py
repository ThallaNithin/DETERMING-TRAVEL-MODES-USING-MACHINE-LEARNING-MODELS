import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import pandas as pd
from pandas import DataFrame
import util.catalogue

def get_opt_cv(model, X, y, k):
    """
    Input parameters: 
    Object model
    X: feature matrix
    y: target
    k: number of k folds
    Output: Return the held-out accuracy maximun and the the best k fold
    """
    # evaluate the model and collect the results
    cv_accuracy = sklearn.model_selection.cross_val_score(model, X, y, scoring='accuracy', cv=k)
    max_acc_held_out = np.amax(cv_accuracy) #Return the max of held-out accuracy 
    max_index = np.argmax(cv_accuracy)
    return (round(max_acc_held_out*100,1),max_index+1)
    
def print_accuracy_cv(kfolds, clf,X_train, y_train):
    """
    Input parameters: 
    kfolds: number of k folds
    clf : classifier
    X_train : input training features X
    y_train : targets Y 
    Output: Print accuracu for the training inputs and k folds
    """
    accuracy_values = get_opt_cv(clf,X_train,y_train, kfolds)
    accuracy_cv = accuracy_values[0]
    cv_opt = accuracy_values[1]

    print("**Accuracy on Train and Held-out with CV**")
    print("held-out accuracy max (%d- best fold): %.1f%%" % (cv_opt, accuracy_cv));
    
def random_search(model, X_train, y_train, param_dist,n_iter, cv, n_jobs, scoring = None):
    """
    random_search function for to apply Randomized search Cross Validation (CV) on hyper parameter, 
    in contrast to GridSearchCV, not all parameter values are tried out.
    
    Input Parameters:
    
    model:        object model instance
    X_train:      training data (features) 
    y_train:      training targets 
    param_dist:   dictionary with parameters names (str) as keys and distributions or lists of parameters to try.    
    n_iter:       Number of parameter settings that are sampled.
    cv:           Number of folds. Determines the cross-validation splitting strategy
    n_jobs:       Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
    
    Output:    
    return: the best estimator (best model)
    
    """
    random_search_cv = sklearn.model_selection.RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs, 
        verbose=1, 
        scoring=scoring, 
        random_state=0)  

    return random_search_cv.fit(X_train, y_train)

def accuracy_score(model, X_tst=None, y_tst=None):
    """
    To get accuracy classification score.
        
    Input Parameters:
        
    model: object model instance 
    X_test:      testing data (features) 
    y_test:      testing targets 
        
    Output:   
    return: the Accuracy classification score. 
    
    """
    
    acc_score = sklearn.metrics.accuracy_score(
        y_true=y_tst,
        y_pred=model.predict(X_tst)
    )
    return acc_score
    
# To get macro precision score for multiclass targets,
def precision_score(model, X_tst=None, y_tst=None, average='macro'):
    """
    Compute the precision score for multiclass targets, using the ROC average
    
    The precision is the ratio tp / (tp + fp) where tp is the number of true positives 
    and fp the number of false positives.
    The precision is intuitively the ability of the classifier not to label as positive 
    a sample that is negative.
    
    Input Parameters:
    
    model: object model instance 
    X_tst:      testing data (features) 
    y_tst:      multi-class targets
        
    Output:   
    return: precision score   
    
    """
    precision = sklearn.metrics.precision_score(
        y_true=y_tst, 
        y_pred=model.predict(X_tst),
        average = average)
    
    return precision

# To get micro Recall score for multiclass targets

def recall_score(model, X_tst=None, y_tst=None, average='macro'):
    """
    Compute the recall score for multiclass targets, using the macro average
    
    The recall is the ratio tp / (tp + fn) where tp is the number of true positives 
    and fn the number of false negatives.
    
    Input Parameters:
    
    model: object model instance 
    X_test:      testing data (features) 
    y_test:      multi-class targets
        
    Output:   
    return: average Recall score   
    
    """
    average_recall = sklearn.metrics.recall_score(
        y_true=y_tst, 
        y_pred=model.predict(X_tst),
        average = average)
    
    return average_recall
    
def average_auroc_score(catalogue, model, X_tst=None, y_tst=None,average='macro'):
    """
    Compute average ROC area for multiclass targets
    
    Input Parameters:
    
    catalogue: dictionary variable to get labels
    model: object model instance 
    X_tst:      testing data (features) 
    y_tst:      testing multi-class targets
        
    Output:   
    return: average ROC area  
    
    """
    # calculate the y_score (calls the classifier to predict_proba method.)
    y_score=model.predict_proba(X_tst)

    #Binarize the output    
    classes = util.catalogue.get_labels(catalogue, 'KHVM').to_numpy()[ : , -2]#call get_labels function to get the labels of the classes 
    y_test_binarize = label_binarize(y_tst, classes=classes)

    # Compute average ROC curve and ROC area
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr[average], tpr[average], _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    roc_auc[average] = auc(fpr[average], tpr[average])
    
    return (roc_auc[average])

    
def predict(model, data = None):
    """
    predict function to compute the prediction 
    
    Input Parameters:
    Feature Data: training or testing data
    
    Output:
    return prediction
    
    """       
    return (model.predict(data))
    
def evaluate_performance(catalogue, model, data=None, targets=None, average='macro'):
    """
    To evaluate the performance of the model 
    
    Input: 
    
    catalogue : dictionary variable to get labels
    model: object model instance 
    data:  testing data (features) 
    targets:  testing multi-class targets

    Output: 
    return (accuracy, recall, precision, macro_AUROC)
    
    """

    accuracy = accuracy_score(model,data, targets) 
    recall = recall_score(model,data, targets, average)
    precision = precision_score(model,data, targets, average)
    average_auroc = average_auroc_score(catalogue, model,data, targets, average)
    
    return (accuracy,
            recall,
            precision,
            average_auroc)

def print_performance_evaluation(catalogue,classifierName, model,data,targets,average='macro'):
    """
    To print the performance of the model 
    
    Input: 
    
    catalogue : dictionary variable to get labels
    classifierName : Name of the classifier
    model: object model instance 
    data:  testing data (features) 
    targets:  testing multi-class targets

    Output: 
    return (accuracy, recall, precision, average_AUROC)
    
    """
    # Evaluate the performance of the Best Model 
    metrics = evaluate_performance(catalogue, model = model, 
                                   data=data, targets=targets, average=average)

    print('Model Performance of Classifier:',classifierName)
    print('Accuracy= {:0.4f}%.'.format(100*metrics[0]))
    print('Recall= {:0.4f}%.'.format(100*metrics[1]))
    print('Precision= {:0.4f}%.'.format(100*metrics[2]))
    print(average+'-average ROC curve= {:0.4f}.'.format(metrics[3]))

# function to get the importance of the features
# this function can be used on Random Forest and SVM methods
def get_features_importances(catalogue, model,data): 
    """
    To get the importances of the features
    
    Input: 
    
    catalogue : dictionary variable to get labels
    model: object model instance 

    Output: 
    return a data frame  the importance of the features in descending order 
    
    """

    #get features Importances 
    # getting the matrix of features X's excluding the target variable KHVM (y) 
    X=data.drop('KHVM',axis=1) 
    # coef_ attribute is the feature importance for Logistic Regression
    if hasattr(model, 'feature_importances_'):
        f_importance = model.feature_importances_
    else:
        f_importance = model.coef_[0]
        
    ## we want to print the series with the key description as a third column 
    # we get the first two columns with this:
    importances= pd.Series(f_importance,index=X.columns) 
    sorted_importances = importances.sort_values(ascending=False) 
    
    # and the column description is here 
    keys = sorted_importances.keys() # to get the features names 
    descriptions = util.catalogue.get_var_descriptions(catalogue,keys) # to get features description 
    
    feature_importance = pd.DataFrame(np.column_stack((keys, descriptions, np.round(sorted_importances,4))))
    feature_importance.columns = ["Feature", "Description", "Importance"]
        
    return feature_importance

def get_cross_validation(classifier, folds, X_train, y_train, X_test, y_test):
    """
    To get the cross validation score
    
    Input: 
    
    classifier : estimator class
    folds: number of K-folds 
    X_train : training input
    y_train : targets
    X_test  : test input
    y_test  : test targets
    Output: 
    return the cross validation score
    
    """
    accuracy_train = sklearn.metrics.accuracy_score(y_true=y_train, 
                                                    y_pred=classifier.predict(X_train))
    accuracy_test =  sklearn.metrics.accuracy_score(y_true=y_test, 
                                                    y_pred=classifier.predict(X_test)) 
    cv_accuracy = sklearn.model_selection.cross_val_score(classifier, 
                                                          X_train, y_train, cv=folds)
    print("**Accuracy on Train and Held-out**")
    print("Training accuracy: %.1f%%" % (accuracy_train*100))
    print("Held-out accuracy (testing):  %.1f%%" % (accuracy_test*100))
    [print("held-out accuracy (%d-fold): %.1f%%" % (i+1, cv_accuracy[i]*100)) 
     for i in range(1, len(cv_accuracy))]
    return accuracy_train, accuracy_test, cv_accuracy

def train_classifiers(base_estimator, X, y, param_name, param_vals):
    """
    Trains multiple instances of `base_estimator` on (X, y) where for each instance
    the parameter named `param_name` is set to a value from `param_vals`.
    Returns the list of trained estimators.
    """
    
    # estimators will hold the cloned estimators
    classifiers = [sklearn.base.clone(base_estimator).set_params(**{param_name:val}) 
                  for val in param_vals]

    return [classifier.fit(X, y) for classifier in classifiers]


def score_classifiers(X, y, classifiers):
    """Scores each estimator on (X, y), returning a list of scores."""
  
    return [sklearn.metrics.accuracy_score(y_true=y, 
                                           y_pred=classifier.predict(X))
                                           for classifier in classifiers]

def evaluate_score_estimators(catalogue, estimators, name_estimators, X_test, y_test,average='macro'): 
    """
    To evaluate the scores of different estimators
    
    Input: 
    catalogue : dictionary variable to get labels
    Estimator,names of estimators X_test, y_test
    Output: 
    
    Accuracy, Recall, Precision, average ROC curve 
    
    """
    scores = [evaluate_performance(catalogue, model = estimator, 
                                   data=X_test, 
                                   targets=y_test,  
                                   average=average) 
              for estimator in estimators ]
    
    scores_estimators = pd.DataFrame(np.column_stack((scores))) 
    scores_estimators.columns = name_estimators 
    scores_estimators.index = ["Accuracy", "Recall","Precision",average + "-average ROC curve"] 
    scores_estimators
        
    return scores_estimators

def get_index_max_score_test(X_test, y_test, estimator):
    # max score of validation set 
    score_test = score_classifiers(X_test, y_test, estimator)
    max_score_test = max(score_test) 
    # index with max score of training set 
    max_index_score_val = score_test.index(max_score_test) 
        
    return (max_index_score_val)
    
def get_tp_by_class(y_test, y_pred):
    """
    To get the True Positive Rates (TPR) 
    and known positives correctly classified (“true positives” TR) 
    for multiclass targets
    
    Input: 
    model: object model instance (best model)
    X_test:      testing data (features) 
    y_test:      testing multi-class targets
        
    Output: 
    return TP, TPR
    """
    conf_matrix = confusion_matrix(y_test, y_pred)

    # known negatives incorrectly classified (“false positives”)
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix) 
    #known positives incorrectly classified (“false negatives”)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix) 
    # known positives correctly classified (“true positives”)
    TP = np.diag(conf_matrix) 
    # known negatives correctly classified (“true negatives”)
    TN = conf_matrix.sum() - (FP + FN + TP)

    # recall, or true positive rate
    TPR = TP/(TP+FN)

    return (TP, TPR)

def comparison_prediction_tpr(catalogue, prediction_classifiers, y_test):
    """
    To get comparison of Predicted Recall (TPT)of each classifier
    
    Input: 
    catalogue : dictionary variable to get labels
    prediction_classifiers: array with the prediction of each classifier
        
    Output: 
    return Table of Comparison of the prediction
    """ 
    # to call "get_tp_by_class" to get the "TPR" for each class and classifier
    true_positive_rate = [get_tp_by_class(y_test, prediction)[1] 
                      for prediction in prediction_classifiers] 
    tpr_estimators = pd.DataFrame(np.column_stack((true_positive_rate))) 
    tpr_estimators.columns = ["LR","RF","SVM"] 
    tpr_estimators.index = util.catalogue.get_labels(catalogue,'KHVM').to_numpy()[ : , -1]
        
    print ( 'Comparison of the prediction ability - Predicted Recall (TPR)')
    return tpr_estimators
    
def get_predictions_classifiers(lr_classifier, rf_classifier, svm_classifier, X_test,X_tst_scaled):
    """
    To get prediction classifiers
    
    Input: 
    lr_classifier  : logistic regression classifier
    rf_classifier  : random forest classifier
    svm_classifier : svm classifier
    X_test         : test input feature
    X_tst_scaled   : scaled test input feature
        
    Output: 
    return array of prediction classifiers
    """ 
    prediction_classifiers = [lr_classifier.predict(X_tst_scaled), #Logistic Regression Prediction 
                    rf_classifier.predict(X_test), #Random Forest Prediction 
                   svm_classifier.predict(X_tst_scaled)] #SVM Prediction 
    return prediction_classifiers
    

