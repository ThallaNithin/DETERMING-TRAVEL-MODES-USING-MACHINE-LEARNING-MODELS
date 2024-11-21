import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import sklearn.preprocessing   # For scale function
from sklearn.preprocessing import label_binarize
import util.catalogue
import util.storage
import training.analysis

def print_best_parameters(model):
    """
    return the best_parameters of the model using Random Search
    """  
    print('Parameter range: ', model.best_params_)

# function to get the best 15 variables of the model
def get_plot_features_importances(model, data):  
    
    """
    To get the importances of the features
    
    Input: 
    model: object model instance
    data : original dataset

    Output: 
    return a plot of the importance of the features in descending order 
    
    """
    
    #get features Importances 
    # getting the matrix of features X's excluding the target variable KHVM (y) 
    X=data.drop('KHVM',axis=1)
    
    # get feature importance depending which module we use
    if hasattr(model,'feature_importances_'):
        f_importance = model.feature_importances_
    else:
        f_importance = model.coef_[0]
    
    #X=data[best_features]
    ## we want to print the series with the key description as a third column 
    # we get the first two columns with this: 
    importances= pd.Series(f_importance,index=X.columns) 
    
    weights = pd.Series(np.round(importances,4), index=X.columns)
    
    plt.title('Feature Importance')
    
    return weights.sort_values()[-20:].plot(kind = 'barh') # # return the 20 most important variables considered by the m
    
def get_confusion_matrix(catalogue, model,X_tst,y_tst, ax=None, xaxis_visible=True, yaxis_visible=True, xlabel_visible=True):
    
    """
    To get the confusion matrix for multiclass targets
    
    Input: 
    catalogue: dictionary variable to get labels
    model: object model instance (best model)
    X_test:      testing data (features) 
    y_test:      testing multi-class targets
        
    Output: 
    return confusion matrix for multi class targets
    
    """
    #plt.figure(figsize=(8,4))
    if ax == None: 
        fig, ax = plt.subplots(1,1)

    y_pred =  model.predict(X_tst)
    conf_matrix = confusion_matrix(y_tst, y_pred)
    sns.heatmap(conf_matrix.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels= (util.catalogue.get_labels(catalogue,'KHVM').to_numpy()[ : , 1:]).reshape(-1),
                yticklabels= (util.catalogue.get_labels(catalogue,'KHVM').to_numpy()[ : , 1:]).reshape(-1),
                ax=ax )
    ax.set_title(model.__class__.__name__, fontweight='bold')
    if xlabel_visible: ax.set_xlabel('true label')
    ax.set_ylabel('predicted label', fontweight='bold')
    ax.get_xaxis().set_visible(xaxis_visible)
    ax.get_yaxis().set_visible(yaxis_visible);

def get_classification_report(catalogue, model, X_tst, y_tst):
    
    """
    To get the cclassification_report for multiclass targets
    
    Input: 
    catalogue: dictionary variable to get labels
    model: object model instance (best model)
    X_test:      testing data (features) 
    y_test:      testing multi-class targets
        
    Output: 
    return classification_report for multi class targets
    
    """
    
    y_pred = model.predict(X_tst)
    
    print('Metrics:')
    
    print('\nAccuracy: {:.2f}\n'.format(sklearn.metrics.accuracy_score(y_tst, y_pred)))
    print('Micro Precision: {:.2f}'.format(sklearn.metrics.precision_score(y_tst, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(sklearn.metrics.recall_score(y_tst, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(sklearn.metrics.f1_score(y_tst, y_pred, average='micro')))
    
    print('Macro Precision: {:.2f}'.format(sklearn.metrics.precision_score(y_tst, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(sklearn.metrics.recall_score(y_tst, y_pred, average='macro'))) 
    print('Macro F1-score: {:.2f}\n'.format(sklearn.metrics.f1_score(y_tst, y_pred, average='macro')))
    
    print('Weighted Precision: {:.2f}'.format(sklearn.metrics.precision_score(y_tst, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(sklearn.metrics.recall_score(y_tst, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(sklearn.metrics.f1_score(y_tst, y_pred, average='weighted')))


    print('\nClassification Report\n')    
    print(classification_report(y_tst, y_pred, target_names=(util.catalogue.get_labels(catalogue, 'KHVM').to_numpy()[ : , 1:]).reshape(-1)))
    
def get_plot_roc_auc(catalogue, model, X_features, y_target):
    
    """"Function  to get the ROC Curve for multiclass data
    
    reference code: https://scikit-learn.org/0.15/auto_examples/plot_roc.html
    (However we modified it according to our project )
    Inputs: 
    catalogue: dictionary variable to get labels
    model: classifier training
    X_features : features (X_test)
    y_target: target (y_test)
    
    Ouput: Plot of ROC curve of class and AUCROC for each class and micro-average and macro-average
    ROC curve and ROC area   
    
    """
    
    # calculate the y_score (calls the classifier to predict_proba method.)
    y_score=model.predict_proba(X_features)

    #Binarize the output    
    classes = util.catalogue.get_labels(catalogue, 'KHVM').to_numpy()[ : , -2]#call get_labels function to get the labels of classes
    labels = util.catalogue.get_labels(catalogue, 'KHVM').to_numpy()[ : , -1] #call get_labels function to get the names of classes
    
    y_test_binarize = label_binarize(y_target, classes=classes)
    n_classes = y_test_binarize.shape[1]

    # Create a ROC curve and compute AUC values for multiple classes

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colors = ['blue', 'green', 'red', 'purple', 'brown', 'yellow', 'gray', 'darkorange']

    plt.figure(figsize=(10,8))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarize[:, i], y_score[:, i]) 
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.5f})'.format(labels[i], roc_auc[i]))
        print('AUC for class {0}: {1:0.5f}'.format(labels[i], auc(fpr[i], tpr[i])))
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.5f})'
             ''.format(roc_auc["micro"]),
             color="deeppink",
             linestyle=":",
             linewidth=4,
            )
    
    # Compute macro-average ROC curve and ROC area
    fpr["macro"], tpr["macro"], _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.5f})'
             ''.format(roc_auc["macro"]),
             color="navy",
             linestyle=":",
             linewidth=4,)
    

    plt.plot([0, 1], [0, 1], "k--")
    #plt.xlim([0.05, 1.0])
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for the different Travel Mode Choices (micro-average AUROC = {0:0.5f})'.format(roc_auc["micro"]), fontsize=11)
    plt.legend(loc="lower right");
    
def get_precision_recall_curve(catalogue, model, X_features, y_target):
    
    """"Function  to get the ROC Curve for multiclass data
    
    reference code: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    (However we modified it according to our project )
    Inputs: 
    catalogue: dictionary variable to get labels
    model: classifier training
    X_features : features (X_test)
    y_target: target (y_test)
    
    Ouput: Plot of precision_recall_curve and Average Precision for each class and micro-average precision-recall    
    
    """
            
    y_score=model.predict_proba(X_features)

    #Binarize the output  
    classes = util.catalogue.get_labels(catalogue,'KHVM').to_numpy()[ : , -2] #call get_labels to get the classes  
    labels_class = util.catalogue.get_labels(catalogue,'KHVM').to_numpy()[ : , -1] #call get_labels function to get the names of classes
    y_test_binarize = label_binarize(y_target, classes=classes)
    n_classes = y_test_binarize.shape[1] 
    
    # For each class 
    precision = dict() 
    recall = dict() 
    average_precision = dict() 
    
    for i in range(n_classes): 
        precision[i], recall[i], _ = precision_recall_curve(y_test_binarize[:, i], y_score[:, i]) 
        average_precision[i] = average_precision_score(y_test_binarize[:, i], y_score[:, i]) 
    
    # A "micro-average": quantifying score on all classes jointly 
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test_binarize.ravel(), y_score.ravel()
    )
    
    average_precision["micro"] = average_precision_score(y_test_binarize, y_score, average="micro")
    
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal","darkorange","brown","red"])
    
    _, ax = plt.subplots(figsize=(10, 8)) 
    
    #f_scores = np.linspace(0.2, 0.8, num=4) 
    #We can present as many iso-F1 curves in the plot of a precision-recall curve as we'd like
    # E.g., one would contain all points for which F1 equals 0.2, the second one all points for
    #which F1 equals 0.4, and so on
    
    lines, labels = [], [] 
    f_scores = np.linspace(0.2, 0.8, num=4) 
    
    lines, labels = [], [] 
    
    for f_score in f_scores: 
        x = np.linspace(0.01, 1) 
        y = f_score * x / (2 * x - f_score) #formula F-score :https://en.wikipedia.org/wiki/F-score
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02)) 
    
    display = sklearn.metrics.PrecisionRecallDisplay(
        recall=recall["micro"], 
        precision=precision["micro"], 
        average_precision=average_precision["micro"],estimator_name=None ) 
    
    display.plot(ax=ax, name="Micro-average precision-recall", color="gold") 
    
    
    for i, color in zip(range(n_classes), colors): 
        display = sklearn.metrics.PrecisionRecallDisplay( 
            recall=recall[i], 
            precision=precision[i], 
            average_precision=average_precision[i],estimator_name=None 
        )
        display.plot(ax=ax, name=f"Precision-recall for class {labels_class[i]}", color=color)
        
        # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels() 
    handles.extend([l]) 
    # iso-F1 curve contains all points in the precision/recall space whose F1 scores are the same. 
    labels.extend(["iso-f1 curves"]) 
    
    # set the legend and the axes 
    
    ax.set_xlim([0.0, 1.00]) 
    ax.set_ylim([0.0, 1.05]) 
    ax.legend(handles=handles, labels=labels, loc="best") 
    ax.set_title("Precision-Recall for the different Travel Mode Choices") 
    
    return()

def get_summary_confusion_matrices(catalogue, lr_classifier, svm_classifier, rf_classifier,X_test,X_tst_scaled,y_test):
    """
    To get all confusion matrices from every single classifier
    
    Input: 
    catalogue: dictionary variable to get labels
    lr_classifier: logistic regression classifier
    svm_classifier: svm classifier
    rf_classifier: random forest classifier
    X_test:       testing data (features) 
    X_tst_scaled: scaled testing data
    y_test:       testing multi-class targets
        
    Output: 
    return confusion matrices plot
    
    """
    fig, axes = plt.subplots(1,3, figsize=(10,6), sharey='row')

    get_confusion_matrix( catalogue, model=lr_classifier.best_estimator_,X_tst=X_tst_scaled,y_tst=y_test, 
                          ax=axes[0], xlabel_visible=False)
    
    get_confusion_matrix( catalogue, model=rf_classifier.best_estimator_,X_tst=X_test,y_tst=y_test, 
                          ax=axes[1], xlabel_visible=False, yaxis_visible=False)
    
    get_confusion_matrix( catalogue, model=svm_classifier.best_estimator_,X_tst=X_tst_scaled,y_tst=y_test, 
                          ax=axes[2], xlabel_visible=False, yaxis_visible=False)
    fig.text(0.6, 0.01, 'true label', fontsize=10, horizontalalignment='center', fontweight='bold')
    
    util.storage.save_fig("Classifiers_Confusion_Matrix");


def get_comparison_predictions_ability(catalogue, predictions_classifiers, y_test):
    """
    To get comparison predictions ability
    
    Input: 
    catalogue: dictionary variable to get labels
    predictions_classifiers: list of classifiers
    y_test:       testing multi-class targets
        
    Output: 
    return comparison predictions ability
    
    """
    return round(100*training.analysis.comparison_prediction_tpr(catalogue,predictions_classifiers,y_test),2)

def get_comparison_proportions_and_predictions(catalogue, data, prediction_classifiers,y_test):
    """
    To get comparison proportions and predictions
    
    Input: 
    catalogue: dictionary variable to get labels
    data: original dataset
    predictions_classifiers: list of classifiers
    y_test:       testing multi-class targets
        
    Output: 
    return comparison proportions and predictions
    
    """
    # actual share 
    actual_share = pd.DataFrame(data["KHVM"].value_counts() / len(data) *100)
    actual_share.columns = ["Actual Share%"]
    actual_share = actual_share.sort_index()
    actual_share.index = util.catalogue.get_labels(catalogue, 'KHVM').to_numpy()[ : , -1]
    actual_share

    # predict share
    pred_share = [100* training.analysis.get_tp_by_class(y_test, prediction)[0]/sum(training.analysis.get_tp_by_class(y_test, prediction)[0])  
                     for prediction in prediction_classifiers] 
    pred_share_classifiers = pd.DataFrame(np.column_stack((pred_share)))
    pred_share_classifiers.columns = ["Predicted Share LR%","Predicted Share RF%","Predicted Share SVM%"] 
    pred_share_classifiers.index = util.catalogue.get_labels(catalogue,'KHVM').to_numpy()[ : , -1]
    pred_share_classifiers

    # concatane actual share and predict share dataframes
    proportion_travel_choice =  pd.concat([actual_share,pred_share_classifiers], axis=1)

    # to get the error % of the predict share vs. actual share
    proportion_travel_choice["LR error%"] = 100 * proportion_travel_choice["Predicted Share LR%"] / proportion_travel_choice["Actual Share%"] - 100
    proportion_travel_choice["RF error%"] = 100 * proportion_travel_choice["Predicted Share RF%"] / proportion_travel_choice["Actual Share%"] - 100
    proportion_travel_choice["SVM error%"] = 100 * proportion_travel_choice["Predicted Share SVM%"] / proportion_travel_choice["Actual Share%"] - 100
    print("Comparison the proportion of travel choices of travelers in the overall data set and predict share with LR,RF, and SVM")
    return round(proportion_travel_choice,2)

def get_variation_percentage(score_lr, score_rf,score_svm):
    """
    To get variation percentage of estimators
    
    Input: 
    score_lr: score logistic regression
    score_rf: score random forest
    score_svm: score support vector machines
        
    Output: 
    return variation percentage of estimators
    
    """
    result = pd.concat([score_lr,score_rf,score_svm ], axis=1)
    result

    # to get the variation percentage  of each metric used to evaluate the performance of the classifiers applied
    # to compare best model SVM vs LR, and best model SVM SVM vs Random Search
    result["Var% SVM vs LR"] = 100 * result["SVM"] / result["LR"] - 100
    result["Var% SVM vs RF"] = 100 * result["SVM"] / result["RF"] - 100
    print("The variation percentage")
    return round(result,2)