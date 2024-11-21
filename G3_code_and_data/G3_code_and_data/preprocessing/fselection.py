import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import util.storage

def get_features_target(data):
    """
    Input parameters: 
    data: original dataset
    Output: Return features and targets
    """
    df_y = data["KHVM"] ##target column i.e travel mode choices of travellers: "1.Car as driver", "2.Car as passenger", ...
    df_X = data.drop("KHVM", axis=1)#independent columns
    return df_X, df_y;
def get_best_features(df_X, df_y):
    """
    Input parameters: 
    df_X: input features
    df_y: targets
    Output: Return best features
    """
    #apply SelectKBest class to extract top 25 best features
    bestfeatures = sklearn.feature_selection.SelectKBest(score_func=chi2, k=25)
    #Run score function on (X, y) and get the appropriate features.
    fit = bestfeatures.fit(df_X,df_y)
    #Scores of features.
    scores = pd.DataFrame(fit.scores_)
    
    columns = pd.DataFrame(df_X.columns)
    #concat two dataframes for better visualization 
    feature_Scores = pd.concat([columns,scores],axis=1)
    feature_Scores.columns = ['Feature','Score']  #naming the dataframe columns
    #pd.DataFrame(feature_Scores.nlargest(25,'Score'))  #print 25th best features
    return feature_Scores
def show_correlation_matrix(data):
    """
    Input parameters: 
    data: original dataset
    Output: Return correlation matrix of all features
    """
    #get correlations of each features in dataset
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(30,30))
    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    util.storage.save_fig("Correlation_Matrix_TargetvsFeatures") #saving the plot to disk
    plt.show();