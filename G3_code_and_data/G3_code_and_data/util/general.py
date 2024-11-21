import sklearn

# Function that normalize training and test input features X
def normalize_x(X_train,X_test):
    scaled = sklearn.preprocessing.StandardScaler().fit(X_train)
    X_train_scaled  = scaled.transform(X_train)
    X_test_scaled = scaled.transform(X_test)
    return X_train_scaled,X_test_scaled