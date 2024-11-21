import numpy as np

# Function to replace whitespace to Nan value
def replaceWhitespaceToNan(data):
    return data.replace(r'^\s*$', np.nan, regex=True);

# Function that replace missing values to the value most used (mode)
def replaceMissingValuesToMode(data,field):
    data[field].fillna(data[field].mode()[0], inplace=True);