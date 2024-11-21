import numpy as np
import util.cleansing

def cleaning_data(data):
    """
    Input parameters: 
    data: original dataset
    Output: original dataset cleaned
    """
    # Set the missing values in these categories to its modes
    util.cleansing.replaceMissingValuesToMode(data,'KREISDUUR')
    util.cleansing.replaceMissingValuesToMode(data,'PARKEERKOSTEN')

    #fixing the data types
    data["KREISDUUR"] = data["KREISDUUR"].astype(np.int64)
    data["PARKEERKOSTEN"] = data["PARKEERKOSTEN"].astype(np.int64)