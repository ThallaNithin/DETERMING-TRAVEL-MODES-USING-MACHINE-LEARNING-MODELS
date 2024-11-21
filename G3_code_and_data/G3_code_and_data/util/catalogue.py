import numpy as np

# Some helper functions to get data from the catalogue 

# Function to get a label given a category and a value
def get_value_catalogue(catalogue, variable, value):
    temp = catalogue[catalogue["Variable"] == variable]
    return(temp[temp["Value"]==value])

# function that return all labels and values of a categorical variable
def get_labels(catalogue,variable):
    temp = catalogue[catalogue["Variable"] == variable]
    return(temp[['Value','Label']])

# Function to get the description label
def get_var_description(var_description,variable):
    temp = var_description[var_description["Variable"] == variable]
    return(temp[['Var_description']])

# function that return a label and a value of a categorical variable
def get_value_label(catalogue, variable, value):
    return(get_value_catalogue(catalogue, variable, value)["Label"][0])

def get_value_measure_level(catalogue, variable, value):
    return(get_value_catalogue(catalogue, variable, value)["Measurement_level"][0])

#get_var_description('KHVM')
def get_var_description(catalogue,variable):
    return(np.unique(catalogue[catalogue["Variable"] == variable]["Var_description"])[0])

def get_var_descriptions(var_description,variables_array):
    return [get_var_description(var_description,variable) for variable in variables_array ]