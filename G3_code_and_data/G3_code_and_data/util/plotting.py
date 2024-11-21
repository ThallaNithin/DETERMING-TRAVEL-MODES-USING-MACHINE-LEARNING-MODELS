import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

# Function that plots a pie chart
def show_pie_chart(title,labels,sizes,explode,figsize):
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title(title, size = 20)
    plt.tight_layout()
    plt.show();
    
# Function that shows the first n values of a Panda DataFrame
def show_first_n_dataframe(dataframe,n, field):
    return pd.DataFrame(dataframe.nlargest(n,field))   