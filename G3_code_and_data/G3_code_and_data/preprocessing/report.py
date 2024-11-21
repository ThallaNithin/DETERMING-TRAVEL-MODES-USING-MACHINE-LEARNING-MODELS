import util.catalogue
import util.plotting

def plot_travel_mode_pie(catalogue, data):
    #call get_labels function to get the value and labels of interest variable 'KHVM'
    value_labels = util.catalogue.get_labels(catalogue,'KHVM')

    v = value_labels[["Value"]]
    value  = tuple(list(v ['Value'])) #values (1,2,3,...)

    l = value_labels[["Label"]]
    labels  = tuple(list(l['Label'])) #labels ("Car as driver", "Car as passenger", ...)

    # variable that counts for each mode of transport that individuals use
    sizes = [data.KHVM[data['KHVM']==v].count() for v in value if data.KHVM[data['KHVM']==v].count()>0]
 
    # only "explode" the 1st slice (i.e. 'Car as driver')
    explode = (0.1,0,0,0,0,0,0,0)
    util.plotting.show_pie_chart('Proportion of travel mode choices of travellers',
                                 labels[0:8],sizes,explode,(9, 5))