## Considerations when loading CSV data
# File Header (i.e. names to each column of data)
# Comments (indicated by a hash(#) at start of a line)
# Delimiter (separates values (e.g. comma (,) character))
# Quotes (default quote character is the double quatation marks character)

#####################################################################
""" Load CSV Files with the Python Standard Library """
#####################################################################
import csv, numpy 
filename = 'pima-indians-diabetes.data.csv' # all fields in this dataset are numeric and there is no header line
raw_data = open(filename,'rb')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')
print(data.shape)

>> (768, 9)

#####################################################################
""" Load CSV Files with Numpy """
#####################################################################
from numpy import loadtxt
filename ='pima-indians-diabetes.data.csv'
raw_data = open(filename,'rb')
data = loadtxt(raw_data, delimiter=",")
print(data.shape)

>> (768, 9)

# Load CSV from URL using NumPy
from numpy import loadtxt
from urllib import urlopen
url ='https://goo.gl/vhm1eU'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=",")
print(dataset.shape)

>> (768, 9)

#####################################################################
""" Load CSV Files with Pandas (Recommendation) """ 
#####################################################################
#  The function returns a pandas.DataFrame7that you can immediately start summarizingand plotting.
from pandas import read_csv
filename ='pima-indians-diabetes.data.csv' # or url ='https://goo.gl/vhm1eU'
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = read_csv(filename, names=names) # or url 
print(data.shape)

>> (768, 9)







