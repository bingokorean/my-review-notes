
#############################################################################
""" Peek at / Dimensions of Your Data / Data Type for Each Attribiute"""
#############################################################################
# View first 20 rows
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

peek = data.head(20)
print(peek)

shape = data.shape
print(shape)

types = data.dtypes # characterize each attribute using the dtypes property.
print(types)

#############################################################################
""" Descriptive Statistics """
#############################################################################
# Statistical Summary
from pandas import read_csv
from pandas import set_option
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

set_option('display.width', 100)
set_option('precision', 3)
description = data.describe()
print(description)

#############################################################################
""" Class Distribution (Classification Only) """
#############################################################################
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

class_counts = data.groupby('class').size()
print(class_counts)

>> class
>> 0 500
>> 1 268

#############################################################################
""" Correlations Between Attributes """
#############################################################################
# Pairwise Pearson correlations
from pandas import read_csv
from pandas import set_option
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

set_option('display.width', 100)
set_option('precision', 3)
correlations = data.corr(method='pearson')
print(correlations)

# it shows diagonal symmetric (correlation) matrix
# Some machine learning algorithms like linear and logistic regression can suffer poor performance if there are highly correlated attributes in your dataset.

#############################################################################
""" Skew of Univariate Distributions """
#############################################################################
# Skew for each attribute
from pandas import read_csv
filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

skew = data.skew() #  calculate the skew of each attribute
print(skew)

# Many machine learning algorithms assume a Gaussian distribution.
# An attribute has a skew may allow you to perform data preparation to correct the skew and later improve the accuracy of your models. 
# The skew result show a positive (right) or negative (left) skew. Values closer to zero show less skew.



#############################################################################
""" Univariate Plots (Histograms, Density, Box and Whisker) """
#############################################################################
from matplotlib import pyplot
from pandas import read_csv
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

### Univariate Histograms
data.hist()
pyplot.show()
# A fast way to get an idea of the distribution of each attribute is to look at histograms.
# You can quickly get a feeling for whether an attribute is Gaussian, skewed or even has an exponential distribution. 

### Univariate Density Plots
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()
# Density plots are another way of getting a quick idea of the distribution of each attribute.

### Box and Whisker Plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()
# Another useful way to review the distribution of each attribute is to use Box and Whisker Plots or boxplots for short.

#############################################################################
""" Multivariate Plots (Correlation Matrix Plot, Scatter Plot Matrix) """
#############################################################################
### Correction Matrix Plot
from matplotlib import pyplot
from pandas import read_csv
import numpy
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

correlations = data.corr()
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# plot correlation matrix
# made more generic by removing these aspects as follows
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
pyplot.show()

# some machine learning algorithms like linear and logistic regression can have poor performance if there are highly correlated input variables in your data.

### Scatterplot Matrix
data = read_csv(filename, names=names)
scatter_matrix(data)
pyplot.show()

# Attributes with structured relationships may also be correlated and good candidates for removal from your dataset.




