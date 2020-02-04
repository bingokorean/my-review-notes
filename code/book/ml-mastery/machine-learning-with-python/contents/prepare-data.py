
########################################################################
""" Data Transforms (Rescale, Standardize, Normalize, Binarize) """
########################################################################
# Many machine learning algorithms make assumptions about your data.
# It is often a very good idea to prepare your data in such way to best expose the structure of the problem to the machine learning algorithms that you intend to use.
# A difficulty is that different algorithms make different assumptions about your data and may require different transforms.

from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

###
### Rescale data (between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])
[[ 0.353 0.744 0.59 0.354 0. 0.501 0.234 0.483]
[ 0.059 0.427 0.541 0.293 0. 0.396 0.117 0.167]
[ 0.471 0.92 0.525 0. 0. 0.347 0.254 0.183]
[ 0.059 0.447 0.541 0.232 0.111 0.419 0.038 0. ]
[ 0. 0.688 0.328 0.354 0.199 0.642 0.944 0.2 ]]
# This is useful for optimization algorithms in used in the core of machine learning algorithms like gradient descent.
# It is also useful for algorithms that weight inputs like regression and neural networks and algorithms that use distance measures like k-Nearest Neighbors.

###
###  Standardize Data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])
[[ 0.64 0.848 0.15 0.907 -0.693 0.204 0.468 1.426]
[-0.845 -1.123 -0.161 0.531 -0.693 -0.684 -0.365 -0.191]
[ 1.234 1.944 -0.264 -1.288 -0.693 -1.103 0.604 -0.106]
[-0.845 -0.998 -0.161 0.155 0.123 -0.494 -0.921 -1.042]
[-1.142 0.504 -1.505 0.907 0.766 1.41 5.485 -0.02 ]]
# Standardization is a useful technique to transform attributes with a Gaussian distribution and differing means and standard deviations to a standard Gaussian distribution with a mean of 0 and a standard deviation of 1.
# It is most suitable for techniques that assume a Gaussian distribution in the input variables and work better with rescaled data, such as linear regression, logistic regression and linear discriminate analysis.

###
###  Normalize Data
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(normalizedX[0:5,:])
[[ 0.034 0.828 0.403 0.196 0. 0.188 0.004 0.28 ]
[ 0.008 0.716 0.556 0.244 0. 0.224 0.003 0.261]
[ 0.04 0.924 0.323 0. 0. 0.118 0.003 0.162]
[ 0.007 0.588 0.436 0.152 0.622 0.186 0.001 0.139]
[ 0. 0.596 0.174 0.152 0.731 0.188 0.01 0.144]]
# Normalizing in scikit-learn refers to rescaling each observation (row) to have a length of 1 (called a unit norm or a vector with the length of 1 in linear algebra). 
# This pre-processing method can be useful for sparse datasets (lots of zeros) with attributes of varying scales when using algorithms that weight input values such as neural networks and algorithms that use distance measures such as k-Nearest Neighbors. 

###
###  Binarize Data (Make Binary)
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(binaryX[0:5,:])
[[ 1. 1. 1. 1. 0. 1. 1. 1.]
[ 1. 1. 1. 1. 0. 1. 1. 1.]
[ 1. 1. 1. 0. 0. 1. 1. 1.]
[ 1. 1. 1. 1. 1. 1. 1. 1.]
[ 0. 1. 1. 1. 1. 1. 1. 1.]]
# It can be useful when you have probabilities that you want to make crisp values.
# It is also useful when feature engineering and you want to add new features that indicate something meaningful.

########################################################################
""" Feature Selection () """
########################################################################
# Feature selection is a process where you automatically select those features in your data that contribute most to the prediction variable or output in which you are interested.
# Irrelevant or partially relevant features can negatively impact model performance, especially linear algorithms like linear and logistic regression.
# by doing Feature Selection, we can get benefits:
# (1) Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
# (2) Improves Accuracy: Less misleading data means modeling accuracy improves.
# (3) Reduces Training Time: Less data means that algorithms train faster.
from pandas import read_csv
from numpy import set_printoptions
# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

###
### Univariate Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])

# You can see the scores for each attribute and the 4 attributes chosen (those with the highest scores): plas, test, mass and age.
# Statistical tests can be used to select those features that have the strongest relationship with the output variable.
# We use the chi-squared (chi2) statistical test for non-negative features to select 4 of the best features from the Pima Indians onset of diabetes dataset

###
### Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_

# You can see that RFE chose the top 3 features as preg, mass and pedi. These are marked True in the support array and marked with a choice 1 in the ranking array
# The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those attributes that remain.
# It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.

###
### Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=3) # select 3 principal components
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)

# You can see that the transformed dataset (3 principal components) bare little resemblance to the source data.
# Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form. (Generally this is called a data reduction technique)
# A property of PCA is that you can choose the number of dimensions or principal components in the transformed result.

###
###  Feature Importance
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

# You can see that we are given an importance score for each attribute where the larger the score, the more important the attribute. 
# Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.
# In the example below we construct a ExtraTreesClassifier classifier

