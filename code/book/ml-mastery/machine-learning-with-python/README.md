# Machine Learning with Python

Jason Brownlee <br>
MACHINE LEARNING MASTERY

## Contents
_Introduction_
* [Welcome](#Welcome)

_Lessons_
* [Python Ecosystem for Machine Learning](#Python-Ecosystem-for-Machine-Learning)
* [Crash Course in Python and SciPy](#Crash-Course-in-Python-and-SciPy)
* [How to Load Machine Learning Data](#How-to-Load-Machine-Learning-Data)
* [Understand Your Data with Descriptive Statistics](#Understand-Your-Data-with-Descriptive-Statistics) (Analyze Data)
* Understand Your Data with Visualization (Analyze Data)
* Prepare Your Data for Machine Learning (Prepare Data)
* Feature Selection for Machine Learning (Prepare Data)
* Evaluate the Performance of Machine Learning Algorithms with Resampling (Evaluate Algorithms)
* Machine Learning Algorithm Performance Metrics (Evaluate Algorithms)
* Spot-Check Classification Algorithms (Evaluate Algorithms)
* Spot-Check Regression Algorithms (Evaluate Algorithms)
* Compare Machine Learning Algorithms - Model Selection (Evaluate Algorithms)
* Automate Machine Learning Workflows with Pipelines (Evaluate Algorithms)
* Improve Performance with Algorithm Tuning (Improve Results)
* Save and Load Machine Learning Models (Present Results)

_Projects_
* Predictive Modeling Project Template
* Your First Machine Learning Project in Python Step-by-Step
...


## Welcome

이 책은 predictive modeling이라는 machine learning sub-field에 초점이 맞춰져 있다. 이 field는 현재 industry에서 매우 유용하게 활용되고 있다. 
* Unlike statistics, where models are used to understand data, predictive modeling is laser focused on developing models that make the most accurate predictions at the expense of explaining why predictions are made.
* Unlike the broader ﬁeld of machine learning that could feasibly be used with data in any format, predictive modeling is primarily focused on tabular data (e.g. tables of numbers like in a spreadsheet).

 A predictive modeling machine learning project can be broken down into 6 top-level tasks:
1. Deﬁne Problem: Investigate and characterize the problem in order to better understand the goals of the project.
2. Analyze Data: Use descriptive statistics and visualization to better understand the data you have available.
3. Prepare Data: Use data transforms in order to better expose the structure of the prediction problem to modeling algorithms.
4. Evaluate Algorithms: Design a test harness to evaluate a number of standard algorithms on the data and select the top few to investigate further.
5. Improve Results: Use algorithm tuning and ensemble methods to get the most out of well-performing algorithms on your data.
6. Present Results: Finalize the model, make predictions and present results.

## Python Ecosystem for Machine Learning

SciPy is an ecosystem of Python libraries for mathematics, science and engineering.
* NumPy: A foundation for SciPy that allows you to eﬃciently work with data in arrays.
* Matplotlib: Allows you to create 2D charts and plots from data.
* Pandas: Tools and data structures to organize and analyze your data.

Scikit-learn library is how you can develop and practice machine learning in Python. 
* It is built upon and requires the SciPy ecosystem. The name scikit suggests that it is a SciPy plug-in or toolkit.
* The focus of the library is machine learning algorithms for classiﬁcation, regression, clustering and more.
* It also provides tools for related tasks such as evaluating models, tuning parameters and pre-processing data. 

## Crash Course in Python and SciPy

Matplotlib Crash Course

```
# basic line plot 
import matplotlib.pyplot as plt 
import numpy 
myarray = numpy.array([1, 2, 3]) 
plt.plot(myarray) 
plt.xlabel('some x axis') 
plt.ylabel('some y axis') 
plt.show()

# basic scatter plot 
import matplotlib.pyplot as plt 
import numpy 
x = numpy.array([1, 2, 3]) 
y = numpy.array([2, 4, 6]) 
plt.scatter(x,y) 
plt.xlabel('some x axis') 
plt.ylabel('some y axis') 
plt.show()
```

Pandas Crash Course

```
# Series - a one dimensional array where the rows and columns can be labeled.
import numpy 
import pandas 
myarray = numpy.array([1, 2, 3]) 
rownames = ['a', 'b', 'c'] 
myseries = pandas.Series(myarray, index=rownames)
print(myseries)
>>>
a  1
b  2
c  3
print(myseries[0]) 
print(myseries['a'])
>>>
1
1

# Dataframe - a multi-dimensional array where the rows and the columns can be labeled.
import numpy 
import pandas 
myarray = numpy.array([[1, 2, 3], [4, 5, 6]]) 
rownames = ['a', 'b'] 
colnames = ['one', 'two', 'three'] 
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames) 
print(mydataframe)
>>>
  one two three 
a   1   2   3 
b   4   5   6
print("method 1:") 
print("one column: %s") % mydataframe['one'] 
print("method 2:") 
print("one column: %s") % mydataframe.one
>>>
method 1: 
a   1 
b   4 
method 2: 
a   1 
b   4
```

## How to Load Machine Learning Data

기계학습 프로젝트에서 csv 파일을 많이 사용합니다. 다양한 종류의 csv 파일 로드 방법이 있습니다.

CSV파일 고려사항

* File Header - 데이터 column 정보임. csv파일 로딩할 때 있는지 여부 명시할 것.
* Comments - csv파일에서 hash(#)로 시작하는 line은 comment임. csv파일 로딩할 때 comment character 명시할 수 있음.
* Delimiter - 데이터를 구분하기 위한 character. 보통 comma(,) 사용. Tab이나 whitespace 등으로 지정할 수 있음.
* Quotes - default quote character는 double quatation mark(")임. 다른 character도 사용 가능.

Load CSV Files with the Python Standard Libarary

```
# Load CSV Using Python Standard Library 
import csv 
import numpy 
filename = 'pima-indians-diabetes.data.csv' 
raw_data = open(filename, 'rb') 
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)      # header는 로드 파일에 없다고 가정.
x = list(reader) 
data = numpy.array(x).astype('float') 
print(data.shape)
>>>
(768, 9)
```

Load CSV Files with NumPy

```
# Load CSV using NumPy 
from numpy import loadtxt 
filename = 'pima-indians-diabetes.data.csv' 
raw_data = open(filename, 'rb') 
data = loadtxt(raw_data, delimiter=",")     # loadtxt()는 no header row 이고, 모든 data는 똑같은 format이어야 함.
print(data.shape)
>>>
(768, 9)

# Load CSV from URL using NumPy 
from numpy import loadtxt 
from urllib import urlopen 
url = 'https://goo.gl/vhm1eU' 
raw_data = urlopen(url) 
dataset = loadtxt(raw_data, delimiter=",") 
print(data.shape)
>>>
(768, 9)
```

Load CSV Files with Pandas

pandas의 rade_csv 함수는 매우 flexible하기 때문에 머신러닝 프로젝트에서 데이터를 로딩할 때 이 방식을 추천한다. 이 함수는 pandas.DataFrame을 리턴하는데 여기에서 데이터를 summarizing과 plotting을 곧바로 할 수 있다.

```
# Load CSV using Pandas 
from pandas import read_csv 
filename = 'pima-indians-diabetes.data.csv' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names) 
print(data.shape)
>>>
(768, 9)

# Load CSV using Pandas 
from URL from pandas 
import read_csv 
url = 'https://goo.gl/vhm1eU' 
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(url, names=names) 
print(data.shape)
>>>
(768, 9)
```

## Understand Your Data with Descriptive Statistics

























A predictive modeling machine learning project can be broken down into 6 top-level tasks:

   * Level1. Define Problem [[**.py**](https://github.com/gritmind/review-code/blob/master/blog/machine-learning-with-python/contents/define-problem.py)]
      * Ch4. How to load machine learning data (e.g. CSV, url)
   * Level2. Analyze Data [[**.py**](https://github.com/gritmind/review-code/blob/master/blog/machine-learning-with-python/contents/analyze-data.py)]
      * Ch5. Summary dataset (e.g. peek, dimensions, type, class distribution, summary, correlations, skewness) 
      * Ch6. Visualize dataset (e.g. Univariate Plots (e.g. Histograms, Density, Box and Whisker), Multivariate Plots (e.g. Correlation Matrix Plot, Scatter Plot Matrix)
   * Level3. Prepare Data [[**.py**](https://github.com/gritmind/review-code/blob/master/blog/machine-learning-with-python/contents/prepare-data.py)]
      * Ch7. Data Transforms (e.g. Rescale, Standardize, Normalize, Binarize)
      * Ch8. Feature Selection (e.g. Univariate Selection, Recursive Feature Elimination, Principle Component Analysis, Feature Importance)
   * Level4. Evaluate Algorithms
      * Ch9. Evaluation with Resampling Methods
      * Ch10. Evaluation Metrics
      * Ch11. Spot-Check Classification Algorithms
      * Ch12. Spot-Check Regression Algorithms
      * Ch13. Model Selection
      * Ch14. Pipelines
   * Level5. Improve Results
      * Ch15. Ensemble Methods
      * Ch16. Parameter Tuning
   * Level6. Present Results
      * Ch17. Save and Load Models


## Projects
* Regression Machine Learning Case Study Project
* Binary Classification Machine Learning Case Study Project
* More Predictive Modeling Projects
