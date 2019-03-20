import sys
import numpy
import pandas
import matplotlib
import scipy
import seaborn
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Sklearn: {}'.format(sklearn.__version__))

#load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load data
data = pd.read_csv('C:\Machine learning\Credit card fraud\creditcard.csv')

#explore the dataset
print(data.columns)

print(data.shape)

print(data.describe())

data1 = data.sample(frac=0.1,random_state=1)
print(data1.shape)

#plot histogram
data1.hist(figsize = (20,20))
plt.show()

data = data1
#determine frauds
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud)/float(len(valid))
print(outlier_fraction)

print('Fraud cases: {}'.format(len(fraud)))
print('Valid cases: {}'.format(len(valid)))

corrmat = data.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax = .8,square = True)
plt.show()

#get columns
columns = data.columns.tolist()

#filter columns
columns = [c for c in columns if c not in ["Class"]]

# store the variable we are predicting
target = "Class"

X = data[columns]
Y = data[target]

print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state
state = 1

#define outlier detection methods
classifiers = {
    "Isolation Forest":IsolationForest(max_samples=len(X),
                                      contamination = outlier_fraction,
                                      random_state = state),
    "Local Outlier Factor":LocalOutlierFactor(
    n_neighbors = 20,
    contamination = outlier_fraction)
}

#fit the model
n_outliers = len(fraud)

for i, (clf_name,clf) in enumerate(classifiers.items()):

    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_

    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    #Reshape the prediction 0 is for valid, 1 is for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred!=Y).sum()

    #Run classification metric
    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))

