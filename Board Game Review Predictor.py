import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

#Load the data
games = pd.read_csv('C:\Machine learning\Board Game\games.csv')

#print the names of the columns
#print(games.columns)

#print(games.shape)

#make a histogram of all the ratings in the average_rating column
#plt.hist(games['average_rating'])
#plt.show()

#print the first row of all games with zero ratings
#print(games[games['average_rating']==0].iloc[0])

#print the first row of games with rating greater than zero
#print(games[games['average_rating']>0].iloc[0])

#remove any rows without any reviews
games = games[games["users_rated"]>0]

#remove any rows with missing vlues
games = games.dropna(axis=0)

#make a histogram
plt.hist(games["average_rating"])
plt.show()

#correlation matrix
correlation_mat = games.corr()
fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_mat,vmax=.8,square=True)
plt.show()

#Get all of the columns from df
columns = games.columns.tolist()

#Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ['bayes_average_rating','average_rating','type','name','id']]

#Store the variable we will be predicting on
target = 'average_rating'

#generate training and testing datasets
from sklearn import model_selection
from sklearn.model_selection import train_test_split

train,test = model_selection.train_test_split(games,test_size = 0.2,random_state = 1)

print(train.shape)
print(test.shape)

#Import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#initialise the model class
clf = LinearRegression()

#fit the model to our training data
clf.fit(train[columns],train[target])

#generate predictions for testing data
predictions = clf.predict(test[columns])

#Compute error between our test prediction and actual values
print('Linear Regression Error',mean_squared_error(predictions, test[target]))

#Import the random forest model
from sklearn.ensemble import RandomForestRegressor

#Initialise the model
classifier = RandomForestRegressor(n_estimators=100,min_samples_leaf=10,random_state=1)

#fit the data
classifier.fit(train[columns],train[target])

#predict
predictions = classifier.predict(test[columns])

#Error
print('Random Forest Error',mean_squared_error(predictions,test[target]))

print(test[columns].iloc[0])

#Make predictions with both models
rating_LR = clf.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_RFR = classifier.predict(test[columns].iloc[0].values.reshape(1,-1))

#Print the predictions
print(rating_LR)
print(rating_RFR)
print(test[target].iloc[0])
