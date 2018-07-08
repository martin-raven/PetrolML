import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.neighbors import KNeighborsClassifier
import json,pickle

# load the boston dataset
file_Name = "DataSet"
fileObject = open(file_Name,'rb')

Data=pickle.load(fileObject)
JSON=json.loads(Data)
X=[]
y=[]
DataSet=dict(JSON)
for i in range(1,DataSet['Totaldata']+1):
	X.append(DataSet[str(i)]['dateFloat'])
	y.append(DataSet[str(i)]['Delhi'])
# splitting X and y into training and testing sets
X= np.asarray(X).reshape(-1, 1).astype(float)
y= np.asarray(y).reshape(-1, 1).astype(float)
# print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)
# create linear regression object
model = linear_model.LinearRegression()
# model=KNeighborsClassifier(n_neighbors=3)
 
# train the model using the training sets
model.fit(X_train, y_train)
# regression coefficients
print('Coefficients: \n', model.coef_)
 
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(model.score(X_test, y_test)))
 
# plot for residual error
 
## setting plot style
plt.style.use('fivethirtyeight')
plt.scatter(X_train,y_train,color = "green", s = 10, label = 'real train data')
plt.scatter(X_train,model.predict(X_train),color = "red", s = 10, label = 'prediction training data')
plt.scatter(X_test,model.predict(X_test),color = "red", s = 10, label = 'prediction data')
plt.scatter(X_test,y_test,color = "green", s = 10, label = 'real test data')
## plotting residual errors in training data
# plt.scatter(model.predict(X_train), model.predict(X_train) - y_train,
#             color = "green", s = 10, label = 'Train data')
 
# ## plotting residual errors in test data
# plt.scatter(model.predict(X_test), model.predict(X_test) - y_test,
#             color = "red", s = 10, label = 'Test data')
# plt.scatter(y,X,color = "cyan", s = 10, label = 'OriginalData')
## plotting line for zero residual error
# plt.hlines(y = 65, xmin =1.5, xmax = 1.52, linewidth = 0.01)
 
## plotting legend
plt.legend(loc = 'upper right')
 
## plot title
plt.title("Data Plot")
 
## function to show plot
plt.show() 