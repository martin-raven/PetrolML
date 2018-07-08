import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
df = pd.read_csv('DelhiPrice.csv')
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()
# print(Real_Price)
prediction_days = 30
df_train= Real_Price[:len(Real_Price)-prediction_days]
df_test= Real_Price[len(Real_Price)-prediction_days:]
# Data preprocess
training_set = df_train.values
# print("Training data",df_train.keys)
training_set = np.reshape(training_set, (len(training_set), 1))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
# print("Training set",training_set)
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
# print("Train data X",X_train)
y_train = training_set[1:len(training_set)]
# print("Train data y",y_train)
X_train = np.reshape(X_train, (len(X_train), 1, 1))
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 48, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 180, epochs = 15000)
# Making the predictions
test_set = df_test.values
print("test Values",test_set)
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
print(sc.inverse_transform(inputs),type(inputs))
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_Petrol_price = regressor.predict(inputs)
predicted_Petrol_price = sc.inverse_transform(predicted_Petrol_price)
print("Predicted Price",predicted_Petrol_price)
# Visualising the results
plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()  
plt.plot(test_set, color = 'red', label = 'Real Petrol Price')
plt.plot(predicted_Petrol_price, color = 'blue', label = 'Predicted Petrol Price')
plt.title('Petrol Price Prediction', fontsize=40)
df_test = df_test.reset_index()
x=df_test.index
labels = df_test['date']
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(18)
plt.xlabel('Time', fontsize=40)
plt.ylabel('Petrol Price(INR)', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()