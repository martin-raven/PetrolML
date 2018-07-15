import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# Importing the Keras libraries and packages
from keras.models import Sequential,model_from_json
# from keras.layers import Dense
# from keras.layers import LSTM
# from matplotlib import pyplot as plt
# pickle.dump( favorite_color, open( "save.p", "wb" ) )
# favorite_color = pickle.load( open( "save.p", "rb" ) )
# sc=MinMaxScaler()
df = pd.read_csv('DelhiPrice.csv')
df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()
# print(Real_Price)
# prediction_days = 1
df_train = Real_Price[len(Real_Price) - 2:]
# sc.fit(df_train)
# df_test = Real_Price[len(Real_Price) - prediction_days:]
# Data preprocess
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
print("Training data",training_set)
sc = pickle.load( open( "MinMaxScaler.dat", "rb" ) )
# print("Training set",training_set)
# sc.fit(training_set)
training_set = sc.transform(training_set)
print(training_set)
X_train = training_set[0]
# print("Train data X",X_train)
y_train = training_set[1]
# print("Train data y",y_train)
X_train = np.reshape(X_train, (len(X_train), 1, 1))
print(X_train,y_train)
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# Fitting the RNN to the Training set
regressor = loaded_model
# Initialising the RNN
# regressor = Sequential()

# # Adding the input layer and the LSTM layer
# regressor.add(LSTM(units=48, activation='sigmoid', input_shape=(None, 1)))

# # Adding the output layer
# regressor.add(Dense(units=1))

# # Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=12000)
# Making the predictions
# test_set = df_test.values
# print("test Values", test_set)
# inputs = np.reshape(test_set, (len(test_set), 1))
# inputs = sc.transform(inputs)
# print(sc.inverse_transform(inputs), type(inputs))
# inputs = np.reshape(inputs, (len(inputs), 1, 1))
# predicted_Petrol_price = regressor.predict(inputs)
# predicted_Petrol_price = sc.inverse_transform(predicted_Petrol_price)
# print("Predicted Price", predicted_Petrol_price)
# serialize model to JSON
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")
