import numpy as np 
import pandas as pd 
#from matplotlib import pyplot as plt
import tensorflow as tf
from keras.backend import tensorflow_backend as K
import time
from keras.models import Sequential,model_from_json
from keras.layers import Dense
from keras.layers import LSTM,Dropout,Activation
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

def build_model(layers):
    model = Sequential()

    model.add(LSTM(units=layers[0],
        input_shape=(None, 1),
        # input_dim=layers[0],
        # output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        layers[1],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        units=layers[2]
        # output_dim=layers[3]
        ))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print ("Compilation Time : ", time.time() - start)
    return model
# Fitting the RNN to the Training set
regressor=build_model([1,2,1])
regressor.fit(X_train, y_train, batch_size = 150, epochs = 12000)
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
# serialize model to JSON
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")
