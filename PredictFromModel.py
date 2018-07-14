import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import time
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted
sc = MinMaxScaler()
df = pd.read_csv('DelhiPrice.csv')
df['date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
group = df.groupby('date')
Real_Price = group['Weighted_Price'].mean()
# print(Real_Price)
prediction_days = 1
df_train= Real_Price[:len(Real_Price)-prediction_days]
df_test= Real_Price[len(Real_Price)-prediction_days:]
training_set = df_train.values
print("Training set",training_set)
training_set = np.reshape(training_set, (len(training_set), 1))
training_set = sc.fit_transform(training_set)
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# Fitting the RNN to the Training set
regressor=loaded_model
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
# serialize model to JSON
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")


