from flask import Flask, request, jsonify
import json
import numpy as np
import pandas as pd
import time
import datetime as dt
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

app = Flask(__name__)


@app.route('/')
def home():
    return "Howdy! It's up!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        predicted_price = str(predict_petrol_price())
        print("Tomorrow's price", predicted_price)
        return jsonify(Price=predicted_price)
    else:
        return "Not authorised."


@app.route('/getprice', methods=['GET', 'POST'])
def get_price():
    if request.method == 'GET':
        df = pd.read_csv('static/dataset/DelhiPrice.csv')
        return jsonify(
            Price=str(df['Weighted_Price'][0])
        )
    else:
        return "Not authorised."


# @app.route('/getlast7days', methods=['GET', 'POST'])
# def get_price_of_last_7_days():
# 	if request.method == 'GET':
# 		df = pd.read_csv('static/dataset/DelhiPrice.csv')
# 		items = [{},{},{},{},{},{},{}]
# 		today = dt.datetime.now()
# 		for i in reversed(range(7)):
# 			items[6-i] = {"date": {"day": (today - dt.timedelta(days=i)).day,
# 			"month": (today - dt.timedelta(days=i)).month, "year": (today - dt.timedelta(days=i)).year	},
# 			"price": str(df['Weighted_Price'][i])}
# 		return jsonify(Items=items)
# 	else:
# 		return "Not authorised."

@app.route('/getlast7days', methods=['GET', 'POST'])
def get_price_of_last_7_days():
    if request.method == 'GET':
        df = pd.read_csv('static/dataset/DelhiPrice.csv')
        items = [{}, {}, {}, {}, {}, {}, {}]
        today = dt.datetime.now()
        for i in reversed(range(7)):
            items[6 - i] = {"date": (today - dt.timedelta(days=i)).strftime("%d/%m/%Y"),
                            "price": str(df['Weighted_Price'][i])}
        return jsonify(Items=items)
    else:
        return "Not authorised."

# @app.route('/predictforweek', methods=['GET', 'POST'])
# def get_predictions_for_a_week():
# 	if request.method == 'GET':
# 		week = predict_petrol_price_week()
# 		items = [{},{},{},{},{},{},{}]
# 		today = dt.datetime.now()
# 		for i in range(7):
# 			items[i] = {"date": {"day": (today + dt.timedelta(days=i+1)).day,
# 			"month": (today + dt.timedelta(days=i+1)).month, "year": (today + dt.timedelta(days=i+1)).year	},
# 			"price": str(week[i]) }
# 		return jsonify(Items=items)
# 	else:
# 		return "Not authorised."


@app.route('/predictforweek', methods=['GET', 'POST'])
def get_predictions_for_a_week():
    if request.method == 'GET':
        week = predict_petrol_price_week()
        items = [{}, {}, {}, {}, {}, {}, {}]
        today = dt.datetime.now()
        for i in range(7):
            items[i] = {"date": (today + dt.timedelta(days=i + 1)).strftime("%d/%m/%Y"),
                        "price": str(week[i])}
        return jsonify(Items=items)
    else:
        return "Not authorised."


@app.route('/uploadmodeljson', methods=['GET', 'POST'])
def upload_model_json():
    if request.method == 'POST':
        uploaded_json = request.get_json()
        with open('static/models/model.json', 'w') as outfile:
            json.dump(uploaded_json, outfile)
        return "200"

    return "Error"


def predict_petrol_price():
    df = pd.read_csv('static/dataset/DelhiPrice.csv')
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    group = df.groupby('date')
    Real_Price = group['Weighted_Price'].mean()
    prediction_days = 30
    df_train = Real_Price[:len(Real_Price) - prediction_days]
    df_test = Real_Price[len(Real_Price) - prediction_days:]
    training_set = df_train.values
    training_set = np.reshape(training_set, (len(training_set), 1))
    training_set = sc.fit_transform(training_set)
    # load json and create model
    json_file = open('static/models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("static/models/model.h5")
    print("Loaded model from disk")
    # Fitting the RNN to the Training set
    regressor = loaded_model
    # Making the predictions
    test_set = df_test.values
    print("test Values", test_set)
    inputs = np.reshape(test_set, (len(test_set), 1))
    inputs = sc.transform(inputs)
    print(sc.inverse_transform(inputs), type(inputs))
    inputs = np.reshape(inputs, (len(inputs), 1, 1))
    predicted_Petrol_price = regressor.predict(inputs)
    predicted_Petrol_price = sc.inverse_transform(predicted_Petrol_price)
    print("Predicted Price\n", predicted_Petrol_price)
    predicted_Petrol_price = predicted_Petrol_price.tolist()
    print(predicted_Petrol_price[-1][0])
    return predicted_Petrol_price[-1][0]


def predict_petrol_price_week():
    week = []
    df = pd.read_csv('static/dataset/DelhiPrice.csv')
    df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    group = df.groupby('date')
    Real_Price = group['Weighted_Price'].mean()
    prediction_days = 30
    df_train = Real_Price[:len(Real_Price) - prediction_days]
    df_test = Real_Price[len(Real_Price) - prediction_days:]
    training_set = df_train.values
    training_set = np.reshape(training_set, (len(training_set), 1))
    training_set = sc.fit_transform(training_set)
    # load json and create model
    json_file = open('static/models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("static/models/model.h5")
    print("Loaded model from disk")
    # Fitting the RNN to the Training set
    regressor = loaded_model
    # Making the predictions
    test_set = df_test.values
    print("test Values", test_set)
    inputs = np.reshape(test_set, (len(test_set), 1))
    inputs = sc.transform(inputs)
    print(sc.inverse_transform(inputs), type(inputs))
    inputs = np.reshape(inputs, (len(inputs), 1, 1))
    predicted_Petrol_price = regressor.predict(inputs)
    predicted_Petrol_price = sc.inverse_transform(predicted_Petrol_price)
    print("Predicted Price\n", predicted_Petrol_price)
    price = predicted_Petrol_price[prediction_days - 1]
    print("DEBUG:\n", price)
    for i in range(7):
        # print("DEBUG:",price[0],price,"\n")
        print("DEBUG:\n", price)
        try:
            week.append(price[0][0])
            price = predict_point_by_point(regressor, price)
        except:
            week.append(price[0])
            price = predict_point_by_point(regressor, price)
    return week


def predict_point_by_point(model, data):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    inputs = np.reshape(data, (len(data), 1))
    inputs = sc.transform(inputs)
    inputs = np.reshape(inputs, (len(inputs), 1, 1))
    predicted = model.predict(inputs)
    predicted = sc.inverse_transform(predicted)
    print("After Prediction: ", predicted)
    return predicted


if __name__ == "__main__":
    app.run(host='0.0.0.0')