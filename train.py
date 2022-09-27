from turtle import shape
import numpy as np
import pandas as pd
import csv
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from sklearn import preprocessing

df = pd.read_csv('Data/TrainData.csv')
flow_min = df['Flow'].min()
flow_max = df['Flow'].max()

# normalise data

# x_train [scats, data, time, conditions]
# y_train [flow]



def get_date(date_string):
    [day, month, year] = date_string.split('/')
    return datetime.date(int(year), int(month), int(day))

def normalise_time(time):
    [hour, minute, second] = time.split(':')
    decimal = float(hour) + (int(minute) / 60)
    return round(decimal / 24, 4)

def scats_encoder(scats):
    # return int(scats)
    return float(scats) / 10000



def normalise_data(file):
    x = []
    y = []
    with open(file) as td:
        reader = csv.reader(td, delimiter=',')
        index = -1
        for row in reader:
            index += 1
            if index < 1:
                continue
            scats = scats_encoder(row[0])
            date = get_date(row[1]).weekday() / 6
            time = normalise_time(row[2])
            flow = (float(row[3]) - flow_min) / (flow_max - flow_min)
            condition = float(row[4])

            x.append([scats, date, time, condition])
            y.append(flow)
    return x, y


x_train, y_train = normalise_data("Data/TrainData.csv")
x_test, y_test = normalise_data("Data/TestData.csv")

print(x_train)

model = keras.Sequential()
model.add(layers.LSTM(4, input_shape=(4, 1), return_sequences=True))
model.add(layers.LSTM(8))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
#input layer
#model.add(layers.LSTM(4, input_shape=(4,), activation='relu', return_sequences=True))
#hidden layers

#output layer


# model.add(layers.LSTM(4, input_shape=(4,), activation='relu', return_sequences=True))
# model.add(layers.LSTM(16, input_shape=(16,), activation='relu', return_sequences=True))
# model.add(layers.LSTM(16, input_shape=(16,), activation='relu'))
# model.add(layers.Dense(units=1, activation='relu'))

# model.add(keras.Input(shape=(4, 1)))


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    metrics=["accuracy"]
)

model.fit(x_train,y_train, batch_size=128, epochs=50, verbose=2)
model.evaluate(x_test,y_test, batch_size=128, verbose=2)


# print("Prediction:", model.predict(x_train[1]))