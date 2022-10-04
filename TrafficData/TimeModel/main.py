"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import argparse
import numpy as np
import pandas as pd
import string
import os

from sklearn.preprocessing import MinMaxScaler
from .data.data import process_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import tensorflow as tf

def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=96, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
        default="0970",
        help="SCATS Number")
    parser.add_argument(
        "--dayindex",
        default="6",
        help="Day index")
    args = parser.parse_args()

    lstm = load_model('model/lstm.h5')
    gru = tf.compat.v1.keras.models.load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']

    file1 = 'data/train-data.csv'
    file2 = 'data/test-data.csv'

    location = int(args.location)
    dayindex = int(args.dayindex)

    _, _, _, _,_,_,X,y_location,flow_scaler, scats_scalar,days_scalar,times_scalar = process_data(file1, file2,scats_id=location,day=dayindex)
    y_location = flow_scaler.inverse_transform(y_location.reshape(-1, 1)).reshape(1, -1)[0]

    days = days_scalar.transform(np.array([dayindex for _ in range(96)]).reshape(-1,1)).reshape(1,-1)[0]
    times = times_scalar.transform(np.array([t*15 for t in range(96)]).reshape(-1,1)).reshape(1,-1)[0]
    scats = scats_scalar.transform(np.array([dayindex for _ in range(96)]).reshape(-1,1)).reshape(1,-1)[0]
    X = np.array([np.array([days[i],times[i],scats[i]]) for i in range(96)])

    y_preds = []
    for name, model in zip(names, models):
        if model is None: continue

        if name == 'SAEs':
            X = np.reshape(X, (X.shape[0], X.shape[1]))
        else:
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X)
        predicted = flow_scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:96])
        print(name)
        eva_regress(y_location[:96], predicted)

    plot_results(y_location[:96], y_preds, names)

if __name__ == '__main__':
    main()
