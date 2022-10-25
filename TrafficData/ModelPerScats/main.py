"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import os
import math
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data_series,process_data_datetime
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


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--location",
        default="0970",
        help="SCATS Number")
    parser.add_argument(
        "--dayindex",
        default="0",
        help="Day index")
    args = parser.parse_args()

    lstm_path = os.path.join(os.path.dirname(__file__),'model','location_models',f'{args.location}-lstm.h5')
    lstm = load_model(lstm_path) if os.path.exists(lstm_path) else None
    gru_path = os.path.join(os.path.dirname(__file__),'model','location_models',f'{args.location}-gru.h5')
    gru = load_model(gru_path) if os.path.exists(gru_path) else None
    saes_path = os.path.join(os.path.dirname(__file__),'model','location_models',f'{args.location}-saes.h5')
    saes = load_model(saes_path) if os.path.exists(saes_path) else None
    new_saes_path = os.path.join(os.path.dirname(__file__),'model','location_models',f'{args.location}-new_saes.h5')
    new_saes = load_model(new_saes_path) if os.path.exists(new_saes_path) else None
    rnn_path = os.path.join(os.path.dirname(__file__),'model','location_models',f'{args.location}-lstm.h5')
    rnn = load_model(rnn_path) if os.path.exists(rnn_path) else None
    average_path = os.path.join(os.path.dirname(__file__),'model','location_models',f'{args.location}-lstm.h5')
    average = load_model(average_path) if os.path.exists(average_path) else None

    models = [lstm, gru, saes,new_saes,rnn,average]
    names = ['LSTM']

    lag = 12
    file1 = os.path.join(os.path.dirname(__file__),'data','locations',f'{args.location}-train.csv')
    file2 = os.path.join(os.path.dirname(__file__),'data','locations',f'{args.location}-test.csv')

    location = int(args.location)
    dayindex = int(args.dayindex)

    _, _, _, _,X_series,y_series,flow_scaler = process_data_series(file1, file2, lag)
    _, _, _, _,_,_,X_datetime,y_datetime,flow_scaler,days_scalar,times_scalar = process_data_datetime(file1, file2,day=dayindex)
    
    y_series = flow_scaler.inverse_transform(y_series.reshape(-1, 1)).reshape(1, -1)[0]
    y_datetime = flow_scaler.inverse_transform(y_datetime.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if model is None: continue

        if name == 'SAEs':
            X_series = np.reshape(X_series, (X_series.shape[0], X_series.shape[1]))
        else:
            X_series = np.reshape(X_series, (X_series.shape[0], X_series.shape[1], 1))
            X_datetime = np.reshape(X_datetime, (X_datetime.shape[0], X_datetime.shape[1], 1))
        
        file = os.path.join(os.path.dirname(__file__),'images',name + '.png')
        plot_model(model, to_file=file, show_shapes=True)
        
        predicted = None
        if name == 'average':
            predicted = model.predict(X_datetime)
            eva_regress(y_datetime, predicted)
        else:
            predicted = model.predict(X_series)
            eva_regress(y_series, predicted)
        predicted = flow_scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        
        y_preds.append(predicted[:96])
        print(name)

    plot_results(y_series[: 96], y_preds, names)


if __name__ == '__main__':
    main(sys.argv)
