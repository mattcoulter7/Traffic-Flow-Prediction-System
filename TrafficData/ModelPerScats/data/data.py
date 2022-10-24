"""
Processing the data
"""
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_data_series(train, test, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    lane_flow_attr = 'Lane 1 Flow (Veh/5 Minutes)'

    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[lane_flow_attr].values.reshape(-1, 1))
    flow1 = flow_scaler.transform(df1[lane_flow_attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = flow_scaler.transform(df2[lane_flow_attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train, test, data = [], [], []
    for i in range(lags, len(flow1)):
        row = [*flow1[i - lags: i],flow1[i]]
        train.append(row)
        data.append(row)

    for i in range(lags, len(flow2)):
        row = [*flow2[i - lags: i],flow2[i]]
        test.append(row)
        data.append(row)

    data = np.array(data)
    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X = data[:, :-1]
    y = data[:,-1]

    X_train = train[:, :-1]
    y_train = train[:, -1]

    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, X,y,flow_scaler

def process_data_datetime(train, test,day=0):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """

    lane_flow_attr = 'Lane 1 Flow (Veh/5 Minutes)'
    date_time_attr = '5 Minutes'

    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    flow_scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[lane_flow_attr].values.reshape(-1, 1))
    flow1 = flow_scaler.transform(df1[lane_flow_attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = flow_scaler.transform(df2[lane_flow_attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    dates1 = list(map(parse_date,df1[date_time_attr].values))
    dates2 = list(map(parse_date,df2[date_time_attr].values))

    days1 = np.array(list(map(lambda d: d.weekday(),dates1)))
    days2 = np.array(list(map(lambda d: d.weekday(),dates2)))
    day_scalar = MinMaxScaler(feature_range=(0, 1)).fit(days1.reshape(-1, 1))
    days1 = day_scalar.transform(days1.reshape(-1, 1)).reshape(1, -1)[0]
    days2 = day_scalar.transform(days2.reshape(-1, 1)).reshape(1, -1)[0]

    times1 = np.array(list(map(lambda d: d.hour * 60 + d.minute,dates1)))
    times2 = np.array(list(map(lambda d: d.hour * 60 + d.minute,dates2)))
    time_scalar = MinMaxScaler(feature_range=(0, 1)).fit(times1.reshape(-1, 1))
    times1 = time_scalar.transform(times1.reshape(-1, 1)).reshape(1, -1)[0]
    times2 = time_scalar.transform(times2.reshape(-1, 1)).reshape(1, -1)[0]

    train, test,data = [], [],[]
    for i in range(len(flow1)):
        row = [days1[i],times1[i],flow1[i]]
        train.append(row)
        data.append(row)

    for i in range(len(flow2)):
        row = [days2[i],times2[i],flow2[i]]
        test.append(row)
        data.append(row)

    data = np.array(data)
    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X = data[:, :-1]
    y = data[:, -1]

    day = day_scalar.transform(np.array([day]).reshape(-1,1)).reshape(1,-1)[0][0]
    day_indices = [i for i in range(len(X)) if X[i][0] == day]
    X_day = X[day_indices]
    y_day = y[day_indices]

    X_train = train[:, :-1]
    y_train = train[:, -1]

    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test,X,y,X_day,y_day, flow_scaler,day_scalar,time_scalar

# Date Format: 3/10/2006 2:45
def parse_date(date_string):
    date,time = date_string.split()
    day,month,year = date.split('/')
    hour,minute = time.split(':')
    return datetime.datetime(int(year),int(month),int(day),int(hour),int(minute))