from enum import Enum
from time import time
import warnings
import numpy as np
import string
import os
import datetime
import math

from sklearn.preprocessing import MinMaxScaler
from TrafficData.SingleModelScats.data.data import process_data_datetime,process_data_series
from keras.models import load_model
warnings.filterwarnings("ignore")

class TrafficFlowModelsEnum(Enum):
    LSTM = 'lstm'
    GRU = 'gru'
    SAES = 'saes'
    NEW_SAES = 'new_saes'
    RNN = 'rnn'
    AVERAGE = 'average'


class TrafficFlowPredictor():
    def __init__(self):
        self.models = {}

        self.flow_scaler:MinMaxScaler = None
        self.days_scaler:MinMaxScaler = None
        self.scats_scaler:MinMaxScaler = None
        self.times_scaler:MinMaxScaler = None
        
        self.lags = 12 # must match whatever the models were trained on

        self.file1 = os.path.join(os.path.dirname(__file__),'SingleModelScats','data','train-data.csv')
        self.file2 = os.path.join(os.path.dirname(__file__),'SingleModelScats','data','test-data.csv')

        self.get_scalars()
        self.get_lookup_data()
        
        self.location_series_data = {}

    def get_model(self,model_name:string):
        if self.models.get(model_name) == None:
            self.models[model_name] = load_model(os.path.join(os.path.dirname(__file__),'SingleModelScats','model',f'{model_name}.h5'))
        return self.models.get(model_name)
    
    def get_scalars(self):
        _, _, _, _,_,_,_,_,self.flow_scaler, self.scats_scaler,self.days_scaler,self.times_scaler = process_data_datetime(self.file1, self.file2)

    def get_lookup_data(self):
        _, _, _, _,self.series_data,_,_,_,_,_ = process_data_series(self.file1, self.file2,self.lags)
        
        
    def predict_traffic_flow(self,location: int,date: datetime,steps:int,model_name: string):
        model = self.get_model(model_name)
        
        X = None
        if model_name == "average":
            X = self.get_datetime_inputs(location,date,steps)
            if X is None: return 0
            y_pred = self.predict_datetime(model,X)
        else:
            X = self.get_timeseries_inputs(location,date,steps)
            if X is None: return 0
            y_pred = self.predict_series(model,X)
        
        return y_pred.sum()

    def get_datetime_inputs(self,location: int,date:datetime,steps:int):
        dayindex = date.weekday() # determine weekday
        actual_time = date.hour * 60 + date.minute # determine time in minutes
        rounded_time = 15 * math.floor(actual_time / 15) # get current 15 minute interval

        days = self.days_scaler.transform(np.array([dayindex for _ in range(steps)]).reshape(-1,1)).reshape(1,-1)[0]
        times = self.times_scaler.transform(np.array([actual_time + t*15 for t in range(steps)]).reshape(-1,1)).reshape(1,-1)[0]
        scats = self.scats_scaler.transform(np.array([location for _ in range(steps)]).reshape(-1,1)).reshape(1,-1)[0]

        X = np.array([np.array([days[i],times[i],scats[i]]) for i in range(steps)])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X

    def predict_datetime(self,model,X):
        y_pred = model.predict(X)
        y_pred = self.flow_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(1, -1)[0]
        return y_pred

    def lookup_location_data(self,location:int):
        scaled_location = self.scats_scaler.transform(np.array([location]).reshape(-1,1)).reshape(1,-1)[0][0]
        if self.location_series_data.get(location) is None:
            location_indices = [i for i in range(len(self.series_data)) if self.series_data[i][self.lags] == scaled_location]
            self.location_series_data[location] = self.series_data[location_indices]
    
        return self.location_series_data[location]
        
    def get_timeseries_inputs(self,location: int,date:datetime,steps:int):
        day = date.day
        actual_time = date.hour * 60 + date.minute # determine time in minutes
        rounded_time = 15 * math.floor(actual_time / 15) # get current 15 minute interval
        time_index = int(rounded_time / 15)
        
        location_X = self.lookup_location_data(location)
        if len(location_X) == 0:
            raise Exception(f"No Data exists for location {location}")
        
        day_X = location_X[(day-1)*96:day*96]
        
        # fix for bad data having incomplete days
        while len(day_X) == 0 and day >= 0:
            day -= 7
            day_X = location_X[(day-1)*96:day*96]

        if len(day_X) == 0:
            return None

        X = np.array([day_X[time_index + i] for i in range(steps)])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X

    def predict_series(self,model,X):
        y_pred = model.predict(X)
        y_pred = self.flow_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(1, -1)[0]
        return y_pred