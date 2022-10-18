import warnings
import numpy as np
import string
import os

from sklearn.preprocessing import MinMaxScaler
from .data.data import process_data
from keras.models import load_model
warnings.filterwarnings("ignore")

class TrafficFlowPredictionModel():
    def __init__(self):
        self.models = {}

        self.flow_scaler:MinMaxScaler = None
        self.days_scaler:MinMaxScaler = None
        self.scats_scaler:MinMaxScaler = None
        self.times_scaler:MinMaxScaler = None
        
        self.get_scalars()

    def get_model(self,model_name:string):
        if self.models.get(model_name) == None:
            self.models[model_name] = load_model(os.path.join(os.path.dirname(__file__),'model',f'{model_name}.h5'))
        return self.models.get(model_name)
    
    def get_scalars(self):
        file1 = os.path.join(os.path.dirname(__file__),'data','train-data.csv')
        file2 = os.path.join(os.path.dirname(__file__),'data','test-data.csv')
        _, _, _, _,_,_,_,_,self.flow_scaler, self.scats_scaler,self.days_scaler,self.times_scaler = process_data(file1, file2)

    def predict_traffic_flow(self,location: int,dayindex: int,time:int,steps:int = 1,model_name: string = "lstm"):
        print(f'location: {location}, day: {dayindex},time: {time}')
        model = self.get_model(model_name)

        days = self.days_scaler.transform(np.array([dayindex for _ in range(steps)]).reshape(-1,1)).reshape(1,-1)[0]
        times = self.times_scaler.transform(np.array([time + t*15 for t in range(steps)]).reshape(-1,1)).reshape(1,-1)[0]
        scats = self.scats_scaler.transform(np.array([location for _ in range(steps)]).reshape(-1,1)).reshape(1,-1)[0]

        X = np.array([np.array([days[i],times[i],scats[i]]) for i in range(steps)])
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        predicted = model.predict(X)
        predicted = self.flow_scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]

        return predicted