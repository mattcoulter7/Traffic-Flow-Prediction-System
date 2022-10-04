import math
import string
import datetime
from TrafficData.TimeModel.TrafficFlowPredictionModel import TrafficFlowPredictionModel

traffic_flow_model:TrafficFlowPredictionModel = TrafficFlowPredictionModel()

def predict_traffic_flow(location: string,date: datetime):
    weekday = date.weekday() # determine weekday
    time = date.hour * 60 + date.minute # determine time in minutes
    time = 15 * math.floor(time / 15) # get current 15 minute interval

    # 4 indictaes 4 * 15 minutes steps = 1 hour
    y_pred = traffic_flow_model.predict_traffic_flow(location,weekday,time,4)

    total_traffic_flow = y_pred.sum()

    return total_traffic_flow