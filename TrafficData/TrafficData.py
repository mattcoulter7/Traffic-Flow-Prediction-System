from random import Random
import string
import datetime
from keras.models import load_model
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from TrafficData.SingleModelScats.data.data import process_data
from TrafficData.TimeModel.main import predict_traffic_flow
from functools import lru_cache

#TODO this is just a dummy filler function for now need to create proper implementation for final
# get traffic data

model: Sequential = None

def open_traffic_model(file_name: string):
    model = load_model('TrafficData/SingleModelScats/model/' + file_name)
    X, y = make_blobs(n_samples=100, centers=2, n_features=13, random_state=1)
    scalar = MinMaxScaler()
    scalar.fit(X)
    # new instances where we do not know the answer
    Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=13, random_state=1)
    Xnew = scalar.transform(Xnew)
    
    lag = 12
    file1 = 'TrafficData/SingleModelScats/data/train-data.csv'
    file2 = 'TrafficData/SingleModelScats/data/test-data.csv'
    
    _, _, _, _, _,_,X_location,y_location,flow_scaler,scats_scalar = process_data(file1, file2, lag,scats_id="0970")
    
    print("X location:", X_location)
    
    # make a prediction
    ynew = model.predict(X_location)


    predicted = flow_scaler.inverse_transform(ynew.reshape(-1, 1)).reshape(1, -1)[0]

    # show the inputs and predicted outputs
    for i in range(len(Xnew)):
        print("X=%s, Predicted=%s, %s" % (Xnew[i], ynew[i], predicted[i]))

# returns the vehicles traveled throught the intersection over the last hour
def get_traffic_flow(scats_number: int, time: float) -> float:

    # x is the input ordered 
    X_new = [[0.2416417, 0.8331284, 0.13294513, 0.203367, 0.07782224, 0.30480586, 0.42202578, 0.06675819, 0.18879954, 0.93919459, 0.31653253, 0.2352255, 0.03451021]]
    # y is the output
    y_new = model.predict(X_new)


    random = Random()
    random.seed(scats_number + time)
    return y_new #random.uniform(1, 100)

def predict_flow(location: string,date: datetime):
    weekday = date.weekday()
    time = date.hour * 60 + date.minute
    y_pred = predict_traffic_flow(location,weekday)

    time_index = round(time/15)

    return y_pred[time_index]