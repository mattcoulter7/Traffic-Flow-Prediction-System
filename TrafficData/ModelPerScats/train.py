"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import os
from data.data import process_data_series,process_data_datetime
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")



def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save(os.path.join(os.path.dirname(__file__),'model','location_models',f'{name}.h5'))
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(os.path.join(os.path.dirname(__file__),'model','location_models',f'{name} loss.csv'), encoding='utf-8', index=False)


def train_seas(models, X_train, y_train, name, config):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(inputs=p.input,
                                       outputs=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config)

def get_locations():
    with open(os.path.join(os.path.dirname(__file__),'data','locations','locations.txt')) as locations_file:
        locations = locations_file.readlines()
        locations = ''.join(locations).split('\n')
        locations = filter(lambda loc: loc != '',locations)
        return list(locations)


def main(argv):
    lag = 12
    config = {"batch": 128, "epochs": 10}

    locations = get_locations()
    for location in locations:
        file1 = os.path.join(os.path.dirname(__file__),'data','locations',f'{location}-train.csv')
        file2 = os.path.join(os.path.dirname(__file__),'data','locations',f'{location}-test.csv')
        X_train_series, y_train_series, _, _, _,_,_ = process_data_series(file1, file2, lag)
        X_train_datetime, y_train_datetime, _, _, _,_,_,_,_,_,_ = process_data_datetime(file1, file2)

        # train each type of model
        models_types = ['rnn']
        test_identifier = '10 epochs'
        for model_type in models_types:
            model_name = f"{location}-{model_type}" if test_identifier == '' else f"{location}-{model_type} ({test_identifier})"
            if model_type == 'lstm':
                X_train2 = np.reshape(X_train_series, (X_train_series.shape[0], X_train_series.shape[1], 1))
                m = model.get_lstm([lag, 64, 64, 1])
                train_model(m, X_train2, y_train_series, model_name, config)
            if model_type == 'rnn':
                X_train2 = np.reshape(X_train_series, (X_train_series.shape[0], X_train_series.shape[1], 1))
                m = model.get_rnn([lag, 64, 64, 1])
                train_model(m, X_train2, y_train_series, model_name, config)
            if model_type == 'gru':
                X_train2 = np.reshape(X_train_series, (X_train_series.shape[0], X_train_series.shape[1], 1))
                m = model.get_gru([lag, 64, 64, 1])
                train_model(m, X_train2, y_train_series, model_name, config)
            if model_type == 'saes':
                X_train2 = np.reshape(X_train_series, (X_train_series.shape[0], X_train_series.shape[1]))
                m = model.get_saes([lag, 400, 400, 400, 1])
                train_seas(m, X_train2, y_train_series, model_name, config)
            if model_type == 'new_saes':
                X_train2 = np.reshape(X_train_series, (X_train_series.shape[0], X_train_series.shape[1], 1))
                m = model.get_new_saes(lag,1,encoder_size=10,auto_encoder_count=3, fine_tuning_layers=[10])
                train_model(m, X_train2, y_train_series, model_name, config)
            if model_type == 'average':
                X_train2 = np.reshape(X_train_datetime, (X_train_datetime.shape[0], X_train_datetime.shape[1]))
                m = model.get_average([2, 400, 400, 400, 1])
                train_model(m, X_train2, y_train_datetime, model_name, config)


if __name__ == '__main__':
    main(sys.argv)
