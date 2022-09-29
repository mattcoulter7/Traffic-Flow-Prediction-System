"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
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

    model.save('model/location_models/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/location_models/' + name + ' loss.csv', encoding='utf-8', index=False)


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
    with open('data/locations/locations.txt') as locations_file:
        locations = locations_file.readlines()
        locations = ''.join(locations).split('\n')
        locations = filter(lambda loc: loc != '',locations)
        return list(locations)


def main(argv):
    lag = 12
    config = {"batch": 128, "epochs": 10}

    locations = get_locations()
    for location in locations:
        file1 = f'data/locations/{location}-train.csv' # file1 = 'data/train.csv' 
        file2 = f'data/locations/{location}-test.csv' # file2 = 'data/test.csv'
        X_train, y_train, _, _, _,_,_,_,_ = process_data(file1, file2, lag,append_scats=False)

        # train each type of model
        models_types = ['saes','lstm','gru']
        for model_type in models_types:
            model_name = f"{location}-{model_type}"
            if model_type == 'lstm':
                X_train2 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                m = model.get_lstm([lag, 64, 64, 1])
                train_model(m, X_train2, y_train, model_name, config)
            if model_type == 'gru':
                X_train2 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                m = model.get_gru([lag, 64, 64, 1])
                train_model(m, X_train2, y_train, model_name, config)
            if model_type == 'saes':
                X_train2 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
                m = model.get_saes([lag, 400, 400, 400, 1])
                train_seas(m, X_train2, y_train, model_name, config)


if __name__ == '__main__':
    main(sys.argv)
