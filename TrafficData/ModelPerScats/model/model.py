"""
Defination of NN model
"""
from keras.layers import Dense, Dropout, Activation,InputLayer
from keras.layers import LSTM, GRU, SimpleRNN
from keras.models import Sequential


def get_lstm(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    for i in range(1,len(units)):
        if i == 1:
            model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
        elif i == len(units) - 1:
            model.add(Dropout(0.2))
            model.add(Dense(units[i], activation='sigmoid'))
        elif i < len(units) - 2:
            model.add(LSTM(units[i], return_sequences=True))
        else:
            model.add(LSTM(units[i]))
        

    return model


def get_gru(units):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """
    model = Sequential()
    for i in range(1,len(units)):
        if i == 1:
            model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True, reset_after=True))
        elif i == len(units) - 1:
            model.add(Dropout(0.2))
            model.add(Dense(units[i], activation='sigmoid'))
        elif i < len(units) - 2:
            model.add(GRU(units[i], return_sequences=True))
        else:
            model.add(GRU(units[i]))

    return model

def get_rnn(units):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    for i in range(1,len(units)):
        if i == 1:
            model.add(SimpleRNN(units[1], input_shape=(units[0], 1), return_sequences=True))
        elif i == len(units) - 1:
            model.add(Dropout(0.2))
            model.add(Dense(units[i], activation='sigmoid'))
        elif i < len(units) - 2:
            model.add(SimpleRNN(units[i], return_sequences=True))
        else:
            model.add(SimpleRNN(units[i]))


    return model


def _get_sae(inputs, hidden, output):
    """SAE(Auto-Encoders)
    Build SAE Model.

    # Arguments
        inputs: Integer, number of input units.
        hidden: Integer, number of hidden units.
        output: Integer, number of output units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(Dense(hidden, input_dim=inputs, name='hidden'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(output, activation='sigmoid'))

    return model


def get_saes(layers):
    """SAEs(Stacked Auto-Encoders)
    Build SAEs Model.

    # Arguments
        layers: List(int), number of input, output and hidden units.
    # Returns
        models: List(Model), List of SAE and SAEs.
    """
    sae1 = _get_sae(layers[0], layers[1], layers[-1])
    sae2 = _get_sae(layers[1], layers[2], layers[-1])
    sae3 = _get_sae(layers[2], layers[3], layers[-1])

    saes = Sequential()
    saes.add(Dense(layers[1], input_dim=layers[0], name='hidden1'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[2], name='hidden2'))
    saes.add(Activation('sigmoid'))
    saes.add(Dense(layers[3], name='hidden3'))
    saes.add(Activation('sigmoid'))
    saes.add(Dropout(0.2))
    saes.add(Dense(layers[4], activation='sigmoid'))

    models = [sae1, sae2, sae3, saes]

    return models

def get_new_saes(inputs, output, auto_encoder_count = 3, encoder_size = 5,fine_tuning_layers = []):
    saes = Sequential()
    # input
    saes.add(InputLayer(inputs))

    # auto encoders
    for i in range(auto_encoder_count):
        saes.add(Dense(encoder_size, name=f'hidden{(i+1) * 2 - 1}')) # encode
        saes.add(Activation('ReLU'))
        saes.add(Dense(inputs, name=f'hidden{(i+1) * 2}')) # decode
        saes.add(Activation('sigmoid'))
    
    # Model Fine Tuning
    for i in range(len(fine_tuning_layers)):
        saes.add(Dense(fine_tuning_layers[i], name=f'hidden{auto_encoder_count * 2 + 1 + (i + 1)}'))
        saes.add(Activation('sigmoid'))

    # output
    saes.add(Dropout(0.2))
    saes.add(Dense(output, activation='sigmoid'))

    return saes
    
def get_average(units):
    model = Sequential()
    # input

    for i in range(len(units) - 1):
        dim = units[i]
        if i==0:
            model.add(InputLayer(dim))
        else:
            model.add(Dense(dim, name=f'hidden{i}')) # encode
            model.add(Activation('ReLU'))
    
    model.add(Dropout(0.2))
    model.add(Dense(units[len(units) - 1], activation='sigmoid'))

    return model
