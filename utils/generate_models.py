import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, Lambda, \
    Convolution2D, Flatten, \
    Reshape, LSTM, Dropout, TimeDistributed, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
import numpy as np

import keras.backend as K
from tensorflow import set_random_seed


def pearson_corr(y_true, y_pred):
    """
    Pearson Correlation as a Keras metric

    Parameters
    ----------
    y_true: matrix of true values
    y_pred: matrix of predictions
    """

    fsp = y_pred - K.expand_dims(K.mean(y_pred, axis=-1), axis=1)
    fst = y_true - K.expand_dims(K.mean(y_true, axis=-1), axis=1)

    devP = K.std(y_pred, axis=-1)
    devT = K.std(y_true, axis=-1)

    return K.mean(fsp*fst, axis=-1)/(devP*devT)


# from https://github.com/NLeSC/mcfly
def generate_DeepConvLSTM_model(dim_length, dim_channels, output_dim,
                                filters, lstm_dims, learning_rate=0.01,
                                regularization_rate=0.01, metrics=[pearson_corr],
                                dropout=0.5, dropout_rnn=None, dropout_cnn=None,
                                alternative_out=False, set_seed=True,
                                clipvalue=None):
    """
    Generate a model with convolution and LSTM layers.
    See Ordonez et al., 2016, http://dx.doi.org/10.3390/s16010115

    Last layer is adapted for prediction instead of classification

    Parameters
    ----------
    dim_length : int
        Number of samples per window for the dataset
    dim_channels : int
        Number of channels (i.e. variables)
    output_dim : int
        Number of samples for the prediction output
    filters : list of ints
        Number of filters for each convolutional layer
        It may be an empty list, so that no filters are added
    lstm_dims : list of ints
        Number of hidden nodes for each LSTM layer
    learning_rate : float
        Learning rate
    regularization_rate : float
        Regularization rate
    metrics : list
        Metrics to calculate on the validation set.
        See https://keras.io/metrics/ for possible values.
    dropout : float, optional
        If different to None, a dropout layer is added after the LSTM chain
    dropout_cnn : float, optional
        If different to None, a dropout is added after every filter in the CNN
    dropout_rnn : float, optional
        If different to None, a dropout is added after every filter in the RNN
    alternative_out : boolean, optional
        If True, a Dense layer is added at the output (instead of a TimeDistributed)

    Returns
    -------
    model : Keras model
        The compiled Keras model
    """
    if set_seed:
        np.random.seed(1234)
        set_random_seed(1234)
    weightinit = 'lecun_uniform'  # weight initialization
    model = Sequential()  # initialize model
    model.add(BatchNormalization(input_shape=(dim_length, dim_channels)))
    # reshape a 2 dimensional array per window/variables into a
    # 3 dimensional array
    if len(filters) > 0:
        model.add(Reshape(target_shape=(dim_length, dim_channels, 1)))

        for filt in filters:
            # filt: number of filters used in a layer
            # filters: vector of filt values
            model.add(Convolution2D(filt, kernel_size=(3, 1), padding='same',
                                    kernel_regularizer=l2(regularization_rate),
                                    kernel_initializer=weightinit))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            if dropout_cnn is not None:
                model.add(Dropout(dropout_cnn))

        # reshape 3 dimensional array back into a 2 dimensional array,
        # but now with more dept as we have the the filters for each channel
        model.add(Reshape(target_shape=(dim_length, filters[-1] * dim_channels)))

    # LSTM's
    for index in range(0,len(lstm_dims)):
        lstm_dim = lstm_dims[index]
        model.add(LSTM(units=lstm_dim, return_sequences=True,
                       activation='tanh'))
        if dropout_rnn is not None:
             model.add(Dropout(dropout_rnn))

    if dropout is not None:
        model.add(Dropout(dropout))  # dropout before the dense layer

    if alternative_out:
        # reshape using last dimension of LSTM cells into a vector)
        model.add(Reshape(target_shape=(dim_length*lstm_dims[-1], )))
        # dense layer over that vector, one cell per prediction step
        model.add(Dense(output_dim))

    else:
        # Dense layer with 1 unit per output_dim,
        # no activation specified means 'linear', which makes sense for regression
        model.add(TimeDistributed(Dense(units=output_dim,
                                        kernel_regularizer=l2(regularization_rate))))
        # take last samples for output_shape
        model.add(Lambda(lambda x: x[:, -1, :], output_shape=[output_dim]))

    # optimizer
    # TODO clipvalue seems to work better, investigate
    # SGD optimizer definitively worse using SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    optimizer=Adam(lr=learning_rate)
    if clipvalue is not None:
        optimizer = Adam(lr=learning_rate, clipvalue=clipvalue)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=metrics)

    return model


# adapted from https://github.com/NLeSC/mcfly
def generate_models(dim_length, dim_channels, output_dim, number_of_models=5,
                    deepconvlstm_min_conv_layers=1, deepconvlstm_max_conv_layers=10,
                    deepconvlstm_min_conv_filters=10, deepconvlstm_max_conv_filters=100,
                    deepconvlstm_min_lstm_layers=1, deepconvlstm_max_lstm_layers=5,
                    deepconvlstm_min_lstm_dims=10, deepconvlstm_max_lstm_dims=100,
                    low_lr=1, high_lr=4, low_reg=1, high_reg=4,
                    dropout_final_max=None, dropout_final_min=None,
                    dropout_cnn_max=None, dropout_cnn_min=None,
                    dropout_rnn_max=None, dropout_rnn_min=None,
                    metrics=[pearson_corr]):
    """
    Generate one or multiple untrained Keras models from hyperparameters.
    Simplified from mcfly

    Parameters
    ----------
    dim_length : int
        Number of samples per window for the dataset
    dim_channels : int
        Number of channels (i.e. variables)
    output_dim : int
        Number of samples for the prediction output
    number_of_models : int
        Number of models to generate
    deepconvlstm_min_conv_layers : int
        minimum number of Conv layers in DeepConvLSTM model
    deepconvlstm_max_conv_layers : int
        maximum number of Conv layers in DeepConvLSTM model
    deepconvlstm_min_conv_filters : int
        minimum number of filters per Conv layer in DeepConvLSTM model
    deepconvlstm_max_conv_filters : int
        maximum number of filters per Conv layer in DeepConvLSTM model
    deepconvlstm_min_lstm_layers : int
        minimum number of Conv layers in DeepConvLSTM model
    deepconvlstm_max_lstm_layers : int
        maximum number of Conv layers in DeepConvLSTM model
    deepconvlstm_min_lstm_dims : int
        minimum number of hidden nodes per LSTM layer in DeepConvLSTM model
    deepconvlstm_max_lstm_dims : int
        maximum number of hidden nodes per LSTM layer in DeepConvLSTM model
    low_lr : float
        minimum of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    high_lr : float
        maximum  of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    low_reg : float
        minimum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`
    high_reg : float
        maximum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`
    dropout_final_max : float
        Maximum value of dropout for final layer after Dense
        If None, no dropout is added for final layer
    dropout_final_min : float
        Minimum value of dropout for final layer after Dense
        If None, no dropout is added for final layer
    dropout_cnn_max : float
        Maximum value of dropout for cnn layer after Dense
        If None, no dropout is added for cnn layer
    dropout_cnn_min : float
        Minimum value of dropout for cnn layer after Dense
        If None, no dropout is added for cnn layer
    dropout_rnn_max : float
        Maximum value of dropout for rnn layer after Dense
        If None, no dropout is added for rnn layer
    dropout_rnn_min : float
        Minimum value of dropout for rnn layer after Dense
        If None, no dropout is added for rnn layer
    metrics : list
        Metrics to calculate on the validation set.
        See https://keras.io/metrics/ for possible values.

    Returns
    -------
    models : list
        List of compiled models
    """
    np.random.seed(0)
    #set_random_seed(0)

    models = []
    for _ in range(0, number_of_models):
        hyperparameters = generate_DeepConvLSTM_hyperparameter_set(
            min_conv_layers=deepconvlstm_min_conv_layers,
            max_conv_layers=deepconvlstm_max_conv_layers,
            min_conv_filters=deepconvlstm_min_conv_filters,
            max_conv_filters=deepconvlstm_max_conv_filters,
            min_lstm_layers=deepconvlstm_min_lstm_layers,
            max_lstm_layers=deepconvlstm_max_lstm_layers,
            min_lstm_dims=deepconvlstm_min_lstm_dims,
            max_lstm_dims=deepconvlstm_max_lstm_dims,
            low_lr=low_lr, high_lr=high_lr, low_reg=low_reg,
            high_reg=high_reg,
            dropout_final_max=dropout_final_max, dropout_final_min=dropout_final_min,
            dropout_cnn_max=dropout_cnn_max, dropout_cnn_min=dropout_cnn_min,
            dropout_rnn_max=dropout_rnn_max, dropout_rnn_min=dropout_rnn_min)
        model = generate_DeepConvLSTM_model(dim_length, dim_channels, output_dim,
                                            hyperparameters['filters'],
                                            hyperparameters['lstm_dims'],
                                            hyperparameters['learning_rate'],
                                            hyperparameters['regularization_rate'],
                                            dropout=hyperparameters['dropout'],
                                            dropout_cnn=hyperparameters['dropout_cnn'],
                                            dropout_rnn=hyperparameters['dropout_rnn'],
                                            metrics=metrics, set_seed=False)
        models.append((model, hyperparameters))
    return models



"""
Utility functions from mcfly https://github.com/NLeSC/mcfly
(Generating random hyperpameters)
"""


def generate_base_hyper_parameter_set(low_lr=1,
                                      high_lr=4,
                                      low_reg=1,
                                      high_reg=4):
    """ Generate a base set of hyperparameters that are necessary for any
    model, but sufficient for none.

    Parameters
    ----------
    low_lr : float
        minimum of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    high_lr : float
        maximum  of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    low_reg : float
        minimum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`
    high_reg : float
        maximum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`

    Returns
    -------
    hyperparameters : dict
        basis hyperpameters
    """
    hyperparameters = {}
    hyperparameters['learning_rate'] = get_learning_rate(low_lr, high_lr)
    hyperparameters['regularization_rate'] = get_regularization(
        low_reg, high_reg)
    return hyperparameters


def get_learning_rate(low=1, high=4):
    """ Return random learning rate 10^-n where n is sampled uniformly between
    low and high bounds.

    Parameters
    ----------
    low : float
        low bound
    high : float
        high bound

    Returns
    -------
    learning_rate : float
        learning rate
    """
    result = 10 ** (-np.random.uniform(low, high))
    return result


def get_regularization(low=1, high=4):
    """ Return random regularization rate 10^-n where n is sampled uniformly
    between low and high bounds.

    Parameters
    ----------
    low : float
        low bound
    high : float
        high bound

    Returns
    -------
    regularization_rate : float
        regularization rate
    """
    return 10 ** (-np.random.uniform(low, high))


def generate_DeepConvLSTM_hyperparameter_set(
        min_conv_layers=1, max_conv_layers=10,
        min_conv_filters=10, max_conv_filters=100,
        min_lstm_layers=1, max_lstm_layers=5,
        min_lstm_dims=10, max_lstm_dims=100,
        low_lr=1, high_lr=4, low_reg=1, high_reg=4,
        dropout_final_max=None, dropout_final_min=None,
        dropout_cnn_max=None, dropout_cnn_min=None,
        dropout_rnn_max=None, dropout_rnn_min=None):
    """
    Generate a hyperparameter set that defines a DeepConvLSTM model.

    Parameters
    ----------
    min_conv_layers : int
        minimum number of Conv layers in DeepConvLSTM model
    max_conv_layers : int
        maximum number of Conv layers in DeepConvLSTM model
    min_conv_filters : int
        minimum number of filters per Conv layer in DeepConvLSTM model
    max_conv_filters : int
        maximum number of filters per Conv layer in DeepConvLSTM model
    min_lstm_layers : int
        minimum number of Conv layers in DeepConvLSTM model
    max_lstm_layers : int
        maximum number of Conv layers in DeepConvLSTM model
    min_lstm_dims : int
        minimum number of hidden nodes per LSTM layer in DeepConvLSTM model
    max_lstm_dims : int
        maximum number of hidden nodes per LSTM layer in DeepConvLSTM model
    low_lr : float
        minimum of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    high_lr : float
        maximum  of log range for learning rate: learning rate is sampled
        between `10**(-low_reg)` and `10**(-high_reg)`
    low_reg : float
        minimum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`
    high_reg : float
        maximum  of log range for regularization rate: regularization rate is
        sampled between `10**(-low_reg)` and `10**(-high_reg)`
    dropout_final_max : float
        Maximum value of dropout for final layer after Dense
        If None, no dropout is added for final layer
    dropout_final_min : float
        Minimum value of dropout for final layer after Dense
        If None, no dropout is added for final layer
    dropout_cnn_max : float
        Maximum value of dropout for cnn layer after Dense
        If None, no dropout is added for cnn layer
    dropout_cnn_min : float
        Minimum value of dropout for cnn layer after Dense
        If None, no dropout is added for cnn layer
    dropout_rnn_max : float
        Maximum value of dropout for rnn layer after Dense
        If None, no dropout is added for rnn layer
    dropout_rnn_min : float
        Minimum value of dropout for rnn layer after Dense
        If None, no dropout is added for rnn layer

    Returns
    ----------
    hyperparameters: dict
        hyperparameters for a DeepConvLSTM model
    """
    hyperparameters = generate_base_hyper_parameter_set(
        low_lr, high_lr, low_reg, high_reg)
    number_of_conv_layers = np.random.randint(
        min_conv_layers, max_conv_layers + 1)
    hyperparameters['filters'] = np.random.randint(
        min_conv_filters, max_conv_filters + 1, number_of_conv_layers).tolist()
    number_of_lstm_layers = np.random.randint(
        min_lstm_layers, max_lstm_layers + 1)
    hyperparameters['lstm_dims'] = np.random.randint(
        min_lstm_dims, max_lstm_dims + 1, number_of_lstm_layers).tolist()
    hyperparameters['dropout'] = None
    if dropout_final_min is not None:
        hyperparameters['dropout'] = np.random.uniform(dropout_final_min, dropout_final_max)
    hyperparameters['dropout_rnn'] = 0.75
    if dropout_rnn_max is not None:
        hyperparameters['dropout_rnn'] = np.random.uniform(dropout_rnn_min, dropout_rnn_max)
    hyperparameters['dropout_cnn'] = 0.75
    if dropout_cnn_max is not None:
        hyperparameters['dropout_cnn'] = np.random.uniform(dropout_cnn_min, dropout_cnn_max)
    return hyperparameters


# deprecated
def generate_simple_lstm_model(units=5, window_size=15,
                               output_dim=1, metrics=[]):
    """
    Generate model with simple LSTM chain plus Dense

    :param units: int
        Number of units of the LSTM chain.
    :param window_size: int
        Size of the temporal window of the input data.
    :param output_dim: int
        Number of prediction steps to be tuned.
    :param metrics: list
        List of additional metrics to be evaluated by the model.
    :return: model: object
    """
    np.random.seed(0)

    model = Sequential()
    # LSTM of X units
    # - first parameter is the number of units = X
    # - second parameter is the input shape = (window_size, 1)
    # - activation layer tanh by default
    # - rest of parameters also by default
    model.add(LSTM(units, input_shape=(window_size, 1)))
    # Dense layer with just one unit, no activation specified means 'linear', which
    # makes sense for regression
    model.add(Dense(output_dim))
    model.summary()

    # build model using keras documentation recommended optimizer initialization
    # either RMSprop or AdaDelta are recommended for LSTMs
    # - for RMSprop, only learning rate may be fine-tuned
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # - for AdaDelta, all parameters should be left to default
    # optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=metrics)

    return model


def get_lstm_simple_model(dim_length, number_of_cells=10,
                          dropout=None, time_distributed=False,
                          regularization_rate=0.001, output_dim=15):
    """
    Generate a simple LSTM with maybe more than one LSTM chain.

    :param dim_length: int
        Number of timesteps in the input (window_size)
    :param number_of_cells: int or list
        Number of cells of the LSTM chain(s)
    :param dropout: float
        Value of dropout for final layer
    :param time_distributed: boolean, optional
        If True, the outpout is a TimeDistributed layer instead of Dense
    :param regularization_rate: float, optional
        Used for the TimeDistributed layer, if it exists
    :param output_dim: int
        Numbero of prediction steps
    :return: model: object
    """
    # init
    np.random.seed(0)
    if output_dim is None:
        output_dim = dim_length

    model = Sequential()
    # LSTM of X units
    # - first parameter is the number of units = X
    # - second parameter is the input shape = (window_size, 1)
    # - activation layer tanh by default
    # - rest of parameters also by default
    if type(number_of_cells) is list:
        # more than 1 LSTM chain
        model.add(LSTM(number_of_cells[0], input_shape=(dim_length, 1)))
        for index in range(1, len(number_of_cells)):
            model.add(LSTM(number_of_cells[index], return_sequences=True))
    else:
        model.add(LSTM(number_of_cells, input_shape=(dim_length, 1)))

    if dropout is not None:
        model.add(Dropout(dropout))

    # Dense layer with just one unit, no activation specified means 'linear', which
    # makes sense for regression
    if time_distributed:
        # TODO (assert len(input_shape) >= 3)
        model.add(TimeDistributed(Dense(units=output_dim,
                                        kernel_regularizer=l2(regularization_rate))))
    else:
        model.add(Dense(output_dim))
    model.summary()


    # build model using keras documentation recommended optimizer initialization
    # either RMSprop or AdaDelta are recommended for LSTMs
    # - for RMSprop, only learning rate may be fine-tuned
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # - for AdaDelta, all parameters should be left to default
    #oprimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=[pearson_corr])

    return model
