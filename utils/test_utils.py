import numpy as np
from utils.data_generator import DataGenerator
from utils.generate_models import generate_DeepConvLSTM_model
from keras.losses import mean_absolute_percentage_error


def __get_ref_generators__(dataset_reduced_std):
    np.random.seed(1234)
    train_names = np.array(['p_17-01-19', 'p_17-01-20'])
    val_names = np.array(['p_17-01-21'])
    test_names = np.array(['p_17-01-22'])
    batch_size = 16
    number_of_predictions = 4
    window_size = 12
    train_gen = DataGenerator(dataset_reduced_std, train_names,
                              "spo2", batch_size=batch_size,
                              number_of_predictions=number_of_predictions,
                              window_size=window_size,
                              step_prediction_dates=1, shuffle=False,
                              rebalance_data=False, debug=False)
    val_gen = DataGenerator(dataset_reduced_std, val_names,
                            "spo2", batch_size=batch_size,
                            number_of_predictions=number_of_predictions,
                            window_size=window_size,
                            step_prediction_dates=1, shuffle=False,
                            rebalance_data=False, debug=False)
    test_gen = DataGenerator(dataset_reduced_std, test_names,
                             "spo2", batch_size=batch_size,
                             number_of_predictions=number_of_predictions,
                             window_size=window_size,
                             step_prediction_dates=1, shuffle=False,
                             rebalance_data=False, debug=False)
    return train_gen, val_gen, test_gen

def __get_ref_model__(dim_length, dim_channels, output_dim):
    # models
    hyperparameters = {}
    regularization_rate = 10 ** -4  # max bound
    hyperparameters['regularization_rate'] = regularization_rate
    learning_rate = 10 ** -4  # max bound
    hyperparameters['learning_rate'] = learning_rate
    filters = []
    hyperparameters['filters'] = filters
    lstm_dims = [48]
    hyperparameters['lstm_dims'] = lstm_dims

    model = generate_DeepConvLSTM_model(dim_length, dim_channels, output_dim,
                                        filters, lstm_dims, learning_rate,
                                        regularization_rate,
                                        metrics=[mean_absolute_percentage_error],
                                        dropout=None, dropout_rnn=0.75, dropout_cnn=0.75)

    return model, hyperparameters


