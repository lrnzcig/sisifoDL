import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from keras.callbacks import EarlyStopping
from tensorflow import set_random_seed
from sklearn.metrics import mean_squared_error

from utils.generate_models import generate_models
from utils.generate_models import generate_DeepConvLSTM_model
from utils.data_generator import DataGenerator

import pickle
import traceback
import random

import tensorflow as tf


def set_seed_secure(reproducible=True,
                    seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    if reproducible:
        # reproducible results only when using 1 thread!!
        # https://github.com/keras-team/keras/issues/2280
        tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1))


def train_models_on_samples(train_gen, val_gen, models,
                            nr_epochs=5, verbose=True,
                            model_path=None, early_stopping=False,
                            metric_name='pearson_corr',
                            use_testset=False, test_gen=None,
                            reproducible=True, seed=1234,
                            early_stopping_patience=20,
                            debug=True,
                            debug_file_suffix='no_cv',
                            dir_name="./"):
    """
    Given a list of compiled models, this function trains
    them all on a subset of the train data. If the given size of the subset is
    smaller then the size of the data, the complete data set is used.

    TODO refactoring should not be used with use_testset == False so that
    TODO val_* outputs correspond to validation set
    TODO (else add new ouputs test_*)

    Parameters
    ----------
    train_gen : DataGenerator for training
    val_gen : DataGenerator for validation
    models : list of model, params, modeltypes
        List of keras models to train
    nr_epochs : int, optional
        nr of epochs to use for training one model
    verbose : bool, optional
        flag for displaying verbose output
    model_path : str, optional
        Directory to store the models as HDF5 files
    early_stopping: bool
        Stop when validation loss does not decrease
        It needs validacion dataset
    batch_size : int
        nr of samples per batch
    metric_name : str
        metric to store in the history object
    use_testset : boolean, optional
        If True, the testset is used for selecting the best architecture
    test_gen: DataGenerator for test
    reproducible: boolean, optional
        if True, results are reproducible but run in 1 thread only
    seed: int, optional
        if not None, a seed is set in a secure way
    early_stopping_patience: int, optional
        patience if early_stopping=True
        (i.e. number of epochs for which loss does not need to improve)
    debug : boolean, optional
        If True (by default) intermediary files are written as pickle
    debug_file_suffix : str, optional
        suffix to add to output pickle file names
    dir_name : str, optional
        directory for output pickle files if debug is True

    Returns
    ----------
    histories : list of Keras History objects
        train histories for all models
    val_metrics : list of floats
        validation accuraracies of the models
        If no validation dataset, gives training looses
    val_losses : list of floats
        validation losses of the models
        If no validation dataset, gives training losses
    """
    histories = []
    val_metrics = []
    val_losses = []
    for i, (model, params) in enumerate(models):
        if verbose:
            print('Training model %d' % i)
            print(params)
            model.summary()
        if early_stopping:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=early_stopping_patience,
                              verbose=verbose, mode='min')]
        else:
            callbacks = []

        if seed is not None:
            set_seed_secure(reproducible, seed)

        history = model.fit_generator(train_gen, steps_per_epoch=len(train_gen),
                                      epochs=nr_epochs, shuffle=True,
                                      validation_data=val_gen, validation_steps=len(val_gen),
                                      verbose=verbose, callbacks=callbacks)
        histories.append(history)

        try:
            history.history[metric_name]
        except KeyError as e:
            raise(Exception("Model metric does not correspond with metric name"))

        if use_testset:
            # evaluation is done using test set
            testing_error = \
                model.evaluate_generator(test_gen, verbose=0)
            val_losses.append(testing_error[0])
            val_metrics.append(testing_error[1])
        else:
            if 'val_loss' in history.history:
                # validation set is available
                val_metrics.append(history.history['val_' + metric_name][-1])
                val_losses.append(history.history['val_loss'][-1])
            else:
                # no validation test
                val_metrics.append(history.history[metric_name][-1])
                val_losses.append(history.history['loss'][-1])

        if model_path is not None:
            # TODO not tested
            model.save(os.path.join(model_path, 'model_{}.h5'.format(i)))

        if debug:
            # write to intermediary files
            with open(dir_name +'model_' + debug_file_suffix + '_' +
                      str(i) + '.pkl', 'wb') as output:
                pickle.dump(params, output, pickle.HIGHEST_PROTOCOL)
            with open(dir_name +'history_' + debug_file_suffix + '_' +
                      str(i) + '.pkl', 'wb') as output:
                if 'val_loss' in history.history:
                    pickle.dump(pd.DataFrame(data={'loss':history.history['loss'],
                                                   'metric':history.history[metric_name],
                                                   'val_loss':history.history['val_loss'],
                                                   'val_metric':history.history['val_' +  metric_name]}),
                                output, pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(pd.DataFrame(data={'loss':history.history['loss'],
                                                   'metric':history.history[metric_name]}),
                                output, pickle.HIGHEST_PROTOCOL)
            with open(dir_name +'val_metrics_' + debug_file_suffix +
                      '.pkl', 'wb') as output:
                pickle.dump(val_metrics, output, pickle.HIGHEST_PROTOCOL)
            with open(dir_name +'val_losses_' + debug_file_suffix +
                      '.pkl', 'wb') as output:
                pickle.dump(val_losses, output, pickle.HIGHEST_PROTOCOL)

    return histories, val_metrics, val_losses


def get_best_model(val_losses, val_metrics, models, metric,
                   verbose=True):
    metric_name = metric.__name__

    best_model_losses_index = np.nanargmin(val_losses)
    best_model_losses, best_params_losses = models[best_model_losses_index]

    if metric_name == "pearson_corr":
        best_model_metrics_index = np.nanargmax(val_metrics)
    elif metric_name == "mean_absolute_percentage_error":
        best_model_metrics_index = np.nanargmin(val_metrics)
    else:
        raise(Exception("Metric " + metric_name + " unknown, don't know if best is max or min"))
    best_model_metrics, best_params_metrics = models[best_model_metrics_index]

    if verbose:
        print('=> Best model for losses')
        _evaluate_print_out(best_model_losses_index, best_params_losses,
                            val_losses, val_metrics, metric)
        if best_model_losses_index == best_model_metrics_index:
            print('=> Best model for losses is the same as the best model for metric...')
        else:
            print('=> Best model for metrics')
            _evaluate_print_out(best_model_metrics_index, best_params_metrics,
                                val_losses, val_metrics, metric)

    return best_model_losses_index, best_model_losses, best_params_losses, \
           best_model_metrics_index, best_model_metrics, best_params_metrics


def find_best_architecture(train_gen, val_gen, test_gen,
                           verbose=True, number_of_models=5,
                           nr_epochs=5, early_stopping=False,
                           model_path=None, metric=None,
                           models=None, use_testset=True,
                           debug=False, debug_file_suffix="no_cv",
                           output_all=False, dir_name = "./",
                           seed=None, test_retrain=True,
                           **kwargs):
    """
    Tries out a number of models on a subsample of the data,
    and outputs the best found architecture and hyperparameters.

    Parameters
    ----------
    train_gen : training set DataGenerator
    val_gen : validation set DataGenerator
    test_gen : test set DataGenerator
    verbose : bool, optional
        flag for displaying verbose output
    number_of_models : int, optiona
        The number of models to generate and test
    nr_epochs : int, optional
        The number of epochs that each model is trained
    early_stopping: bool
        Stop when validation loss does not decrease
    model_path: str, optional
        Directory to save the models as HDF5 files
    metric: function, optional
        metric that is used to evaluate the model on the validation set.
        See https://keras.io/metrics/ for possible metrics
    models: object, optional
        if informed, it does not generate new models but uses the models on
        the list
    use_testset: boolean
        If True, the testset is used for selecting the best architecture
    debug: boolean, optional
        if True, generates more outputs for debugging
    debug_file_suffix : str, optional
        suffix to add to output pickle file names
    output_all: boolean, optional
        if True, there is an additional output with intermediary results
    dir_name: str, optional
        route for output pickle files and log file
    seed: int, optional
        if not None, a seed is set in a secure way
    **kwargs: key-value parameters
        parameters for generating the models
        (see docstring for utils.generate_models.generate_models)

    Returns
    ----------
    best_model_losses : Keras model
        Best performing model, already trained on a small sample data set,
        using validation loss
    best_params_losses : dict
        Dictionary containing the hyperparameters for the best model, using
        validation loss
    best_model_metrics : Keras model
        Best performing model, already trained on a small sample data set,
        using validation metric
    best_params_metrics : dict
        Dictionary containing the hyperparameters for the best model, using
        validation metric
    debug_output : dict
        For debugging, it includes all models, all the history objects from
        training, and validation losses and metrics
    """
    if models is None:
        # generate models if none are provided
        models = generate_models(train_gen.get_window_size(),
                                 train_gen.get_number_of_channels(),
                                 train_gen.get_number_of_predictions(),
                                 number_of_models=number_of_models,
                                 metrics=[metric],
                                 **kwargs)
    metric_name = metric.__name__
    if seed is not None:
        set_seed_secure(True, seed)
    try:
        histories, val_metrics, val_losses = \
            train_models_on_samples(train_gen,
                                    val_gen,
                                    models,
                                    nr_epochs,
                                    verbose=verbose,
                                    early_stopping=early_stopping,
                                    model_path=model_path,
                                    metric_name=metric_name,
                                    use_testset=False,
                                    test_gen=test_gen,
                                    debug=debug,
                                    debug_file_suffix=debug_file_suffix,
                                    dir_name=dir_name,
                                    seed=None)
    except Exception as e:
        try:
            logf = open(dir_name +"logfile.log", "a")
            traceback.print_exc(file=logf)
            raise(e)
        finally:
            logf.close()

    test_histories = []
    test_metrics = []
    test_losses = []
    if use_testset:
        # evaluate test set for each model and redo val_metrics & val_losses
        # TODO decide which mechanism is best
        if not test_retrain:
            # straight forward approach: use the model as trained
            for model, _ in models:
                testing_error = \
                    model.evaluate_generator(test_gen, verbose=0)
                test_losses += [testing_error[0]]
                test_metrics += [testing_error[1]]
        else:
            all_gen = train_gen.get_merged_generator(val_gen)
            for index in range(0, len(models)):
                # create a model from scratch with the same parameters
                _, hyperparameters = models[index]
                model = generate_DeepConvLSTM_model(train_gen.get_window_size(),
                                                    train_gen.get_number_of_channels(),
                                                    train_gen.get_number_of_predictions(),
                                                    hyperparameters['filters'],
                                                    hyperparameters['lstm_dims'],
                                                    hyperparameters['learning_rate'],
                                                    hyperparameters['regularization_rate'],
                                                    dropout=hyperparameters['dropout'],
                                                    dropout_cnn=hyperparameters['dropout_cnn'],
                                                    dropout_rnn=hyperparameters['dropout_rnn'],
                                                    metrics=[metric], set_seed=False)
                models_index = [(model, hyperparameters)]

                # deduce the number of epochs
                nr_epochs_index = len(histories[0].history['loss'])

                # retrain the model
                histories_index, test_metrics_index, test_losses_index = \
                    train_models_on_samples(all_gen,    # train_gen
                                            [],         # val_gen
                                            models_index,
                                            nr_epochs_index,
                                            verbose=verbose,
                                            early_stopping=early_stopping,
                                            model_path=model_path,
                                            metric_name=metric_name,
                                            use_testset=use_testset,
                                            test_gen=test_gen,
                                            debug=False,
                                            debug_file_suffix=debug_file_suffix + '_disregard',
                                            dir_name=dir_name,
                                            seed=None)
                test_losses += test_losses_index
                test_metrics += test_metrics_index
                test_histories += histories_index

    if verbose:
        print(val_metrics)
        print(val_losses)

    _, best_model_losses, best_params_losses, \
    _, best_model_metrics, best_params_metrics = \
        get_best_model(val_losses, val_metrics, models, metric,
                       verbose=verbose)

    if debug or output_all:
        debug_o = {"models" : models,
                   "histories": histories,
                   "val_metrics": val_metrics,
                   "val_losses": val_losses,
                   "test_metrics": test_metrics,
                   "test_losses": test_losses,
                   "test_histories": test_histories}
        return best_model_losses, best_params_losses, best_model_metrics, best_params_metrics, \
            debug_o
    else:
        return best_model_losses, best_params_losses, best_model_metrics, best_params_metrics


def get_generators(dataset_reduced_std,
                   start_train_date,
                   end_train_date,
                   start_val_date,
                   end_val_date,
                   start_test_date,
                   end_test_date,
                   target_variable,
                   batch_size=16,
                   number_of_predictions=15,
                   window_size=30):
    np.random.seed(1234)
    train_gen = DataGenerator(dataset_reduced_std, start_train_date,
                              end_train_date,
                              target_variable, batch_size=batch_size,
                              number_of_predictions=number_of_predictions,
                              window_size=window_size,
                              step_prediction_dates=1, shuffle=True,
                              shuffle_and_sample=False, debug=False)

    val_gen = DataGenerator(dataset_reduced_std, start_val_date,
                            end_val_date,
                            target_variable, batch_size=batch_size,
                            number_of_predictions=number_of_predictions,
                            window_size=window_size,
                            step_prediction_dates=1, shuffle=False,
                            shuffle_and_sample=False, debug=False)

    test_gen = DataGenerator(dataset_reduced_std, start_test_date,
                             end_test_date,
                             target_variable, batch_size=batch_size,
                             number_of_predictions=number_of_predictions,
                             window_size=window_size,
                             step_prediction_dates=1, shuffle=False,
                             shuffle_and_sample=False, debug=False)

    return train_gen, val_gen, test_gen


def find_best_architecture_cv(dataset_reduced_std,
                              start_train_date, end_train_date,
                              start_val_date, end_val_date,
                              start_test_date, end_test_date,
                              models, cv_fold_size,
                              target_variable,
                              batch_size=16,
                              number_of_predictions=15,
                              window_size=30,
                              verbose=True,
                              nr_epochs=5, early_stopping=False,
                              model_path=None, metric=None,
                              use_testset=True, debug=True,
                              dir_name="./", seed=1234,
                              test_retrain=True):
    """
    For the list of models, tries out the different architectures using
    forward-chaining-cross-validation.
    It ouputs the best architecture found, based on performance of:
    - either testset, if use_testset=True
    - or mean of validation folds, if use_testset=False

    :param dataset_reduced_std: pandas dataframe
        Standarized input dataframe, which includes a column called "date"
    :param start_train_date: date
        Start date of training dataset (fixed)
    :param end_train_date: date
        End date of training dataset (fixed)
    :param start_val_date: date
        Start date of validation dataset
        Note that actually the validation dataset is divided into folds
    :param end_val_date: date
        End date of validation dataset
        Note that actually the validation dataset is divided into folds
    :param start_test_date: date
        Start date of test dataset
        It will be used for evaluation only if use_testset is True. If it
        is not, this data is reserved and may be used as validation for
        the choosen architecture
    :param end_test_date:
        End date of test dataset
        It will be used for evaluation only if use_testset is True. If it
        is not, this data is reserved and may be used as validation for
        the choosen architecture
    :param models: list of tuple (model, hyperparamets)
        List of models to be tested
    :param cv_fold_size: int
        Size of validation data per fold. User of the function takes care that
        the size of the full validation set is multiple of fold size; no extra
        checks in the code
    :param target_variable: str
        Name of target variable in the input dataset
    :param batch_size: int
        Batch size for DataGenerators
    :param number_of_predictions: int
        Number of timesteps of the output prediction of the model
    :param window_size: int
        Size of the backwards window of the prediction
    :param verbose: boolean, optional
        If true, outputs messages from the model training and for the results
    :param nr_epochs: int
        Maximum number of epochs for training
    :param early_stopping: boolean
        If true, it stops early, if validation performance decreases
    :param model_path: str
        Obsolete
    :param metric: function
        Metric used as an evaluation in addition to RMSE
    :param use_testset: boolean
        If True, the performance for the reserved data in the test set is used
        for choosing the best architecture. If False, the mean of performances
        for validation folds is used for choosing the best arch.
    :param debug: boolean
        If True, it outputs a number of pkl files with the results. Recommended if
        training in the cloud (robust to notebook disconnect)
    :param dir_name: str, optional
        Directory name for the output files if debug is set to True
    :param seed: int, optional
        If not None, a seed is set in a secure way
    :return:
        best_model_losses_index: index of best model according to losses
        best_model_metrics_index: index of best model according to metric
        val_losses_list: full list of results per model/folds for losses
            2D list with number_of_folds, number_of_models
        val_metrics_list: full list of results per model/folds for metrics
            2D list with number_of_folds, number_of_models
    """
    if seed is not None:
        set_seed_secure(True, seed)

    _, val_gen, _ = \
        get_generators(dataset_reduced_std,
                       start_train_date,
                       end_train_date,
                       start_val_date,
                       end_val_date,
                       start_test_date,
                       end_test_date,
                       target_variable=target_variable,
                       batch_size=batch_size,
                       number_of_predictions=number_of_predictions,
                       window_size=window_size)

    val_losses_list = []
    val_metrics_list = []
    test_losses_list = []
    test_metrics_list = []
    all_dates = pd.to_datetime(dataset_reduced_std["date"])
    number_of_folds = np.ceil(len(val_gen) * batch_size / cv_fold_size)
    if verbose:
        print("number_of_folds: " + str(number_of_folds))

    for fold in range(0, int(number_of_folds)):
        end_train_date_index = all_dates[all_dates == end_train_date].index[0]
        start_val_date_index = all_dates[all_dates == start_val_date].index[0]
        new_end_train_date = all_dates.iloc[end_train_date_index + cv_fold_size * fold]
        new_start_val_date = all_dates.iloc[start_val_date_index + cv_fold_size * fold]
        new_end_val_date = all_dates.iloc[start_val_date_index + cv_fold_size * (fold + 1) - 1]
        if verbose:
            print(new_end_train_date)
            print(new_start_val_date)
            print(new_end_val_date)
        train_gen, val_gen, test_gen = \
            get_generators(dataset_reduced_std,
                           start_train_date,
                           new_end_train_date,
                           new_start_val_date,
                           new_end_val_date,
                           start_test_date,
                           end_test_date,
                           target_variable=target_variable,
                           batch_size=batch_size,
                           number_of_predictions=number_of_predictions,
                           window_size=window_size)
        x_train, y_train = train_gen.get_all_batches()
        x_val, y_val = val_gen.get_all_batches()
        x_test, y_test = test_gen.get_all_batches()
        if verbose:
            print(y_train.shape)
            print(y_val.shape)
            print(y_test.shape)

        test_retrain_fold = False
        last_fold = fold == int(number_of_folds) - 1
        if use_testset and test_retrain and last_fold:
            # take into account test results only for the last fold, since the model is retrained
            test_retrain_fold = True
        _, _, _, _, debug_o = \
            find_best_architecture(train_gen, val_gen, test_gen,
                                   verbose=verbose, number_of_models=None, nr_epochs=nr_epochs,
                                   early_stopping=early_stopping, metric=metric,
                                   models=models, use_testset=use_testset, model_path=model_path,
                                   debug=debug, debug_file_suffix="fold" + str(fold),
                                   output_all=True, dir_name=dir_name,
                                   seed=None, test_retrain=test_retrain_fold)
        val_losses_list += [debug_o['val_losses']]
        val_metrics_list += [debug_o['val_metrics']]
        if use_testset:
            # TODO actually, only makes sense for the last fold
            test_losses_list += [debug_o['test_losses']]
            test_metrics_list += [debug_o['test_metrics']]

    val_losses = pd.DataFrame(val_losses_list).mean(axis=0).tolist()
    val_metrics = pd.DataFrame(val_metrics_list).mean(axis=0).tolist()
    if verbose:
        print(val_losses_list)
        print(val_losses)
        print(val_metrics_list)
        print(val_metrics)
        print(test_losses_list)
        print(test_metrics_list)

    if debug:
        with open(dir_name + 'val_losses_list.pkl', 'wb') as output:
            pickle.dump(val_losses_list, output, pickle.HIGHEST_PROTOCOL)
        with open(dir_name + 'val_metrics_list.pkl', 'wb') as output:
            pickle.dump(val_metrics_list, output, pickle.HIGHEST_PROTOCOL)
        with open(dir_name + 'test_losses_list.pkl', 'wb') as output:
            pickle.dump(test_losses_list, output, pickle.HIGHEST_PROTOCOL)
        with open(dir_name + 'test_metrics_list.pkl', 'wb') as output:
            pickle.dump(test_metrics_list, output, pickle.HIGHEST_PROTOCOL)
        with open(dir_name + 'val_losses.pkl', 'wb') as output:
            pickle.dump(val_losses, output, pickle.HIGHEST_PROTOCOL)
        with open(dir_name + 'val_metrics.pkl', 'wb') as output:
            pickle.dump(val_metrics, output, pickle.HIGHEST_PROTOCOL)

    if use_testset:
        test_losses = test_losses_list[-1] # last fold
        test_metrics = test_metrics_list[-1] # last fold
        best_model_losses_index, _, _, \
        best_model_metrics_index, _, _ = \
            get_best_model(test_losses, test_metrics, models, metric,
                           verbose=verbose)
    else:
        best_model_losses_index, _, _, \
        best_model_metrics_index, _, _ = \
            get_best_model(val_losses, val_metrics, models, metric,
                           verbose=verbose)

    return best_model_losses_index, best_model_metrics_index, \
           val_losses_list, val_metrics_list, \
           test_losses_list, test_metrics_list


def _evaluate_print_out(best_model_index, best_params, val_losses, val_metrics,
                        metric=None):
    """
    Internal. It prints out several parameters for evaluation.

    Parameters
    ----------
    best_model_index : int
        Index of the best model, for accessing the lists for losses and metrics
    best_params : dict
        Hyperparameters for the model
    val_losses : list
        List of validation losses for each of the models
    val_metrics : list
        List of validation metric results for each of the models
    metric : object, optional
        Metric used

    Returns
    ----------
    None
    """
    print('Best model: model ', best_model_index)
    print('Hyperparameters: ', best_params)
    print('Loss on validation set: ', val_losses[best_model_index])
    if metric is not None:
        print(metric.__name__ + ' on validation set: ',
              val_metrics[best_model_index])


def evaluate_model(train_gen, val_gen, test_gen,
                   model, verbose=True):
    """
    Evaluates train, val and test + gives back predictions for each.

    :param train_gen: DataGenerator
        Training generator
    :param val_gen: DataGenerator
        Validation generator
    :param test_gen: DataGenerator
        Test generator
    :param model: object
        Model that has already been trained
    :param verbose: boolean, optional
        If true, shows results of the evaluation
    :return: train_predict: numpy array
        Prediction for training set
    :return: val_predict: numpy array
        Prediction for validation set
    :return: test_predict: numpy array
        Prediction for test set
    """
    training_error = model.evaluate_generator(train_gen, verbose=0)
    validation_error = model.evaluate_generator(val_gen, verbose=0)
    testing_error = model.evaluate_generator(test_gen, verbose=0)

    if verbose:
        print('training error = ' + str(training_error))
        print('validation error = ' + str(validation_error))
        print('testing error = ' + str(testing_error))

    train_predict = model.predict_generator(train_gen)
    val_predict = model.predict_generator(val_gen)
    test_predict = model.predict_generator(test_gen)

    return train_predict, val_predict, test_predict

def corr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_plot(original_series, val_predict, test_predict,
                  end_test_date=None,
                  plot_window=200, target_variable=None,
                  prediction=None, y_test=None,
                  do_denormalize=True, do_plot=True,
                  metric=None):
    """
    Given a list of compiled models, this function trains
    them all on a subset of the train data. If the given size of the subset is
    smaller then the size of the data, the complete data set is used.

    Parameters
    ----------
    original_series : pandas dataframe
        Reference dataframe
    val_predict : numpy array of shape (num_samples_val, number_of_predictions)
        Prediction for the validation dataset
    test_predict : numpy array of shape (num_samples_test, number_of_predictions)
        Prediction for the test dataset
    plot_window : int, optional
        Number of samples shown in plot for training test (200 by default)
    target_variable : str, optional
        Name of target variable TODO cannot be optional
    prediction : int
        Index for prediction, i.e. index in the test_predict and y_test arrays
    y_test : numpy array of shape (num_samples_test, number_of_predictions)
        True value for test dataset ouput
    do_denormalize : boolean
        If True, values are denormalized using original_series
    do_plot : boolean
        If False, no plots are shown

    Returns
    ----------
    rmse_test : float
        RMSE for test set
    corr_test : float
        Correlation for test set
    """
    # init
    val_size = len(val_predict)
    val_length = 0 if val_size == 0 else val_predict.shape[1]
    test_size = len(test_predict)
    test_length = test_predict.shape[1]

    # plot original series
    original_series_2plot = original_series
    if end_test_date is not None:
        original_series_2plot = \
            original_series_2plot[pd.to_datetime(original_series_2plot["date"]) <= end_test_date]
    max_series_limit = original_series_2plot.shape[0]
    if prediction is not None and end_test_date is None:
        max_series_limit += prediction-test_size
    plt.plot(np.arange(max_series_limit-plot_window,
                       max_series_limit, 1),
             original_series_2plot[max_series_limit-plot_window:max_series_limit][target_variable],
             color = 'k')

    split_pt = max_series_limit - val_length - test_length

    # plot validation set prediction
    # if val_predict is an empty list it means there is no validation set
    val_predict_2plot = val_predict
    if val_size != 0:
        if do_denormalize:
            val_predict_dolar = de_normalize_prediction(original_series, val_predict[prediction],
                                                        target_variable)
            val_predict_2plot = val_predict_dolar
        elif prediction is not None:
            val_predict_2plot = val_predict[prediction]
        plt.plot(np.arange(split_pt,
                           split_pt + len(val_predict_2plot),1),
                 val_predict_2plot, color = 'y')

    # plot test set prediction
    test_predict_2plot = test_predict
    if do_denormalize and y_test is not None:
        y_test_dolar = de_normalize_prediction(original_series, y_test[prediction],
                                               target_variable)
        test_predict_dolar = de_normalize_prediction(original_series, test_predict[prediction],
                                                     target_variable)
        test_predict_2plot = test_predict_dolar
    elif prediction is not None:
        test_predict_2plot = test_predict[prediction]
    plt.plot(np.arange(split_pt + len(val_predict_2plot),
                       split_pt + len(val_predict_2plot) + len(test_predict_2plot),1),
             test_predict_2plot, color = 'r')

    # evaluate RMSE/correlation
    title = None
    rmse_test = None
    corr_test = None
    if metric is None:
        metric = corr

    if y_test is not None:
        if do_denormalize:
            rmse_test = np.sqrt(mean_squared_error(y_test_dolar, test_predict_dolar))
            corr_test = metric(y_test_dolar, test_predict_dolar)
        else:
            rmse_test = np.sqrt(mean_squared_error(y_test[prediction], test_predict[prediction]))
            corr_test = metric(y_test[prediction], test_predict[prediction])

    title="RMSE= " + str(rmse_test) + ", " + metric.__name__ + "= " + str(corr_test)

    # pretty up graph
    plt.xlabel('day')
    plt.ylabel(target_variable)
    if val_size != 0:
        plt.legend(['original series', 'validation fit', 'testing fit'],
                   loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(['original series', 'testing fit'],
                   loc='center left', bbox_to_anchor=(1, 0.5))

    if title is not None:
        plt.title(title)
    if do_plot:
        plt.show()

    return rmse_test, corr_test


def de_normalize_predictions(original_series, y_train, y_val, y_test,
                             train_predict, val_predict, test_predict,
                             target_variable, verbose=True,
                             calculate_correlations=True):
    """
    De-normalize predictions using the original series
    TODO it would be better to do this using sklearn's scaler

    :param original_series: pandas dataframe
        Reference dataframe
    :param y_train: numpy array
        Real value of the target variable for the training set
    :param y_val: numpy array
        Real value of the target variable for the validation set
    :param y_test: numpy array
        Real value of the target variable for the test set
    :param train_predict: numpy array
        Prediction for the training set
    :param val_predict: numpy array
        Prediction for the validation set
    :param test_predict: numpy array
        Prediction for the test set
    :param target_variable: str
        Name of the target variable 
    :param verbose: boolean, optional
        If true, results for RMSE and correlation are shown
    :param calculate_correlations: boolean, optional
        If true, results for correlations are shown
    :return: train_predict_dolar: numpy array
        Prediction in dollars for the training set
    :return: val_predict_dolar: numpy array
        Prediction in dollars for the validation set
    :return: test_predict_dolar: numpy array
        Prediction in dollars for the test set
    """
    train_predict_dolar = de_normalize_prediction(original_series,
                                                  train_predict,
                                                  target_variable)
    val_predict_dolar = de_normalize_prediction(original_series,
                                                val_predict,
                                                target_variable)
    test_predict_dolar = de_normalize_prediction(original_series,
                                                 test_predict,
                                                 target_variable)
    y_train_dolar = de_normalize_prediction(original_series,
                                            y_train,
                                            target_variable)
    y_val_dolar = de_normalize_prediction(original_series,
                                          y_val,
                                          target_variable)
    y_test_dolar = de_normalize_prediction(original_series,
                                           y_test,
                                           target_variable)

    if verbose:
        print("RMSE train : " + str(np.sqrt(mean_squared_error(y_train_dolar, train_predict_dolar))))
        print("RMSE val   : " + str(np.sqrt(mean_squared_error(y_val_dolar, val_predict_dolar))))
        print("RMSE test  : " + str(np.sqrt(mean_squared_error(y_test_dolar, test_predict_dolar))))

        if calculate_correlations:
            print("Corr train : " + str(np.corrcoef(y_train_dolar, np.transpose(train_predict_dolar))[0,1]))
            print("Corr val   : " + str(np.corrcoef(y_val_dolar, np.transpose(val_predict_dolar))[0,1]))
            print("Corr test  : " + str(np.corrcoef(y_test_dolar, np.transpose(test_predict_dolar))[0,1]))

    return train_predict_dolar, val_predict_dolar, test_predict_dolar


def de_normalize_prediction(original_series, series,
                            target_variable):
    """
    De-normalize predictions using the original series
    TODO it would be better to do this using sklearn's scaler

    :param original_series: pandas dataframe
        Reference dataframe
    :param series: pandas dataframe
        Dataframe to be de-nomalized
    :param target_variable: str
        Name of the target variable
    :return: pandas Series
        De-normalized variable
    """
    y_min = min(original_series[target_variable])
    y_max = max(original_series[target_variable])

    return series * (y_max - y_min) + y_min


