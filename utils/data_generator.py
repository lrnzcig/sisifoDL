import numpy as np
import pandas as pd
import keras
import random
import pickle

from datetime import timedelta, datetime


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data, names, target_column_name,
                 batch_size=32, number_of_predictions=15,
                 window_size=30, step_prediction_dates=1, shuffle=False,
                 shuffle_and_sample=False, debug=False):
        'Initialization'
        self.data_backup = data
        self.data = data.drop(["datetime", "name"], axis=1)
        self.names = names
        self.target_column_name = target_column_name
        self.batch_size = batch_size
        self.number_of_predictions = number_of_predictions
        self.window_size = window_size
        self.step_prediction_dates = step_prediction_dates
        self.shuffle = shuffle
        self.shuffle_and_sample = shuffle_and_sample
        self.debug = debug
        self.n_channels = data.shape[1] - 2  # excluding datetime and name
        self.datetimes = None
        self.all_datetimes = pd.DataFrame({"datetime" : pd.to_datetime(data["datetime"]),
                                           "name" : data["name"]})
        self.valid_datetimes = self._get_valid_datetimes_(data)
        self.total_number_of_rows = len(self.valid_datetimes)
        self.__cache_data_generation()
        self.on_epoch_end()
        self.debug_iteration = 0

    def _get_valid_datetimes_(self, data):
        valid_datetime_indexes = pd.Index([])
        for name in self.names:
            name_indexes = data.index[data["name"] == name]
            # remove beginning samples for allowing window_size
            # and ending samples for number_of_predictions
            valid_datetime_indexes_name = name_indexes[self.window_size:len(name_indexes)-self.number_of_predictions+1]
            if len(valid_datetime_indexes_name) == 0:
                raise(Exception("name " + name + " does not have enough rows"))
            valid_datetime_indexes = valid_datetime_indexes.append(valid_datetime_indexes_name)
        output = self.all_datetimes.iloc[valid_datetime_indexes]
        if self.debug:
            print(set([(o.day, o.month) for o in output["datetime"]]))
        return output

    def __len__(self):
        # Denotes the number of batches per epoch
        # NOTE: whould be np.floor instead of np.ceil!
        # - with np.floor and shuffle=True, a few samples are lost in every iteration, but not overall;
        #       however with np.ceil there are small batches that may cause convergence problems
        # - with np.floor and shuffle=False, the last few samples would be lost;
        #       some batches may be be small and may cause convergence problems
        if self.shuffle:
            return int(np.floor(self.total_number_of_rows /
                                (self.step_prediction_dates * self.batch_size)))
        else:
            return int(np.ceil(self.total_number_of_rows /
                               (self.step_prediction_dates * self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if self.shuffle_and_sample:
            datetimes = random.sample(self.valid_datetimes, self.batch_size)
        else:
            datetimes = self.datetimes.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        #print(len(self.valid_datetimes))
        #print(len(self.datetimes))
        #print((index + 1) * self.batch_size)

        #print(datetimes)
        # Generate data
        X, y = self.__get_data_from_cache(datetimes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # Select valid dates and use step for selecting less dates
        # (at the moment not really using it)
        indexes =  self.all_datetimes.index[(self.all_datetimes["datetime"].isin(self.valid_datetimes["datetime"])) & \
                                            (self.all_datetimes["name"].isin(self.valid_datetimes["name"]))]
        if self.step_prediction_dates > 1:
            raise(Exception("Not implemented yet"))
        #print("indexes " + str(len(indexes)))
        #print("len(valid_datetimes) " + str(len(self.valid_datetimes)))
        #print(self.data[self.all_datetimes.duplicated()])
        self.datetimes = self.all_datetimes.iloc[indexes]
        self.datetimes = self.datetimes.reset_index(drop=True)
        if self.shuffle:
            np.random.shuffle(self.datetimes)
        return

    def __cache_data_generation(self):
        self.X_cache = np.empty((max(self.valid_datetimes.index)+1, self.window_size, self.n_channels))
        self.y_cache = np.empty((max(self.valid_datetimes.index)+1, self.number_of_predictions))
        for i, row in self.valid_datetimes.iterrows():
        #for i in self.valid_datetimes.index:
            #print(i)
            #print(self.valid_datetimes.loc[i])
            #datetime_i = self.valid_datetimes.loc[i]
            X, y = self.__data_generation(row)
            self.X_cache[i,] = X
            self.y_cache[i,] = y

    def __get_data_from_cache(self, datetimes):
        #print(datetimes["datetime"])
        #print(datetimes["name"])
        #print(self.valid_datetimes)
        indexes = self.valid_datetimes[self.valid_datetimes["datetime"].isin(datetimes["datetime"]) & \
            self.valid_datetimes["name"].isin(datetimes["name"])].index
        X = self.X_cache[indexes]
        y = self.y_cache[indexes]
        if np.isnan(X).any() or np.isnan(y).any():
            raise Exception("Fatal error: nan values!! Indexes: ", str(indexes))

        if self.debug:
            print(self.debug_iteration)
            with open("tmp/dates_" + str(self.debug_iteration) + ".pkl", "wb") as output:
                pickle.dump(datetimes, output, pickle.HIGHEST_PROTOCOL)
            with open("tmp/X_" + str(self.debug_iteration) + ".pkl", "wb") as output:
                pickle.dump(X, output, pickle.HIGHEST_PROTOCOL)
            with open("tmp/y_" + str(self.debug_iteration) + ".pkl", "wb") as output:
                pickle.dump(y, output, pickle.HIGHEST_PROTOCOL)
            self.debug_iteration += 1

        return X, y

    def __data_generation(self, datetime_row):
        'Generates data containing 1 sample'
        # Initialization
        X = np.empty((1, self.window_size, self.n_channels))
        y = np.empty((1, self.number_of_predictions))

        # Generate data
        datetime_i = datetime_row["datetime"]
        datetime_n = datetime_row["name"]
        datetime_index = self.all_datetimes[(self.all_datetimes["datetime"] == datetime_i) & \
                                            (self.all_datetimes["name"] == datetime_n)].index[0]
        # Store sample
        x_indexes = range(datetime_index - self.window_size, datetime_index)
        X[0,] = self.data.iloc[x_indexes]

        # Store target
        y_indexes = range(datetime_index, datetime_index + self.number_of_predictions)
        data_y = self.data.iloc[y_indexes]
        y[0,] = data_y[self.target_column_name]

        if self.debug:
            print(datetime_i.strftime("%Y-%m-%d %H:%M:%S"))
            print([datetime_j.strftime("%Y-%m-%d %H:%M:%S") for datetime_j in self.all_datetimes.iloc[x_indexes]["datetime"]])
            print(X[0,])
            print([datetime_j.strftime("%Y-%m-%d %H:%M:%S") for datetime_j in self.all_datetimes.iloc[y_indexes]["datetime"]])
            print(data_y)

        return X, y

    def get_all_batches(self):
        indexes =  self.all_datetimes.index[(self.all_datetimes["datetime"].isin(self.valid_datetimes["datetime"])) & \
                                            (self.all_datetimes["name"].isin(self.valid_datetimes["name"]))]
        return self.X_cache[indexes], self.y_cache[indexes]

    def get_all_batches_debug(self):
        X = []
        y = []
        for i in range(0, self.__len__()):
            X_i, y_i = self.__getitem__(i)
            X = np.append(X, X_i)
            y = np.append(y, y_i)
        X = X.reshape(int(np.ceil(X.shape[0]/self.window_size/self.n_channels)),
                      self.window_size, self.n_channels)
        y = y.reshape(int(np.ceil(y.shape[0]/self.number_of_predictions)),
                      self.number_of_predictions)
        return X, y

    def get_merged_generator(self, second):
        merged_names = set(np.append(self.names, second.names))
        return DataGenerator(self.data_backup, merged_names, self.target_column_name, batch_size=self.batch_size,
                             number_of_predictions=self.number_of_predictions, window_size=self.window_size,
                             step_prediction_dates=self.step_prediction_dates, shuffle=self.shuffle,
                             shuffle_and_sample=self.shuffle_and_sample, debug=self.debug)

    def get_number_of_predictions(self):
        return self.number_of_predictions

    def get_window_size(self):
        return self.window_size

    def get_number_of_channels(self):
        data_shape = self.data.shape
        if len(data_shape) == 1:
            return 1
        else:
            return data_shape[1]

