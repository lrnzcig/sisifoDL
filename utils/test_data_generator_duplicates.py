from data_generator import DataGenerator
from get_dataset_pulsi import get_dataset_pulsi
import unittest
import pandas as pd
import numpy as np

import pickle

class Test0(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test0, self).__init__(*args, **kwargs)
        # read dataset
        columns = np.array(['bpm', 'spo2'])
        self.dataset_reduced_std, _ = get_dataset_pulsi(columns,
                                                        filename='./test_data/42nights.csv')

        # instatiate DataGenerator
        self.names = np.array(['h_17-04-27', 'p_17-04-27'])
        self.batch_size = 3
        self.number_of_predictions = 4
        self.window_size = 12
        self.generator = DataGenerator(self.dataset_reduced_std, self.names,
                                       "spo2", batch_size=self.batch_size,
                                       number_of_predictions=self.number_of_predictions,
                                       window_size=self.window_size,
                                       step_prediction_dates=1, shuffle=False,
                                       rebalance_data=False, debug=False)

    def test_generator(self):
        filtered_by_names = self.dataset_reduced_std[self.dataset_reduced_std["name"].isin(self.names)]
        expected_lines = len(filtered_by_names) - len(self.names) * (self.window_size + self.number_of_predictions - 1)
        expected = np.ceil(expected_lines / self.batch_size)
        # checks
        self.assertEqual(len(self.generator), expected)

        X0, y0 = self.generator.__getitem__(0)
        self.assertEqual(X0.shape, (self.batch_size, self.window_size, 2)) # 2 columns
        self.assertEqual(y0.shape, (self.batch_size, self.number_of_predictions))
        #print(X0)
        #print(self.dataset_reduced_std[self.dataset_reduced_std["name"] == 'h_17-04-27'][0:20])
        #print(y0)

        with open('./test_data/X0.pkl', 'rb') as f:
            X0_ref = pd.read_pickle(f)
        self.assertTrue((X0_ref == X0).all())
        with open('./test_data/y0.pkl', 'rb') as f:
            y0_ref = pd.read_pickle(f)
        self.assertTrue((y0_ref == y0).all())

        X_second_last, y_second_last = self.generator.__getitem__(len(self.generator) - 2)
        self.assertEqual(X_second_last.shape, (self.batch_size, self.window_size, 2)) # 2 columns
        self.assertEqual(y_second_last.shape, (self.batch_size, self.number_of_predictions))
        #print(X_second_last)
        #print(self.dataset_reduced_std[self.dataset_reduced_std["name"] == 'p_17-04-27'][-21:])
        #print(y_second_last)


        with open('./test_data/X_second_last_dupl.pkl', 'rb') as f:
            X_second_last_ref = pd.read_pickle(f)
        self.assertTrue((X_second_last_ref == X_second_last).all())
        with open('./test_data/y_second_last_dupl.pkl', 'rb') as f:
            y_second_last_ref = pd.read_pickle(f)
        self.assertTrue((y_second_last_ref == y_second_last).all())

        X_last, y_last = self.generator.__getitem__(len(self.generator) - 1)
        self.assertEqual(X_last.shape, (3, self.window_size, 2)) # could be 2 or 1 instead of 3
        self.assertEqual(y_last.shape, (3, self.number_of_predictions)) # could be 2 or 1 instead of 3
        #print(X_last)
        #print(self.dataset_reduced_std[self.dataset_reduced_std["name"] == 'p_17-04-27'][-20:])
        #print(y_last)

        with open('./test_data/X_last_dupl.pkl', 'wb') as f:
            pickle.dump(X_last, f, pickle.HIGHEST_PROTOCOL)
        with open('./test_data/y_last_dupl.pkl', 'wb') as f:
            pickle.dump(y_last, f, pickle.HIGHEST_PROTOCOL)
        with open('./test_data/X_last_dupl.pkl', 'rb') as f:
            X_last_ref = pd.read_pickle(f)
        self.assertTrue((X_last_ref == X_last).all())
        with open('./test_data/y_last_dupl.pkl', 'rb') as f:
            y_last_ref = pd.read_pickle(f)
        self.assertTrue((y_last_ref == y_last).all())

    def test_all_batches(self):
        X, y = self.generator.get_all_batches()
        X_b, y_b = self.generator.get_all_batches_debug()
        self.assertTrue((X == X_b).all())
        self.assertTrue((y == y_b).all())

    def test_merge(self):
        X, y = self.generator.get_all_batches()

        names_1 = np.array(['h_17-04-27'])
        generator_1 = DataGenerator(self.dataset_reduced_std, names_1,
                                    "spo2", batch_size=3,
                                    number_of_predictions=4, window_size=12,
                                    step_prediction_dates=1, shuffle=False,
                                    rebalance_data=False, debug=False)
        names_2 = np.array(['p_17-04-27'])
        generator_2 = DataGenerator(self.dataset_reduced_std, names_2,
                                    "spo2", batch_size=3,
                                    number_of_predictions=4, window_size=12,
                                    step_prediction_dates=1, shuffle=False,
                                    rebalance_data=False, debug=False)
        generator_all = generator_1.get_merged_generator(generator_2)
        X_b, y_b = generator_all.get_all_batches()
        self.assertTrue((X == X_b).all())

        generator_all_bis = generator_2.get_merged_generator(generator_1)
        X_c, y_c = generator_all_bis.get_all_batches()
        self.assertTrue((X == X_c).all())

