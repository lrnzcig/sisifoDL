from data_generator import DataGenerator
from get_dataset_pulsi import get_dataset_pulsi
import unittest
import pandas as pd
import numpy as np

import random

class Test0(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test0, self).__init__(*args, **kwargs)

    def setUp(self):
        # read dataset
        columns = np.array(['bpm', 'spo2'])
        self.dataset_reduced_std, _ = get_dataset_pulsi(columns,
                                                        filename='./test_data/42nights_reduced.csv')

        # instatiate DataGenerator
        self.names = np.array(['p_17-01-19', 'p_17-01-20', 'p_17-01-25'])
        self.batch_size = 4
        self.number_of_predictions = 2
        self.window_size = 4
        self.rebalance_threshold = 0.8
        random.seed(1234)
        self.generator = DataGenerator(self.dataset_reduced_std, self.names,
                                       "spo2", batch_size=self.batch_size,
                                       number_of_predictions=self.number_of_predictions,
                                       window_size=self.window_size,
                                       step_prediction_dates=1, shuffle=False,
                                       rebalance_data=True, rebalance_threshold=self.rebalance_threshold,
                                       debug=False)

    def test_generator(self):
        filtered_by_names = self.dataset_reduced_std[self.dataset_reduced_std["name"].isin(self.names)]
        expected_lines = len(filtered_by_names) - len(self.names) * (self.window_size + self.number_of_predictions - 1)
        expected = np.ceil(expected_lines / self.batch_size)
        # checks
        self.assertEqual(len(self.generator), expected)

        X0, y0 = self.generator.__getitem__(0)
        self.assertEqual(X0.shape, (self.batch_size, self.window_size, 2)) # 2 columns
        self.assertEqual(y0.shape, (self.batch_size, self.number_of_predictions))
        #print(y0)
        self.assertEqual(sum(np.apply_along_axis(self.generator.rebalance_select_row, 1, y0)),
                         self.batch_size/2)

        with open('./test_data/X0_balanced.pkl', 'rb') as f:
            X0_ref = pd.read_pickle(f)
        self.assertTrue((X0_ref == X0).all())
        with open('./test_data/y0_balanced.pkl', 'rb') as f:
            y0_ref = pd.read_pickle(f)
        self.assertTrue((y0_ref == y0).all())

        X_second_last, y_second_last = self.generator.__getitem__(len(self.generator) - 2)
        self.assertEqual(X_second_last.shape, (self.batch_size, self.window_size, 2)) # 2 columns
        self.assertEqual(y_second_last.shape, (self.batch_size, self.number_of_predictions))
        #print(y_second_last)
        self.assertEqual(sum(np.apply_along_axis(self.generator.rebalance_select_row, 1, y_second_last)),
                         self.batch_size/2)

        with open('./test_data/X_second_last_balanced.pkl', 'rb') as f:
            X_second_last_ref = pd.read_pickle(f)
        self.assertTrue((X_second_last_ref == X_second_last).all())
        with open('./test_data/y_second_last_balanced.pkl', 'rb') as f:
            y_second_last_ref = pd.read_pickle(f)
        self.assertTrue((y_second_last_ref == y_second_last).all())

        X_last, y_last = self.generator.__getitem__(len(self.generator) - 1)
        self.assertEqual(X_last.shape, (self.batch_size, self.window_size, 2)) # 2 columns
        self.assertEqual(y_last.shape, (self.batch_size, self.number_of_predictions))
        #print(y_last)
        self.assertEqual(sum(np.apply_along_axis(self.generator.rebalance_select_row, 1, y_last)),
                         self.batch_size/2)

        with open('./test_data/X_last_balanced.pkl', 'rb') as f:
            X_last_ref = pd.read_pickle(f)
        self.assertTrue((X_last_ref == X_last).all())
        with open('./test_data/y_last_balanced.pkl', 'rb') as f:
            y_last_ref = pd.read_pickle(f)
        self.assertTrue((y_last_ref == y_last).all())


