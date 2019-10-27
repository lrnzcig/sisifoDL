from data_generator import DataGenerator
from get_dataset_pulsi import get_dataset_pulsi

import numpy as np
import pickle
import unittest


class test_shifted(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)

    def setUp(self):
        columns = np.array(['bpm', 'spo2'])
        self.dataset_reduced_std, _ = get_dataset_pulsi(columns,
                                                        filename='./test_data/42nights_shifted.csv')

        test_names_orig = \
            np.array(['p_17-02-19', 'p_17-02-20', 'p_17-02-21', 'p_17-02-22',
                      'p_17-02-23', 'p_17-02-24', 'p_17-02-25', 'p_17-04-27'])
        test_names = [n + "_" + str(i) for n in test_names_orig for i in range(0, 15)]

        batch_size = 32
        number_of_predictions = 4  # numero de predicciones en la salida
        window_size = 12  # ventana de valores para la prediccion

        self.test_gen = \
            DataGenerator(self.dataset_reduced_std, test_names,
                          "spo2", batch_size=batch_size,
                          number_of_predictions=number_of_predictions,
                          window_size=window_size,
                          step_prediction_dates=1, shuffle=False,
                          rebalance_data=False,
                          debug=False)

        train_names_orig = \
            np.array(['p_17-01-19', 'p_17-01-20', 'p_17-01-21', 'p_17-01-22', 'p_17-01-23', 'p_17-01-24', 'p_17-01-25',
                      'p_17-01-26', 'p_17-01-27', 'p_17-01-28', 'p_17-01-29', 'p_17-01-30', 'p_17-01-31', 'p_17-02-01',
                      'p_17-02-02', 'p_17-02-03', 'p_17-02-04', 'p_17-02-05', 'p_17-02-06', 'p_17-02-07', 'p_17-02-08',
                      'p_17-02-09', 'p_17-02-10'])
        train_names = [n + "_" + str(i) for n in train_names_orig for i in range(0, 15)]

        self.train_gen = \
            DataGenerator(self.dataset_reduced_std, train_names,
                                  "spo2", batch_size=batch_size,
                                  number_of_predictions=number_of_predictions,
                                  window_size=window_size,
                                  step_prediction_dates=1, shuffle=False,
                                  rebalance_data=True, rebalance_threshold=0.5,
                                  debug=False)

        val_names_orig = \
            np.array(['p_17-02-11', 'p_17-02-12', 'p_17-02-13', 'p_17-02-14',
                      'p_17-02-15', 'p_17-02-16', 'p_17-02-17', 'p_17-02-18'])
        val_names = [n + "_" + str(i) for n in val_names_orig for i in range(0, 15)]

        self.val_gen = DataGenerator(self.dataset_reduced_std, val_names,
                                "spo2", batch_size=batch_size,
                                number_of_predictions=number_of_predictions,
                                window_size=window_size,
                                step_prediction_dates=1, shuffle=False,
                                rebalance_data=True, rebalance_threshold=0.5,
                                debug=False)

    def test_basic(self):
        with open("test_gen.pkl", 'wb') as output:
            pickle.dump(self.test_gen, output, pickle.HIGHEST_PROTOCOL)

        X, y = self.test_gen.get_all_batches()

        self.assertEqual(np.isclose(X.sum(), 2238170.909015533), True)
        self.assertEqual(np.isclose(y.sum(), 517260.953926506), True)

        with open("val_gen.pkl", 'wb') as output:
            pickle.dump(self.val_gen, output, pickle.HIGHEST_PROTOCOL)

        X, y = self.val_gen.get_all_batches()

        self.assertEqual(np.isclose(X.sum(), 2267565.531491879), True)
        self.assertEqual(np.isclose(y.sum(), 516429.0304805267), True)

        with open("train_gen.pkl", 'wb') as output:
            pickle.dump(self.train_gen, output, pickle.HIGHEST_PROTOCOL)

        X, y = self.train_gen.get_all_batches()

        self.assertEqual(np.isclose(X.sum(), 6315931.116053891), True)
        self.assertEqual(np.isclose(y.sum(), 1438494.9115140745), True)
