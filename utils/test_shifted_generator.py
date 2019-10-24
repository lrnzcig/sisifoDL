from data_generator import DataGenerator
from get_dataset_pulsi import get_dataset_pulsi

import numpy as np
import pickle
import unittest


class test_shifted(unittest.TestCase):

    def __init__(self, *args, **kwargs):
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

    def test_something(self):
        with open("test_gen.pkl", 'wb') as output:
            pickle.dump(self.test_gen, output, pickle.HIGHEST_PROTOCOL)

        X, y = self.test_gen.get_all_batches()

        self.assertEqual(np.isclose(X.sum(), 93417.1367913111), True)
        self.assertEqual(np.isclose(y.sum(), 93417.1367913111), True)
