from data_generator import DataGenerator
from get_dataset_pulsi import get_dataset_pulsi
from generate_models import generate_DeepConvLSTM_model
import unittest
import numpy as np
from tensorflow import set_random_seed
from scipy.stats.stats import pearsonr

class Test0(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test0, self).__init__(*args, **kwargs)

    def setUp(self):
        # read dataset
        columns = np.array(['bpm', 'spo2'])
        self.dataset_reduced_std, _ = get_dataset_pulsi(columns,
                                                        filename='./test_data/42nights.csv')

        # instatiate DataGenerator
        self.train_names = np.array(['p_17-01-19', 'p_17-01-20'])
        self.test_names = np.array(['p_17-01-21'])
        self.batch_size = 16
        self.number_of_predictions = 4
        self.window_size = 12
        self.number_of_columns = 2
        self.train_gen = DataGenerator(self.dataset_reduced_std, self.train_names,
                                       "spo2", batch_size=self.batch_size,
                                       number_of_predictions=self.number_of_predictions,
                                       window_size=self.window_size,
                                       step_prediction_dates=1, shuffle=False,
                                       debug=False)
        self.test_gen = DataGenerator(self.dataset_reduced_std, self.test_names,
                                      "spo2", batch_size=self.batch_size,
                                      number_of_predictions=self.number_of_predictions,
                                      window_size=self.window_size,
                                      step_prediction_dates=1, shuffle=False,
                                      debug=False)

        regularization_rate = 10 ** -4
        learning_rate = 10 ** -4
        filters = []
        lstm_dims = [48]
        dim_length = self.window_size
        dim_channels = self.number_of_columns
        output_dim = self.number_of_predictions
        np.random.seed(0)
        set_random_seed(1234)
        self.model = generate_DeepConvLSTM_model(dim_length, dim_channels, output_dim,
                                                 filters, lstm_dims, learning_rate,
                                                 regularization_rate,
                                                 dropout=None, dropout_rnn=0.75, dropout_cnn=0.75)
        nrepochs = 5
        history = self.model.fit_generator(self.train_gen, steps_per_epoch=len(self.train_gen),
                                           epochs=nrepochs, shuffle=True, verbose=0)
        self.X_test, self.y_test = self.test_gen.get_all_batches()

    def predict_and_check(self, index):
        test_predict = self.model.predict(self.X_test[index].reshape(1, self.window_size, self.number_of_columns))
        pearson = pearsonr(self.y_test[index], test_predict.reshape(self.number_of_predictions))[0]
        eval = self.model.evaluate(self.X_test[index].reshape(1, self.window_size, self.number_of_columns),
                                   self.y_test[index].reshape(1, self.number_of_predictions),
                                   verbose=0)[1]
        self.assertAlmostEqual(pearson, eval, places=5)
        return pearson, eval

    def test(self):
        # test single values
        self.predict_and_check(7) # careful with flat values at the beginning of the file
        self.predict_and_check(8)

        # test mean of two values
        pearson_0, _ = self.predict_and_check(7)
        pearson_1, _ = self.predict_and_check(8)
        pearson_mean = np.mean([pearson_0, pearson_1])
        eval = self.model.evaluate(self.X_test[7:9].reshape(2, self.window_size, self.number_of_columns),
                                   self.y_test[7:9].reshape(2, self.number_of_predictions),
                                   verbose=0)[1]
        self.assertAlmostEqual(pearson_mean, eval, places=5)

