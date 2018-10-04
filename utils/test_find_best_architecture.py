import unittest
import numpy as np
from get_dataset_pulsi import get_dataset_pulsi
from validate_models import find_best_architecture, set_seed_secure
from test_utils import __get_ref_generators__
import keras
from keras.losses import mean_absolute_percentage_error


class Test0(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Test0, self).__init__(*args, **kwargs)
        # read dataset
        columns = np.array(['bpm', 'spo2'])
        self.dataset_reduced_std, _ = get_dataset_pulsi(columns,
                                                        filename='./test_data/42nights.csv')

        self.window_size = 12
        self.number_of_predictions = 4
        self.dim_length = self.window_size
        self.number_of_columns = self.dataset_reduced_std.shape[1] - 2  # remove "datetime" & "name"
        self.dim_channels = self.number_of_columns
        self.output_dim = self.number_of_predictions


    def test_mape(self):
        set_seed_secure(True)
        train_gen, val_gen, test_gen = __get_ref_generators__(self.dataset_reduced_std)
        best_model_losses, best_params_losses, best_model_metrics, best_params_metrics,\
            debug_o = \
            find_best_architecture(train_gen, val_gen, test_gen,
                                   verbose=False, number_of_models=3, nr_epochs=5,
                                   early_stopping=False, metric=mean_absolute_percentage_error,
                                   models=None, use_testset=True, debug=False, output_all=True,
                                   seed=None, test_retrain=False,
                                   deepconvlstm_min_conv_layers=1, deepconvlstm_max_conv_layers=1,
                                   deepconvlstm_min_conv_filters=10, deepconvlstm_max_conv_filters=100,
                                   deepconvlstm_min_lstm_layers=1, deepconvlstm_max_lstm_layers=1,
                                   deepconvlstm_min_lstm_dims=10, deepconvlstm_max_lstm_dims=200,
                                   low_lr=1, high_lr=4, low_reg=1, high_reg=4,
                                   dropout_rnn_max=0.9, dropout_rnn_min=0.2,
                                   dropout_cnn_max=0.9, dropout_cnn_min=0.2)

        self.assertIsInstance(best_model_losses, keras.engine.sequential.Sequential)
        self.assertIsInstance(best_model_metrics, keras.engine.sequential.Sequential)

        self.assertAlmostEqual(best_params_losses['learning_rate'], 0.002257145150581335)
        self.assertAlmostEqual(best_params_metrics['learning_rate'], 0.002257145150581335)

        self.assertAlmostEqual(debug_o['val_losses'][0], 0.006559250258926984, 4)
        self.assertAlmostEqual(debug_o['val_metrics'][0], 4.667705343480696, 4)

        self.assertAlmostEqual(debug_o['test_losses'][0], 0.0073970343518470014, 4)
        self.assertAlmostEqual(debug_o['test_metrics'][0], 5.519604462875432, 4)

    def test_no_testset(self):
        set_seed_secure(True)
        train_gen, val_gen, test_gen = __get_ref_generators__(self.dataset_reduced_std)
        best_model_losses, best_params_losses, best_model_metrics, best_params_metrics,\
            debug_o = \
            find_best_architecture(train_gen, val_gen, test_gen,
                                   verbose=False, number_of_models=3, nr_epochs=5,
                                   early_stopping=False, metric=mean_absolute_percentage_error,
                                   models=None, use_testset=False, debug=False, output_all=True,
                                   seed=None, test_retrain=False,
                                   deepconvlstm_min_conv_layers=1, deepconvlstm_max_conv_layers=1,
                                   deepconvlstm_min_conv_filters=10, deepconvlstm_max_conv_filters=100,
                                   deepconvlstm_min_lstm_layers=1, deepconvlstm_max_lstm_layers=1,
                                   deepconvlstm_min_lstm_dims=10, deepconvlstm_max_lstm_dims=200,
                                   low_lr=1, high_lr=4, low_reg=1, high_reg=4,
                                   dropout_rnn_max=0.9, dropout_rnn_min=0.2,
                                   dropout_cnn_max=0.9, dropout_cnn_min=0.2)

        self.assertIsInstance(best_model_losses, keras.engine.sequential.Sequential)
        self.assertIsInstance(best_model_metrics, keras.engine.sequential.Sequential)

        self.assertAlmostEqual(best_params_losses['learning_rate'], 0.002257145150581335, 4)
        self.assertAlmostEqual(best_params_metrics['learning_rate'], 0.002257145150581335, 4)

        self.assertAlmostEqual(debug_o['val_losses'][0], 0.006559250258926984, 4)
        self.assertAlmostEqual(debug_o['val_metrics'][0], 4.667705343480696, 4)

        self.assertEqual(len(debug_o['test_losses']), 0)
        self.assertEqual(len(debug_o['test_metrics']), 0)


    def test_mape_testretrain(self):
        set_seed_secure(True)
        train_gen, val_gen, test_gen = __get_ref_generators__(self.dataset_reduced_std)
        best_model_losses, best_params_losses, best_model_metrics, best_params_metrics,\
            debug_o = \
            find_best_architecture(train_gen, val_gen, test_gen,
                                   verbose=False, number_of_models=3, nr_epochs=5,
                                   early_stopping=False, metric=mean_absolute_percentage_error,
                                   models=None, use_testset=True, debug=False, output_all=True,
                                   seed=None, test_retrain=True,
                                   deepconvlstm_min_conv_layers=1, deepconvlstm_max_conv_layers=1,
                                   deepconvlstm_min_conv_filters=10, deepconvlstm_max_conv_filters=100,
                                   deepconvlstm_min_lstm_layers=1, deepconvlstm_max_lstm_layers=1,
                                   deepconvlstm_min_lstm_dims=10, deepconvlstm_max_lstm_dims=200,
                                   low_lr=1, high_lr=4, low_reg=1, high_reg=4,
                                   dropout_rnn_max=0.9, dropout_rnn_min=0.2,
                                   dropout_cnn_max=0.9, dropout_cnn_min=0.2)

        self.assertIsInstance(best_model_losses, keras.engine.sequential.Sequential)
        self.assertIsInstance(best_model_metrics, keras.engine.sequential.Sequential)

        self.assertAlmostEqual(best_params_losses['learning_rate'], 0.002257145150581335, 4)
        self.assertAlmostEqual(best_params_metrics['learning_rate'], 0.002257145150581335, 4)

        self.assertAlmostEqual(debug_o['val_losses'][0], 0.006559250258926984, 4)
        self.assertAlmostEqual(debug_o['val_metrics'][0], 4.667705343480696, 4)

        self.assertAlmostEqual(debug_o['test_losses'][0], 0.004608124492637555, 4)
        self.assertAlmostEqual(debug_o['test_metrics'][0], 5.162087056440709, 4)



