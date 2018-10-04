from get_dataset_pulsi import get_dataset_pulsi
from generate_models import generate_DeepConvLSTM_model
from validate_models import train_models_on_samples
from test_utils import __get_ref_generators__, __get_ref_model__
import unittest
import numpy as np



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
        self.number_of_columns = self.dataset_reduced_std.shape[1] - 2 # remove "datetime" & "name"
        self.dim_channels = self.number_of_columns
        self.output_dim = self.number_of_predictions

        model, self.hyperparameters = __get_ref_model__(self.dim_length, self.dim_channels, self.output_dim)
        self.models = [(model, self.hyperparameters)]

    def test_base(self):
        train_gen, val_gen, test_gen = __get_ref_generators__(self.dataset_reduced_std)
        histories, val_metrics, val_losses = \
            train_models_on_samples(train_gen, val_gen, self.models,
                                    nr_epochs=20, verbose=False,
                                    model_path=None, early_stopping=False,
                                    metric_name='mean_absolute_percentage_error',
                                    debug=False, use_testset=True,
                                    test_gen=test_gen)

        history_model = histories[0].history
        self.assertEqual(len(history_model['loss']), 20)
        self.assertEqual(len(history_model['mean_absolute_percentage_error']), 20)
        self.assertEqual(len(history_model['val_loss']), 20)
        self.assertEqual(len(history_model['val_mean_absolute_percentage_error']), 20)

        self.assertAlmostEqual(history_model['loss'][0], 0.7767009734269942)
        self.assertAlmostEqual(history_model['mean_absolute_percentage_error'][0], 87994.5229075557)
        self.assertAlmostEqual(history_model['val_loss'][0], 0.47735168890048785)
        self.assertAlmostEqual(history_model['val_mean_absolute_percentage_error'][0], 78.18934104874818)

        self.assertAlmostEqual(val_metrics[0], 8.455293159361009)
        self.assertAlmostEqual(val_losses[0], 0.01196039453173032)

    def test_base_notestset(self):
        train_gen, val_gen, test_gen = __get_ref_generators__(self.dataset_reduced_std)
        histories, val_metrics, val_losses = \
            train_models_on_samples(train_gen, val_gen, self.models,
                                    nr_epochs=20, verbose=False,
                                    model_path=None, early_stopping=False,
                                    metric_name='mean_absolute_percentage_error',
                                    debug=False, use_testset=False,
                                    test_gen=None)

        history_model = histories[0].history
        self.assertEqual(len(history_model['loss']), 20)
        self.assertEqual(len(history_model['mean_absolute_percentage_error']), 20)
        self.assertEqual(len(history_model['val_loss']), 20)
        self.assertEqual(len(history_model['val_mean_absolute_percentage_error']), 20)

        self.assertAlmostEqual(history_model['loss'][0], 0.7767009734269942)
        self.assertAlmostEqual(history_model['mean_absolute_percentage_error'][0], 87994.5229075557)
        self.assertAlmostEqual(history_model['val_loss'][0], 0.47735168890048785)
        self.assertAlmostEqual(history_model['val_mean_absolute_percentage_error'][0], 78.18934104874818)

        self.assertAlmostEqual(val_metrics[0], 8.29117184513892)
        self.assertAlmostEqual(val_losses[0], 0.013613081621216204)

    def test_early_stopping(self):
        train_gen, val_gen, test_gen = __get_ref_generators__(self.dataset_reduced_std)
        histories, val_metrics, val_losses = \
            train_models_on_samples(train_gen, val_gen, self.models,
                                    nr_epochs=100, verbose=False,
                                    model_path=None, early_stopping=True,
                                    metric_name='mean_absolute_percentage_error',
                                    debug=False, use_testset=True,
                                    test_gen=test_gen,
                                    early_stopping_patience=0)

        history_model = histories[0].history
        self.assertEqual(len(history_model['loss']), 6)
        self.assertEqual(len(history_model['mean_absolute_percentage_error']), 6)
        self.assertEqual(len(history_model['val_loss']), 6)
        self.assertEqual(len(history_model['val_mean_absolute_percentage_error']), 6)

        self.assertAlmostEqual(history_model['loss'][0], 0.7767009734269942)
        self.assertAlmostEqual(history_model['mean_absolute_percentage_error'][0], 87994.5229075557)
        self.assertAlmostEqual(history_model['val_loss'][0], 0.477351688900487859)
        self.assertAlmostEqual(history_model['val_mean_absolute_percentage_error'][0], 78.18934104874818)

        self.assertAlmostEqual(val_metrics[0], 12.742151500652362)
        self.assertAlmostEqual(val_losses[0], 0.01858853024482985)

    def test_early_stopping_notestset(self):
        train_gen, val_gen, test_gen = __get_ref_generators__(self.dataset_reduced_std)
        histories, val_metrics, val_losses = \
            train_models_on_samples(train_gen, val_gen, self.models,
                                    nr_epochs=100, verbose=False,
                                    model_path=None, early_stopping=True,
                                    metric_name='mean_absolute_percentage_error',
                                    debug=False, use_testset=False,
                                    test_gen=None,
                                    early_stopping_patience=0)

        history_model = histories[0].history
        self.assertEqual(len(history_model['loss']), 6)
        self.assertEqual(len(history_model['mean_absolute_percentage_error']), 6)
        self.assertEqual(len(history_model['val_loss']), 6)
        self.assertEqual(len(history_model['val_mean_absolute_percentage_error']), 6)

        self.assertAlmostEqual(history_model['loss'][0], 0.7767009734269942)
        self.assertAlmostEqual(history_model['mean_absolute_percentage_error'][0], 87994.5229075557)
        self.assertAlmostEqual(history_model['val_loss'][0], 0.477351688900487859)
        self.assertAlmostEqual(history_model['val_mean_absolute_percentage_error'][0], 78.18934104874818)

        self.assertAlmostEqual(val_metrics[0], 10.43860529817535)
        self.assertAlmostEqual(val_losses[0], 0.014492817285315833)

