# import os
# os.environ["PYSPARK_PYTHON"]='C:\\Users\\XXXX\\AppData\\Local\\Continuum\\anaconda3\\envs\\feci\\python.exe'

from pyspark_utils.utils_local import get_spark_context, get_spark_sql_context_from_sc, \
    get_test_data, get_modelling_database, get_modelling_pipeline

from pyspark_utils.utils_h2o import get_h2o_context, get_h2o_train_test, \
    get_example_h2o_grid_search

from utils.lime_h2o_predict_proba_wrapper import lime_h2o_predict_proba_wrapper

import lime.lime_tabular

# 0. Configuración inicial
sc = get_spark_context()
sqlContext = get_spark_sql_context_from_sc(sc)


# 1. cargamos datos para generar modelo
trns = get_test_data(sqlContext, 10000)
bbdd = get_modelling_database(trns)
train, test, pipeline_columns = get_modelling_pipeline(bbdd)

# 2. Ajustamos un gbm con h2o
hc = get_h2o_context(sc)
train_h2o, test_h2o = get_h2o_train_test(hc, train, test, pipeline_columns)
gbm_grid = get_example_h2o_grid_search(train_h2o, test_h2o)
gbm_model = gbm_grid.models[0]


# 3. Interpretación modelo H20
# https://marcotcr.github.io/lime/tutorials/Tutorial_H2O_continuous_and_cat.html
feature_names = train_h2o.drop('label').columns
train_pandas_df = train_h2o[feature_names].as_data_frame()
train_numpy_array = train_pandas_df.values

test_pandas_df = test_h2o[feature_names].as_data_frame()
test_numpy_array = test_pandas_df.values

explainer = lime.lime_tabular.LimeTabularExplainer(train_numpy_array,
                                                   feature_names=feature_names,
                                                   class_names=['react', 'aband'],
                                                   discretize_continuous=True)

h2o_drf_wrapper = lime_h2o_predict_proba_wrapper(gbm_model, feature_names)

# explicación de un ejemplo cualquiera de test
i = 27
exp = explainer.explain_instance(test_numpy_array[i], h2o_drf_wrapper.predict_proba, num_features=len(feature_names), top_labels=1)
exp.as_pyplot_figure()


# 4. Interpretamos modelo (K-LIME Manual)
# https://resources.oreilly.com/oriole/interpretable-machine-learning-with-python-xgboost-and-h2o/blob/master/lime.ipynb


