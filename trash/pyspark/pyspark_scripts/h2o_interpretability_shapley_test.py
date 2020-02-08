# import os
# os.environ["PYSPARK_PYTHON"]='C:\\Users\\XXXX\\AppData\\Local\\Continuum\\anaconda3\\envs\\feci\\python.exe'

import shap

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

import pickle

from pyspark_utils.utils_local import get_spark_context, get_spark_sql_context_from_sc, \
    get_test_data, get_modelling_database, get_modelling_pipeline

from pyspark_utils.utils_h2o import get_h2o_context, get_h2o_train_test, \
    get_example_h2o_grid_search

from pyspark_utils.utils_h2o_sparkling import get_example_pysparkling_h2o_gbm

# 0. Configuración inicial
sc = get_spark_context(log_level="WARN")
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
contributions = gbm_model.predict_contributions(test_h2o)

# 3. plots a partir de las contribuciones
# https://github.com/h2oai/h2o-3/blob/master/h2o-py/demos/shap_values_drf.ipynb

# convert the H2O Frame to use with shap's visualization functions
contributions_matrix = contributions.as_data_frame().values
# shap values are calculated for all features
# TODO mejorar y coger todas las columnas menos la última
shap_values = contributions_matrix[:,0:(contributions_matrix.shape[1]-1)]
# expected values is the last returned column
expected_value = contributions_matrix[:,(contributions_matrix.shape[1]-1)].min()
with open("data/shap_values_h2o.pickle", "wb") as f:
    pickle.dump(shap_values, f)
with open("data/expected_value_h2o.pickle", "wb") as f:
    pickle.dump(expected_value, f)

test_driver = hc.as_spark_frame(test_h2o).toPandas().drop('label', axis=1)
with open("data/test_driver_h2o.pickle", "wb") as f:
    pickle.dump(test_driver, f)

# visualize the first prediction's explanation
shap.force_plot(expected_value, shap_values[0,:], test_driver.iloc[0,:], matplotlib=True)

# visualize the training set predictions
shap.force_plot(expected_value, shap_values, test_driver, show=False)

# summarize the effects of all the features
shap.summary_plot(shap_values, test_driver)

# create a SHAP dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot("n_active_cv_months", shap_values, test_driver)

shap.summary_plot(shap_values, test_driver, plot_type="bar")



# 4. H2O obtiene contribuciones de fábrica de manera alternativa...
# SEGURAMENTE DESCARTAR
# TODO cómo hacer plots a paritr de las contribuciones
# TODO H2OGridSearch en pysparkling...
# TODO de momento H2OGBM en pysparkling, cómo aproxima replicar hiperparámetros de un ajuste h2o puro
model = get_example_pysparkling_h2o_gbm(hc, train_h2o, test_h2o)
predictions = model.transform(hc.as_spark_frame(test_h2o))
predictions.select("detailed_prediction").head(5)


# 5. Interpretamos modelo (Shapley values)
# TODO problemas de rendimiento para llegar a shapley values 0
# https://github.com/slundberg/shap
shap.initjs()

# ... necesitamos DF de test en local y una función para predecir a partir de pandas
train_driver = hc.as_spark_frame(train_h2o).toPandas()
test_driver = hc.as_spark_frame(test_h2o).toPandas()
def predict(df):
    if not isinstance(df, DataFrame):
        df = pd.DataFrame(df)
    return model.transform(sqlContext.createDataFrame(df)).select('detailed_prediction.p0').toPandas()
predict(test_driver)

# Shapley con KernelExplainer (a partir de las predicciones, sin entender estructura
explainer = shap.KernelExplainer(predict, train_driver.drop('label', axis=1), link="logit")
# ... el cálculo de Shapley values funciona, pero va muy despacio (un montón de llamadas a Spark
#     en la función `predict`)
shap_values = explainer.shap_values(test_driver.drop('label', axis=1), nsamples=100)
with open("data/shap_values_driver.pickle", "wb") as f:
    pickle.dump(shap_values, f)

# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(explainer.expected_value, shap_values[0,:],
                test_driver.drop('label', axis=1).iloc[0,:], link="logit")
plt.show() # TODO no se ve, en todo caso los valores están a 0

# plot the SHAP values for the Setosa output of all instances
shap.force_plot(explainer.expected_value[0], shap_values[0],
                train_driver.drop('label', axis=1), link="logit")

# summarize the effects of all the features
shap.summary_plot(shap_values, test_driver.drop('label', axis=1))
