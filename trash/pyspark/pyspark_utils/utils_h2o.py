# Databricks notebook source
import h2o as h2o
from pysparkling import H2OContext, H2OConf
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from pyspark.sql.functions import when



def get_h2o_context(sc):
    """
    Crea el contexto de H20 a partir del de Spark
    TODO SparkSession en lugar de SparkContext

    :param sc:
    :return:
    """
    hc = H2OContext.getOrCreate(sc, H2OConf(sc).set_internal_cluster_mode())
    # asegurarse de que está conectado (H2O ya podría estar levantado por otro cluster
    hc._H2OContext__h2o_connect()
    return hc


def get_h2o_grid_summary_as_dataframe(sqlContext, grid, test_h2o,
                                      measure="auc"):
    """
    Graba el resultado de un grid en un dataframe

    :param sqlContext: .
    :param grid:
    :param test_h2o:
    :param measure: medida de precisión que se añade al df
    :return: __DataFrame__ pyspark dataframe
    """
    import pandas as pd
    table = []
    for model in grid.models:
        model_summary = model._model_json["output"]["model_summary"]
        r_values = list(model_summary.cell_values[0])
        r_values[0] = model.model_id
        measure_value = getattr(model, measure)(**{'valid':test_h2o})
        r_values.append(measure_value)
        table.append(r_values)
    summary_df = sqlContext.createDataFrame(
        pd.DataFrame(table, columns=['model_id'] + model_summary.col_header[1:] + [measure]))
    summary_df.show()
    return summary_df


def grid_save_all_models(grid,
                         relative_path):
    """
    Graba todos los modelos del grid

    :param grid: H2O grid
    :param relative_path: path relativo; los modelos se graban como subirectorios
        p.e. "dbfs:/FileStore/models_gbm_grid_20203112"
    :return:
    """
    for model in grid.models:
        h2o.save_model(model, path=relative_path + model.model_id, force=True)


def get_aml_leaderboard_as_dataframe(sqlContext, aml,
                                     relative_path,
                                     leaderboard_name=None):
    """
    Obtener el leadaerboard a partir del objeto aml y guardarlo en parquet

    :param sqlContext: spark
    :param aml: objeto aml
    :param relative_path: path relativo donde guardar el modelo leader
    :param parquet_file_name: nombre de fichero parquet, sobre el path relativo
    :return:
    """

    # grabar el modelo líder
    h2o.save_model(aml.leader, path=relative_path + "/aml.leader.model", force=True)

    # grabar leaderboard como dataframe (con "save")
    leaderboard = sqlContext.createDataFrame(aml.leaderboard.as_data_frame())
    leaderboard.show()
    if leaderboard_name is not None:
        leaderboard.write.mode("overwrite").parquet(relative_path + leaderboard_name)

    return leaderboard


def get_h2o_train_test(hc, train, test,
                       pipeline_columns,
                       target_variable='label',
                       target_label_1='reactive',
                       target_label_0='churn'):
    """
    Obtener train y test para h2o

    :param hc: objeto H2O
    :param train: dataframe train pyspark
    :param test: dataframe test pyspark
    :param pipeline_columns: columnas en el pipeline
    :param target_variable: nombre de la variable objetivo
    :param target_label_1: valor de la variable objetivo para la clase positiva
    :param target_label_0: valor de la variable objetivo para la clase negativa
    :return:
    """
    pipeline_columns = [c.replace("classVec", "Index")  if "indexclassVec" in c else c for c in pipeline_columns]
    
    # convertimos el target a string
    test = test.withColumn(target_variable,
                           when(test.label == 1, target_label_1).otherwise(target_label_0))
    # tiramos variables no útiles
    test_mod = test.select(pipeline_columns + [target_variable])
    # convertimos a typo H2OFrame
    test_h2o = hc.as_h2o_frame(test_mod)

    # cambiamos el tipo de la variable objetivo
    test_h2o[target_variable] = test_h2o[target_variable].asfactor()

    if train is not None:
      # el mismo proceso para train
      train = train.withColumn(target_variable,
                               when(train.label == 1, target_label_1).otherwise(target_label_0))
      train_mod = train.select(pipeline_columns + [target_variable])
      train_h2o = hc.as_h2o_frame(train_mod)
      train_h2o[target_variable] = train_h2o[target_variable].asfactor()
    else:
      train_h2o = None

    return train_h2o, test_h2o


def get_example_h2o_grid_search(train_h2o, test_h2o):
    gbm_params = {'max_depth': [5, 7, 9, 10],
                  'learn_rate': [0.01, 0.1],
                  'ntrees': [100]  # , 200, 300, 500]
                  }

    gbm_grid = H2OGridSearch(model=H2OGradientBoostingEstimator,
                             grid_id='power_plant_gbm_grid',
                             hyper_params=gbm_params)

    gbm_grid.train(x=train_h2o.drop('label').columns,
                   y="label",
                   training_frame=train_h2o,
                   validation_frame=test_h2o,
                   score_each_iteration=False,
                   stopping_metric="AUC",
                   stopping_rounds=10)

    return gbm_grid
