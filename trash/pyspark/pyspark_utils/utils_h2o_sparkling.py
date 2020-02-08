# Databricks notebook source
from pyspark.ml.pipeline import Pipeline
from ai.h2o.sparkling.ml.algos import H2OGridSearch
from ai.h2o.sparkling.ml.algos import H2OGBM
from ai.h2o.sparkling.ml.params import H2OGridSearchParams


def get_example_pysparkling_h2o_grid_search(hc, train_h2o, test_h2o):
    """
    # http://docs.h2o.ai/sparkling-water/2.2/latest-stable/doc/tutorials/grid_gbm_pipeline.html
    # https://github.com/h2oai/sparkling-water/blob/master/py/src/ai/h2o/sparkling/ml/algos/H2OGBM.py
    # https://github.com/h2oai/sparkling-water/blob/master/py/src/ai/h2o/sparkling/ml/params/H2OGBMParams.py
    # https://github.com/h2oai/sparkling-water/blob/master/py/src/ai/h2o/sparkling/ml/algos/H2OGridSearch.py

    :param train_h2o:
    :param test_h2o:
    :return:
    """
    #TODO gridSearch como poner hiperparámetros
    
    gbm_params = {#'maxDepth': [5, 7, 9, 10],
                  #'learn_rate': [0.01, 0.1],
                  'ntrees': [100]  # , 200, 300, 500]
                  }

    gbm_grid = H2OGridSearch(algo=H2OGBM().setMaxDepth(30),
                             #hyperParameters=gbm_params,
                             withDetailedPredictionCol=True,
                             labelCol='label',
                             featuresCols=train_h2o.drop('label').columns,
                             stoppingMetric="AUC")

    gbm_grid = H2OGridSearch()\
        .setLabelCol("label") \
        .setHyperParameters(gbm_params)\
        .setAlgo(H2OGBM().setMaxDepth(30))

    model_pipeline = Pipeline().setStages([gbm_grid])
    model = model_pipeline.fit(hc.as_spark_frame(train_h2o))


    gbm_grid = H2OGridSearch().setAlgo(H2OGBM())\
        .setLabelCol('label')\
        .setFeaturesCols(train_h2o.drop('label').columns)\
        .setStoppingMetric("AUC")\
        .setWithDetailedPredictionCol(True)#.setHyperParameters(gbm_params)
    models = gbm_grid.fit(hc.as_spark_frame(train_h2o))#, {'algo': "GBM"})

    temp = H2OGridSearchParams().setHyperParameters(gbm_params)

    #model_pipeline = Pipeline().setStages([gbm_grid])
    #model = model_pipeline.fit(hc.as_spark_frame(train_h2o), gbm_params)

    return models


def get_example_pysparkling_h2o_gbm(hc, train_h2o, test_h2o):
    """
    # http://docs.h2o.ai/sparkling-water/2.2/latest-stable/doc/tutorials/grid_gbm_pipeline.html
    # https://github.com/h2oai/sparkling-water/blob/master/py/src/ai/h2o/sparkling/ml/algos/H2OGBM.py
    # https://github.com/h2oai/sparkling-water/blob/master/py/src/ai/h2o/sparkling/ml/params/H2OGBMParams.py
    # https://github.com/h2oai/sparkling-water/blob/master/py/src/ai/h2o/sparkling/ml/algos/H2OGridSearch.py

    :param train_h2o:
    :param test_h2o:
    :return:
    """
    gbm_model = H2OGBM(labelCol='label',
                      withDetailedPredictionCol=True).setLearnRate(0.01).setMaxDepth(5).setNtrees(100)

    return gbm_model.fit(hc.as_spark_frame(train_h2o))

    #model_pipeline = Pipeline().setStages([gbm_model])
    #model = model_pipeline.fit(hc.as_spark_frame(train_h2o))

    #return model.stages[0]
