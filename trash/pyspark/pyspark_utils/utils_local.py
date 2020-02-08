# utilidades comunes a PySpark
# IMPLEMENTACIÓN LOCAL

import pyspark
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext

import pyspark.sql.functions as F
from pyspark.sql.functions import when
from pyspark.sql.types import FloatType, IntegerType, StringType

from pyspark.ml.feature import VectorAssembler, OneHotEncoderEstimator, StringIndexer
from pyspark.ml import Pipeline

import re
import numpy as np

from pyspark_utils.utils_hypothesis import process_hypothesis_2_safe_cv_high_low, extract_cv_months_udf


def get_spark_context(log_level="WARN"):
    conf = SparkConf()
    conf.setMaster("local").setAppName("test")
    conf.set("spark.sql.shuffle.partitions", 3)
    conf.set("spark.default.parallelism", 3)
    conf.set("spark.debug.maxToStringFields", 100)
    sc = pyspark.SparkContext(conf=conf)
    sc.setLogLevel(log_level)
    return sc


def get_spark_sql_context():
    sc = get_spark_context()
    return get_spark_sql_context_from_sc(sc)


def get_spark_sql_context_from_sc(sc):
    sqlContext = SQLContext(sc)
    return sqlContext


def get_test_data(sqlContext, number_of_rows):
    if number_of_rows not in [10, 100, 1000, 10000]:
        raise RuntimeError("número de filas no disponible para test: " + str(number_of_rows))
    trns = sqlContext.read \
        .format('com.databricks.spark.csv') \
        .option('header', 'true') \
        .option('delimiter', ',') \
        .option('inferSchema', 'true') \
        .load('data/sales_agg_product_year_month_limit_' + str(number_of_rows) + '.csv')
    return trns


def get_modelling_database(trns,
                           param_safe_inactivity_window = 12,
                           param_standstill = 12,
                           param_cv = 12):
    out, _ = process_hypothesis_2_safe_cv_high_low(trns,
                                                   param_standstill=param_standstill,
                                                   param_safe_inactivity_window=param_safe_inactivity_window,
                                                   param_cv=param_cv,
                                                   high_low_trim_trailing_zeros=False,
                                                   high_low_bound=None)
    # las cuentas que queremos son las que tienen valor 'True' o 'False' para la hipótesis 1 estricta
    out = out.filter((out.stricted_hypothesis1 == 'True') |
                     (out.stricted_hypothesis1 == 'False')).cache()

    inv_columns = [column for column in out.columns if bool(re.search(r'^\d{6}$', column))]
    out = out.withColumn('inv_cv_months',
                         extract_cv_months_udf(
                             F.array([out[column] for column in inv_columns]),
                             F.lit(12)
                         ))

    out = out.withColumn('reactiva',
                         when(F.col('stricted_hypothesis1') == 'False', 1)
                         .when(F.col('stricted_hypothesis1') == 'True', 0))

    # variables de inversion de los últimos 12 meses
    bbdd = out.select(out.inv_cv_months[0].alias('mes1'),
                      out.inv_cv_months[1].alias('mes2'),
                      out.inv_cv_months[2].alias('mes3'),
                      out.inv_cv_months[3].alias('mes4'),
                      out.inv_cv_months[4].alias('mes5'),
                      out.inv_cv_months[5].alias('mes6'),
                      out.inv_cv_months[6].alias('mes7'),
                      out.inv_cv_months[7].alias('mes8'),
                      out.inv_cv_months[8].alias('mes9'),
                      out.inv_cv_months[9].alias('mes10'),
                      out.inv_cv_months[10].alias('mes11'),
                      out.inv_cv_months[11].alias('mes12'),
                      'inv_cv_months',
                      'reactiva') \
        .withColumnRenamed('reactiva', 'label') \
        .cache()

    # numero de meses con actividad antes del parón
    bbdd = bbdd.withColumn('n_active_cv_months',
                           F.udf(lambda x: sum([bool(inv != 0) for inv in x]), IntegerType())(F.col('inv_cv_months')))

    # media de actividad en los meses de cv
    bbdd = bbdd.withColumn('mean_inv_cv_months',
                           F.udf(lambda x: (np.mean(x)).item().__round__(0), FloatType())(F.col('inv_cv_months')))

    # maximo de actividad en los meses de cv
    bbdd = bbdd.withColumn('max_inv_cv_months',
                           F.udf(lambda x: max(x), FloatType())(F.col('inv_cv_months')))

    # minimo de actividad en los meses de cv (ignoramos meses con actividad)
    bbdd = bbdd.withColumn('min_inv_cv_months',
                           F.udf(lambda x: min([inv for inv in x if inv != 0]), FloatType())(F.col('inv_cv_months')))

    # suma de inversion total realizada
    bbdd = bbdd.withColumn('sum_inv_cv_months',
                           F.udf(lambda x: sum(x), FloatType())(F.col('inv_cv_months')))

    # numero de meses de cv (implicitamente es el mes de inicio del parón)
    bbdd = bbdd.withColumn('standstill_month_index',
                           F.udf(lambda x: 'standtill_mes' + str(len(x)), StringType())(F.col('inv_cv_months')))

    # numero de meses de cv (implicitamente es el mes de inicio del parón)
    bbdd = bbdd.withColumn('n_cv_months',
                           F.udf(lambda x: len(np.trim_zeros(x, 'f')), IntegerType())(F.col('inv_cv_months')))
    return bbdd


def get_modelling_pipeline(bbdd):
    stages = []  # lista de pasos para el pipeline
    # codificamos variables categoricas
    categoricalColumns = ['standstill_month_index']
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],
                                         outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]

    # agrupamos variables
    pipeline_columns = ['mes' + str(i + 1) for i in range(12)] + \
                       ['n_active_cv_months', 'mean_inv_cv_months', 'max_inv_cv_months',
                        'min_inv_cv_months', 'sum_inv_cv_months', 'n_cv_months'] + \
                       [c + "classVec" for c in categoricalColumns]
    assembler = VectorAssembler(inputCols=pipeline_columns,
                                outputCol="features")
    stages += [assembler]

    # definimos el pipeline (ml workflow of Transformers and Estimators)
    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(bbdd)
    df = pipeline_model.transform(bbdd)
    # dividimos en train y test
    train, test = df.randomSplit([0.8, 0.2], seed=777)
    print("Training Dataset Count: " + str(train.count()))
    print("Test Dataset Count: " + str(test.count()))
    return train, test, pipeline_columns


