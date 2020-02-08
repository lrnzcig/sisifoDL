
import pyspark
from pyspark.sql.types import ArrayType, FloatType, StructType
import pyspark.sql.functions as F
from pyspark.sql.functions import when
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

from pyspark.conf import SparkConf
from pyspark.sql import SQLContext

from pyspark_utils.utils_hypothesis import process_hypothesis_2_safe_cv_high_low, extract_cv_months_udf

import re

# Iniciamos configuracion de spark y set de datos pequeño para trabajar en local
conf = SparkConf()
conf.setMaster("local").setAppName("test")
conf.set("spark.sql.shuffle.partitions", 3)
conf.set("spark.default.parallelism", 3)
sc = pyspark.SparkContext(conf=conf)
sqlContext = SQLContext(sc)

trns = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .option('header', 'true') \
    .option('delimiter', ',') \
    .option('inferSchema', 'true') \
    .load('data/sales_agg_product_year_month_limit_1000.csv')


# 1. Filtramos cuentas que tengan un parón de 12 meses y al menos 12 meses de cv y 12 meses de colchón
param_safe_inactivity_window = 12
param_standstill = 12
param_cv = 12
out, _ = process_hypothesis_2_safe_cv_high_low(trns,
                                               param_standstill=param_standstill,
                                               param_safe_inactivity_window=param_safe_inactivity_window,
                                               param_cv=param_cv,
                                               high_low_trim_trailing_zeros=False,
                                               high_low_bound=None)
# las cuentas que queremos son las que tienen valor 'True' o 'False' para la hipótesis 1 estricta
out = out.filter((out.stricted_hypothesis1 == 'True') |
                 (out.stricted_hypothesis1 == 'False')).cache()

# 2. Extraemos meses de actividad antes del parón
inv_columns = [column for column in out.columns if bool(re.search(r'^\d{6}$', column))]
out = out.withColumn('inv_cv_months',
                     extract_cv_months_udf(
                         F.array([out[column] for column in inv_columns]),
                         F.lit(12)
                      ))
# tomamos solo los últimos 12 meses de cv
out = out.withColumn('inv_cv_months',
                     F.udf(lambda x: x[-12:],
                           ArrayType(FloatType()))(F.col('inv_cv_months')))

# 3. Creamos target (1 reactiva, 0 no reactiva en proximos 12 meses)
out = out.withColumn('reactiva',
                     when(F.col('stricted_hypothesis1') == 'False', 1)
                     .when(F.col('stricted_hypothesis1') == 'True', 0))

# 4. Ajustamos modelo de regresión logística
stages = []  # para el pipeline
bbdd_reactivation_model = out.select(out.inv_cv_months[0].alias('mes1'),
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
                                     'reactiva')\
    .withColumnRenamed('reactiva', 'label')\
    .cache()

assembler = VectorAssembler(inputCols=['mes'+str(i+1) for i in range(12)], outputCol="features")
stages += [assembler]

# definimos el pipeline (ml workflow of Transformers and Estimators)
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(bbdd_reactivation_model)
df = pipeline_model.transform(bbdd_reactivation_model)
# dividimos en train y test
train, test = df.randomSplit([0.8, 0.2], seed=777)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))
# ajustamos modelo
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)
lrModel = lr.fit(train)

# visualización de coeficientes
import matplotlib.pyplot as plt
import numpy as np
beta = np.sort(lrModel.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()

# precisión en Train
trainingSummary = lrModel.summary
trainingSummary.accuracy





