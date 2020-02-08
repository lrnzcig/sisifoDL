import pyspark
import pyspark.sql.functions as F

from pyspark.conf import SparkConf
from pyspark.sql import SQLContext

from pyspark_utils.utils_hypothesis import cumulative_inactive_months_udf, check_1st_consecutive_inactive_streaks_udf

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

# DESCRIPCIÓN:
# Hipótesis5 versión 1: cuando una cuenta esperimenta X meses sin actividad, el parón llega a ser de Y meses, donde Y
# es razonablemente mayor que X

# 0. Configuración Inicial
param_standstill = 6
aux_list_inv_columns = trns.columns
aux_list_inv_columns.remove('CUENTA')


# 1. Detectar si experimenta un parón de 'param_standstill'
# ... contamos meses de inactividad (jumps) en ventanas de 'param_standstill' meses
trns = trns.withColumn('cumulative_inactive_months',
                       cumulative_inactive_months_udf(F.array([trns[column] for column in aux_list_inv_columns]))
                       )


# 2. Detectar cuando una cuenta alcanza 12 meses seguidos sin inactividad, qué proporción de clientes alcanzan 24 meses
# sin inactividad
trns = trns.withColumn('consecutive_inactive_streaks',
                       check_1st_consecutive_inactive_streaks_udf(trns['cumulative_inactive_months'],
                                                                  F.lit(param_standstill)))

# 3. Recogemos el numero de
output = trns.select('consecutive_inactive_streaks').toPandas()

# import matplotlib.pyplot as plt
# data = output.loc[:, 'consecutive_inactive_streaks']
# data
# plt.figure()
# p = data.hist(bins=30)
# p.plot()



