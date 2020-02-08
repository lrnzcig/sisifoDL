import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import array, udf, first
from pyspark.sql.types import FloatType, ArrayType

from pyspark.conf import SparkConf
from pyspark.sql import SQLContext

import numpy as np
import pandas as pd
from dtw import dtw
from sklearn.cluster import AgglomerativeClustering
from math import log

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
    .load('data/sales_agg_product_year_month_limit_10.csv')


def _trans_log(x):
    if x >= 0:
        return log(x + 1)
    else:
        return (-1) * log(1 - x)


def trans_log(x):
    return [_trans_log(el) for el in x]


trans_log_udf = udf(trans_log, ArrayType(FloatType()))


aux_list_inv_columns = trns.columns
aux_list_inv_columns.remove('CUENTA')

trns_sample_mod = trns.withColumn('inv_row', trans_log_udf(array([trns[column] for column in aux_list_inv_columns])))
print(trns_sample_mod.head())

mat = trns_sample_mod[['CUENTA', 'inv_row']].toDF('CUENTA1', 'inv_row1') \
    .crossJoin(trns_sample_mod[['CUENTA', 'inv_row']].toDF('CUENTA2', 'inv_row2')) \
    .filter(F.col('CUENTA1') > F.col('CUENTA2'))
print(mat.columns)
print(mat.count())
print((10 * 10 - 10) / 2)

print(mat.select('CUENTA1').distinct().count())
print(mat.select('CUENTA2').distinct().count())

l1 = [row["CUENTA1"] for row in mat.select('CUENTA1').distinct().collect()]
l2 = [row["CUENTA2"] for row in mat.select('CUENTA2').distinct().collect()]
[e for e in l1 if e not in l2]
mat.filter(F.col('CUENTA1') == '1248').select('CUENTA2').collect()
[e for e in l2 if e not in l1]
mat.filter(F.col('CUENTA2') == '1016').select('CUENTA1').collect()


def dtw_dist(inv_row1, inv_row2):
    return dtw(inv_row1, inv_row2).distance.item()


dtw_dist_udf = udf(dtw_dist, FloatType())

res = mat.withColumn('d', dtw_dist_udf(mat['inv_row1'], mat['inv_row2']))
print(res[['d']].head())

res_pivot = res.groupby('CUENTA1').pivot('CUENTA2').agg(first("d")).cache()
print(res_pivot.head(5))

distance_matrix = res_pivot.toPandas()

distance_matrix.sort_values(by="CUENTA1")
np.max(distance_matrix['CUENTA1'])
np.min(distance_matrix.columns)
distance_matrix[str(np.max(distance_matrix['CUENTA1']))] = np.nan
distance_matrix = distance_matrix.append(pd.DataFrame({"CUENTA1": int(np.min(distance_matrix.columns))},
                                                      index=[distance_matrix.shape[0]]), sort=True)
distance_matrix = distance_matrix.sort_values(by="CUENTA1").set_index("CUENTA1")
distance_matrix.index.name = None
print(distance_matrix)

# convertir en matriz de distancias simetrica
distance_matrix_symmetric = np.array(distance_matrix.fillna(0)) + np.array(distance_matrix.fillna(0).transpose())
print(pd.DataFrame(distance_matrix_symmetric))

# prueba tonta clusters
n_clusters = 3
acm = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete', affinity='precomputed')
clustering = acm.fit(distance_matrix_symmetric)
print(clustering.labels_)
