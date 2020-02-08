from pysparkling import *
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext

import pyspark

conf = SparkConf()
conf.setMaster("local").setAppName("test")
conf.set("spark.sql.shuffle.partitions", 3)
conf.set("spark.default.parallelism", 3)
sc = pyspark.SparkContext(conf=conf)
sqlContext = SQLContext(sc)

hc = H2OContext.getOrCreate(sc)

trns = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .option('header', 'true') \
    .option('delimiter', ',') \
    .option('inferSchema', 'true') \
    .load('data/sales_agg_product_year_month_limit_1000.csv')

h2o_trns = hc.as_h2o_frame(trns)