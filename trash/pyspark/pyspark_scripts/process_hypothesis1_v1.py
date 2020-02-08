import pyspark
import pyspark.sql.functions as F


from pyspark.conf import SparkConf
from pyspark.sql import SQLContext

from pyspark_utils.utils_hypothesis import process_hypothesis1

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



# check window of 6
trns6 = process_hypothesis1(trns, 6)
w6 = trns6.groupby(trns6.stricted_hypothesis1).count()

w6.show()

w6c = w6.collect()
print(w6c)

# compose all windows
w = w6.select(F.col('stricted_hypothesis1'), F.col('count').alias('count_w' + str(6))).fillna("null")
other_windows = [12, 18, 24, 30, 36]
for window in other_windows:
  trns_w = process_hypothesis1(trns, window)
  w_w = trns_w.groupby(trns_w.stricted_hypothesis1).count()
  w = w.join(w_w.select(F.col('stricted_hypothesis1'), F.col('count').alias('count_w' + str(window))).fillna("null"),
             on="stricted_hypothesis1", how='outer')

w.show()
wc = w.collect()
print(wc)



