import pyspark
import pyspark.sql.functions as F


from pyspark.conf import SparkConf
from pyspark.sql import SQLContext

from pyspark_utils.utils_hypothesis import process_hypothesis1, divide_sample_by_avg_month_inv

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


trns_high, trns_low = divide_sample_by_avg_month_inv(trns, bound=62.5)

# check window of 6
trns6_high = process_hypothesis1(trns_high, 6)
trns6_low = process_hypothesis1(trns_low, 6)
w6h = trns6_high.groupby(trns6_high.stricted_hypothesis1).count()
w6l = trns6_low.groupby(trns6_low.stricted_hypothesis1).count()


w6ch = w6h.collect()
print(w6ch)

w6cl = w6l.collect()
print(w6cl)


