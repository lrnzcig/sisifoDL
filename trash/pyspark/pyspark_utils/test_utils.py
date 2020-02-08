import unittest
import pyspark
from pyspark.conf import SparkConf
from pyspark.sql import SQLContext
import warnings


class TestPySparkBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls,
                   log_level="OFF"):
        warnings.filterwarnings("ignore")
        conf = SparkConf()
        conf.setMaster("local").setAppName("test")
        conf.set("spark.sql.shuffle.partitions", 3)
        conf.set("spark.default.parallelism", 3)
        cls.sc = pyspark.SparkContext.getOrCreate(conf=conf)

        # https://stackoverflow.com/questions/32512684/how-to-turn-off-info-from-logs-in-pyspark-with-no-changes-to-log4j-properties
        # https://stackoverflow.com/questions/25193488/how-to-turn-off-info-logging-in-spark/32208445#32208445
        # https://github.com/apache/spark/blob/master/conf/log4j.properties.template
        #logger = cls.sc._jvm.org.apache.log4j
        #logger.LogManager.getRootLogger().setLevel(logger.Level.OFF)
        #logger.LogManager.getLogger("org").setLevel(logger.Level.OFF)
        # https://stackoverflow.com/a/37836847/3519000
        cls.sc.setLogLevel(log_level)

        cls.sqlContext = SQLContext(cls.sc)

        cls.trns_1000row = cls.sqlContext.read \
            .format('com.databricks.spark.csv') \
            .option('header', 'true') \
            .option('delimiter', ',') \
            .option('inferSchema', 'true') \
            .load('test/data/sales_agg_product_year_month_limit_1000.csv')

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()

    def compare_spark_dfs(self, left, right,
                          columns, number_of_rows_to_check,
                          test_name):
        for column in columns:
            for row in range(0, number_of_rows_to_check):
                self.assertEqual(left[row][column], right[row][column],
                                 test_name + ": error in row " + str(row) + " column " + column)
