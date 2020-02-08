import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import udf

from pyspark.conf import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import BooleanType

from pyspark_utils.utils_hypothesis import counter_jumps_udf, calculate_account_life_udf, \
    calculate_n_months_cv_udf, calculate_n_months_safe_inactivity_udf, divide_sample_by_avg_month_inv, \
    process_hypothesis1

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


# 0. Inicializamos hiper-parámetros
param_standstill = 6  # parámetro de parón (meses sin actividad)
param_safe_inactivity_window = 6  # parámetro de 'seguridad' (minimo numero de meses que debe mantenerse un parón)
param_cv = 12  # parámetro de actividad antes del primer parón
param_bound = 62.5  # media de inversión mensual para dividir las cuentas por alto y bajo valor
aux_list_inv_columns = trns.columns
aux_list_inv_columns.remove('CUENTA')


# 1. Detectar si experimenta un parón de 'param_standstill'
# ... contamos meses de inactividad (jumps) en ventanas de 'param_standstill' meses
trns = trns.withColumn('jumps',
                       counter_jumps_udf(F.array([trns[column] for column in aux_list_inv_columns]),
                                         F.lit(param_standstill)))
# ... flag para saber si hay un parón de 'param_standstill' meses
trns = trns.withColumn('exist_standstill',
                       udf(lambda x: param_standstill in x,
                           BooleanType())(trns['jumps']))
# ... check visual
# trns.take(10)


# 2. Eliminar cuentas que tienen menos de 'param_standstill' + 'param_safe_inactivity_window' + 'param_cv' meses de
# vida (no hay suficiente histórico)
# ... calculamos meses de vida
trns = trns.withColumn('n_months_account_life',
                       calculate_account_life_udf(F.array([trns[column] for column in aux_list_inv_columns])))
# ... filtramos ('~' sirve para negar)
trns = trns.filter(~(  # negamos
        trns.n_months_account_life < (param_cv +
                                      param_standstill +
                                      param_safe_inactivity_window)  # no tiene meses suficientes de histórico
                     )
                   )
# ... check visual (los clientes que no tienen parón tiene suficientes meses vida)
# trns.filter(~ trns.exist_standstill).groupBy('n_months_account_life').count().show()


# 3. Eliminar cuentas que tienen un parón y o tienen menos de 'param_cv' meses de histórico antes del parón o menos de
# 'param_safe_inactivity_window' meses de histórico después del parón
# ... calculamos meses entre la primera actividad y el inicio del parón (cv)
trns = trns.withColumn('n_months_cv',
                       calculate_n_months_cv_udf(
                           F.array([trns[column] for column in aux_list_inv_columns]),
                           F.lit(param_standstill))
                       )
# ... calculamos meses entre el parón (si existe) y el final de histórico (a esto nos referimos como meses de 'colchon')
trns = trns.withColumn('n_months_safe_inactivity',
                       calculate_n_months_safe_inactivity_udf(
                           F.array([trns[column] for column in aux_list_inv_columns]),
                           F.lit(param_standstill)
                       ))
# ... filtramos
trns = trns.filter(~(  # negamos
        trns.exist_standstill &  # hay parón y ...
        ((trns.n_months_cv < param_cv) |  # (1) o no hay suficientes meses antes del paron
         (trns.n_months_safe_inactivity < param_safe_inactivity_window))  # (2) o no hay suficientes meses despues
                   ))
# ... check visual
# trns.select('jumps', 'n_months_account_life', 'exist_standstill', 'n_months_cv', 'n_months_safe_inactivity').take(20)
# trns.take(20)
# trns.count()


# 4. Procesamos hipótesis2
# ... eliminamos variables auxiliares creadas para filtrar una muestra de cuentas homogeneas
trns = trns.drop('jumps', 'exist_standstill', 'n_months_account_life', 'n_months_cv', 'n_months_safe_inactivity')
# ... dividimos muestra en clientes de alto valor y clientes de bajo valor
trns_high, trns_low = divide_sample_by_avg_month_inv(trns, bound=param_bound)
# ... check de la hipótesis para cada submuestra
trns_high_processed = process_hypothesis1(trns_high, param_standstill)
trns_low_processed = process_hypothesis1(trns_low, param_standstill)
wh = trns_high_processed.groupby(trns_high_processed.stricted_hypothesis1).count()
wl = trns_low_processed.groupby(trns_low_processed.stricted_hypothesis1).count()

wh.show()
wl.show()
