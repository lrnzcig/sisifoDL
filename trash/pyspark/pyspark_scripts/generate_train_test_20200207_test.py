
# 0. configuracion inicial
import pyspark
import pyspark.sql.functions as F

from pyspark.ml.feature import VectorAssembler, OneHotEncoderEstimator, StringIndexer
from pyspark.ml import Pipeline

import re
import pandas as pd
from collections import Counter
from functools import reduce
from pyspark.sql.functions import expr, array

from utils.utils_data_structures import wide_sales_4y_schema
from utils.utils_eda import get_var_names_info
from pyspark_utils.utils_local import get_spark_sql_context
from pyspark_utils.utils_feature_engineering import *
from pyspark_utils.utils_hypothesis import data_preparation_cv_safe, process_hypothesis1, \
    extract_cv_months_udf

sqlContext = get_spark_sql_context()

# 1. cargamos muestra de transaccional y la información de campañas
data_file = "data/sales_agg_product_year_month_limit_1000.csv"

trns_aggregated = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .option('header', 'true') \
    .option('delimiter', ',') \
    .option('inferschema', 'true') \
    .load(data_file)

# el formato es tan sencillo que no merece la pena generar un fichero, creamos un diccionario
campaigns_dict = {'ventas_privadas': [12, 6], # Diciembre y Junio
                  'navidad': [11, 12, 1], # Noviembre, Diciembre y Enero
                  'black_friday': [11], # Noviembre
                  'rebajas': [1, 2, 7, 8], # Enero, Febrero, Julio y Agosto
                  'vuelta_al_cole': [6, 7, 8, 9], # Junio, Julio, Agosto y Septiembre
                  'dias_sin_iva': [1] # Enero
                 }
# contamos cuantas campañas hay en cada uno de los meses del año
_list = []
for element in list(campaigns_dict.values()):
  _list = _list + element
count_month_campaigns = Counter(_list)
print(count_month_campaigns)

# 2. filtramos cuentas que tengan un parón de 12 meses y al menos 12 meses de cv y de colchón
# definimos parámetros para elegir cuentas para el modelo
param_safe_inactivity_window = 12
param_standstill = 12
param_cv = 12
# seleccionamos cuentas objetivo del modelo
out = data_preparation_cv_safe(trns_aggregated,
                               param_standstill=param_standstill,
                               param_safe_inactivity_window=param_safe_inactivity_window,
                               param_cv=param_cv)
# determinamos qué cuentas validad (abandonan) o no (reactivan) la hipótesis de abandono en 12 meses
out = process_hypothesis1(out,
                          window=param_standstill,
                          safe_inactivity_window=param_safe_inactivity_window
                          )

trns = out

# 3. extraemos meses de actividad antes del parón
# columnas de inversión agregada
inv_columns = [column for column in trns.columns if bool(re.search(r'^\d{6}$', column))] # columnas que son exactamente 6 dígitos
# columnas de inversión de TC
inv_tc_columns = [column for column in trns.columns if bool(re.search(r'^\d{6}_card$', column))]
# columnas de inversión de LC
inv_lc_columns = [column for column in trns.columns if bool(re.search(r'^\d{6}_lc$', column))]
# columnas de inversión de noTC
inv_rest_columns = [column for column in trns.columns if bool(re.search(r'^\d{6}_feci$', column))]

# actividad total
trns = trns.withColumn('inv_cv_months',
                       extract_cv_months_udf(
                         F.array([trns[column] for column in inv_columns]),
                         F.lit(param_standstill)
                       ))
# actividad en tarjeta de compra (LC)
trns = trns.withColumn('inv_lc_cv_months',
                       extract_cv_months_udf(
                         F.array([trns[column] for column in inv_lc_columns]),
                         F.lit(param_standstill)
                       ))
# actividad en otro productos financieros
trns = trns.withColumn('inv_rest_cv_months',
                       extract_cv_months_udf(
                         F.array([trns[column] for column in inv_rest_columns]),
                         F.lit(param_standstill)
                       ))

# tiramos variables de inversión individuales (las acabamos de agregar)
trns = trns.drop(*inv_columns)\
    .drop(*inv_tc_columns)\
    .drop(*inv_lc_columns)\
    .drop(*inv_rest_columns)

# 4. creamos target
trns = trns.withColumn('label',
                       F.when(F.col('stricted_hypothesis1') == 'False', 1)
                       .when(F.col('stricted_hypothesis1') == 'True', 0))\
  .drop('stricted_hypothesis1')

bbdd = trns

# 5. feature engineering
# definimos productos a distinguir (actualmente 3 categorías: tc, lc y resto)
products = ['']  # el transaccional que tenemos en este script está agregado, no distinguimos productos
# definimos los lags para los que sacar la inversión
lags = range(1, 13)
# definimos los quantiles que sacar de inversiones en meses de cv
q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# calculamos inversiones en tc en los últimos 12 meses antes del parón
bbdd = calculate_inv_cv_product_lag(bbdd,
                                    lags=lags,
                                    products=products)

# numero de meses con inversión antes del parón
bbdd = calculate_n_active_cv_months(bbdd,
                                    products=products)
# media de inversión en los meses de cv
bbdd = calculate_mean_inv_cv_months(bbdd,
                                    products=products)
# mínimo de inversión no negativa en los meses de cv
bbdd = calculate_min_inv_cv_months(bbdd,
                                   products = products)
# suma de inversion total realizada
bbdd = calculate_sum_inv_cv_months(bbdd,
                                   products = products)
# mes de inicio del parón
bbdd = calculate_standstill_month_index(bbdd,
                                        products = products)
# numero de meses de cv
bbdd = calculate_n_cv_months(bbdd,
                             products = products)

# quantiles de la inversión
bbdd = calculate_quantile_inv_cv(bbdd,
                                 products = products,
                                 quantiles = q_list)

# calculamos numero de campañas en el mes antes del parón de actividad en cada producto financiero
bbdd = calculate_campaign_weight_standstill(bbdd,
                                            count_month_campaigns = count_month_campaigns,
                                            products = products)

bbdd.take(5)

# 6. Pipeline
stages = []  # lista de pasos para el pipeline
# codificamos variables categoricas
categoricalColumns = ['standstill_month_index']
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols = [stringIndexer.getOutputCol()], outputCols = [categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

# 1. variables de inversión en TC los meses de cv
# 2. variables de inversión en LC los meses de cv
# 3. variables de inversión en el resto de productos en los meses de cv
# 4. quantiles, estadísticos y otras variables generadas a partir del CV
# 5. variables de tipo fecha (de momento numéricas)
# 6. otros
# 7. variables categóricas condificadas
pipeline_columns = ['inv_cv_month_lag'+str(i) for i in lags] + \
                   ['quantile' + str(int(q*100)) + '_inv' + product + '_cv_months' for product in products for q in q_list] + \
                   ['n_active' + product + '_cv_months' for product in products] + \
                   ['mean_inv' + product + '_cv_months' for product in products] + \
                   ['min_inv' + product + '_cv_months' for product in products] + \
                   ['sum_inv' + product + '_cv_months' for product in products] + \
                   ['n' + product + '_cv_months' for product in products] + \
                   ['weight_campaign_standstill' + product + '_month_index' for product in products] + \
                   [c + "classVec" for c in categoricalColumns]

# agrupamos variables
assembler = VectorAssembler(inputCols = pipeline_columns,
                            outputCol = "features")
stages += [assembler]

# definimos el pipeline (ml workflow of Transformers and Estimators)
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(bbdd)
df = pipeline_model.transform(bbdd)
# dividimos en train y test
train, test = df.randomSplit([0.8, 0.2], seed = 777)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

""" TODO
# guardamos listado de columnas
sqlContext.createDataFrame(pd.DataFrame(pipeline_columns, columns=["column_name"]))\
  .write.mode("overwrite").parquet("FileStore/reactivation_model_20200207/pipeline_columns.parquet")
# guardamos split de train y test
train.write.mode("overwrite").parquet('FileStore/reactivation_model_20200207/train.parquet')
test.write.mode("overwrite").parquet('FileStore/reactivation_model_20200207/test.parquet')
"""