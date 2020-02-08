# import os
# os.environ["PYSPARK_PYTHON"]='C:\\Users\\XXXX\\AppData\\Local\\Continuum\\anaconda3\\envs\\feci\\python.exe'
from functools import reduce
import pyspark.sql.functions as F
from pyspark.sql.functions import expr, array

from utils.utils_data_structures import wide_sales_4y_schema
from utils.utils_eda import get_var_names_info

from pyspark_utils.utils_local import get_spark_sql_context

sqlContext = get_spark_sql_context()

data_file = "data/MODELO_ABANDONO_4A.csv"

wide_data = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .option('header', 'true') \
    .option('delimiter', ';') \
    .schema(wide_sales_4y_schema) \
    .load(data_file)

print("Generamos muestras del transaccional mensual agregado (sin distinguir producto)")
#print(wide_data.take(2))
wide_data_all = wide_data.fillna(0)

year_month = get_var_names_info(wide_data, level='year_month')
#print(year_month)
for date in year_month:
    # seleccionamos variables a agrupar
    #print(date)
    vars2group = [col for col in wide_data.columns if date in col]
    # agrupamos inversiones
    expression = reduce(lambda x, y: x + " + " + y, vars2group)
    #print(expression)
    expression = expr(expression)
    wide_data_all = wide_data_all.withColumn(date, expression)

wide_data_all.select('CUENTA', '201612').take(2)

output = wide_data_all.select(['CUENTA'] + year_month)
print(output.columns)

# problemas de permisos
#output.limit(10).write.csv("data/sales_agg_product_year_month_limit_10.csv")
output.limit(10).toPandas().to_csv("data/sales_agg_product_year_month_limit_10.csv", index=False)
output.limit(1000).toPandas().to_csv("data/sales_agg_product_year_month_limit_1000.csv", index=False)
output.limit(1000).select(array([output[col] for col in year_month])) \
    .toPandas().to_csv("data/sales_agg_product_year_month_limit_1000_as_array.csv", index=False, header=False)
output.limit(10000).toPandas().to_csv("data/sales_agg_product_year_month_limit_10000.csv", index=False)


# escribir a parquet
#output.write.save('/FileStore/tables/sales_agg_product_year_month', format='parquet', mode='overwrite')

print("Generamos muestras del transaccional mensual dividido en Tarjeta y resto de productos")
# dividir en transaccional + no transaccional
products = get_var_names_info(wide_data, level='product')
print(products)
products_card = ["TC"]
products_feci = [p for p in products if p not in products_card]

wide_data_card_feci = wide_data.fillna(0)

for date in year_month:
    # seleccionamos variables a agrupar
    vars2group_card = \
        [col for col in wide_data.columns if (date in col) and any([p for p in products_card if p in col])]
    vars2group_feci = \
        [col for col in wide_data.columns if (date in col) and any([p for p in products_feci if p in col])]
    # agrupamos inversiones
    expression_card = reduce(lambda x, y: x + " + " + y, vars2group_card)
    expression_feci = reduce(lambda x, y: x + " + " + y, vars2group_feci)
    expression_card = expr(expression_card)
    expression_feci = expr(expression_feci)
    wide_data_card_feci = wide_data_card_feci.withColumn(date + "_card", expression_card)
    wide_data_card_feci = wide_data_card_feci.withColumn(date + "_feci", expression_feci)

output_card_feci = wide_data_card_feci.select(['CUENTA'] + [s + "_card" for s in year_month] +
                                              [s + "_feci" for s in year_month])
print(output_card_feci.columns)

# comprobar que los DFs son coheerentes sin y con separación, al menos para una fecha
sanity = output.join(output_card_feci, on="CUENTA")
assert(sanity.filter((F.col('201601') - F.col('201601_card') - F.col('201601_feci')) > 1e-5).count() == 0)

# para todas las fechas, cudidado con el cache, muy bestia en local
"""
sanity = output.join(output_card_feci, on="CUENTA").cache()
for date in year_month:
    assert(sanity.filter((F.col(date) - F.col(date + '_card') - F.col(date + '_feci')) > 1e-5).count() == 0)

# TODO debería ser sc.catalog.clearCache()
sqlContext.clearCache()
"""


print("Generamos muestras del transaccional mensual dividido en Tarjeta, Linea de crédito y resto de productos")
products = get_var_names_info(wide_data, level='product')
print(products)
products_card = ["TC"]
products_lc = ["LC"]
products_feci = [p for p in products if (p not in products_card) and (p not in products_lc)]

wide_data_card_lc_feci = wide_data.fillna(0)

for date in year_month:
    # seleccionamos variables a agrupar
    vars2group_card = \
        [col for col in wide_data.columns if (date in col) and any([p for p in products_card if p in col])]
    vars2group_lc = \
        [col for col in wide_data.columns if (date in col) and any([p for p in products_lc if p in col])]
    vars2group_feci = \
        [col for col in wide_data.columns if (date in col) and any([p for p in products_feci if p in col])]
    # agrupamos inversiones
    expression_card = reduce(lambda x, y: x + " + " + y, vars2group_card)
    expression_lc = reduce(lambda x, y: x + " + " + y, vars2group_lc)
    expression_feci = reduce(lambda x, y: x + " + " + y, vars2group_feci)
    expression_card = expr(expression_card)
    expression_lc = expr(expression_lc)
    expression_feci = expr(expression_feci)
    wide_data_card_lc_feci = wide_data_card_lc_feci.withColumn(date + "_card", expression_card)
    wide_data_card_lc_feci = wide_data_card_lc_feci.withColumn(date + "_lc", expression_lc)
    wide_data_card_lc_feci = wide_data_card_lc_feci.withColumn(date + "_feci", expression_feci)

output_card_lc_feci = wide_data_card_lc_feci.select(['CUENTA'] + [s + "_card" for s in year_month] +
                                                    [s + "_lc" for s in year_month] +
                                                    [s + "_feci" for s in year_month])
print(output_card_lc_feci.columns)

# guardamos
output_card_lc_feci.limit(10).toPandas().to_csv("data/sales_agg_product_year_month_limit_10_card_lc_feci.csv",
                                                index=False)
output_card_lc_feci.limit(1000).toPandas().to_csv("data/sales_agg_product_year_month_limit_1000_card_lc_feci.csv",
                                                  index=False)
output_card_lc_feci.limit(10000).toPandas().to_csv("data/sales_agg_product_year_month_limit_10000_card_lc_feci.csv",
                                                   index=False)
# para todas las fechas, cudidado con el cache, muy bestia en local
"""
sanity = output.join(output_card_lc_feci, on="CUENTA").cache()
for date in year_month:
    assert(sanity.filter((F.col(date) - F.col(date + '_card') - F.col(date + '_feci')) > 1e-5).count() == 0)

# TODO debería ser sc.catalog.clearCache()
sqlContext.clearCache()
"""
