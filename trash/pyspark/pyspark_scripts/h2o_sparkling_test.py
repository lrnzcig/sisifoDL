# import os
# os.environ["PYSPARK_PYTHON"]='C:\\Users\\XXXX\\AppData\\Local\\Continuum\\anaconda3\\envs\\feci\\python.exe'

from pyspark_utils.utils_local import get_spark_context, get_spark_sql_context_from_sc, \
    get_test_data, get_modelling_database, get_modelling_pipeline

from pyspark_utils.utils_h2o import get_h2o_context, get_h2o_train_test

from pyspark_utils.utils_h2o_sparkling import get_example_pysparkling_h2o_grid_search

# 0. Configuración inicial
sc = get_spark_context(log_level="WARN")
sqlContext = get_spark_sql_context_from_sc(sc)


# 1. cargamos datos para generar modelo
trns = get_test_data(sqlContext, 10000)
bbdd = get_modelling_database(trns)
train, test, pipeline_columns = get_modelling_pipeline(bbdd)

# 2. Ajustamos un gbm con h2o
hc = get_h2o_context(sc)
train_h2o, test_h2o = get_h2o_train_test(hc, train, test, pipeline_columns)
gbm_grid = get_example_pysparkling_h2o_grid_search(hc, train_h2o, test_h2o)
