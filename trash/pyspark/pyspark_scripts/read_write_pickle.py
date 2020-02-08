# import os
# os.environ["PYSPARK_PYTHON"]='C:\\Users\\XXXX\\AppData\\Local\\Continuum\\anaconda3\\envs\\feci\\python.exe'
#
# https://youtrack.jetbrains.com/issue/PY-30628?_ga=2.9692339.766777321.1579518226-1779259590.1551095517

from pyspark_utils.utils_local import get_spark_context, get_spark_sql_context_from_sc, \
    get_test_data, get_modelling_database, get_modelling_pipeline

from pyspark_utils.utils_h2o import get_h2o_context, get_h2o_grid_summary_as_dataframe, \
    grid_save_all_models, get_aml_leaderboard_as_dataframe, get_h2o_train_test, \
    get_example_h2o_grid_search

sc = get_spark_context()
sqlContext = get_spark_sql_context_from_sc(sc)

trns = get_test_data(sqlContext, 10000)

bbdd = get_modelling_database(trns)

train, test, pipeline_columns = get_modelling_pipeline(bbdd)

hc = get_h2o_context(sc)
train_h2o, test_h2o = get_h2o_train_test(hc, train, test, pipeline_columns)
gbm_grid = get_example_h2o_grid_search(train_h2o, test_h2o)

# escribir summary a DF
summary_df = get_h2o_grid_summary_as_dataframe(sqlContext, gbm_grid, test_h2o)
summary_df.show()

# grabar todos los modelos del grid
grid_save_all_models(gbm_grid, "models/")

# TODO escribir y leer a un RDD, pruebas de otros objetos como pickles
import pickle
from collections import namedtuple
Record = namedtuple("Record", ["object_name", "content"])

"""
gbm_grid_to_object = [Record(c, (pickle.dumps(getattr(gbm_grid, c)()) \
                                 if callable(getattr(gbm_grid, c)) else pickle.dumps(getattr(gbm_grid, c))))
                      for c in ["summary"]]
                      #for c in dir(gbm_grid) if (c[0] != "_") and (c not in ["aic", "biases", "algo_params",
                      #                                                       "build_model", "catoffsets",
                      #                                                       "coefficients_table"])]
gbm_grid_rdd = sc.parallelize(gbm_grid_to_object)
gbm_grid_df = gbm_grid_rdd.map(lambda rec: Record(rec.object_name, bytearray(rec.content))).toDF()
"""
"""
object_names = [getattr(row, "object_name") for row in gbm_grid_df.collect()]
for object_name in object_names:
    print(object_name)
    print(pickle.loads(getattr(gbm_grid_df.filter(F.col("object_name") == object_name).collect()[0], "content")))
    print("==>")
"""
# TODO gbm_grid_df.write.parquet("dbfs:/FileStore/xxxx")
#gbm_grid_reloaded = {getattr(row, "object_name"): pickle.loads(getattr(row, "content")) for row in gbm_grid_df.collect()}


# Run AutoML for X seconds
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_runtime_secs = 60)
aml.train(x = train_h2o.drop('label').columns,
          y = 'label',
          training_frame = train_h2o
         )
# Print Leaderboard (ranked by xval metrics)
aml.leaderboard.summary()
# (Optional) Evaluate performance on a test set
perf = aml.leader.model_performance(test_h2o)
perf.auc()

# grabar el modelo líder
get_aml_leaderboard_as_dataframe(sqlContext, aml,
                                 relative_path="models/")


# guardar parámetros del objeto AML TODO revisar si algo es necesario
aml_to_object = [Record(c, pickle.dumps(getattr(aml, c))) for c in dir(aml)
                 if (c[0] != "_") and (c not in ["detach", "download_mojo",
                                                 "download_pojo", "event_log",
                                                 "leader", "leaderboard",
                                                 "modeling_steps", "predict",
                                                 "train"])]
aml_rdd = sc.parallelize(aml_to_object)
aml_df = aml_rdd.map(lambda rec: Record(rec.object_name, bytearray(rec.content))).toDF()
# TODO aml_df.write.parquet("dbfs:/FileStore/xxxx")
"""
object_names = [getattr(row, "object_name") for row in aml_df.collect()]
for object_name in object_names:
    print(object_name)
    print(pickle.loads(getattr(aml_df.filter(F.col("object_name") == object_name).collect()[0], "content")))
    print("==>")
    """
aml_reloaded = {getattr(row, "object_name"): pickle.loads(getattr(row, "content")) for row in aml_df.collect()}
aml_reloaded.keys()
aml_reloaded["project_name"]
