# import os
# os.environ["PYSPARK_PYTHON"]='C:\\Users\\XXXX\\AppData\\Local\\Continuum\\anaconda3\\envs\\feci\\python.exe'

from pyspark.ml.classification import LogisticRegression  # , LogisticRegressionModel

import matplotlib.pyplot as plt

from utils.utils_bus_eval import get_profit_curve_df_2_models, plot_profit_curve_2_models, \
    get_model_results_to_driver, get_model_results_to_driver_h2o, merge_profit_curves

from pyspark_utils.utils_local import get_spark_context, get_spark_sql_context_from_sc,\
    get_test_data, get_modelling_database, get_modelling_pipeline

from pyspark_utils.utils_h2o import get_h2o_context, get_h2o_train_test, get_example_h2o_grid_search

sc = get_spark_context()
sqlContext = get_spark_sql_context_from_sc(sc)

trns = get_test_data(sqlContext, 10000)

bbdd = get_modelling_database(trns)

train, test, pipeline_columns = get_modelling_pipeline(bbdd)

lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=1000)
lrModel = lr.fit(train)

# lrModel.write().overwrite().save("models/ml_logit_initial")
# lrModel = LogisticRegressionModel.load("models/lr_simple")

# precisión en Test
predictions_sorted_driver = get_model_results_to_driver(lrModel, test)

# import pickle
# with open("test/data/predictions_sorted_driver.pickle", "wb") as f:
#    pickle.dump(predictions_sorted_driver, f, protocol=pickle.HIGHEST_PROTOCOL)

# parámetros de coste-beneficio
coste_camp_react = 1
coste_camp_aband = 15
benef_clie_react = 50
benef_clie_aband = 400
prob_camp_react_clie_react = 0.1
prob_camp_react_clie_aband = 0.01
prob_camp_aband_clie_aband = 0.075
prob_camp_aband_clie_react = 0.25
baseline_react = 0.57

# presupuesto
presupuesto_react = 50 # a coste 1, 50 clientes
presupuesto_aban = 90 # a coste 15,  6 clientes

# visualización de los modelos
predictions_df_2_models = get_profit_curve_df_2_models(predictions_sorted_driver,
                                                       prob_m1_column='prob_react', prob_m2_column='prob_aban',
                                                       cost_m1=coste_camp_react, p_m1_c1=prob_camp_react_clie_react,
                                                       p_m1_c2=prob_camp_react_clie_aband,
                                                       cost_m2=coste_camp_aband, p_m2_c1=prob_camp_aband_clie_react,
                                                       p_m2_c2=prob_camp_aband_clie_aband,
                                                       profit_c1=benef_clie_react, profit_c2=benef_clie_aband,
                                                       random_m1_p=baseline_react, random_m2_p=1-baseline_react)  # c1 c. reactivan c2 c. abandonan
f = plot_profit_curve_2_models(predictions_df_2_models, xlabel="Número de clientes objetivo")
plt.show()

# visualización con presupuesto
predictions_df_2_models_bis = get_profit_curve_df_2_models(predictions_sorted_driver,
                                                           prob_m1_column='prob_react', prob_m2_column='prob_aban',
                                                           cost_m1=coste_camp_react, p_m1_c1=prob_camp_react_clie_react,
                                                           p_m1_c2=prob_camp_react_clie_aband,
                                                           cost_m2=coste_camp_aband, p_m2_c1=prob_camp_aband_clie_react,
                                                           p_m2_c2=prob_camp_aband_clie_aband,
                                                           profit_c1=benef_clie_react, profit_c2=benef_clie_aband,
                                                           budget_m1=presupuesto_react, budget_m2=presupuesto_aban,
                                                           random_m1_p=0.57, random_m2_p=0.43)  # c1 c. reactivan c2 c. abandonan
f = plot_profit_curve_2_models(predictions_df_2_models_bis, xlabel="Número de clientes objetivo (con presupuesto limitado)")
plt.show()

# mmodelo generado con H2O (ejecutar read_write_pickle.py)
hc = get_h2o_context(sc)
train_h2o, test_h2o = get_h2o_train_test(hc, train, test, pipeline_columns)
gbm_grid = get_example_h2o_grid_search(train_h2o, test_h2o)

best_h2o_gbt_model = gbm_grid.models[0]

predictions_h2o_sorted_driver = get_model_results_to_driver_h2o(best_h2o_gbt_model, test_h2o,
                                                                sqlContext, hc)

predictions_h2o_2_models = get_profit_curve_df_2_models(predictions_h2o_sorted_driver,
                                                        prob_m1_column='prob_react', prob_m2_column='prob_aban',
                                                        cost_m1=coste_camp_react, p_m1_c1=prob_camp_react_clie_react,
                                                        p_m1_c2=prob_camp_react_clie_aband,
                                                        cost_m2=coste_camp_aband, p_m2_c1=prob_camp_aband_clie_react,
                                                        p_m2_c2=prob_camp_aband_clie_aband,
                                                        profit_c1=benef_clie_react, profit_c2=benef_clie_aband,
                                                        random_m1_p=0.57, random_m2_p=0.43)  # c1 c. reactivan c2 c. abandonan

predictions_merged = merge_profit_curves([predictions_df_2_models, predictions_h2o_2_models],
                                         ["lr", "gbt_h2o"])
f = plot_profit_curve_2_models(predictions_merged,
                               xlabel="Número de clientes objetivo",
                               y_columns=['profit_lr', 'profit_gbt_h2o'],
                               y_labels=["logistic regression", "gradient boosting tree h2o"])
plt.show()

predictions_h2o_2_models_bis = get_profit_curve_df_2_models(predictions_h2o_sorted_driver,
                                                            prob_m1_column='prob_react', prob_m2_column='prob_aban',
                                                            cost_m1=coste_camp_react,
                                                            p_m1_c1=prob_camp_react_clie_react,
                                                            p_m1_c2=prob_camp_react_clie_aband,
                                                            cost_m2=coste_camp_aband,
                                                            p_m2_c1=prob_camp_aband_clie_react,
                                                            p_m2_c2=prob_camp_aband_clie_aband,
                                                            profit_c1=benef_clie_react, profit_c2=benef_clie_aband,
                                                            budget_m1=presupuesto_react, budget_m2=presupuesto_aban,
                                                            random_m1_p=0.57, random_m2_p=0.43)  # c1 c. reactivan c2 c. abandonan
predictions_merged_bis = merge_profit_curves([predictions_df_2_models_bis, predictions_h2o_2_models_bis],
                                             ["lr", "gbt_h2o"])
f = plot_profit_curve_2_models(predictions_merged_bis,
                               xlabel="Número de clientes objetivo (con presupuesto",
                               y_columns=['profit_lr', 'profit_gbt_h2o'],
                               y_labels=["logistic regression", "gradient boosting tree h2o"])
plt.show()

