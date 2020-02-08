# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pyspark.sql.functions as F
from pyspark.sql.window import Window as W
from pyspark.sql.functions import udf, monotonically_increasing_id
from pyspark.sql.types import FloatType
from pyspark.sql import DataFrame


def get_profit_curve_df_2_models(predictions_df,
                                 prob_m1_column, prob_m2_column,
                                 cost_m1, p_m1_c1, p_m1_c2,
                                 cost_m2, p_m2_c1, p_m2_c2,
                                 profit_c1, profit_c2,
                                 random_m1_p, random_m2_p,
                                 random_c1_p=None, random_c2_p=None,
                                 budget_m1=None, budget_m2=None,
                                 label_c1_column='label', label_c2_column='inv_label'):
    """
    Genera curvas de coste beneficio a partir de probabilidades resultado del modelo predictivo

    Asume 2 modelos (aunque pueden ser el mismo con la probabilidad invertida); en el caso del
    modelo de reactivación, son las probabilidades de reactivación vs abandono

    Para cada cliente, se escoge la mayor probabilidad de los dos modelos, y se lanza la campaña
    que corresponda (p.e. si se asume que el cliente vva a reactivar, se lanza campaña de reactivación)

    :param predictions_df: __DataFrame__ con probabilidades del modelo y resultado esperado
    :param prob_m1_column: __str__ nombre de columna de `predictions_df` para el modelo 1 -p.e. reactivación
    :param prob_m2_column: __str__ nombre de columna de `predictions_df` para el modelo 2 -p.e. abandono
    :param cost_m1: __float__ coste de la campaña 1 -p.e. reactivación
    :param p_m1_c1: __float__ probabilidad de conversión de un cliente de probabilidad mayor de modelo 1 y que
        realmente es 1
        (p.e. prob. conversión cliente con salida de modelos presume reactivación y realmente es de reactivación)
    :param p_m1_c2: __float__ probabilidad de conversión de un cliente de probabilidad mayor de modelo 1 y que
        realmente es 2
        (p.e. prob. conversión cliente con salida de modelos presume reactivación y realmente es de abandono)
    :param cost_m2: __float__ coste de la campaña 2 -p.e. abandono
    :param p_m2_c1: __float__ probabilidad de conversión de un cliente de probabilidad mayor de modelo 2 y que
        realmente es 1
        (p.e. prob. conversión cliente con salida de modelos presume abandono y realmente es de reactivación)
    :param p_m2_c2: __float__ probabilidad de conversión de un cliente de probabilidad mayor de modelo 2 y que
        realmente es 2
        (p.e. prob. conversión cliente con salida de modelos presume abandono y realmente es de abandono)
    :param profit_c1: __float__ beneficio futuro cliente tipo 1 (p.e. reactivación)
    :param profit_c2: __float__ beneficio futuro cliente tipo 2 (p.e. abandono)
    :param random_m1_p: __float__ para el modelo aleatorio, probabilidad de elegir modelo 1
        (p.e. 0.57 clientes se catalogarían como reactivación en un modelo aleatorio)
    :param random_m2_p: __float__ para el modelo aleatorio, probabilidad de elegir modelo 1
        (p.e. 0.43 clientes se catalogarían como abandono en un modelo aleatorio)
    :param random_c1_p: __float__ para el modelo aleatorio, prob. de que el cliente sea de tipo 1 -p.e. reactivación
        Si a None, se aume igual a `random_m1_p`
    :param random_c2_p: __float__ para el modelo aleatorio, prob. de que el cliente sea de tipo 2 -p.e. abandono
        Si a None, se aume igual a `random_m2_p`
    :param budget_m1: __float__ presupuesto modelo/campaña 1 -p.e. reactivación
    :param budget_m2: __float__ presupuesto modelo/campaña 2 -p.e. abandono
    :param label_c1_column: __str__ nombre de la columna de `predictions_df` de la etiqueta de clientes tipo 1
        (p.e. el valor de la columna será 1 si el cliente realmente es de reactivación)
    :param label_c2_column: __str__ nombre de la columna de `predictions_df` de la etiqueta de clientes tipo 1
        (p.e. el valor de la columna será 1 si el cliente realmente es de reactivación)
    :return: __DataFrame__ con columnas añadidas según sigue y ordenada descendente por probabilidad "max_prob"
        "model": modelo elegido
        "max_prob": probabilidad máxima de los dos clientes para el modelo
        "x": ordinal de las filas
        "profit": beneficio del cliente
        "random": beneficio del modelo aleatorio
        "perfect": beneficio del modelo perfecto
    """
    # inicialización
    if random_c1_p is None:
        random_c1_p = random_m1_p
    if random_c2_p is None:
        random_c2_p = random_m2_p

    # obtener la campaña que va a aplicar a cada cliente a partir de las probabilidades del modelo
    # y ordenar clientes por la máxima probabilidad (según la campaña elegida)
    predictions_df['model'] = predictions_df[[prob_m1_column, prob_m2_column]].idxmax(axis=1)
    predictions_df['max_prob'] = predictions_df[[prob_m1_column, prob_m2_column]].max(axis=1)
    predictions_df = predictions_df.sort_values('max_prob', ascending=False).reset_index(drop=True)

    if (budget_m1 is not None) or (budget_m2 is not None):
        if (cost_m1 == 0) or (cost_m2 == 0):
            raise (BaseException("si se indica presupuesto, los costes no pueden ser 0"))
        predictions_df_1 = predictions_df.loc[predictions_df["model"] == prob_m1_column].reset_index(drop=True)
        predictions_df_2 = predictions_df.loc[predictions_df["model"] == prob_m2_column].reset_index(drop=True)
        predictions_df = predictions_df_1.loc[predictions_df_1.index < (budget_m1 / cost_m1)] \
            .append(predictions_df_2.loc[predictions_df_2.index < (budget_m2 / cost_m2)])
        predictions_df = predictions_df.sort_values('max_prob', ascending=False).reset_index(drop=True)

    # beneficios de cada una de las situaciones posibles
    profits = [profit_c1 * p_m1_c1 - cost_m1,
               profit_c2 * p_m1_c2 - cost_m1,
               profit_c2 * p_m2_c2 - cost_m2,
               profit_c1 * p_m2_c1 - cost_m2]

    # beneficio aculumado de los modelos
    predictions_df['profit'] = \
        np.select([(predictions_df['model'] == prob_m1_column) & (predictions_df[label_c1_column] == 1),  # m1 c1
                   (predictions_df['model'] == prob_m1_column) & (predictions_df[label_c1_column] == 0),  # m1 c2
                   (predictions_df['model'] == prob_m2_column) & (predictions_df[label_c2_column] == 1),  # m2 c2
                   (predictions_df['model'] == prob_m2_column) & (predictions_df[label_c2_column] == 0)],  # m2 c1
                  profits,
                  default=None).cumsum().astype(int)

    # variable x para las gráficas
    predictions_df['x'] = predictions_df.index + 1
    predictions_df = predictions_df.sort_values(by='x')

    # modelo aleatorio: para cada cliente en media el beneficio será lo que marque `random_step`
    if budget_m1 is None:
        random_step = np.dot([random_m1_p * random_c1_p,
                              random_m1_p * random_c2_p,
                              random_m2_p * random_c2_p,
                              random_m2_p * random_c1_p],
                             profits)
    else:
        random_step = np.dot([random_m1_p * (budget_m1 / cost_m1),
                              random_m2_p * (budget_m1 / cost_m1),
                              random_m2_p * (budget_m2 / cost_m2),
                              random_m1_p * (budget_m2 / cost_m2)],
                             profits) / (budget_m1 / cost_m1 + budget_m2 / cost_m2)
    predictions_df['random'] = predictions_df['x'] * random_step

    # modelo perfecto: acierta todas las veces en si un cliente es de reactivación o no
    perfect_profits = np.array([random_m1_p,
                                0,
                                random_m2_p,
                                0]).T * np.array(profits)
    # los primeros `max_profit_threshold` clientes aplicarán el beneficio máximo, mientras que al resto aplicará el
    # beneficio mínimo (es decir, suponemos que ordenamos los clientes para obtener el máximo beneficio
    m1_higher_benefit = np.max(perfect_profits) == perfect_profits[0]
    if budget_m1 is None:
        max_profit_threshold = (random_m1_p if m1_higher_benefit else random_m2_p) * \
                               np.max(predictions_df['x'])
    else:
        max_profit_threshold = (budget_m1 / cost_m1) if m1_higher_benefit else (budget_m2 / cost_m2)

    predictions_df['perfect'] = np.where(predictions_df['x'] <= max_profit_threshold,
                                         np.max(np.array(profits)[perfect_profits != 0]),
                                         np.min(np.array(profits)[perfect_profits != 0])).cumsum()

    # comprobación numérica del total de beneficios del modelo
    total_pr = np.dot([sum((predictions_df["model"] == prob_m1_column) & (predictions_df[label_c1_column] == 1)),
                       sum((predictions_df["model"] == prob_m1_column) & (predictions_df[label_c1_column] == 0)),
                       sum((predictions_df["model"] == prob_m2_column) & (predictions_df[label_c2_column] == 1)),
                       sum((predictions_df["model"] == prob_m2_column) & (predictions_df[label_c2_column] == 0))],
                      profits)

    # añadir valor 0
    predictions_df = predictions_df.append(pd.DataFrame({'x': 0, 'profit': 0, 'perfect': 0},
                                                        index=[-1]),
                                           sort=True)
    predictions_df = predictions_df.sort_values(by='x')

    return predictions_df


def plot_profit_curve_2_models(predictions_df,
                               xlabel="Numero de clientes objetivo",
                               y_columns=['profit'],
                               y_labels=['modelo']):
    """
    Dibuja curvas de coste beneficio para los modelos, añadiendo el modelo aleatorio
    y el modelo perfecto

    :param predictions_df: __DataFrame__ resultado de `get_profit_curve_df_2_models`
    :param xlabel: __str__ literal del eje x
    :param y_columns: __list__ columnas de `predictions_df` que contienen modelos
    :param y_labels: __list__ literales en la leyenda de los modelos de `predictions_df`
    :return: __object__ objeto de matplotlib para dibujar (en `pyspark` hacer display(f)
    """
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    sns.lineplot(x="x", y="perfect", data=predictions_df,
                 ax=ax, label="perfecto")
    sns.lineplot(x="x", y="random", data=predictions_df,
                 ax=ax, label="aleatorio")
    for i in range(0, len(y_columns)):
        sns.lineplot(x="x", y=y_columns[i], data=predictions_df,
                     ax=ax, label=y_labels[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Beneficios acumulados")
    return fig


probability_of_one = udf(lambda v: float(v[1]), FloatType())


def get_model_results_to_driver(model, test,
                                relative_path=None,
                                model_name=None):
    """
    TODO TODO TODO asume que el modelo es de reactivacion
    TODO también asume que la variable objetivo es "label"

    :param predictions:
    :return:
    """
    predictions_raw = model.transform(test)

    predictions = predictions_raw.withColumn('prob_react',
                                         probability_of_one(F.col('probability')))

    predictions = predictions.withColumn('correct',
                                         F.when(predictions.label == predictions.prediction, 1).otherwise(0))
    print('Precisión en Test : {}'.format(predictions.agg(F.mean('correct')).collect()[0]["avg(correct)"]))

    if relative_path is not None:
      predictions.write.mode("overwrite").parquet(relative_path + model_name + "_predictions.parquet")
      predictions_raw.write.mode("overwrite").parquet(relative_path + model_name + "_predictions_raw.parquet")

    return _prepare_df_for_driver(predictions)


def _prepare_df_for_driver(predictions,
                           label_reactive="reactive"):
    out = predictions.select('prob_react', 'label').toPandas()
    out['prob_aban'] = 1 - out['prob_react']
    if out['label'].dtype not in ["int", "int32"]:
        if label_reactive not in out.groupby('label').count().index:
            raise RuntimeError("Re-define label_reactive (currently set to " + str(label_reactive) + ")")
        out['label'] = np.where(out['label'] == label_reactive, 1, 0)
    out['inv_label'] = 1 - out['label']
    return out


def get_predictions_to_driver(sqlContext, parquet_file_name):
    """
    Obtener predicciones a partir de un fichero parquet y llevar al driver

    :param sqlContext:
    :param parquet_file_name: path completo dbfs
    :return:
    """
    predictions = sqlContext.read.format('parquet').load(parquet_file_name)
    print('Precisión en Test : {}'.format(predictions.agg(F.mean('correct')).collect()[0]["avg(correct)"]))
    return _prepare_df_for_driver(predictions)


def get_model_results_to_driver_h2o(model, test,
                                    sqlContext, hc,
                                    parquet_file_name=None,
                                    label_reactive='reactive'):
    """
    TODO TODO TODO asume que el modelo es de reactivacion
    TODO también asume que la variable objetivo es "label"

    :param model:
    :param test: puede ser H2O o no! TODO TODO TODO
    :param sqlContext:
    :param hc:
    :param parquet_file_name:
    :param label_reactive:
    :return:
    """
    # convertimos a typo H2OFrame
    if isinstance(test, DataFrame):
        # @Deprecated
        raise RuntimeError("Deprecated!")
    else:
        # TODO refactorizar después de quitar el @Deprecated
        test_h2o = test
        test = hc.as_spark_frame(test_h2o)

    # cambiamos el tipo de la variable objetivo
    test_h2o['label'] = test_h2o['label'].asfactor()

    h2o_predictions = sqlContext.createDataFrame(model.predict(test_h2o).as_data_frame())
    test_bis = test.withColumn("row_id", monotonically_increasing_id())
    test_bis = test_bis.withColumn("row_id", F.row_number().over(W.orderBy("row_id")))
    h2o_predictions = h2o_predictions.withColumn("row_id", monotonically_increasing_id())
    h2o_predictions = h2o_predictions.withColumn("row_id", F.row_number().over(W.orderBy("row_id")))
    h2o_predictions = test_bis.join(h2o_predictions, "row_id").drop("row_id")

    predictions = h2o_predictions.withColumnRenamed(label_reactive, 'prob_react')
    if predictions.count() != test_bis.count():
        raise RuntimeError("Panic !!! -malformed join join with row_id")

    predictions = \
        predictions.withColumn('correct',
                               F.when(h2o_predictions.label == h2o_predictions.predict, 1).otherwise(0))
    print('Precisión en Test : {}'\
          .format(predictions.agg(F.mean('correct')).collect()[0]["avg(correct)"]))
    
    if parquet_file_name is not None:
      predictions.write.mode("overwrite").parquet(parquet_file_name)

    return _prepare_df_for_driver(predictions, label_reactive='reactive')


def merge_profit_curves(list_of_profit_curves,
                        model_suffixes):
    out = list_of_profit_curves[0][['x', 'profit', 'random', 'perfect']]\
        .rename(columns={'profit': 'profit_' + model_suffixes[0]})
    for i in range(1, len(list_of_profit_curves)):
        out = pd.merge(out,
                       list_of_profit_curves[i][['x', 'profit']]\
                       .rename(columns={'profit': 'profit_' + model_suffixes[i]}),
                       on='x')
    return out
