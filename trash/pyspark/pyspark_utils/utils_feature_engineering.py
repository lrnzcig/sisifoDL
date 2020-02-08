# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType, StringType

import numpy as np
from itertools import chain


def calculate_inv_cv_product_lag(bbdd,
                                 lags,
                                 products=['tc', 'lc', 'rest']):

    """
    genera una variable para cada lag en 'lags' de los valores en la columna 'col_invs_cv' de 'bbdd'
    :param bbdd: __sparkDataFrame__
    :param lags: __list__ últimos valores de inversión que extraer como variables
    :param products: __list__ strings con los productos financieros a distinguir
    """
  
    # para cada producto
    for product in products:

        # para cada lag
        for lag in lags:

            new_var_name = 'inv_' + product + '_cv_month_lag' + str(lag)
            new_var_name = new_var_name.replace('__', '_')
            # variable con las inversiones en cv de cada producto
            inv_cv_name = 'inv_' + product + '_cv_months'
            inv_cv_name = inv_cv_name.replace('__', '_')

            # creamos una variable con el valor en el lag i-ésimo
            bbdd = bbdd.withColumn(new_var_name,
                                   # estraemos la inversión del lag i-ésimo
                                   F.udf(lambda invs, lag:
                                         # si intentamos extraer un lag superior al máximo posible devuelve 0.0
                                         0.0 if len(invs) < lag
                                         else invs[-lag], FloatType())(F.col(inv_cv_name),
                                                                       F.lit(lag)))

    return bbdd


def calculate_n_active_cv_months(bbdd,
                                 products = ['tc', 'lc', 'rest']):
  """
  calcula el número de meses de cv con actividad en cada producto financiero
  :param bbdd: __sparkDataFrame__
  :param products: __list__ strings con los productos financieros a distinguir
  """
  # para cada producto
  for product in products:
    
    # nombre de la nueva variable
    new_var_name = 'n_active_' + product + '_cv_months'
    new_var_name = new_var_name.replace('__', '_')
    # variable con las inversiones en cv de cada producto
    inv_cv_name = 'inv_' + product + '_cv_months'
    inv_cv_name = inv_cv_name.replace('__', '_')
    
    # numero de meses con actividad antes del parón
    bbdd = bbdd.withColumn(new_var_name, 
                           # contamos valores distintos de cero en el vector de inversiones de cv
                           F.udf(lambda x: sum([bool(inv!=0) for inv in x]), 
                                 IntegerType())(F.col(inv_cv_name)))
    
  return bbdd


def calculate_mean_inv_cv_months(bbdd,
                                 products = ['tc', 'lc', 'rest']):
  """
  calcula la media de inversión en los meses de cv de cada producto financiero
  :param bbdd: __sparkDataFrame__
  :param products: __list__ strings con los productos financieros a distinguir
  """
  # para cada producto
  for product in products:
    # nombre de la nueva variable
    new_var_name = 'mean_inv_' + product + '_cv_months'
    new_var_name = new_var_name.replace('__', '_')
    # variable con las inversiones en cv de cada producto
    inv_cv_name = 'inv_' + product + '_cv_months'
    inv_cv_name = inv_cv_name.replace('__', '_')
    
    # numero de meses con actividad antes del parón
    bbdd = bbdd.withColumn(new_var_name, 
                           F.udf(lambda x: 
                                 # si no hay inversiones de cv para algún producto devuelve 0.0
                                 0.0 if (len(x)==0)
                                 else (np.mean(x)).item().__round__(0), 
                                 FloatType())(F.col(inv_cv_name)))
    
  return bbdd


def calculate_min_inv_cv_months(bbdd,
                                products = ['tc', 'lc', 'rest']):
  """
  calcula la mínima de inversión no negativa (ignoramos devoluciones) en los meses de cv de cada producto financiero
  :param bbdd: __sparkDataFrame__
  :param products: __list__ strings con los productos financieros a distinguir
  """
  # para cada producto
  for product in products:
    # nombre de la nueva variable
    new_var_name = 'min_inv_' + product + '_cv_months'
    new_var_name = new_var_name.replace('__', '_')
    # variable con las inversiones en cv de cada producto
    inv_cv_name = 'inv_' + product + '_cv_months'
    inv_cv_name = inv_cv_name.replace('__', '_')
    
    # minima inversión no negativa
    bbdd = bbdd.withColumn(new_var_name, 
                           F.udf(lambda x: 
                                 # ignoramos inversiones negativas (devoluciones)
                                 0.0 if (len([inv for inv in x if inv!=0])==0)
                                 else min([inv for inv in x if inv!=0]), 
                                 FloatType())(F.col(inv_cv_name)))

  return bbdd


def calculate_sum_inv_cv_months(bbdd,
                                products = ['tc', 'lc', 'rest']):
  """
  calcula la suma de inversión en los meses de cv de cada producto financiero
  :param bbdd: __sparkDataFrame__
  :param products: __list__ strings con los productos financieros a distinguir
  """
  for product in products:
    # nombre de la nueva variable
    new_var_name = 'sum_inv_' + product + '_cv_months'
    new_var_name = new_var_name.replace('__', '_')
    # variable con las inversiones en cv de cada producto
    inv_cv_name = 'inv_' + product + '_cv_months'
    inv_cv_name = inv_cv_name.replace('__', '_')

    # suma de la inversión realizada en los meses de cv
    bbdd = bbdd.withColumn(new_var_name, 
                       F.udf(lambda x: sum(x), FloatType())(
                         F.col(inv_cv_name)))

  return bbdd


def calculate_standstill_month_index(bbdd,
                                     products = ['tc', 'lc', 'rest']):
  """
  calcula el índice del mes anterior al parón de cada producto financiero. El índice es 1 en los Eneros, 2 en los Febreros, ... y 12 en los Diciembres
  :param bbdd: __sparkDataFrame__
  :param products: __list__ strings con los productos financieros a distinguir
  """
  
  def calculate_index(inv_cv):
    # si no hay inversiones devuelve 'Missing'
    if len(np.trim_zeros(inv_cv, 'b'))==0:
      return 'Missing'
    else:
      # calculamos numero de meses de cv módulo 12
      inv_cv_mod = np.trim_zeros(inv_cv, 'b')
      monthly_index_inv_cv = len(inv_cv_mod) % 12
      # si el parón empezó en un diciembre corregimos el output a 12, en vez de 0
      if monthly_index_inv_cv == 0:
        return '12'
      else:
        return str(monthly_index_inv_cv)
  
  # para cada producto
  for product in products:
    # nombre de la nueva variable
    new_var_name = 'standstill_' + product + '_month_index'
    new_var_name = new_var_name.replace('__', '_')
    # variable con las inversiones en cv de cada producto
    inv_cv_name = 'inv_' + product + '_cv_months'
    inv_cv_name = inv_cv_name.replace('__', '_')
    
    # calculamos el índice anual del mes
    bbdd = bbdd.withColumn(new_var_name, 
                       F.udf(lambda x: 'standstill_month' + calculate_index(x), 
                             StringType())(F.col(inv_cv_name)))

  return bbdd


def calculate_n_cv_months(bbdd,
                          products = ['tc', 'lc', 'rest']):
  """
  calcula el número de meses de cv antes del parón de cada producto financiero
  :param bbdd: __sparkDataFrame__
  :param products: __list__ strings con los productos financieros a distinguir
  """
  # para cada producto
  for product in products:
    # nombre de la nueva variable
    new_var_name = 'n_'+ product + '_cv_months'
    new_var_name = new_var_name.replace('__', '_')
    # variable con las inversiones en cv de cada producto
    inv_cv_name = 'inv_' + product +'_cv_months'
    inv_cv_name = inv_cv_name.replace('__', '_')
    
    bbdd = bbdd.withColumn(new_var_name, 
                           # quitamos primeros meses de cv sin actividad
                           F.udf(lambda x: len(np.trim_zeros(x, 'f')), 
                                 IntegerType())(F.col(inv_cv_name)))

  return bbdd


def calculate_quantile_inv_cv(bbdd,
                              products = ['tc', 'lc', 'rest'],
                              quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
  """
  calcula el percentil en inversión en los meses de cv antes del parón de cada producto financiero
  :param bbdd: __sparkDataFrame__
  :param products: __list__ strings con los productos financieros a distinguir
  :param quantiles: __list__ floats con los percentiles de inversión de cv a obtener
  """
  # para cada producto
  for product in products:
    # para cada percentil
    for q in quantiles:
      
      # nombre de la nueva variable
      new_var_name = 'quantile' + str(int(q*100)) + '_inv_' + product + '_cv_months'
      new_var_name = new_var_name.replace('__', '_')
      # variable con las inversiones en cv de cada producto
      inv_cv_name = 'inv_' + product + '_cv_months'
      inv_cv_name = inv_cv_name.replace('__', '_')
      
      # calculamos percentil
      bbdd = bbdd.withColumn(new_var_name, 
                         F.udf(lambda x: 
                               # si no hay inversiones en cv devuelve 0.0
                             0.0 if (len(x)==0)
                             else np.quantile(x, q).item(), FloatType())(F.col(inv_cv_name)))

  return bbdd


# ... funcion original
def calculate_campaign_weight_standstill(bbdd, 
                                         count_month_campaigns, 
                                         products = ['tc', 'lc', 'rest']):
  """
  calcula el número de campañas (weight) en el més antes del parón
  :param bbdd: __spark.DataFrame__ 
  :param count_month_campaigns: __dict__ auxiliar con keys = indice del mes (1,2,...,12) y values = número de campañas distintas
  :param products: __list__ strings con los productos financieros a distinguir
  """
  # para cada producto
  for product in products:
    # nombre de la nueva variable
    new_var_name = 'weight_campaign_standstill_' + product + '_month_index'
    new_var_name = new_var_name.replace('__', '_')
    # variable con el mes de parón en cada producto
    standstill_month_index_var_name = 'standstill_' + product + '_month_index'
    standstill_month_index_var_name = standstill_month_index_var_name.replace('__', '_')

    
    # extraemos índice del mes de parón (1 es Enero, 2 es Febrero, ..., 12 es Diciembre)
    bbdd = bbdd.withColumn(new_var_name, 
                           F.regexp_extract(F.col(standstill_month_index_var_name), r'(\d+)$', 1))
    # convertimos al tipo entero
    bbdd = bbdd.withColumn(new_var_name, 
                           F.col(new_var_name).cast('int'))
    # definimos map para sustituir valores con diccionario y aplicamos
    mapping = F.create_map([F.lit(x) for x in chain(*count_month_campaigns.items())])
    bbdd = bbdd.withColumn(new_var_name, 
                           mapping[F.col(new_var_name)])
    # si no hay campañas el peso es 0
    bbdd = bbdd.fillna(0, subset = [new_var_name])
  
  return bbdd

def discretize_value(value,
                     break_values,
                     prefix = 'age_range_'):
  """
  determina el rango de valores de 'break_points' en el que está 'value'
  :param value: __int__ valor a clasificar
  :param break_values: __list__ valores que emplear para la clasificación
  """
  
  # valores auxiliares
  n_ranges = len(break_values)
  min_value = break_values[0]
  max_value = break_values[-1]
  
  # casos nulos
  if value is None:
    return 'missing'
  
  # casos extremos
  if value < min_value:
    # inferior al mínimo de edad declarado
    return prefix + 'under_' + str(min_value)
  if value >= max_value:
    # superior al máximo de edad declarado
    return prefix + 'above_' + str(max_value)
  
  # casos intermedios
  i = 0
  while value >= break_values[i]:
    i += 1
  return prefix + str(break_values[i-1]) + '_' + str(break_values[i])


def discretize_value_udf(_list):
  """
  versión en udf para poder pasar lista de valores ('break_values') como input
  """
  return F.udf(lambda age, prefix: discretize_value(value = age, 
                                                    break_values = _list, 
                                                    prefix = prefix), StringType())


def calculate_age_range(bbdd, 
                        list_ages_break = [18, 22, 28, 35, 40, 50, 60],
                        var_nacimiento = 'NACIMIENTO',
                        var_alta_cliente = 'ALTA_CLIENTE'):
  """
  discretiza la variable de 'var_nacimiento' respecto al rango de valores de 'list_ages_break' en el que están sus valores
  :param bbdd: __sparkDataframe__
  :param list_ages_break: __list__ valores que definen los límites de cada rango de valores (frontera de las categorías)
  :param var_nacimiento: __string__ nombre de la variable de nacimiento a discretizar
  :param var_alta_cliente: __string__ nombre de la variable de alta de cliente
  """
  # calculamos año actual (de ejecución)
  from datetime import date
  current_year = date.today().strftime('%Y')
  
  # creamos edad del cliente
  bbdd = bbdd.withColumn('current_year_age', F.year(F.current_date()) - F.year(var_nacimiento))
  # discretizamos por el rango de edad
  bbdd = bbdd.withColumn('age_range', 
                         discretize_value_udf(list_ages_break)(F.col('current_year_age'),
                                                               F.lit('age_range_')))
  # imputamos clase 'missing' a los registros dudodos o incongruentes
  imput_rows_1900 = F.year(var_nacimiento) == 1900
  imput_rows_1930 = F.year(var_nacimiento) == 1930
  under_age = F.year(var_nacimiento) > (int(current_year) - 18)
  birth_date_af_signup_date = F.year(var_nacimiento) > F.year(var_alta_cliente)

  bbdd = bbdd.withColumn('age_range', F.when(imput_rows_1900 | imput_rows_1930 | under_age | birth_date_af_signup_date, 'missing')\
                         .otherwise(F.col('age_range')))
  
  # eliminamos variable auxiliar
  bbdd = bbdd.drop('current_year_age')
  
  return bbdd


def calculate_seniority_range(bbdd, 
                              list_ages_break = [10, 20, 30, 50],
                              var_alta_cliente = 'ALTA_CLIENTE'):
  """
  discretiza la variable de 'var_alta_cliente'. Para cada cuenta se calcula la antiguedad y se asigna al rango de valores de 
  'list_ages_break' al que pertenezca
  :param bbdd: __sparkDataframe__
  :param list_ages_break: __list__ valores que definen los límites de cada rango de valores (frontera de las categorías)
  :param var_alta_cliente: __string__ nombre de la variable de alta de cliente
  """
  
  # creamos edad del cliente
  bbdd = bbdd.withColumn('seniority', F.year(F.current_date()) - F.year(var_alta_cliente))
  # discretizamos por el rango de edad
  bbdd = bbdd.withColumn('seniority_range',
                         discretize_value_udf(list_ages_break)(F.col('seniority'),
                                                               F.lit('seniority_range_')))
  
  # tiramos variables auxiliares
  bbdd = bbdd.drop('seniority')
  
  return bbdd


