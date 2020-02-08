# Databricks notebook source
def get_trns_data(base_path,
                  name_splitted_trns,
                  name_agg_trns = None):
  """
  lectura de datos de transaccional dividido por producto y agregado
  :param base_path: ruta base que apunta a los ficheros de transaccional
  :param name_splitted_trns: nombre del fichero de transaccional dividido por producto
  :param name_agg_trns: nombre del fichero de transaccional agregado
  """
  
  # lectura de transaccional dividido por productos
  trns = sqlContext.read.format('parquet').load(base_path + name_splitted_trns)
  
  if name_agg_trns:
    # lectura de transaccional agregado
    trns_aggregated = sqlContext.read.format('parquet').load(base_path + name_agg_trns)
    return trns, trns_aggregated
  
  return trns


def get_customer_data(base_path,
                      name_customer_data,
                      trns = None):
  """
  lectura de datos de transaccional dividido por producto y agregado
  :param base_path: ruta base que apunta a los ficheros de transaccional
  :param name_customer_data: nombre del fichero con datos personales
  :param trns: __sparkDataFrame__ con info de transaccional (no importa si está o no agregado)
  """
  
  # lectura de datos personales
  customer_data = sqlContext.read.format('parquet').load(base_path + name_customer_data)
  
  # Check: cuentas de las que no tenemos Datos de Clientes
  if trns:
    accounts_wo_customer_info = trns.join(customer_data, on = 'CUENTA', how = 'left_anti')
    print('Número de cuentas de las que tenemos disponible la info de transaccional pero no la personal: {}'\
          .format(accounts_wo_customer_info.count()))
  
  return customer_data


def get_campaign_info():
  """
  generar diccionario con el número de campañas diferentes que hay en cada mes. Keys: índice del mes, values: número de campañas
  """
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
  
  return count_month_campaigns


def filter_portuguese_accounts(trns):
  """
  eliminamos cuentas de Portugal
  :param trns: transaccional desagregado
  """
  
  # Check: conteo de cuentas de Portugal
  n_PT_accounts = trns.where(F.when(F.col('CUENTA').rlike(r'^(9|0095).*'), True)\
                             .otherwise(False)).count()
  print('Numero de cuentas de Portugal filtradas de la tabla de transaccional alojada en el BS: {}'\
        .format(n_PT_accounts))
  
  # eliminamos cuentas de portugal
  trns = trns.where(F.when(F.col('CUENTA').rlike(r'^(9|0095).*'), False)\
                    .otherwise(True))
  
  return trns


def get_model_samples(trns,
                      trns_aggregated,
                      param_standstill,
                      param_safe_inactivity_window,
                      param_cv):
  """
  filtramos registros de la tabla del transaccional que tengan un parón de 'param_standstill' meses, al menos 'param_safe_inactivity_window' meses
  de histórico después del parón y al menos 'param_cv' meses de histórico anterior al parón. Además de eso etiquetamos los registros en función de
  si se reactivan después del parón (1) o si no reactivan (0) en una ventana de 'param_safe_inactivity_window' meses
  :param trns: __sparkDataframe__ transaccional desagregado
  :param trns_aggregated: __sparkDataframe__ transaccional agregado
  :param param_standstill: __int__ número de meses sin actividad para considerar parón
  :param param_safe_inactivity_window: __int__ mínimo número de meses de histórico necesarios después del parón
  :param param_cv: __int__ mínimo número de meses de histórico necesarios antes del parón
  """
  
  # seleccionamos cuentas objetivo del modelo
  out = data_preparation_cv_safe(trns_aggregated,
                                 param_standstill = param_standstill,
                                 param_safe_inactivity_window = param_safe_inactivity_window,
                                 param_cv = param_cv)
  
  # determinamos qué cuentas validad (abandonan) o no (reactivan) la hipótesis de abandono en 12 meses
  out = process_hypothesis1(out, 
                            window = param_standstill,
                            safe_inactivity_window = param_safe_inactivity_window)
  
  # seleccionamos solo CUENTA y la columna con el target
  out = out.select('CUENTA', 'stricted_hypothesis1')\
    .withColumnRenamed('stricted_hypothesis1', 'label')\
    .withColumn('label',
                when(F.col('label') == 'False', 1)
                .when(F.col('label') == 'True', 0))
  
  # print info
  print('Número de cuentas total en la muestra: {}'.format(out.count()))
  out.groupby('label').count().show()
  
  # añadimos la columna con el target a la tabla con el transaccional desagregado
  trns = trns.join(out, on = 'CUENTA', how = 'inner')  # con el inner join filtramos a los de Portugal en trns
  
  return trns


def create_unified_bbdd(trns,
                        customer_data,
                        create_copy = False):
  """
  consolida el __sparkDataframe__ con todas las variables a poder emplear en la modelización
  :param trns: __sparkDataframe__ transaccional desagregado
  :param customer_data: __sparkDataframe__ transaccional agregado
  :param create_copy: __bool__ si es True genera una copia de bbdd como segundo argumento (útil para rescatar 
  versión de bbdd antes del feature engineering)
  """
  
  # unimos información adicional sobre el transaccional
  bbdd = trns.join(
    customer_data, on = 'CUENTA', how = 'inner')\
    .cache()
  
  # punto de control
  if create_copy:
    bbdd_copy = bbdd
    # forzamos la copia de cache con el action '.count'
    print('Número de filas para entrenar: {}'.format(bbdd_copy.count()))
  else:
    bbdd_copy = None
    # forzamos la copia de cache con el action '.count'
    print('Número de filas para entrenar: {}'.format(bbdd.count()))
  
  return bbdd, bbdd_copy


def count_null_values_per_column(bbdd,
                                 column_names,
                                 save_output = False):
  """
  cuenta el número de valores nulos de las variables en 'column_names' de 'bbdd'
  :param bbdd: __sparkDataframe__
  :param column_names: __list__ nombres de las variables a analizar
  :param save_output: __bool__ si es True devuelve la tabla con los conteos
  """
  
  # comprobamos campos nulos
  null_count = bbdd.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in column_names]).toPandas()
  
  if save_output:
    return null_count
  else:
    display(null_count)
  
  
  
