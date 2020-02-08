# Databricks notebook source
# SCRIPT DE UTILS PAAR EDA
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_pcrt_missings(df,
                        output=False):
    """
    obtener porcentaje de valores missing de un DataFrame

    :param df: __DataFrame__
    :param output: __bool__ si es true devuelve el porcentaje de valores missing de df
    :return: __float__ porcentaje de valores missing
    """

    pcrt_missings = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    message = 'Hay un {}% de valores missing'.format(str(np.round(100 * pcrt_missings, 1)))

    if output:
        print(message)
        return pcrt_missings


def get_var_names(df):
    """
    obtener nombres de variables de transaccionalidad a esxlusión del campo CUENTA

    :param df: __DataFrame__ de transaccionalidad
    :return: __list__ con los nombres de las variables
    """

    var_names = df.columns
    if not isinstance(var_names, list):
        var_names = var_names.to_list()
    if 'CUENTA' in var_names:
        var_names.remove('CUENTA')  # quitamos la columna de CUENTA
    if '_NAME_' in var_names:
        var_names.remove('_NAME_')  # quitamos la columna de _NAME_
    return var_names


def get_var_names_info(df,
                       level):
    """
    obtener informacion del numero de variables en cada level

    :param df: __DataFrame__ de transaccionalidad
    :param level: __str__ valor en ['year_month', 'product', 'uneco']
    :return: __Series__ conteo de variables en cada level
    """

    # extraemos nombres de variables de interes
    var_names = get_var_names(df)

    # extraemos informacion de los nombres de variables de transaccionalidad disponibles
    if level == 'year_month':
        output = pd.value_counts([re.split('_', var)[0] for var in var_names])
    elif level == 'uneco':
        output = pd.value_counts([re.split('_', var)[-1] for var in var_names])
    elif level == 'product':
        output = pd.value_counts([re.split('_', var)[1] for var in var_names])
    else:
        print("El valor de level especificado debe ser alguno de los siguientes: year_month, product o uneco")
        output = np.NaN

    output = output.index.to_list()
    output.sort()
    return output


def supress_product_level(df,
                          actualizar=False):
    """
    agrupar variables de inversión de un producto en un mes y uneco en comun

    :param df: __DataFrame__ de transaccionalidad
    :param actualizar: __bool__ si es True, actualiza la salida, si es False lee el último archivo generado
    :return: __DataFrame__ de transaccionalidad sin el nivel producto
    """

    # extraemos nombres de variables de interes
    var_names = get_var_names(df)
    # extraemos informacion adicional de las variables
    unecos = get_var_names_info(df, level='uneco')
    year_month = get_var_names_info(df, level='year_month')

    trns_wo_products = df.copy()
    if actualizar:
        t = time.time()
        for date in year_month:
            for uneco in unecos:
                # seleccionamos todas las variables que coincidan en fecha y uneco
                vars2group = [var for var in var_names if bool(re.search(date, var)) & bool(re.search(uneco, var))]
                if len(vars2group) > 1:
                    trns_wo_products[date + '_' + uneco] = trns_wo_products[vars2group].fillna(0).apply(sum, axis=1)
                else:
                    trns_wo_products[date + '_' + uneco] = trns_wo_products[vars2group].fillna(0)
                # tiramos las variables despues de agrupar
                trns_wo_products = trns_wo_products.drop(columns=vars2group)
                print(date + '_' + uneco)
                print(vars2group)
        # print del tiempo de ejecución
        print("Tiempo de ejecucion: {} min".format((time.time() - t) / 60))
        trns_wo_products.to_csv('data/trns_wo_products.csv', index=False, sep=';')
    else:
        trns_wo_products = pd.read_csv('data/trns_wo_products.csv', sep=";")

    return trns_wo_products


def convert_chunk_to_sparse(chunk, verbose=False):
    """
    convierte el fichero en columnas a formato sparse
    asume una estructura:
    - CUENTA como clave primaria
    - _NAME_ texto (legacy) - se elimina la columna
    - Otras columnas con datos

    :param chunk: fichero (o un subcojunto de filas para evitar problemas de memoria)
    :param verbose: si a True muestra información sobre el chunk
    :return: chunk cambiado a sparse
    """
    # se elimina _NAME_ !
    out = (pd.concat([chunk[['CUENTA']],
                      chunk.drop(['CUENTA', '_NAME_'], axis=1).astype(pd.SparseDtype(np.half, fill_value=0))],
                     axis=1))
    if verbose:
        print(out.info())
    return out


def concat_chunk_tuple(chunk1, chunk2, verbose=True):
    """
    concat de dataframes 2 a 2, se trata de evitar hacer algo como
        trns = pd.concat([convert_chunk_to_sparse(chunk) for chunk in trns_p])
    y sustituirlo por
        trns = reduce(concat_chunk_tuple, [convert_chunk_to_sparse(chunk) for chunk in trns_p])
    sin embargo, siendo el `reduce` mejor que `pd.concat`, sigue habiendo problemas de memoria
    óptimo con un bucle for y haciendo el concat 2 a 2, de manera que no hay que guardar toda
    la lista de chunks

    :param chunk1: va a pd.concat
    :param chunk2: va a pd.concat
    :param verbose: si a True muestra info sobre el df generado
    :return:
    """
    out = pd.concat([chunk1, chunk2])
    if verbose:
        print(out.shape)
        print(out.info())
    return out


def group_variables_by_year_month_level(trns):
    """
    agrupamos sumando por filas las columnas de un mismo mes. Bajamos así el nivel de informacion de las variables
    a mes por año (tiramos informacion de producto y uneco)
    :param trns: __DataFrame__ de transaccional
    :return: trns transformado
    """
    year_month = get_var_names_info(trns, level='year_month')
    t = time.time()  # iniciamos contador
    for date in year_month:
        # seleccionamos variables a agrupar
        vars2group = [col for col in trns.columns if date in col]
        # agrupamos inversiones
        trns[date] = trns[vars2group].sum(axis=1)
    trns = trns[["CUENTA"] + year_month]
    print("Tiempo de ejecución: {}".format(time.time() - t))
    return trns


def counter_jumps(inv_row, window=6):
    """
    contar el numero de meses sin actividad en una ventana de 6 meses
    :param inv_row:
    :param window:  __int__ numero de meses hacia atrás que coger
    :return:
    """
    # calculamos meses inactivos acumulados
    jumps = np.cumsum(inv_row == 0)
    jumps = np.array(([max(jump, 0) for jump in jumps]))
    jumps = jumps[window:] - jumps[:-window]
    # marcamos posiciones de los meses desde el primer mes que hubo inversión
    filter_aux = np.cumsum(inv_row > 0) > 0
    filter_aux = filter_aux[window:]  # si no hay al menos window meses de histórico los ignoramos
    jumps = jumps[filter_aux]
    return list(jumps)


def check_hypothesis1(jumps, window=6):
    """
    comprobar que cuando un cliente alcanza :param windows meses sin actividad, no vuelve a tener actividad (hipotesis
    de abandono 1)
    :param jumps:
    :param window: __int__ numero de meses hacia atrás que coger
    :return:
    """
    if window in jumps:  # ha estado 6 meses sin actividad en algun momento
        # detectamos meses en los que se cumplen a tener 'window' meses de inactividad
        long_inactive = np.array([jump >= window for jump in jumps])
        # filtro de meses desde el primer mes en el que se cumple una inactividad de 'window' meses
        filter_bool = np.cumsum(long_inactive) > 0
        list_aux = [jump for (jump, boolean) in zip(jumps, filter_bool) if boolean]
        output = list_aux == sorted(list_aux)
        return output
    else:
        return np.NaN


def process_hypothesis1(trns, window=6):
    """
    proceso completo para comprobar qué cuentas cumplen la hipotesis1
    :param trns: __DataFrame__ con transaccional
    :param window: __int__ numero de meses a pasado en los que comprobar si se cumple la hipotesis1
    :return:
    """

    print("HIPOTESIS1 CON {} MESES SIN ACTIVIDAD".format(window))

    # 1. contamos meses sin actividad
    print("1. Contamos meses sin actividad")
    t = time.time()
    trns['jumps'] = trns.drop('CUENTA', axis=1).apply(counter_jumps, axis=1, window=window)
    print("... tiempo de ejecución: {}".format(time.time() - t))

    # 2. comprobamos que cuentas (registros) cumplen la hipotesis 1
    print("2. Comprobamos cuentas que cumplen hipotesis1")
    trns['hypothesis1'] = trns.jumps.apply(check_hypothesis1, window=window)
    print(trns['hypothesis1'].value_counts(dropna=False))

    # 3. contamos el número de meses seguidos que se mantiene un parón de 6 meses
    # ... filtramos cuentas que NO contradicen la hipótesis
    trns['count_jumps'] = trns['jumps'].apply(lambda jumps: jumps.count(window))

    return trns


def count_end_values_streak(lst, value):
    """
    contamos número de veces seguidas que un valor aparece al final de una lista
    :param lst: __list__ de integers en la que contar el valor
    :param value: __int__ valor que buscar
    :return: __int__ conteo
    """

    i = len(lst)

    if value in lst:
        # iniciamos conteo
        count = 0
        # miramos desde el final hacia atrás cuantos valores seguidos son 'value'
        while lst[i-1] == value:
            count += 1
            i += -1
        return count
    else:
        return np.NaN


def count_max_jumps_streak(trns, window=6):
    """
    numero de meses seguidos con un parón de al menos 'window' meses al del histórico
    :param trns: __DataFrame__ con transaccional y columna de jumps (output de process_hypothesis1)
    :param window: __int__ numero de meses a pasado en los que comprobar si se cumple la hipotesis1
    :return:
    """
    trns['count_max_jumps'] = trns['jumps'].apply(count_end_values_streak, value=window)

    return trns












