# Databricks notebook source
import numpy as np
from itertools import islice
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
from pyspark.sql.functions import array, when
from pyspark.sql.types import ArrayType, FloatType, IntegerType, BooleanType, StringType


# ... funcion original
def counter_jumps(inv_row, window=6):
    """
    contar el numero de meses sin actividad en una ventana de 'window' meses hacia atrás

    :param inv_row: fila del DataFrame con las columnas de inversiones mensuales
    :param window:  __int__ numero de meses hacia atrás que coger
    :return: __list__ número de meses sin actividad
    """
    # borrar 0s iniciales para empezar a contar desde el primer mes con info
    inv_row_clean = np.trim_zeros(inv_row, 'f')
    # suma los 0s en la ventana entre z y z+window,
    # donde z va del inicio de inv_row hasta el final menos la ventana
    out = [sum([(i == 0) for i in islice(inv_row_clean, z, z + window)])
           for z in range(0, len(inv_row_clean) - window + 1)]
    return (out)


# ... udf version
counter_jumps_udf = udf(counter_jumps, ArrayType(IntegerType()))  # StringType()


# ... funcion original
def check_hypothesis1(jumps, window=6):
    """
    comprobar que cuando un cliente alcanza 'window' meses sin actividad, no vuelve a tener actividad (hipotesis de
    abandono 1)

      :param jumps:
      :param window: __int__ numero de meses hacia atrás que coger
      :return:
    """
    # detectamos meses en los que se cumplen a tener 'window' meses de inactividad
    if window in jumps:  # ha estado 'window' meses sin actividad en algun momento
        # detectamos meses en los que se cumplen a tener 'window' meses de inactividad
        long_inactive = np.array([jump >= window for jump in jumps])
        # filtro de meses desde el primer mes en el que se cumple una inactividad de 'window' meses
        filter_bool = np.cumsum(long_inactive) > 0
        list_aux = [jump for (jump, boolean) in zip(jumps, filter_bool) if boolean]
        output = list_aux == sorted(list_aux)
        return output
    else:
        return None


# ... udf version
check_hypothesis1_udf = udf(check_hypothesis1, BooleanType())


# ... funcion original
def count_end_values_streak(lst, value=6):
    """
    contamos número de veces seguidas que un valor aparece al final de una lista

    :param lst: __list__ de integers en la que contar el valor
    :param value: __int__ valor que buscar
    :return: __int__ conteo
    """

    # añadimos un elemento nulo al principio de lista para evitar bucle infinito (cuando
    # lst = [value,value, ..., value])
    lst = [None] + lst

    i = len(lst)
    if value in lst:
        # iniciamos conteo
        count = 0
        # miramos desde el final hacia atrás cuantos valores seguidos son 'value'
        while lst[i - 1] == value:
            count += 1
            i += -1
        return count
    else:
        return None


# ... udf version
count_end_values_streak_udf = udf(count_end_values_streak, IntegerType())


# ... funcion original
def calculate_avg_monthly_inv(inv_row, trim_trailing_zeros=False):
    """
    calculamos la inversion media mensual de cada cuenta desde el primer mes de actividad

    :param inv_row: fila del DataFrame con las columnas de inversiones
    :param trim_trailing_zeros: si True, no se tienen en cuenta los 0's al final de la vida
        de la cuenta
    :return: __float__ la media de inversión mensual
    """
    # meses de la cuenta desde la primera actividad registrada
    trim = 'f'
    if trim_trailing_zeros:
        trim += 'b'
    return np.mean(np.trim_zeros(inv_row, trim=trim)).item()


# ... udf version
calculate_avg_monthly_inv_udf = udf(calculate_avg_monthly_inv, FloatType())


# ... funcion original
def calculate_account_life(inv_row):
    """
    calcular el número de meses de histñorico que hay desde la primera actividad de la cuenta
    :param inv_row: fila del DataFrame con las columnas de inversiones
    :return: __int__ con el numero de meses
    """
    # meses de la cuenta desde la primera actividad registrada
    return len(np.trim_zeros(inv_row, 'f'))


# ... udf version
calculate_account_life_udf = udf(calculate_account_life, IntegerType())


def count_initial_months_wo_activity(inv_row):
    """
    numero de meses iniciales seguidos sin inversion
    :param inv_row: fila del DataFrame con inversiones
    :return:
    """
    i = 0
    while inv_row[i] == 0:
        i += 1

    return i


# ... funcion original
def calculate_n_months_cv(inv_row, param_standstill):
    """
    calcular el número de meses entre la primera actividad y el primer parón de 'param_standstill' meses. Si no hay
    parón devuelve 'None'
    :param inv_row: fila del DataFrame con inversiones
    :param param_standstill: __int__ número de meses seguidos sin inversión para considerar un parón
    :return: __int__ con el numero de meses
    """
    inv_row_clean = np.trim_zeros(inv_row, 'f')  # limpiar 0's iniciales
    max_result = len(inv_row_clean)
    i = 0
    inv_row_slice = inv_row_clean
    while (i < max_result):
        try:
            i += inv_row_slice.index(0)
        except ValueError:
            # no hay más 0 en la serie
            return max_result
        number_of_zeros_starting_index = sum([(i == 0) for i in islice(inv_row_clean, i, i + param_standstill)])
        if (number_of_zeros_starting_index == param_standstill):
            return i
        inv_row_slice = np.trim_zeros(inv_row_clean[i:], 'f')
        i = len(inv_row_clean) - len(inv_row_slice)
    return max_result


# ... udf version
calculate_n_months_cv_udf = udf(calculate_n_months_cv, IntegerType())


def check_posterior_inactivity(hypotesis1, count_max_jumps, safe_inactivity_window=6):
    """
    Comprueba que en una ventana de seguridad hay inactividad

    :param hypotesis1: valor de columna
    :param count_max_jumps: valor de columna
    :param safe_inactivity_window: ventana de colcón
    :return:
    """
    if hypotesis1 and (count_max_jumps < safe_inactivity_window):
        return ('doubt')
    else:
        return (str(hypotesis1))


# ... udf version
check_posterior_inactivity_udf = udf(check_posterior_inactivity, StringType())


def process_hypothesis1(trns,
                        window=6,
                        safe_inactivity_window=6):
    # calculamos meses de parón
    trns = trns.withColumn('jumps',
                           counter_jumps_udf(array([trns[column] for column in trns.columns if column != 'CUENTA']),
                                             F.lit(window)))
    # check hipótesis1
    trns = trns.withColumn('hypothesis1', check_hypothesis1_udf(trns['jumps'], F.lit(window)))

    # calculamos meses de parón seguidos al final del histórico
    trns = trns.withColumn('count_max_jumps', count_end_values_streak_udf(trns['jumps'], F.lit(window)))
    trns = trns.withColumn('stricted_hypothesis1', check_posterior_inactivity_udf(trns['hypothesis1'],
                                                                                  trns['count_max_jumps'],
                                                                                  F.lit(safe_inactivity_window)))
    return (trns)


def divide_sample_by_avg_month_inv(trns,
                                   trim_trailing_zeros=False,
                                   bound=62.5):
    '''
    Divide la muestra de 'trns' en 2 DataFrames; Cuentas con inversión media mensual superior e inferior a 'bound'

    SE UTILIZA SOLO EN LA V1 DE LA HIPÓTESIS 2, que carece de interés

    trns: __DataFrame__ de transaccionalidad
    trim_trailing_zeros: __boolean__ si a True, se excluyen de los cálculos de medias los 0s finales
    bound: __float__ inversion mensual media que divide la población de cuentas
    result: __tuple__ DataFrames; (cuentas de alto valor, cuentas de bajo valor)
    '''

    # calculamos inversion media anual
    trns = trns.withColumn('avg_monthly_inv',
                           calculate_avg_monthly_inv_udf(
                               array([trns[column] for column in trns.columns if column != 'CUENTA']),
                               F.lit(trim_trailing_zeros))
                           )

    # etiquetamos clientes de alto valor y bajo valor
    label_customer_value = when(F.col('avg_monthly_inv') > bound, 'high').otherwise('low')
    trns = trns.withColumn('label_customer_value', label_customer_value)

    # dividimos la muestra en 2 poblaciones: cuentas de valor alto y cuentas de valor bajo
    trns_high = trns.filter(trns.label_customer_value == 'high')
    trns_low = trns.filter(trns.label_customer_value == 'low')

    # eliminamos columnas auxiliares
    trns_high = trns_high.drop('label_customer_value', 'avg_monthly_inv')
    trns_low = trns_low.drop('label_customer_value', 'avg_monthly_inv')

    return (trns_high, trns_low)


# ... funcion original
def calculate_n_months_safe_inactivity(inv_row,
                                       param_standstill):
    """
    si hay un parón, calcula el número de meses de histórico que hay después del parón. Si no hay parón devuelve None

    :param inv_row: fila del DataFrame con inversiones
    :param param_standstill: __int__ numero de meses seguidos sin actividad para considerar parón
    :return:
    """
    # 0. check input
    if len(inv_row) < param_standstill:
        return None

    # 1. detectar si hay o no paron de al menos 'param_standstill' meses sin actividad
    jumps = counter_jumps(inv_row, window=param_standstill)
    exist_standstill = param_standstill in jumps

    # 2. meses de vida total de la cuenta
    n_months_account_life = calculate_account_life(inv_row)

    # 3. meses de cv (antre la primera actividad y el inicio del primer parón
    n_months_cv = calculate_n_months_cv(inv_row, param_standstill)

    if exist_standstill:
        # numero de meses total de vida de la cuenta menos los meses antes del parón (cv) y los del parón.
        # Nota: Restamos el 'param_standstill', que son el mínimo numero de meses sin actividad para considerar un
        # parón, el parón real que experimenta la cuenta puede ser mayor
        return n_months_account_life - n_months_cv - param_standstill
    else:
        return None


# ... udf version
calculate_n_months_safe_inactivity_udf = udf(calculate_n_months_safe_inactivity, IntegerType())


def data_preparation_cv_safe(trns,
                             param_standstill=6,
                             param_safe_inactivity_window=6,
                             param_cv=12):
    """
    Filtra los datos de cuentas:
    - que tengan histórico suficiente (meses param_cv + param_stanstill + param_safe_inactivity_window)
    - si tienen parada, que tengan suficiente param_cv y param_safe_inactivity_window

    :param trns: __DataFrame__
    :param param_standstill: __int__ meses de inactividad
    :param param_safe_inactivity_window: __int__ colchón después de la inactividad
    :param param_cv: __int__ meses con compras
    :return: __DataFrame__ aplicados los filtros
    """
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

    # 2. Eliminar cuentas que no tienen parón y que tienen menos de 'param_standstill' + 'param_safe_inactivity_window' +
    # 'param_cv' meses de vida (no hay suficiente histórico)
    # ... calculamos meses de vida
    trns = trns.withColumn('n_months_account_life',
                           calculate_account_life_udf(F.array([trns[column] for column in aux_list_inv_columns])))
    # ... filtramos ('~' sirve para negar)
    trns = trns.filter(trns.n_months_account_life >= (param_cv +
                                                      param_standstill +
                                                      param_safe_inactivity_window))  # registros con suficiente histórico

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
    trns = trns.filter(  # negamos
        trns.exist_standstill &  # hay parón y ...
        (trns.n_months_cv >= param_cv) &  # hay suficientes meses antes del paron
        (trns.n_months_safe_inactivity >= param_safe_inactivity_window)  # hay suficientes meses despues
    )

    # ... eliminamos variables auxiliares creadas para filtrar una muestra de cuentas homogeneas
    trns = trns.drop('jumps', 'exist_standstill', 'n_months_account_life', 'n_months_cv', 'n_months_safe_inactivity')

    return (trns)


# ... funcion original
def cumulative_inactive_months(inv_row):
    """
    meses seguidos sin actividad alcanzados en cada mes de histñorico desde el primer mes de actividad

    :param inv_row:
    :return:
    """

    # check casos especiales
    # ... Nunca hay actividad
    if all(inv == 0 for inv in inv_row):
        return None

    # 0. eliminar primeros meses de inactividad
    inv_row = np.trim_zeros(inv_row, 'f')

    # 1. detectar qué meses sin actividad
    months_wo_activity = [bool(inv == 0) for inv in inv_row]

    # 2. contar el numero de meses seguidos sin actividad
    aux_count = 0  # cuenta de meses sin actividad seguidos
    cumulative_inactivity = []
    for flag_no_activity in months_wo_activity:
        if flag_no_activity:
            aux_count += 1
        else:
            aux_count = 0  # cuando pasamos por un mes con actividad la cuenta se resetea

        cumulative_inactivity.append(aux_count)

    return cumulative_inactivity


# ... udf version
cumulative_inactive_months_udf = udf(cumulative_inactive_months, ArrayType(IntegerType()))


def process_hypothesis_2_safe_cv_high_low(trns,
                                          param_standstill,
                                          param_safe_inactivity_window=6,
                                          param_cv=12,
                                          high_low_trim_trailing_zeros=False,
                                          high_low_bound=125,
                                          hypothesis_function=process_hypothesis1):
    """
    Procesa la hipótesis 2 (dividiendo entre clientes de alto y bajo valor) pero además
    haciendo el filtrado en `data_preparation_cv_safe`

    :param trns: __DataFrame__
    :param param_standstill: __int__ meses de inactividad
    :param param_safe_inactivity_window: __int__ colchón después de la inactividad
    :param param_cv: __int__ meses con compras
    :param high_low_trim_trailing_zeros: si a True, a la hora de calcular media de compras, no se cuentan los 0's finales
    :param high_low_bound: límite de media de compras para considerar un cliente alto / bajo valor
    :param hypothesis_function: función para validar hipótesis sobre clientes alto / bajo valor
    :return:
    """
    trns_prepared = data_preparation_cv_safe(trns,
                                             param_standstill=param_standstill,
                                             param_safe_inactivity_window=param_safe_inactivity_window,
                                             param_cv=param_cv)

    if high_low_bound is None:
        return hypothesis_function(trns_prepared, param_standstill), None

    trns_high_prepared, trns_low_prepared = divide_sample_by_avg_month_inv(trns_prepared,
                                                                           trim_trailing_zeros=high_low_trim_trailing_zeros,
                                                                           bound=high_low_bound)

    trns_w_h = hypothesis_function(trns_high_prepared, param_standstill)
    trns_w_l = hypothesis_function(trns_low_prepared, param_standstill)
    return trns_w_h, trns_w_l

# ... función original
def check_1st_consecutive_inactive_streaks(v_cumulative_inactive_months,
                                           streak_1st=12):
    """
    contar el numero de meses sin actividad seguidos al primer parón de 'streak_1st' meses sin actividad

    :param v_cumulative_inactive_months: vector de meses seguidos sin actividad
    :param streak_1st: __int__ numero de meses seguidos sin actividad para considerar un primer parón
    :return:
    """

    # 0. si no hay parón de 'streak_1st' meses salimos
    if streak_1st not in v_cumulative_inactive_months:
        return None
    n_months = len(v_cumulative_inactive_months)

    # 1. Detectar el índice de la primera racha de 12 meses sin actividad
    i = 0
    while v_cumulative_inactive_months[i] != streak_1st:
        i += 1
    ind_1st_streak = i

    # 2. Índice de meses seguidos sin actividad a continuación del primer parón de 'streak_1st' meses
    while (v_cumulative_inactive_months[i] >= streak_1st) & ((i+1) < n_months):
        i += 1
    ind_max_streak = i - 1

    # 3. Número de meses sin actividad seguidos al parón
    n_streak_inactive_months = ind_max_streak - ind_1st_streak

    return n_streak_inactive_months


# ... udf version
check_1st_consecutive_inactive_streaks_udf = udf(check_1st_consecutive_inactive_streaks, IntegerType())


def count_model_candidates(trns,
                           param_standstill=12,
                           param_cv=12,
                           lags=range(1, 13)):
    """
    Cuenta de candidatos para el modelo de reactivación
    Cumplen que han tenido una ventana de parada de duración `param_standstill`, han tenido una compra exactamente un
    mes antes de la ventana de parada, y tienen cifra de venta al menos `param_cv` meses antes de la parada

    :param trns: transacciones
    :param param_standstill: tamaño de ventana de parada
    :param param_cv: tamaño de ventana dd compras
    :param lags: ventanas para las que se calcula
    :return: __dict__ resultados por lag
    """
    aux_list_inv_columns = trns.columns
    aux_list_inv_columns.remove('CUENTA')

    trns_mod = trns.withColumn('inv_row', array(
        [trns[column] for column in aux_list_inv_columns]))
    trns_mod = trns_mod.withColumn('jumps',
                                   counter_jumps_udf(trns_mod['inv_row'],
                                                     F.lit(param_standstill))).cache()

    res = {}

    def last_jump(x, lag):
        return x[-lag] if len(x) >= lag else 0
    last_jump_udf = udf(last_jump, IntegerType())

    def last_cv(x, lag):
        return x[-lag] > 0
    last_cv_udf = udf(last_cv, BooleanType())

    accounts = None
    for lag in lags:
        trns_loc = trns_mod.withColumn('last_jumps', last_jump_udf(trns_mod['jumps'],
                                                                   F.lit(lag)))
        trns_loc = trns_loc.filter(F.col('last_jumps') == param_standstill)  # última ventana de parón de X meses
        trns_loc = trns_loc.withColumn('last_cv_udf', last_cv_udf(trns_loc['inv_row'],
                                                                  F.lit(param_standstill+lag)))
        trns_loc = trns_loc.filter(F.col('last_cv_udf'))  # han tenido una compra hace X+lag meses
        trns_loc = trns_loc.withColumn('n_months_account_life',
                                       calculate_account_life_udf(trns_loc['inv_row']))
        min_account_life = param_cv + param_standstill + lag - 1
        trns_loc = trns_loc.filter(F.col('n_months_account_life') >= min_account_life)  # tienen una ventana de cv mayor que 12 meses
        if accounts is None:
            accounts = trns_loc.select(['CUENTA']).distinct()
        else:
            accounts = accounts.join(trns_loc.select(['CUENTA']).distinct(),
                                     on='CUENTA', how="outer")
            accounts = accounts.select(['CUENTA']).distinct()
        res[lag] = trns_loc.count()
        #print(accounts.count())
        #print(sum(res.values()))
        if sum(res.values()) != accounts.count():
            print(accounts.count())
            print(sum(res.values()))
            print(res)
            raise RuntimeError("Panic !!!!")
    #print("final")
    #print(accounts.head(5))
    trns_mod.unpersist()
    # TODO transformar según display(pd.DataFrame({'mes': list(res.keys()), 'count': list(res.values())}))
    return res


# ... funcion original
def extract_cv_months(inv_row,
                      param_standstill=12):
    """
    Extraemos las inversiones en los meses de cv (meses antes del primer parón)
    :param inv_row: __array__ con todas las inversiones
    :param param_standstill: __int__ meses seguidos sin actividad para considerar parón
    #:param jumps: __list__ con los meses sin actividad (depende del valor de param_standstill tomado)
    :return: __list__ con las inversiones
    """

    # filtramos las inversiones desde el primer mes con actividad
    inv_row_mod = np.trim_zeros(inv_row, 'f')
    n_trim_zeros = len(inv_row) - len(inv_row_mod)
    # detectamos el primer parón de param_standstill meses
    i = 0
    while not all([bool(inv == 0) for inv in inv_row_mod[(0+i):(param_standstill+i)]]):
        i += 1  # vamos moviendo la ventana hasta encontrar el primer parón
    # filtramos inversiones hasta el primer parón de standstill meses
    output = inv_row[:(n_trim_zeros + i)]

    return output


# ... udf version
extract_cv_months_udf = udf(extract_cv_months, ArrayType(FloatType()))

