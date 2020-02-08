from pyspark.sql import functions as F


def process_hypothesis_windows(trns, windows,
                               hypothesis_function):
    """
    Genera tabla resumen de resultados para diferentes valores de ventana de parada

    :param trns: __DataFrame__
    :param windows: __list__ lista de meses de ventana sobre la que hacer el cálculo
    :param hypothesis_function: función de cálculo
    :return: __DataFrame__ resumen de resultados
    """
    out = None
    for window in windows:
        trns_w = hypothesis_function(trns, window)
        out = _group_by_hyp(out, trns_w, window)
    return(out)


def process_hypothesis_windows_high_low_cv_safe(trns, windows,
                                                hypothesis_function,
                                                param_standstill=None,
                                                param_safe_inactivity_window=6,
                                                param_cv=12,
                                                high_low_trim_trailing_zeros=False,
                                                high_low_bound=125):
    """
    Genera tabla resumen de resultados para diferentes valores de ventana de parada y dividiendo
    a los clientes en alto / bajo valor

    :param trns: __DataFrame__
    :param windows: __list__ lista de meses de ventana sobre la que hacer el cálculo
    :param hypothesis_function: función de cálculo
    :param param_standstill: __int__ ventana de parada; si a None, se ejecuta una iteración por cada valor de `windows`
    :param param_safe_inactivity_window: __int__ colchón después de la inactividad; si a None, se ejecuta una iteración
        por cada valor de `windows`
    :param param_cv: __int__ meses con compras
    :param high_low_trim_trailing_zeros: si a True, a la hora de calcular media de compras, no se cuentan los 0's finales
    :param high_low_bound: límite de media de compras para considerar un cliente alto / bajo valor
    :return: __DataFrame__ resumen de resultados
    """
    out_high = None
    out_low = None
    for window in windows:
        # elegir a qué parámetro afecta la ventana (dependiendo de cuál esté a None)
        if param_standstill is None:
            param_standstill_loop = window
            param_safe_inactivity_window_loop = param_safe_inactivity_window
        elif param_safe_inactivity_window is None:
            param_standstill_loop = param_standstill
            param_safe_inactivity_window_loop = window

        trns_w_h, trns_w_l = hypothesis_function(trns,
                                                 param_standstill=param_standstill_loop,
                                                 param_safe_inactivity_window=param_safe_inactivity_window_loop,
                                                 param_cv=param_cv,
                                                 high_low_trim_trailing_zeros=high_low_trim_trailing_zeros,
                                                 high_low_bound=high_low_bound)
        out_high = _group_by_hyp(out_high, trns_w_h, window)
        out_low = _group_by_hyp(out_low, trns_w_l, window)
    return(out_high, out_low)


def _group_by_hyp(out, trns_w, window):
    """
    Función de uso interno
    A partir de los resultados de los cálculos, genera una nueva columna para la tabla resumen

    :param out: __DataFrame__ tabla resumen, contiene columnas para los resultados calculados hasta este punto
    :param trns_w: __DataFrame__ filas por cliente con cálculo de resultados para un valor de ventaa
    :param window: __int__ tamaño de ventana
    :return: __DataFrame__ out con una columna más
    """
    if trns_w is None:
        return out
    w_w = trns_w.groupby(trns_w.stricted_hypothesis1) \
        .count() \
        .select(F.col('stricted_hypothesis1'),
                F.col('count').alias('count_w' + str(window))).fillna("null")
    if out is None:
        out = w_w
    else:
        out = out.join(w_w, on='stricted_hypothesis1', how='outer')
    return out
