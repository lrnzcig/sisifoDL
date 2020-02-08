import numpy as np
import pandas as pd
import unittest
from utils.utils_eda import convert_chunk_to_sparse, get_var_names_info


def counter_jumps(inv_row, window=6):
    # calculamos meses inactivos acumulados
    jumps = np.cumsum(inv_row == 0)
    jumps = np.array(([max(jump, 0) for jump in jumps]))
    jumps = jumps[window:] - jumps[:-window]
    # marcamos posiciones de los meses desde el primer mes que hubo inversión
    filter_aux = np.cumsum(inv_row > 0) > 0
    filter_aux = filter_aux[window:]  # si no hay al menos 6 meses de histórico los ignoramos
    jumps = jumps[filter_aux]
    return list(jumps)


class TestCounterJumps(unittest.TestCase):

    def setUp(self):
        trns_p = pd.read_csv('../data/MODELO_ABANDONO_4A.csv', sep=";", nrows=10)
        trns = convert_chunk_to_sparse(trns_p)

        # agrupamos variables a nivel mensual (tiramos info de producto y uneco)
        self.year_month = get_var_names_info(trns, level='year_month')
        for date in self.year_month:
            # seleccionamos variables a agrupar
            vars2group = [col for col in trns.columns if date in col]
            # agrupamos inversiones
            trns[date] = trns[vars2group].sum(axis=1)
        self.trns_ym = trns[["CUENTA"] + self.year_month]

    def test1(self):
        r = self.trns_ym[self.trns_ym.CUENTA == 1016][self.year_month].values[0]
        res = counter_jumps(r)
        self.assertEqual(res,
                         [2, 2, 2, 2, 2, 2, 2, 1, 1, 0,
                          0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         "error caso de prueba ventana 6")

    def test2(self):
        print(self.trns_ym.iloc[1][self.year_month].values)

        self.assertEqual(0, 1, "error")


if __name__ == '__main__':
    unittest.main()
