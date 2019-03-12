import array

import numpy as np
from scipy import sparse


class CooMatrixBuilder:
    def __init__(self, col_map={}, fit=True):
        self.rows = array.array("L")
        self.cols = array.array("L")
        self.data = array.array("f")
        self.col_map = col_map or {}
        self.col_names = []
        self.row_map = {}
        self.row_names = []
        self.fit = fit

    def _get_next_col_id(self):
        return len(self.col_map)

    def _get_next_row_id(self):
        return len(self.row_map)

    def add(self, row, col, val):
        # if columns name not in map
        # add it if it is fitting stage
        # else ignore it
        if col not in self.col_map:
            if self.fit:
                self.col_map[col] = self._get_next_col_id()
                self.col_names.append(col)
            else:
                return

        if row not in self.row_map:
            self.row_map[row] = self._get_next_row_id()

        self.rows.append(self.row_map[row])
        self.cols.append(self.col_map[col])
        self.data.append(val)
        return

    def get_coo_matrix(self):
        shape = (len(self.row_map), len(self.col_map))
        return sparse.coo_matrix(
            (np.array(self.data), (np.array(self.rows), np.array(self.cols))),
            shape=shape,
        )
