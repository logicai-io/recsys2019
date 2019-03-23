import joblib
import numpy as np


def jaccard(a, b):
    return len(a & b) / (len(a | b) + 1)


class JaccardItemSim:
    def __init__(self, path):
        self.imm = joblib.load(path)

    def list_to_item(self, other_items, item):
        if other_items:
            return np.mean([jaccard(self.imm[a], self.imm[item]) for a in other_items])
        else:
            return 0

    def two_items(self, a, b):
        if b != 0:
            return jaccard(self.imm[a], self.imm[b])
        else:
            return 0

    def list_to_item_star(self, v):
        other_items, item = v
        return self.list_to_item(other_items, item)

    def two_items_star(self, v):
        a, b = v
        return self.two_items(a, b)

    def list_to_item_star_chunk(self, vs):
        return [self.list_to_item_star(v) for v in vs]

    def two_items_star_chunk(self, vs):
        return [self.two_items_star(v) for v in vs]
