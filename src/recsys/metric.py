import numpy as np


def calculate_mean_rec_err(queries, clicks):
    recs = []
    for click, query in zip(clicks, queries):
        if query and click in query:
            rec = 1 / (query.index(click) + 1)
            rec = min(1, rec)
            recs.append(rec)
    return np.array(recs).mean()
