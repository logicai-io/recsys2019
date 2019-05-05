import pandas as pd
import numpy as np
from recsys.transformers import FeaturesAtAbsoluteRank

def test_absolute_rank():
    trans = FeaturesAtAbsoluteRank()

    df = pd.DataFrame({
        'clickout_id': [1,1,1,2,2,2],
        'rank': [1,2,3,1,2,3],
        'x1': [1,2,3,4,5,6],
        'x2': [6,5,4,3,2,1],
    })

    X_ = trans.fit_transform(df)
    assert np.allclose(X_['x1_rank_1'],np.array([1,1,1,4,4,4]))


if __name__ == '__main__':
    test_absolute_rank()