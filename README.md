# recsys2019

Current process
---------------

1. cd data/
2. ./download_data.sh
3. cd ../../src/recsys/data_prep
4. ./run_data_prep.sh
5. cd ..
6. python generate_training_data.py (or pypy generate_training_data.py which is 2x faster)
7. python split_events_sorted_trans.py (pypy is good)
8. python vectorize_datasets.py
9. ./make_predictions.sh
10. python make_blend.py

Best submissions
---------------

Best model 1 ranking model (LGBMRanker) model_2_val.py / model_2_submit.py
MRR: Validation 0.6162 Leaderboard 0.6740

Python dataset models
```
Train AUC 0.8903
Val AUC 0.8845
Val MRR 0.6162
```

Feather
---------------

Run `to_feather.py` in data/ for process raw data into feather files for faster experiments.
`wr_eda.ipynb` in notebooks/ shows some EDA on intersections of IDs and session lengths.
