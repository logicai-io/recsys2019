# recsys2019

Current process (full)
----------------------

1. cd data/
2. ./download_data.sh
3. cd ../../src/recsys/data_prep
4. ./run_data_prep.sh
5. cd ..
6. python generate_training_data.py (or pypy generate_training_data.py which is 2x faster)
7. python split_events_sorted_trans.py (pypy is good)
8. python vectorize_datasets.py
9. python model_2_val.py (validate model)
10. python model_2_submit.py (make test predictions)
11. python make_blend.py (prepare submission file)

Quick validation (1 million rows)
----------------------

1. cd data/
2. ./download_data.sh
3. cd ../../src/recsys/data_prep
4. ./run_data_prep.sh
5. cd ..
6. python generate_training_data.py --limit 1000000
7. python quick_validate.py

Best submissions
---------------

Last improvements
```
1m validation   0.6196 -> 0.6233
full validation 0.6174 -> 0.6226
leaderboard     0.6750 -> 0.6776
```

Feather
---------------

Run `to_feather.py` in data/ for process raw data into feather files for faster experiments.
`wr_eda.ipynb` in notebooks/ shows some EDA on intersections of IDs and session lengths.
