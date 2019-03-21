# recsys2019

Current process
---------------

1. cd data/
2. ./download_data.sh
3. python join_datasets.py
4. python convert_item_metadata_to_sets.py
5. python extract_hotel_dense_features.py
6. cd ../src/recsys
7. python generate_training_data.py (or pypy generate_training_data.py which is 2x faster)

Model validation
---------------

Selecting 100k users from training and combining with test users
```
python validate_model.py --n_users 100000 --action validate 
```

Selecting 100k rows from training (useful for quick testing)
```
python validate_model.py --n_debug 100000 --action validate 
```

Model validation
---------------

Selecting 100k users from training and combining with test users - creates `submission.csv`
```
python validate_model.py --n_users 100000 --action submit
```

Selecting all users from training and combining with test users - creates `submission.csv`
```
python validate_model.py --action submit
```

Best submissions
---------------

Best model is a result of this command (the parameter of models n_trees is set to 1600.

```bash
(recsys) pawel@recsys1:~/recsys2019/src/recsys$ python validate_model.py --n_users 1 --action validate
Mem. usage decreased to 49.51 Mb (46.2% reduction)
n_users=1
action=validate
n_jobs=-2
reduce_df_memory=True
load_feather=False
Mem. usage decreased to 11337.71 Mb (33.6% reduction)
Training on 116896 users
Training data shape (8839810, 47)
[Reading training data] done in 473 s
[splitting timebased] done in 3 s
7955877it [00:36, 215301.22it/s]
item_similarity_to_last_clicked_item 0.2920764232514977
7955877it [04:18, 30741.17it/s]
avg_similarity_to_interacted_items 0.4049730973529921
7955877it [02:43, 48549.74it/s]
avg_similarity_to_interacted_session_items 0.3985459158905435
[calculating item similarity] done in 541 s
883933it [00:04, 218947.22it/s]
item_similarity_to_last_clicked_item 0.28455394435164905
883933it [00:28, 31340.12it/s]
avg_similarity_to_interacted_items 0.3762883052824564
883933it [00:18, 48943.24it/s]
avg_similarity_to_interacted_session_items 0.39230373573458166
[calculating item similarity] done in 63 s
0.8921479669737543
0.8875941007492508
7955877it [00:46, 169526.62it/s]
item_similarity_to_last_clicked_item 0.2920764232514977
7955877it [05:18, 25001.45it/s]
avg_similarity_to_interacted_items 0.4049730973529921
7955877it [02:44, 48413.63it/s]
avg_similarity_to_interacted_session_items 0.3985459158905435
[calculating item similarity] done in 639 s
883933it [00:04, 212546.62it/s]
item_similarity_to_last_clicked_item 0.28455394435164905
883933it [00:28, 31313.83it/s]
avg_similarity_to_interacted_items 0.3762883052824564
883933it [00:17, 49704.59it/s]
avg_similarity_to_interacted_session_items 0.39230373573458166
[calculating item similarity] done in 66 s
0.8458901479367109
0.8542202805284564
MRR 0.591433
[validating models] done in 32968 s
```

Leaderboard result 0.6585


Feather
---------------

Run `to_feather.py` in data/ for process raw data into feather files for faster experiments.
`wr_eda.ipynb` in notebooks/ shows some EDA on intersections of IDs and session lengths.
