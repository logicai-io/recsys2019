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
8. cd ../../scala
9. SBT_OPTS="-Xms512M -Xmx100G -Xss2M -XX:MaxMetaspaceSize=1024M" sbt run
10. cd ../src/recsys
11. ./make_predictions.sh
12. python make_blend.py

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

Best model is a result of blending of 3 models (see make_blend.py)
MRR: Validation 0.6066 Leaderboard 0.6686

Python dataset models
```
Train AUC 0.9038
Val AUC 0.8915
Val MRR 0.6033

Train AUC 0.8631
Val AUC 0.8724
Val MRR 0.6046
```

Scala dataset models
```
Train AUC 0.8922
Val AUC 0.8831
Val MRR 0.5927

Train AUC 0.8506
Val AUC 0.8601
Val MRR 0.5937
```

Feather
---------------

Run `to_feather.py` in data/ for process raw data into feather files for faster experiments.
`wr_eda.ipynb` in notebooks/ shows some EDA on intersections of IDs and session lengths.
