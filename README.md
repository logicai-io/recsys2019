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

