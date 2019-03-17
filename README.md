# recsys2019

Current process:

1. cd data/
2. ./download_data.sh
3. python join_datasets.py
4. python convert_item_metadata_to_sets.py
5. python extract_hotel_dense_features.py
6. cd ../src/recsys
7. python generate_training_data.py (or pypy generate_training_data.py which is 2x faster)
8. python validate_model.py 

The last script generates submission.csv and prints the validation.
