# recsys2019

Current process:

1. cd data/
2. ./download_data.sh
3. python join_datasets.py
4. python convert_item_metadata_to_sets.py
5. cd ../src/recsys
6. python generate_training_data.py (or pypy generate_training_data.py which is 2x faster)
7. python validate_model.py 

The last script generates submission.csv and prints the validation.

### WR:

Run `to_feather.py` in data/ for process raw data into feather files for faster experiments.
`wr_eda.ipynb` in notebooks/ shows some EDA on intersections of IDs and session lengths.
