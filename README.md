# recsys2019

Current process:

1. cd data/
2. ./download_data.sh
3. python join_datasets.py
4. cd ../src/recsys
5. python generate_training_data.py
6. python validate.py 

The last script generates submission.csv and prints the validation.

`data/to_arrow.py` creates .parquet files for quick loading.
