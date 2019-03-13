# recsys2019

Current process:

1. cd data/
2. ./download_data.sh
3. python join_datasets.py
4. cd ../src/recsys
5. python generate_training_data.py (or pypy generate_training_data.py which is 2x faster)
6. python validate_model.py 

The last script generates submission.csv and prints the validation.
