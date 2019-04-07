#!/bin/sh

python generate_training_data.py
python split_events_sorted_trans.py
python vectorize_datasets.py
python model_val.py
python model_submit.py
python make_blend.py
