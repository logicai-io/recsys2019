#!/bin/sh

python validate_model.py --data python --n_users 1 --n_trees 1600 --action validate
python validate_model.py --data scala --n_users 1 --n_trees 1600 --action validate
python validate_model.py --data python --n_users 1 --n_trees 1600 --action submit
python validate_model.py --data scala --n_users 1 --n_trees 1600 --action submit
