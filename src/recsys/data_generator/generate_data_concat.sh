#!/bin/sh

python generate_training_data.py --limit 1000000 --hashn 0 &
python generate_training_data.py --limit 1000000 --hashn 1 &
python generate_training_data.py --limit 1000000 --hashn 2 &
python generate_training_data.py --limit 1000000 --hashn 3 &
python generate_training_data.py --limit 1000000 --hashn 4 &
python generate_training_data.py --limit 1000000 --hashn 5 &
python generate_training_data.py --limit 1000000 --hashn 6 &
python generate_training_data.py --limit 1000000 --hashn 7 &

