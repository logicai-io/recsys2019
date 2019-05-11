#!/bin/sh

alias PYPY=~/.pyenv/versions/recsys-pypy/bin/python
alias PY3=~/.pyenv/versions/recsys/bin/python

cd data_generator; PYPY generate_data_parallel_all.py; cd -
PY3 split_events_sorted_trans.py
PY3 vectorize_datasets.py
PY3 model_val.py
PY3 model_submit.py
PY3 make_blend.py
