#!/bin/sh

alias PYPY=~/.pyenv/versions/recsys-pypy/bin/python
alias PY3=~/.pyenv/versions/recsys/bin/python

PYPY generate_training_data.py
PYPY split_events_sorted_trans.py
PY3 vectorize_datasets.py
PY3 model_val.py
PY3 model_submit.py
PY3 make_blend.py
