#!/bin/sh

alias PYPY=~/.pyenv/versions/recsys-pypy/bin/python
alias PY3=~/.pyenv/versions/recsys/bin/python

PYPY generate_data_v2_run_acc.py
PY3 generate_data_v2_vectorize.py
