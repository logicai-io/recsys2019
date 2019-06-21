#!/bin/sh

sudo su pawel

alias PY3=~/.pyenv/versions/recsys/bin/python
alias PIP=~/.pyenv/versions/recsys/bin/pip

PIP install google-cloud-storage

cd /home/pawel/recsys2019/src/recsys/automation
git stash
git checkout automation

MODEL_CONFIG=$(curl http://metadata/computeMetadata/v1/instance/attributes/model_config -H "Metadata-Flavor: Google")
VALIDATION=$(curl http://metadata/computeMetadata/v1/instance/attributes/validation -H "Metadata-Flavor: Google")
STORAGE_PATH=$(curl http://metadata/computeMetadata/v1/instance/attributes/storage_path -H "Metadata-Flavor: Google")

screen
PY3 run_model.py --model_config "$MODEL_CONFIG" --validation $VALIDATION --storage_path $STORAGE_PATH
