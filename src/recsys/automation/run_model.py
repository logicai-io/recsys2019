import json
import sys

import click
from recsys.automation.models import run_model
from recsys.automation.utils import download_data, parse_model_instance, upload_data
from recsys.log_utils import get_logger
import os

@click.command()
@click.option("--model_config", type=str)
@click.option("--validation", type=int)
@click.option("--storage_path", type=str)
def main(model_config, validation, storage_path):
    logger = get_logger('/tmp/run.log')
    mat_path = '/tmp/Xcsr.h5'
    meta_path = '/tmp/meta.h5'
    predictions_path = '/tmp/predictions.csv'
    model_path = '/tmp/model.joblib'
    model_config = json.loads(model_config)
    download_data(model_config['dataset_path_matrix'], mat_path)
    download_data(model_config['dataset_path_meta'], meta_path)
    model_instance = parse_model_instance(model_config)
    run_model(mat_path=mat_path,
              meta_path=meta_path,
              model_instance=model_instance,
              predictions_path=predictions_path,
              model_path=model_path,
              val=validation,
              logger=logger)
    upload_data(predictions_path, storage_path)
    upload_data(model_path, storage_path)
    os.system("shutdown now")


if __name__ == '__main__':
    main()
