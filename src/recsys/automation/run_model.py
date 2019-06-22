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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "run", "recsys.log")
    logger = get_logger(log_path)
    mat_path = os.path.join(dir_path, "run", "Xcsr.h5")
    meta_path = os.path.join(dir_path, "run", "meta.h5")
    predictions_path = os.path.join(dir_path, "run", "predictions.csv")
    model_path = os.path.join(dir_path, "run", "model.joblib")
    model_config = json.loads(model_config)
    model_config["validation"] = validation
    config_path = os.path.join(dir_path, "run", "config.json")
    with open(config_path, "wt") as out:
        out.write(json.dumps(model_config))
    download_data(model_config["dataset_path_matrix"], mat_path)
    download_data(model_config["dataset_path_meta"], meta_path)
    model_instance = parse_model_instance(model_config)
    run_model(
        mat_path=mat_path,
        meta_path=meta_path,
        model_instance=model_instance,
        predictions_path=predictions_path,
        model_path=model_path,
        val=validation,
        logger=logger,
    )
    upload_data(predictions_path, storage_path + "predictions.csv")
    upload_data(model_path, storage_path + "model.joblib")
    upload_data(config_path, storage_path + "config.json")
    upload_data(log_path, storage_path + "recsys.log")
    os.system("sudo shutdown now")


if __name__ == "__main__":
    main()
