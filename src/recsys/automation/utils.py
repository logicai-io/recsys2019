import calendar
import hashlib
import time

from google.cloud import storage
from lightgbm import LGBMRanker, LGBMRankerMRR, LGBMRankerMRR2, LGBMRankerMRR3


def parse_model_instance(model_config):
    model_class = model_config["model_class"]
    model_params = model_config["model_params"]
    if model_class == "LGBMRanker":
        model_instance = LGBMRanker(**model_params)
    elif model_class == "LGBMRankerMRR":
        model_instance = LGBMRankerMRR(**model_params)
    elif model_class == "LGBMRankerMRR2":
        model_instance = LGBMRankerMRR2(**model_params)
    elif model_class == "LGBMRankerMRR3":
        model_instance = LGBMRankerMRR3(**model_params)
    else:
        assert False
    return model_instance


def download_data(src_path, dst_path):
    client = storage.Client()
    bucket = client.get_bucket("logicai-recsys2019")
    blob = bucket.get_blob(src_path)
    blob.download_to_filename(dst_path)


def upload_data(src_path, dst_path):
    client = storage.Client()
    bucket = client.get_bucket("logicai-recsys2019")
    blob = bucket.blob(dst_path)
    blob.upload_from_filename(src_path)


def get_timestamp():
    return calendar.timegm(time.gmtime())


def str_to_hash(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
