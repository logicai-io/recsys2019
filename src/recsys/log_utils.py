import logging


def get_logger():
    log = logging.getLogger()
    ch = logging.StreamHandler()
    log.addHandler(ch)
    fh = logging.FileHandler("recsys.log")
    log.addHandler(fh)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    return log
